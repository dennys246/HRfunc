import scipy.linalg, scipy.stats, json, mne, random, re, os, nilearn, time
import numpy as np
import matplotlib.pyplot as plt
from .hrtree import tree, HRF, _flatten_context_value
from ._utils import standardize_name, _is_oxygenated, _LIB_DIR
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from itertools import compress
from glob import glob


def localize_hrfs(nirx_obj, max_distance = 0.01, verbose = False, **kwargs):
    """
    Localize HRFs to the optodes in a fNIRS montage given a nirx object

    Arguments:
        nirx_obj (mne raw object) - NIRS file loaded through mne
        max_distance (float) - Maximum distance in milimeter's a previously estimated HRF can be attached to an optode
        verbose (bool) - If True, print out extra information during localization
        **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)
    
    Returns:
        montage (montage object) - Montage object with localized HRF's
    """
    # Build a montage
    _montage = montage(nirx_obj, **kwargs)
    _montage.localize_hrfs(max_distance, verbose = verbose)
    return _montage

def load_montage(json_filename, rich = False, **kwargs):
    """ 
    Load montage with the given json filename 
    
    Arguments:
        json_filename (str) - Path to json file to load montage from
        rich (bool) - If True, load the full HRF information including estimates and locations, if False only load mean and std
        **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)
    
    Returns:
        montage (montage object) - Montage object with loaded HRF's
    """
    # Read in json
    with open(json_filename, 'r') as file:
        json_contents = json.load(file)

    if not isinstance(json_contents, dict) or len(json_contents) == 0:
        raise ValueError(
            f"load_montage: {json_filename!r} must contain a non-empty JSON "
            "object keyed by '<ch_name>-<doi>'"
        )

    # Initialize an empty montage object
    _montage = montage(**kwargs)

    # Grab info from json contents
    first_key = next(iter(json_contents))
    first_hrf = json_contents[first_key]
    if not isinstance(first_hrf, dict) or 'sfreq' not in first_hrf:
        raise ValueError(
            f"load_montage: first entry {first_key!r} is missing required "
            "field 'sfreq'"
        )
    sfreq = first_hrf['sfreq']

    # Assess channel names
    ch_names = ['-'.join(key.split('-')[:-1]) for key in json_contents.keys()]

    # Assess which channels are oxygenated and deoxygenated
    _montage.hbo_channels = [ch for ch in ch_names if _is_oxygenated(ch) == True]
    _montage.hbr_channels = [ch for ch in ch_names if _is_oxygenated(ch) == False]

    # Update montage with saved info.
    # M5: any failure inside the per-entry loop aborts the whole load. The
    # local _montage is dropped on the exception path so the caller never
    # sees a half-populated object; no explicit rollback is needed because
    # _montage and its trees are unreferenced after we raise.
    required_top = ('hrf_mean', 'hrf_std', 'sfreq', 'location', 'context')
    for key, channel in json_contents.items():
        try:
            key_split = key.split('-')
            doi = key_split.pop()
            ch_name = '-'.join(key_split)

            # Skip if canonical HRF
            if ch_name == 'canonical':
                continue

            if not isinstance(channel, dict):
                raise ValueError(
                    f"entry {key!r} must be a JSON object, "
                    f"got {type(channel).__name__}"
                )
            for field in required_top:
                if field not in channel:
                    raise ValueError(
                        f"entry {key!r} is missing required field {field!r}"
                    )
            if not isinstance(channel['context'], dict):
                raise ValueError(
                    f"entry {key!r} has non-object 'context' field"
                )
            if 'duration' not in channel['context']:
                raise ValueError(
                    f"entry {key!r} is missing required field "
                    "'context.duration'"
                )

            if rich == False:
                channel['estimates'] = []
                channel['locations'] = []
            else:
                for field in ('estimates', 'locations'):
                    if field not in channel:
                        raise ValueError(
                            f"entry {key!r} is missing required field "
                            f"{field!r} (rich=True)"
                        )

            # Create an HRF node from the saved channel. We must pass
            # channel['context'] through — pre-fix this was omitted, so
            # every loaded HRF fell back to the default template context
            # inside HRF.__init__. After NE-002 the hasher was populated
            # correctly with channel context VALUES, but the node itself
            # carried no task/stimulus/demographics metadata, so
            # compare_context / filter / branch comparisons downstream
            # silently failed to match on the real values. Caught by the
            # cross-branch audit on fix/tree-edge-cases.
            estimated_hrf = HRF(
                doi,
                ch_name,
                channel['context']['duration'],
                channel['sfreq'],
                np.asarray(channel['hrf_mean'], dtype=np.float64),
                np.asarray(channel['hrf_std'], dtype=np.float64),
                channel['location'],
                channel['estimates'],
                channel['locations'],
                channel['context'],
            )

            # Insert hrf into tree and attach pointer to channel. Populate
            # the tree's hasher keyed by the channel's own context VALUES
            # (NE-002 fix — pre-fix populated by context dict KEYS, which
            # tree.branch never searches for).
            oxygenation = _is_oxygenated(ch_name)
            if oxygenation:
                _montage.channels[ch_name] = _montage.hbo_tree.insert(estimated_hrf)
                target_tree = _montage.hbo_tree
            elif oxygenation == False:
                _montage.channels[ch_name] = _montage.hbr_tree.insert(estimated_hrf)
                target_tree = _montage.hbr_tree
            else:
                target_tree = None

            if target_tree is not None:
                channel_context = channel.get('context', {})
                if isinstance(channel_context, dict):
                    for ctx_value in channel_context.values():
                        for hashable in _flatten_context_value(ctx_value):
                            target_tree.hasher.add(hashable, _montage.channels[ch_name])
        except Exception as exc:
            raise ValueError(
                f"load_montage: failed to load entry {key!r}: {exc}"
            ) from exc

    _montage.sfreq = sfreq # Sampling frequency

    _montage.configured = True

    return _montage

class montage(tree):
    """
    Class functions:
        - localize_hrfs() - Localizes HRFs to the optodes in a fNIRS montage given a nirx object
        - estimate_hrf() - Deconvolves a fNIRS signal and impulse function to derive the underlying HRF
        - estimate_activity() - Deconvlve a fNIRS scan using estimated HRF's localized to optodes location to gain a neural activity estimate
        - generate_distribution() - Calculates an average HRF and it's standard deviation across time
        - save() - Saves the current montage HRFs
        - load() - Loads a montage of HRFs
        - _merge_montage() - Merges two montages
        - correlate_hrf() - Correlates the HRF estimates across the subject pool to assess similarity
        - correlate_canonical() - Correlates the HRF estimates with a canonical HRF to assess similarity
        - configure() - Configures the montage object to a nirx object

    
    Class attributes:
        - nirx_obj (mne raw object) - NIRX object loaded in via MNE python library
        - sfreq (float) - Sampling frequency of the fNIRS object
        - channels (list) - fNIRS montage channel names
        - subject_estimates (list) - List of subject event-wise HRF estimate
        - channel_estimates (list) - List of channel HRF distribution estimates (position 0 is mean and 1 is std)
    """

    def __init__(self, nirx_obj = None, **kwargs):

        self.root = None # Set an empty root

        # Save runtime parameters to object
        self.lib_dir = _LIB_DIR

        # Set data context
        self.context = {
                'method': 'toeplitz',
                'doi': 'temp',
                'study': None,
                'task': None,
                'conditions': None,
                'stimulus': None,
                'intensity': 1.0,
                'duration': 30.0,
                'protocol': None,
                'age_range': None,
                'demographics': None
        }
        self.context = {**self.context, **kwargs} # Add user input
        self.context_weights = {context: 1.0 for context in self.context.keys()}

        self.channels = {} # Create variable for holding poiners to each channel
        
        # Load the HRtree's
        self.hbo_tree = tree(f"{self.lib_dir}/hrfs/hbo_hrfs.json", **kwargs)
        self.hbr_tree = tree(f"{self.lib_dir}/hrfs/hbr_hrfs.json", **kwargs)

        self.configured = False
        if nirx_obj:
            # Configure to nirx object passed in
            self.configure(nirx_obj)
            
            # Echo the montage object
            self.__repr__()

    def __repr__(self):
        """
        String representation of the montage object. Safe to call on both
        configured and unconfigured instances (H3).

        Returns:
            str - String representation of the montage object
        """
        context_str = '\n'.join(
            f'{key} - {value} - {self.context_weights[key]}'
            for key, value in self.context.items()
        )
        sfreq = getattr(self, 'sfreq', None)
        hbo_channels = getattr(self, 'hbo_channels', [])
        hbr_channels = getattr(self, 'hbr_channels', [])
        configured = getattr(self, 'configured', False)
        state = 'configured' if configured else 'unconfigured'
        return (
            f" - Montage object ({state}) - \n"
            f"Number of channels: {len(self.channels)}\n"
            f" Sampling frequency: {sfreq}\n"
            f"HbO channels (count of {len(hbo_channels)}): {hbo_channels}\n"
            f" HbR channels (count of {len(hbr_channels)}): {hbr_channels}\n"
            f" - Contexts - \n{context_str}\n"
        )

    def localize_hrfs(self, max_distance = 0.01, verbose = False):
        """
        Tries to find local HRFs to each of the fNIRS optodes using the tree structure
        functionality to quickly find nearby HRF's. If it can't it will default to a
        global HRF estimated.

        Arguments:
            max_distance (float) - maximum distance in milimeter's a previously estimated HRF can be attached to an optode
            verbose (bool) - If True, print out extra information during localization

        Returns:
            None
        """

        # S4: fetch canonicals from the unified helper instead of
        # inlining a second Glover HRF generation path here. The helper
        # generates at the correct sample rate for the scan and caches
        # the result so repeated calls within a localize pass are cheap.
        canonical_duration = float(self.context['duration'])

        for ch_name, optode in self.channels.items(): # Iterate through channels apart of nirx data
            if verbose: print(f"Searching for nodes close to optode {optode.ch_name}")
            oxygenation = _is_oxygenated(ch_name)
            if oxygenation:
                hrf, distance = self.hbo_tree.nearest_neighbor(optode, max_distance, verbose = verbose) # Search in space for similar HRF
            else:
                hrf, distance = self.hbr_tree.nearest_neighbor(optode, max_distance, verbose = verbose)

            if hrf is not None: # If found (nearest_neighbor now returns None on miss, S4)
                if verbose: print(f"HRF {hrf.ch_name} found at {distance} distance")

                optode.trace = hrf.trace # Add mean and std to montage for channel
                optode.trace_std = hrf.trace_std

            else: # If no local HRF within max_distance, fall back to canonical
                if verbose: print(f"Local HRF couldn't be found for channel {ch_name}, using canonical")

                target_tree = self.hbo_tree if oxygenation else self.hbr_tree
                canonical_node = target_tree.get_canonical_hrf(
                    oxygenation, self.sfreq, canonical_duration
                )
                optode.trace = canonical_node.trace
                optode.trace_std = canonical_node.trace_std
                optode.context['method'] = 'canonical'


    def solve_lstsq(self, lhs, rhs, cond_thresh = None):
        # Compute condition number
        if cond_thresh:
            start = time.time()
            cond_number = np.linalg.cond(lhs)
            end = time.time()
            print(f"np.linalg.cond elapsed time: {end - start:.6f} seconds")
        
        if cond_thresh == None or cond_number < cond_thresh:
            # Stable enough to use least squares with pseudoinverse
            start = time.time()
            estimate, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            end = time.time()
            print(f"np.linalg.lstsq elapsed time: {end - start:.6f} seconds")
        else:
            # Same least squares but with smoothing
            start = time.time()
            estimate = scipy.linalg.pinv(lhs) @ rhs
            end = time.time()
            print(f"scipy.linalg.pinv elapsed time: {end - start:.6f} seconds")
        
        return estimate

    def estimate_hrf(self, nirx_obj, events, duration = 30.0, lmbda = 1e-3, edge_expansion = 0.15, preprocess = True, progress_callback = None):
        """
        Estimate an HRF subject wise given a nirx object and event impulse series using toeplitz
        deconvolution with regularization.

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan file loaded in through mne
            events (list) - Event impulse series indicating event occurences during fNIRS scan
            duration (float) - Duration in seconds of the HRF to estimate
            lmbda (float) - Regularization parameter to apply during deconvolution
            edge_expansion (float) - Fraction of the duration to expand the events and duration by to account for toeplitz edge artifacts
            preprocess (bool) - If True, preprocess the fNIRS data before estimating the HRF
            progress_callback (callable or None) - Optional callable invoked at the start of each
                channel iteration with (current_index, total_channels, channel_name). current_index
                is 0-indexed; the final iteration is (total_channels - 1, total_channels, name).
                channel_name is the standardized form (matching self.channels keys) so callers
                see the same naming convention as estimate_activity. Used by GUI/batch tooling to
                surface progress; exceptions raised by the callback propagate. Default None (no-op).

        Returns:
            None
        """
        if isinstance(duration, float) is False and isinstance(duration, int) is False:
            raise ValueError(f"ERROR: Duration passed in must be a float or integer, duration passed in is of type {type(duration)}")

        if isinstance(duration, int): duration = float(duration)

        if duration <= 0:
            raise ValueError(f"ERROR: duration must be > 0, got {duration}")

        if isinstance(events, list) is False:
            raise ValueError(f"ERROR: Events passed in must be of type list, object of type {type(events)} was passed in...")

        if len(events) == 0:
            raise ValueError("ERROR: events list must not be empty")

        if lmbda <= 0:
            raise ValueError(f"ERROR: lmbda must be > 0 for Tikhonov regularization, got {lmbda}")

        # Check montage still needs to be configured
        if self.configured is False:
            self.configure(nirx_obj)

        # Convert events to numpy array
        events = np.array(events) 

        # Expand event and duration to account for toeplitz edge artifacts (removed later)
        timeshift = int(round((self.sfreq * duration) * edge_expansion, 0))
        new_events = np.zeros_like(events)
        for ind in range(events.shape[0]): # Iterate through all events
            if events[ind] != 0: # if we found an event
                if (ind - timeshift) < 0: # Check if we can expand the event
                    print("WARNING: An event has been omitted due to edge expansion falling outside of the scan timeframe")
                    continue
                new_events[ind - timeshift] = 1
        
        # Update events and duration to reflect expansion
        events = new_events

        # Update new time HRF estimation duration to account for edge expansion
        duration *= (1 + 2 * edge_expansion)

        if preprocess:
            nirx_obj = preprocess_fnirs(nirx_obj, deconvolution=True)
            if nirx_obj is None:
                return  # Skip subject if all channels are bad

        nirx_obj.load_data()      # Load nirx object (after preprocessing so data reflects preproc output)
        data = nirx_obj.get_data() # Grab data

        hrf_len = int(round(self.sfreq * duration, 0))  # Calculate HRF length
        scan_len = data.shape[1] # Grab single channel signal length

        if events.shape[0] > scan_len:
            events = events[:scan_len]
            print(f"WARNING: Shortening events for {nirx_obj}")
        elif events.shape[0] != scan_len:
            raise ValueError(f"ERROR: Expected events to be of length {scan_len} but got length {events.shape[0]}...")

        # Build Toeplitz matrix
        X = scipy.linalg.toeplitz(events, np.zeros(hrf_len))
        total_channels = len(nirx_obj.info['chs'])
        for i, (fnirs_signal, channel) in enumerate(zip(data[:], nirx_obj.info['chs'])): # For each channel
            if progress_callback is not None:
                progress_callback(i, total_channels, standardize_name(channel['ch_name']))
            print(f"Deconvolving HRF from channel {channel}")
            # Grab channel data and normalize
            #Y = fnirs_signal / np.max(np.abs(fnirs_signal))
            mean = np.mean(fnirs_signal)
            std = np.std(fnirs_signal)
            Y = (fnirs_signal - mean) / std

            # Define regularized least squares equation
            lhs = X.T @ X + lmbda * np.eye(X.shape[1])
            rhs = X.T @ Y

            hrf_estimate = self.solve_lstsq(lhs, rhs)

            # Denormalize HRF estimate
            #hrf_estimate = hrf_estimate * np.max(np.abs(fnirs_signal))
            #hrf_estimate = hrf_estimate * std + mean

            # Adjust the remove the added edges from the hrf_estimate
            start = timeshift
            end = hrf_len - timeshift
            hrf_estimate = hrf_estimate[start:end]

            # Append estimate to channel estimates
            optode = self.channels[standardize_name(channel['ch_name'])]
            optode.estimates.append(list(hrf_estimate))

            # Calculate new centroid for optode given locations used for estimate
            optode.locations.append(list(channel['loc'][:3]))


    def estimate_activity(self, nirx_obj, lmbda = 1e-4, hrf_model = 'toeplitz', preprocess = True, cond_thresh = None, timeout = 30, progress_callback = None):
        """
        Deconvlve a fNIRS scan using estimated HRF's localized to optodes location
        to gain a neural activity estimate

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan loaded through mne
            lmbda (float) - Regularization parameter to apply during deconvolution
            hrf_model (str) - HRF model to use during deconvolution, either 'toeplitz' for localized HRF's or 'canonical' for a standard canonical HRF
            preprocess (bool) - If True, preprocess the fNIRS data before estimating the neural activity
            cond_thresh (float or None) - Condition number threshold for falling back to pinv in solve_lstsq
            timeout (float) - Seconds to wait for a single channel's lstsq solve before dropping that channel.
                Default 30 is a generous ceiling — realistic fNIRS inputs solve in tens of milliseconds,
                so this fires only on genuinely pathological matrices. Can be tightened once empirical
                solve-time data is collected from real runs.
            progress_callback (callable or None) - Optional callable invoked at the start of each
                channel iteration with (current_index, total_channels, channel_name). current_index
                is 0-indexed and counts every entry in self.channels including 'global' entries that
                are skipped internally, so the callback fires exactly total_channels times. Exceptions
                raised by the callback propagate. Default None (no-op).

        Returns:
            nirx_obj (mne raw object) - Raw with deconvolved neural activity, or None if skipped
        """

        if lmbda <= 0:
            raise ValueError(f"ERROR: lmbda must be > 0 for Tikhonov regularization, got {lmbda}")

        # Check montage still needs to be configured
        if self.configured is False:
            self.configure(nirx_obj)

        nirx_obj.load_data()
        if preprocess:
            nirx_obj = preprocess_fnirs(nirx_obj, deconvolution=True)
            if nirx_obj is None:
                return None  # Skip subject if all channels are bad

        # success is declared at estimate_activity scope so the nested deconvolution
        # closure can write to it via `nonlocal` — otherwise success=True/False inside
        # the closure would create a closure-local and the outer drop-channel check
        # would always read None (ND-003).
        success = None

        # Define hrf deconvolve function to pass nirx object
        def deconvolution(nirx):
            nonlocal success

            # Normalize input z-score
            mean = np.mean(nirx)
            std = np.std(nirx)
            Y = (nirx - mean) / std
            Y = np.asarray(Y, dtype=float)

            # Pad HRF to match nirx length
            hrf_kernel = hrf.trace / np.max(np.abs(hrf.trace))
            hrf_kernel = np.asarray(hrf_kernel, dtype=float)

            # Construct Toeplitz convolution matrix (design matrix)
            n_time = len(Y)
            n_hrf = len(hrf_kernel)

            first_col = np.r_[hrf_kernel, np.zeros(n_time - n_hrf)]
            first_row = np.r_[hrf_kernel[0], np.zeros(n_time - 1)]
            A = scipy.linalg.toeplitz(first_col, first_row)
            A = np.asarray(A, dtype=float)

            # Solve the inverse problem with regularization
            lhs = A.T @ A + float(lmbda) * np.eye(A.shape[1])
            rhs = A.T @ Y

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.solve_lstsq, lhs, rhs, cond_thresh)
                try:
                    deconvolved_signal = future.result(timeout)
                    success = True
                except TimeoutError:
                    print(f"lstsq exceeded {timeout}s timeout, dropping channel")
                    deconvolved_signal = nirx
                    success = False
                except Exception as exc:
                    # Any other solve failure (numpy LinAlgError, ValueError on
                    # malformed inputs, etc.) should be treated as a channel
                    # drop, not propagated out of estimate_activity. Matching
                    # M4's "no orphans after failure" contract — if we let the
                    # exception escape, the outer loop's cleanup never runs.
                    print(f"lstsq failed for channel: {type(exc).__name__}: {exc}; dropping channel")
                    deconvolved_signal = nirx
                    success = False

            return deconvolved_signal # Return recovered neural signal

        # Apply deconvolution and return the nirx object.
        # Iterate a snapshot of channel names so we can safely pop orphaned
        # entries inside the loop when deconvolution fails (M4).
        dropped_channels = []
        all_channels = list(self.channels.keys())
        total_channels = len(all_channels)
        for i, ch_name in enumerate(all_channels):
            if progress_callback is not None:
                progress_callback(i, total_channels, ch_name)
            hrf = self.channels[ch_name]
            success = None  # reset per channel so stale state can't leak from prior iteration

            if 'global' in ch_name: continue # Skip if global hrf estimate

            # Detect degenerate HRF traces that would produce NaN (zero-length,
            # missing, or all-zeros) so we fall back to canonical instead of
            # silently dividing by zero in the deconvolution closure (H1).
            trace_invalid = (
                hrf.trace is None
                or len(hrf.trace) == 0
                or np.max(np.abs(hrf.trace)) == 0
            )
            if trace_invalid and hrf_model != 'canonical':
                print(
                    f"WARNING: HRF trace for channel {ch_name} is empty or all-zero; "
                    "falling back to canonical HRF"
                )

            # If canonical HRF requested (or forced by a degenerate trace)
            if hrf_model == 'canonical' or trace_invalid:
                print(f"WARNING: Using canonical HRF for channel {ch_name} in {nirx_obj}")
                estimate_hrf = hrf # Temporarily replace HRF
                # S4: fetch a canonical generated at the scan's own sfreq
                # (and the montage's duration context) rather than the
                # old hardcoded root.right sentinel which was locked to
                # 7.81 Hz / 30 s regardless of the calling scan.
                canonical_duration = float(self.context.get('duration', 30.0))
                if _is_oxygenated(ch_name):
                    hrf = self.hbo_tree.get_canonical_hrf(
                        True, self.sfreq, canonical_duration
                    )
                else:
                    hrf = self.hbr_tree.get_canonical_hrf(
                        False, self.sfreq, canonical_duration
                    )

            # Figure out which channel to apply to. Track the match explicitly:
            # a bare for/break leaves nirx_channel bound to the LAST iterated
            # channel when ch_name matches nothing, which would silently apply
            # this HRF's deconvolution to an unrelated channel. self.channels
            # can legitimately contain a channel absent from this scan — e.g. a
            # montage loaded via load_montage (configured=True, so configure()
            # is skipped), or estimate_activity reused across scans with
            # differing layouts — so a no-match must skip, not act on a stale
            # loop variable.
            matched_channel = None
            for nirx_channel in nirx_obj.info['chs']:
                if ch_name == standardize_name(nirx_channel['ch_name']):
                    matched_channel = nirx_channel
                    break
            if matched_channel is None:
                print(
                    f"WARNING: channel {ch_name} not found in scan {nirx_obj}; "
                    "skipping deconvolution for it"
                )
                continue

            print(f"Deconvolving channel {ch_name}...") # Apply deconvolution
            nirx_obj.apply_function(deconvolution, picks = [matched_channel['ch_name']]) # Apply deconvolution for channel

            # Remove channel if neural activity estimation failed to converge.
            # M4: also drop the orphaned entry from self.channels and the
            # hbo/hbr channel lists so downstream iterators (correlate_hrf,
            # generate_distribution) don't trip on a stale pointer. The
            # spatial tree copy of the HRF is left in place for now — the
            # broken tree.delete path is fixed in fix/tree-delete-filter, and
            # tree orphans are harmless until that lands because nothing in
            # this release iterates the full tree except montage.branch()
            # which rebuilds from self.channels.
            if success is False:
                nirx_obj.drop_channels([matched_channel['ch_name']])
                dropped_channels.append(ch_name)

            # Replace the canonical HRF estimate temporarily used with the HRF estimate
            if hrf_model == 'canonical':
                hrf = estimate_hrf # Replace the original HRF

        for ch_name in dropped_channels:
            self.channels.pop(ch_name, None)
            if ch_name in self.hbo_channels:
                self.hbo_channels.remove(ch_name)
            if ch_name in self.hbr_channels:
                self.hbr_channels.remove(ch_name)

        return nirx_obj

    def generate_distribution(self, plot_dir = None):
        """
        Calculate average and standard deviation of HRF across subjects for each channel

        Arguments:
            plot_dir (str) - If provided, will save a plot of each channel's HRF to the given directory
        """
        hbr_estimates = []
        hbo_estimates = []

        for channel in self.channels.keys():
            # Grab channel optode attached to montage
            optode = self.channels[channel]

            # Calculate average HRF estimate and standard deviation
            optode.trace = np.mean(optode.estimates, axis = 0)
            optode.trace_std = np.std(optode.estimates, axis = 0)
            
            # Update centroid attached to optode with average location
            optode.update_centroid()

            if plot_dir: # Plot if requested
                # HRF.plot was refactored to accept a full file path
                # instead of a directory. Construct the per-channel
                # filename here and ensure the directory exists.
                # show=False so cluster / headless runs don't block on
                # plt.show() and don't keep figure state around between
                # channels.
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f"{optode.ch_name}_hrf.png")
                optode.plot(plot_path, show=False)
                plt.close('all')
            
            if optode.oxygenation: # Append data
                hbo_estimates.append(optode.trace)
            else:
                hbr_estimates.append(optode.trace)

        # Calculate global HRF mean and standard deviation
        for oxygenation, estimates in zip([True, False], [hbo_estimates, hbr_estimates]):
            type_estimates = np.vstack(estimates)
            global_mean = np.mean(type_estimates, axis = 0)
            global_std = np.std(type_estimates, axis = 0)

            # Create a global HRF variable
            global_location = [360 + random.random(), 360 + random.random(), 360 + random.random()]
            global_hrf = HRF(
                doi = self.context['doi'],
                ch_name = ("global_hbo" if oxygenation else "global_hbr"),
                duration = self.context['duration'],
                sfreq = self.sfreq,
                trace = global_mean,
                trace_std = global_std,
                location = global_location,
                estimates = [list(global_mean)],
                locations = [list(global_location)]
            )
            #Insert global hrf into tree and attach pointer to channels dict
            if oxygenation:
                self.channels['global_hbo'] = self.insert(global_hrf)
            else:
                self.channels['global_hbr'] = self.insert(global_hrf)

    def correlate_hrf(self, plot_filename = "montage_correlation.png"):
        """
        Correlate the HRF estimates across the subject pool to assess similarity

        Arguments:
            plot_filename (str) - Filename to save correlation plot to
        
        Returns:
            corr_matrix (np.array) - Correlation matrix of HRF estimates
        """
        corr_matrix = np.zeros((len(self.hbo_channels), len(self.hbr_channels), 2))
        
        # Calculate correlation coefficients and p-values between HbO and HbR channels
        for hbo_ind, hbo_channel in enumerate(self.hbo_channels):
            hbo_hrf = self.channels[hbo_channel].trace

            for hbr_ind, hbr_channel in enumerate(self.hbr_channels):
                hbr_hrf = self.channels[hbr_channel].trace
                
                corr_coefficient, p_value = scipy.stats.spearmanr(hbo_hrf, hbr_hrf)
                
                corr_matrix[hbo_ind, hbr_ind, 0] = corr_coefficient
                corr_matrix[hbo_ind, hbr_ind, 1] = p_value

        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, :, 0], cmap='viridis', aspect='auto')
        plt.colorbar(label='Correlation Coefficient')
        plt.title('Correlation Matrix of HRF Estimates')
        plt.xlabel('HbR Channels')
        plt.ylabel('HbO Channels')
        plt.xticks(range(len(self.hbr_channels)), self.hbr_channels, rotation=90)
        plt.yticks(range(len(self.hbo_channels)), self.hbo_channels)
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()

        # Plot p-values
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, :, 1], cmap='viridis', aspect='auto')
        plt.colorbar(label='P-value')
        plt.title('P-values of Correlation between HRF Estimates')
        plt.xlabel('HbR Channels')
        plt.ylabel('HbO Channels')
        plt.xticks(range(len(self.hbr_channels)), self.hbr_channels, rotation=90)
        plt.yticks(range(len(self.hbo_channels)), self.hbo_channels)
        plt.tight_layout()
        plt.savefig(plot_filename.replace(".png", "_pvalues.png"))
        plt.close()

        # Save the correlation matrix to a file
        with open("correlation_matrix.json", "w") as f:
            json.dump(corr_matrix.tolist(), f, indent=4)
        
        return corr_matrix

    def correlate_canonical(self, plot_filename = "canonical_correlation.png", duration = 30.0):
        """
        Correlate the HRF estimates with a canonical HRF to assess similarity

        Arguments:
            plot_filename (str) - Filename to save correlation plot to
            duration (float) - Duration in seconds of the canonical HRF to generate
        Returns:
            None
        """
        # Guard against unconfigured / empty montage: self.root is None
        # until _merge_montages runs, and was previously dereferenced
        # directly below. Caught by the mypy sweep.
        if self.root is None or not self.channels:
            raise ValueError(
                "correlate_canonical requires a configured montage with "
                "at least one channel; call configure() or estimate_hrf() "
                "first so self.channels is populated."
            )
        # Generate canonical HRF
        time_stamps = np.arange(0, len(self.root.trace), 1)

        # Parameters for the double-gamma HRF
        peak1 = scipy.stats.gamma.pdf(time_stamps, 6) # peak at ~6s
        peak2 = scipy.stats.gamma.pdf(time_stamps, 16) / 6.0 # undershoot at ~16s

        canonical_hrf = peak1 - peak2
        canonical_hrf /= np.max(canonical_hrf)  # Normalize peak to 1
        corr_matrix = np.zeros((len(self.hbo_channels) + len(self.hbr_channels), 2))
        for ind, ch_name in enumerate(self.hbo_channels + self.hbr_channels):
            hrf = self.channels[ch_name]

            corr_coefficient, p_value = scipy.stats.spearmanr(canonical_hrf, hrf.trace)
            corr_matrix[ind, 0] = corr_coefficient
            corr_matrix[ind, 1] = p_value

        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, 0][np.newaxis, :], cmap='viridis', aspect='auto')
        plt.colorbar(label='Correlation Coefficient')
        plt.title('Correlation Matrix of HRF Estimates with Canonical HRF')
        plt.xlabel('Montage Channels')
        plt.ylabel('Canonical HRF')
        plt.xticks(range(len(self.hbo_channels) + len(self.hbr_channels)), self.hbo_channels + self.hbr_channels, rotation=90)
        plt.yticks(range(1), ['Canonical'])
        plt.tight_layout()

        plt.savefig(plot_filename)
        plt.close()

        # Plot p-values
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, 1][np.newaxis, :], cmap='viridis', aspect='auto')
        plt.colorbar(label='P-value')
        plt.title('P-values of Correlation with Canonical HRF')
        plt.xlabel('Montage Channels')
        plt.ylabel('Canonical HRF')
        plt.xticks(range(len(self.hbo_channels) + len(self.hbr_channels)), self.hbo_channels + self.hbr_channels, rotation=90)
        plt.yticks(range(1), ['Canonical'])
        plt.tight_layout()
        plt.savefig(plot_filename.replace(".png", "_pvalues.png"))
        plt.close()
        return

    def configure(self, nirx_obj, **kwargs):
        """
        Configure the montage object to a nirx object.

        M6: commit-on-success pattern. Any failure during configure
        (channel-name error, _merge_montages exception, etc.) rolls the
        montage back to its pre-call state, including undoing any
        freshly-inserted tree nodes via `tree.delete`. This pairs with
        the KI-009 fix in this branch — the rollback relies on a
        working `_delete_recursive`.

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan file loaded in through
            **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)

        Returns:
            None"""
        print(f"Configuring HRfunc montage...")

        _SENTINEL = object()
        prev_sfreq = getattr(self, 'sfreq', _SENTINEL)
        prev_hbo_channels = getattr(self, 'hbo_channels', _SENTINEL)
        prev_hbr_channels = getattr(self, 'hbr_channels', _SENTINEL)
        prev_channels = dict(self.channels)
        prev_configured = self.configured
        prev_root = self.root
        # Use identity, not value, to recognize nodes that already belonged
        # to the montage's tree before this call. _merge_montages can either
        # reuse existing nodes (via nearest_neighbor match) or insert new
        # ones; only the inserted ones need to be rolled back via delete.
        prev_node_ids = {id(node) for node in prev_channels.values()}

        # Compute the new channel lists locally first so a standardize_name
        # / _is_oxygenated failure raises before we mutate self.
        try:
            new_sfreq = nirx_obj.info['sfreq']
            new_hbo_channels = [
                standardize_name(ch) for ch in nirx_obj.ch_names
                if _is_oxygenated(ch) == True
            ]
            new_hbr_channels = [
                standardize_name(ch) for ch in nirx_obj.ch_names
                if _is_oxygenated(ch) == False
            ]
        except Exception:
            # self is untouched — nothing to roll back
            raise

        # Commit the easy state, then run _merge_montages under a rollback
        # guard. _merge_montages reads self.sfreq and writes self.channels /
        # self.root, so it must see the new state in place.
        self.sfreq = new_sfreq
        self.hbo_channels = new_hbo_channels
        self.hbr_channels = new_hbr_channels
        try:
            self._merge_montages(nirx_obj) # Add empty HRF nodes to the tree for each HRF
        except Exception:
            if prev_root is None:
                # First-time configure. Any tree state originated from this
                # failed call — drop the whole tree in one move rather than
                # deleting node-by-node (which would expose the canonical
                # sentinel auto-inserted by tree.insert at line 164).
                self.root = None
            else:
                # Re-configure. Undo only the newly-inserted nodes via
                # tree.delete so existing pre-call state survives. The
                # kd-tree delete-by-copy may shuffle payload among nodes
                # in the process, but channel-level mappings remain
                # consistent with the pre-call snapshot after we restore
                # self.channels below.
                newly_added = set(self.channels.keys()) - set(prev_channels.keys())
                for ch_name in newly_added:
                    node = self.channels.get(ch_name)
                    if node is None or id(node) in prev_node_ids:
                        continue  # reused existing node — don't delete it
                    try:
                        self.delete(node)
                    except Exception:
                        pass  # best-effort cleanup; fail-quiet in rollback path

            # Restore scalar / list / dict state
            self.channels = prev_channels
            if prev_sfreq is _SENTINEL:
                del self.sfreq
            else:
                self.sfreq = prev_sfreq
            if prev_hbo_channels is _SENTINEL:
                del self.hbo_channels
            else:
                self.hbo_channels = prev_hbo_channels
            if prev_hbr_channels is _SENTINEL:
                del self.hbr_channels
            else:
                self.hbr_channels = prev_hbr_channels
            self.configured = prev_configured
            raise

        self.configured = True

    def save(self, filename = 'montage_hrfs.json'):
        """
        Save the hrf montage

        Arguments:
            Filename (str) - Filename to save the montage HRFs as

        Returns:
            None
        """
        hrfs = self.gather(self.root)
        # Save to a JSON file
        with open(filename, "w") as file:
            json.dump(hrfs, file, indent=4)
        return
    
    def _merge_montages(self, nirx_obj, verbose = False):
        """
        Function to merge a NIRX object montage with the HRfunc montage.
        This function should only be used when initializing an empty
        montage or if merging nirx objects with the same NIRS montage layout
        and with different channel names (useful when dealing with multiple
        data collection sites with slightly different setups in channel naming).
        
        WARNING: Merging two distinctly different montages is not recommended.
        Inaccurate HRF may be estimated depending on how the merged montage
        is used.

        Arguments:
            nirx_obj (mne NIRX object) - MNE NIRS scan recording loading in through MNE
            verbose (bool) - If True, print out extra information during localization
        
        Returns:
            None
        """
        # Add each nirx object channel to the hrfunc.montage
        for channel in nirx_obj.info['chs']:
                # (Re)set runtime variables
                results = None

                # Grab pertinent info from nirx header
                ch_name = standardize_name(channel['ch_name'])
                location = channel['loc'][:3]

                # Skip if canonical HRF
                if ch_name == 'canonical':
                    continue

                empty_hrf = HRF(
                    self.context['doi'],
                    ch_name, 
                    self.context['duration'], 
                    self.sfreq,
                    [], 
                    [], 
                    location,
                    [],
                    [],
                    []
                )

                # Check if an HRF in this area already exists
                # NOTE: This is necessary to localize nodes with slight
                # channel name differences in the same location
                print(f"Searching for {ch_name}")
                best_node, distance = self.nearest_neighbor(empty_hrf, max_distance = 1e-9, verbose = verbose)
                # S4: nearest_neighbor returns (None, inf) on miss now
                # that the canonical sentinel is gone. Check None
                # explicitly so the intent is obvious.
                if best_node is None:
                    print(f"No matching HRF found, inserting new HRF")
                    self.channels[ch_name] = self.insert(empty_hrf)
                elif best_node.ch_name[:9] != 'canonical':
                    print(f"Local HRF found in the channel {best_node.ch_name}, merging with optode {ch_name}")
                    self.channels[ch_name] = best_node # Attach node to channel
                else:
                    # Matched an entry whose ch_name still starts with
                    # 'canonical' (e.g. a legacy JSON that stored a
                    # canonical record). Treat as no match and insert
                    # a fresh empty HRF.
                    print(f"Only canonical match found, inserting new HRF")
                    self.channels[ch_name] = self.insert(empty_hrf)

    def _merge_trees(self, filename = 'tree_hrfs.json'):
        """
        Merge montage, HbO and HbR trees. This function is meant to be used
        by the creators of HRfunc to merge submitted HRF estimates with the
        HRF toolbox

        Arguments:
            Filename (str) - Filename to save the montage HRFs as
        """
        hrfs = self.gather(self.hbo_tree.root)
        hrfs |= self.gather(self.hbr_tree.root)
        hrfs |= self.gather(self.root)
        # Save to a JSON file
        with open(filename, "w") as file:
            json.dump(hrfs, file, indent=4)
        return

    def branch(self, **kwargs):
        """
        Branch the montage on a specific context, creating a new montage
        with deep copies of all HRF data including trace_std.

        Arguments:
            **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)

        Returns:
            branch (montage object) - A new montage object filtered on the requested context
        """
        if kwargs:
            self.context = {**self.context, **kwargs}

        # Create a new empty montage
        branch = montage()
        branch.sfreq = self.sfreq if hasattr(self, 'sfreq') else None
        branch.hbo_channels = list(self.hbo_channels) if hasattr(self, 'hbo_channels') else []
        branch.hbr_channels = list(self.hbr_channels) if hasattr(self, 'hbr_channels') else []
        branch.context = dict(self.context)
        branch.context_weights = dict(self.context_weights)
        branch.configured = self.configured

        # Deep copy channels with all data including trace_std
        for ch_name, node in self.channels.items():
            node_copy = node.copy()
            branch.channels[ch_name] = branch.insert(node_copy)

        self.branched = True
        return branch

def preprocess_fnirs(scan, deconvolution = False):
    """
    Preprocess fNIRS data in an MNE Raw object.

    Steps:
    - Optical density conversion
    - Scalp coupling index evaluation and bad channel marking
    - Motion artifact correction using TDDR
    - Optional polynomial detrending for deconvolution
    - Haemoglobin conversion via Beer-Lambert Law
    - Optional bandpass filtering for GLM-based analysis

    Parameters:
    - scan (mne.io.Raw) - The raw fNIRS MNE object to preprocess.
    - deconvolution (bool) If True, performs detrending and skips filtering.

    Returns:
    - haemo (mne.io.Raw) - Preprocessed data with haemoglobin concentration channels.
    """

    scan.load_data()

    raw_od = mne.preprocessing.nirs.optical_density(scan)

    # scalp coupling index
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.95))

    if len(raw_od.info['bads']) == len(scan.ch_names):
        print("All channels are bad, skipping subject...")
        return

    if len(raw_od.info['bads']) > 0:
        subject_info = raw_od.info.get('subject_info')
        subject_id = subject_info['his_id'] if subject_info else 'unknown'
        print("Bad channels in subject", subject_id, ":", raw_od.info['bads'])

    # Interpolate bad channels
    raw_od.interpolate_bads(reset_bads=False)

    # temporal derivative distribution repair (motion attempt)
    od = mne.preprocessing.nirs.tddr(raw_od)

    # If running deconvolution, polynomial detrend to remove pysiological without cutting into the frequency spectrum
    if deconvolution:
        od = polynomial_detrend(od, order=3)

    # haemoglobin conversion using Beer Lambert Law 
    haemo = mne.preprocessing.nirs.beer_lambert_law(od.copy(), ppf=0.1)

    haemo = baseline_correct(haemo, baseline=(None, 0.0))

    if not deconvolution:
        haemo.filter(0.01, 0.2)

    return haemo

def baseline_correct(raw, baseline=(None, 0.0)):
    """
    Apply baseline correction to fNIRS data.
    
    Parameters:
    - raw (mne.io.Raw) - The raw fNIRS MNE object to baseline correct.
    - baseline (tuple) - The time interval for baseline correction. 
    
    Returns:
    - raw (mne.io.Raw) - Baseline corrected fNIRS data."""
    return raw.apply_function(lambda x: mne.baseline.rescale(x, times=raw.times, baseline=baseline, mode='mean'), picks='data')

def polynomial_detrend(raw, order = 1):
    raw_detrended = raw.copy()
    times = raw.times
    times_scaled = (times - times.mean()) / times.std()  # or just (times - mean)
    X = np.vander(times_scaled, N=order + 1, increasing=True)

    for idx in range(len(raw.ch_names)):
        y = raw.get_data(picks = idx)[0]
        beta = np.linalg.lstsq(X.T @ X, X.T @ y, rcond = None)[0]
        y_detrended = y - X @ beta
        raw_detrended._data[idx] = y_detrended

    return raw_detrended