import json, random, math, re, nilearn, scipy, os
from . import hrhash
from ._utils import standardize_name, _is_oxygenated, _LIB_DIR
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from collections import deque
from nilearn.glm.first_level import spm_hrf


def _flatten_context_value(value):
    """
    Yield hashable hasher keys from a context dict value. Lists and tuples
    are flattened one level; None and empty containers are skipped; dicts
    are passed through as-is (caller is responsible for not indexing them).

    This exists to bridge the HRF context schema (where values may be
    scalars like 'flanker' or lists like [20, 30] for age_range) and the
    hasher's key requirement (any hashable). Used by both load_hrfs and
    load_montage when populating the hasher under NE-002.
    """
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is None:
                continue
            yield item
        return
    yield value

class tree:

    def __init__(self, hrf_filename = None, *, rich = False, **kwargs):
        """
        A k-d tree data structure for storing HRF estimates across NIRX space

        Arguments:
            - hrf_filename (str) - Filepath to json file containing HRF estimates
            - rich (bool) - When True, retain the per-subject ``estimates`` +
                ``locations`` lists from the JSON. Defaults to False for
                back-compat (the v1.2 / v1.3.0 default), which strips both
                lists at load time to save memory. Set rich=True when
                downstream code (e.g. ROI averaging over subject estimates)
                needs the underlying data.
            - context arguments - Any context item you'd like to include in the HRF search
            
        Functions:
            - compare() - Filter the HRF file for contexts of interest
            - insert() - Insert a new HRF node into the tree
            - nearest_neighbor() - Find the nearest neighbor to a target point in the 3D k-d tree
            - radius_search() - Find all HRFs within a certain radius of a target point in the 3D k-d tree
            - gather() - Gather all HRFs in the tree and return them as a dictionary
            - save() - Save the HRFs in the tree to a json file
            - split_save() - Split the tree into oxygenated and deoxygenated HRFs and save the outputs
            - traverse() - In-order traversal of the tree for printing purposes
            - merge() - Merge another tree into this one
            - delete() - Delete a node from the 3D k-d tree based on spatial position
            - filter() - Filter on experimental contexts
            - branch() - Branch the tree on a specific context to reduce search space

        Attributes:
            - root (HRF object) - The root node of the tree
            - hrf_filename (str) - The filename of the HRF json to load into the tree
            - context (dict) - The context to filter on when inserting new HRFs into the tree
            - context_weights (dict) - Weights to attach to each context during similarity comparison
            - hasher (hrhash object) - A hash table for quickly finding HRFs based on context
            - branched (bool) - Whether the tree has been branched on a specific context
        """
        # Find hrfunc install library
        self.lib_dir = _LIB_DIR

        self.hrf_filename = hrf_filename

        self.root = None
        self.branched = False     

        # Set and update context
        self.context = {
            'method': 'toeplitz',
            'doi': 'temp',
            'ch_name': 'global',
            'study': None,
            'task': None,
            'conditions': None,
            'stimulus': None,
            'intensity': None,
            'duration': 30.0,
            'protocol': None,
            'age_range': None,
            'demographics': None
        }
        self.context = {**self.context, **kwargs} 
        self.context_weights = {key: 1.0 for key in self.context.keys()}

        self.hasher = hrhash.hasher(self.context)

        # S4: lazy canonical HRF cache, keyed on (oxygenation, sfreq,
        # duration). Populated on demand by get_canonical_hrf.
        self._canonical_cache = {}

        if self.hrf_filename and os.path.exists(self.hrf_filename):
            self.load_hrfs(self.hrf_filename, rich=rich)
            print(f"Tree initialized with HRFs from {hrf_filename}")
        else:
            print(f"Tree initialized without HRFs loaded...")

    def get_canonical_hrf(self, oxygenation, sfreq, duration):
        """
        Lazily generate and cache a canonical Glover HRF matching the
        calling scan's sample rate and duration (S4).

        Pre-S4 the canonical was eagerly constructed at t_r=0.128
        (~7.81 Hz, 30 s) during the first tree.insert, which meant every
        canonical kernel used downstream was locked to that one sample
        rate regardless of the actual scan. For scans at 5 Hz or 10 Hz
        the kernel length didn't match and downstream deconvolution
        math silently produced wrong results.

        Arguments:
            oxygenation (bool) - True for HbO (positive Glover HRF),
                False for HbR (negated).
            sfreq (float) - Sampling frequency the canonical should be
                generated at. Glover's t_r is 1/sfreq.
            duration (float) - Kernel duration in seconds. Kernel length
                comes out to approximately sfreq * duration samples.

        Returns:
            HRF - A cached HRF node at sentinel location
                [359.0, 359.0, 359.0] with context['method']='canonical'.
                Safe to insert into a kd-tree if needed; the sentinel
                location is far outside realistic MNE meter-scale head
                coordinates so it won't collide with real optodes.

        Thread-safety:
            The cache is a plain dict without locking. This is safe for
            the current call pattern — get_canonical_hrf is only invoked
            from the outer channel loop in montage.estimate_activity and
            montage.localize_hrfs, never from inside the per-channel
            ThreadPoolExecutor closure. If a future caller parallelizes
            montages on the same tree instance, wrap the cache access in
            a threading.Lock. The worst-case failure today is a double-
            generate, not corruption — generated kernels are pure
            functions of the key.
        """
        key = (bool(oxygenation), float(sfreq), float(duration))
        cached = self._canonical_cache.get(key)
        if cached is not None:
            return cached

        canonical_trace = list(nilearn.glm.first_level.glover_hrf(
            t_r = 1.0 / float(sfreq),
            oversampling = 1,
            time_length = float(duration),
        ))
        if not oxygenation:
            canonical_trace = [-point for point in canonical_trace]

        ch_name = 'canonical_hbo' if oxygenation else 'canonical_hbr'

        node = HRF(
            'canonical',
            ch_name,
            float(duration),
            float(sfreq),
            np.asarray(canonical_trace, dtype=np.float64),
            trace_std = np.zeros(len(canonical_trace), dtype=np.float64),
            location = [359.0, 359.0, 359.0],
            estimates = [],
            locations = [],
        )
        node.context['method'] = 'canonical'
        self._canonical_cache[key] = node
        return node

    def load_hrfs(self, hrf_filename, similarity_threshold = 0.0, oxygenated = None, rich = False):
        """
        Orchestrate building the HRF tree while filtering for specific context

        Arguments:
            hrf_filename (str) - Filename of the HRF json to load into the tree
            sim_threshold (float) - Threshold to allow or exclude HRF's based on context, defaults to 0.0 or no threshold
            oxygenated (bool) - If True, load only oxygenated HRFs, if False load only deoxygenated HRFs, if None load all HRFs
            context_weights (dict) - Weight to attach to each context during similarity comparison

        Returns:
            None
        """

        with open(hrf_filename, 'r') as json_file:
            hrfs_json = json.load(json_file) # Load HRFs from json
        
        for key, channel in hrfs_json.items():
            
            # Grab channel and doi info
            split = key.split('-')
            doi = split.pop()
            ch_name = ' '.join(split)


            # Skip if oxygenation/deoxygenation filtering requested
            oxygenation = _is_oxygenated(ch_name)
            if oxygenated == False and oxygenation:
                continue
            if oxygenated and oxygenation == False:
                continue

            # If similarity check requested
            if similarity_threshold > 0.0:
                context_similarity = self.compare_context(self.context, channel['context'], self.context_weights)
                if context_similarity < similarity_threshold:
                    print(f"Skipping {ch_name}, similarity threshold not met")
                    continue

            if rich == False:
                channel['estimates'] = []
                channel['locations'] = []

            # create a new hrf node
            new_hrf = HRF(
                doi, 
                ch_name, 
                float(channel['context']['duration']), 
                float(channel['sfreq']), 
                np.asarray(channel['hrf_mean'], dtype=np.float64), 
                np.asarray(channel['hrf_std'], dtype=np.float64), 
                channel['location'], 
                channel['estimates'],
                channel['locations'],
                channel['context']
                )
            
            # Insert hrf node into tree
            node = self.insert(new_hrf)

            # Add newly added node into HRHash table, keyed by the VALUES
            # in the channel's own context dict — not by the tree's context
            # dict KEYS, which was the NE-002 bug. After the fix,
            # `hasher.search('flanker')` returns every node whose
            # context dict contained 'flanker' anywhere.
            channel_context = channel.get('context', {})
            if isinstance(channel_context, dict):
                for ctx_value in channel_context.values():
                    for hashable in _flatten_context_value(ctx_value):
                        self.hasher.add(hashable, node)

    def insert(self, hrf, depth = 0, node = None):
        """Insert a new node into the 3D k-d tree based on spatial position.
        
        Arguments:
            hrf (HRF) - The HRF node to insert
            depth (int) - Current depth in the tree
            node (HRF) - Internal argument for passing the node to insert into

        Returns:
            node (HRF) - The inserted HRF node
        """

        if self.root is None:
            print(f"Setting root... {hrf.ch_name}")
            self.root = hrf
            # S4: canonical HRFs are now generated on demand via
            # tree.get_canonical_hrf(oxygenation, sfreq, duration) so
            # they match the calling scan's sample rate. Pre-fix this
            # branch eagerly constructed a canonical at t_r=0.128
            # (7.81 Hz) and stashed it at self.root.right, which made
            # downstream callers rely on the sentinel being present and
            # locked every canonical kernel to a single sample rate
            # regardless of the scan. Removed entirely — the tree is
            # now a pure kd-tree of user HRFs.
            return self.root

        if node is None:
            node = self.root

        axis = depth % 3  # Cycle through x, y, z

        h_val = (hrf.x, hrf.y, hrf.z)[axis]
        n_val = (node.x, node.y, node.z)[axis]

        # Handle duplicates by jittering location.
        # 3.5: the pre-fix loop `for val in (hrf.x, hrf.y, hrf.z): val += 1e-10`
        # mutated the loop variable, not the HRF's coordinates, so the
        # jitter never took effect. Assign directly instead.
        if h_val == n_val and hrf.x == node.x and hrf.y == node.y and hrf.z == node.z:
            if node.oxygenation == hrf.oxygenation:
                print(f"WARNING: Jittering location for {hrf.ch_name}, same location as the following node...\n{node.ch_name}")
                hrf.x += 1e-10
                hrf.y += 1e-10
                hrf.z += 1e-10
                # Refresh the axis comparison value for this pass so the
                # jittered coordinate actually informs the left/right
                # decision below.
                h_val = (hrf.x, hrf.y, hrf.z)[axis]
            
        # If the current node is less than the new node
        if h_val < n_val: 
            if node.left is None: # If the left node is empty
                node.left = hrf
                return node.left
            else: # If the left node is not empty
                return self.insert(hrf, depth + 1, node.left)
            
        # If the current node is greater than the new node
        else: 
            if node.right is None:
                node.right = hrf
                return node.right
            else:
                return self.insert(hrf, depth + 1, node.right)

    def filter(self, similarity_threshold = 0.95, node = None, **kwargs):
        """
        Filter on experimental contexts

        Arguments:
            similarity_threshold (float) - Threshold to allow or exclude HRF's based on context, defaults to 0.95
            node (HRF object) - Internal argument for passing the node to filter
            **kwargs - Any context keyword value pair to filter on (i.e. doi, age, etc)

        Returns:
            None
        """
        if node is None: # Set up filtering
            if self.root is None: # If nothing loaded yet
                raise ValueError("No HRFs loaded yet, nothing to filter")

            node = self.root # Set root at node
            self.context = {**self.context, **kwargs}

            # Historical no-op: the pre-fix code here called self.branch()
            # and discarded the result — the intent was to pre-filter to
            # a smaller sub-tree before compare_context, but filter()
            # always ran against self.root regardless, so the "branch"
            # was just a flag flip. Preserved as a direct assignment
            # after the hasher-branch-correctness fix made branch() with
            # no kwargs return an empty tree by design.
            self.branched = True

        if node.left: # If there's a left node
            self.filter(similarity_threshold, node.left)

        if node.right: # If there's a right node
            self.filter(similarity_threshold, node.right)

        # Check if the hrf matches the context
        context_similarity = self.compare_context(self.context, node.context, self.context_weights)
        if context_similarity < similarity_threshold: # If not similar enough to requested context
            self.delete(node) # Exclude derived HRF

    def compare_context(self, first_context, second_context, context_weights=None):
        """
        Compare two contexts to see how similar they are.

        3.1: Context values may be scalars ('flanker'), lists ([20, 30]),
        or None. Scalars are auto-wrapped to single-element lists before
        comparison so the inner `for value in values:` loop behaves the
        same way regardless of the raw type. A key missing from
        second_context (or set to None there) counts as zero matches.

        Arguments:
            first_context (dict) - Context to compare against
            second_context (dict) - Context to compare
            context_weights (dict) - Weights to attach to each context during similarity comparison

        Returns:
            float - Similarity score between 0.0 and 1.0 (1.0 being identical contexts)
        """
        weights = context_weights if context_weights is not None else self.context_weights
        context_similarity = []
        for key, values in first_context.items():
            if values is None: # Exclude from similarity comparison
                continue

            # Auto-wrap scalars so the inner loop doesn't iterate over
            # characters of a string or crash on a non-iterable.
            if not isinstance(values, (list, tuple)):
                values = [values]
            if len(values) == 0:
                continue

            other = second_context.get(key) if isinstance(second_context, dict) else None
            if other is None:
                other_values = []
            elif not isinstance(other, (list, tuple)):
                other_values = [other]
            else:
                other_values = list(other)

            same = 0 # Create a context specific similarity value
            for value in values:
                if value in other_values:
                    if weights: # If a context weight provided
                        same += 1 * weights.get(key, 1.0) # Weight similarity score
                    else:
                        same += 1

            # Calculate context-specific similarity and append (ND-002: use len(values) not len(first_context))
            context_similarity.append(same / len(values))

        return sum(context_similarity) / len(context_similarity) if len(context_similarity) > 0 else 0.0

    def branch(self, **kwargs):
        """
        Build a new tree filtered to nodes whose context dict contains
        every user-specified kwarg value.

        Semantics:
        - kwargs are ANDed together: a node must match all user-specified
          keys to be included.
        - Values within a single kwarg are ORed: `branch(task=['a','b'])`
          returns nodes whose task is 'a' or 'b'.
        - Only the kwargs passed to this call act as filters. The tree's
          own `self.context` defaults (e.g. `method='toeplitz'`) are not
          treated as implicit filters — that pre-fix behavior made
          `branch(task='nothing_matches')` return every node whose method
          happened to equal 'toeplitz', because the loop iterated every
          context-dict entry including defaults.
        - Matching uses the hasher populated by `load_hrfs` / `load_montage`
          keyed by the channel context VALUES (NE-002 fix).

        Arguments:
            **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)

        Returns:
            branch (tree object) - A new tree object filtered on the requested context
        """
        if kwargs:
            self.context = {**self.context, **kwargs} # Update context for downstream readers

        branch = tree()
        self.branched = True

        if not kwargs:
            # No filter specified — return an empty branch rather than the
            # whole tree. Callers wanting a full copy should use the
            # montage.branch() API or a dedicated clone method.
            return branch

        # For each kwarg, gather the set of candidate nodes (ORed across
        # that kwarg's values). Then intersect across kwargs to get the
        # final ANDed set.
        candidate_lists = []
        for key, values in kwargs.items():
            if values is None:
                continue
            if not isinstance(values, (list, tuple)):
                values = [values]
            matches = []
            seen_ids = set()
            for value in values:
                if value is None:
                    continue
                for node in self.hasher.search(value):
                    if id(node) not in seen_ids:
                        seen_ids.add(id(node))
                        matches.append(node)
            candidate_lists.append(matches)

        if not candidate_lists:
            return branch

        final_nodes = candidate_lists[0]
        for cl in candidate_lists[1:]:
            cl_ids = {id(n) for n in cl}
            final_nodes = [n for n in final_nodes if id(n) in cl_ids]

        for node in final_nodes:
            node_copy = node.copy()
            branch.insert(node_copy)
            # Populate the sub-tree's hasher keyed by the copied node's
            # context VALUES — mirrors load_hrfs (NE-002 semantics).
            if isinstance(node_copy.context, dict):
                for ctx_value in node_copy.context.values():
                    for hashable in _flatten_context_value(ctx_value):
                        branch.hasher.add(hashable, node_copy)

        return branch

    def nearest_neighbor(self, optode, max_distance, node = 'root', depth = 0, best = None, verbose = False):
        """
        Find the nearest neighbor to a target point in the 3D k-d tree.
        
        Arguments:
            optode (obj) -
            max_distance (float) - 
            node (obj) - Internal argument for passing the node to search
            depth (int) - Current dimensional orientation of the k-d tree
            best (tuple) - Best node found
        
        Returns:
            best (tuple) - The best node and distance found so far
        """
        # If first call, attach root to node
        if node == 'root':
            # NE-007: explicit empty-tree early return. If the caller
            # invokes nearest_neighbor on a tree that has never had a
            # node inserted, self.root is None and we short-circuit
            # before the recursive base case. Keeps the control flow
            # obvious for future readers.
            if self.root is None:
                if verbose: print(f"nearest_neighbor called on empty tree — no match")
                return None, float("inf")
            if verbose: print(f"Attaching root ({self.root.ch_name}) to search node for {optode.ch_name} search")
            node = self.root

        # Handle base cases
        if node is None:
            if best is not None:
                # ``best`` may be the ``(None, inf)`` sentinel returned by a
                # deeper-recursion empty-tree or no-match path, not a real
                # ``(node, distance)`` tuple. Guard the verbose-print branch
                # so it doesn't dereference ``ch_name`` on None — pre-fix
                # this crashed ``test_localization.py`` at collection time
                # whenever an optode had no in-radius match under verbose.
                if verbose:
                    if best[0] is not None:
                        print(f"No further branch, returning running best {best[0].ch_name}")
                    else:
                        print("No further branch, propagating empty-result sentinel")
                return best
            else:
                if verbose: print("No further branch and no best yet")
                return None, float("inf")
        
        k = 3 
        axis = depth % k

        #Define current and target points
        point = (node.x, node.y, node.z)
        target_point = (optode.x, optode.y, optode.z)

        # Calculate euclidian distance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([optode.x, optode.y, optode.z], [node.x, node.y, node.z])))
        if verbose: print(f"__________\nOptode {optode.ch_name}: {target_point} \nSearch Node {node.ch_name}: {point}\nDistance: {distance}\n_________")

        # Figure out which side needs exploring
        if target_point[axis] < point[axis]:
            near_branch = node.left
            far_branch = node.right
        else:
            near_branch = node.right
            far_branch = node.left

        # Check if this node is closer than the best found so far
        if (best is None or distance < best[1]) and optode.oxygenation == node.oxygenation:
            best = (node, distance)
            if verbose: print(f"New best found: {best[0].ch_name} - distance {best[1]}")

        # Search nearest branch
        best = self.nearest_neighbor(optode, max_distance, near_branch, depth + 1, best, verbose)

        # Check if far branch needs to be explored
        if best is None or abs(target_point[axis] - point[axis]) < best[1]:
            
            best = self.nearest_neighbor(optode, max_distance, far_branch, depth + 1, best, verbose)

        if best and best[1] <= max_distance:
            return best  # return the node only
        else:
            # S4: no canonical sentinel lives in the tree anymore. Callers
            # that want a canonical fallback should call get_canonical_hrf
            # with their own sfreq/duration and handle None explicitly.
            if verbose: print(f"No node within {max_distance} of {optode.ch_name}")
            return None, float("inf")

    def radius_search(self, optode, radius, node = None, depth=0, results=None):
        """
        Collect all HRFs within a radius and return them

        Arguments:
            node (HRF object) - HRF estimate to compare against
            optode (HRF object) - HRF optode object passed in
            radius (float) - Maximum euclidian distance of radius 
            depth (int) - Current depths of the search (range 0 - 2)
            results (list) - Nodes found to be within a range passed through resursions

        Returns:
            results (list) - List of tuples of HRF nodes and their distances within the radius
        """
        if node is None:
            return results or []

        if results is None:
            results = []

        axis = depth % 3
        node_coords = (node.x, node.y, node.z)
        optode_coords = (optode.x, optode.y, optode.z)

        # Check distance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(optode_coords, node_coords)))
        if distance <= radius:
            results.append((node, distance))

        # Decide which branches to explore
        if optode_coords[axis] - radius < node_coords[axis]:
            self.radius_search(optode, radius, node.left, depth + 1, results)
            
        if optode_coords[axis] + radius > node_coords[axis]:
            self.radius_search(optode, radius, node.right, depth + 1, results)

        return results

    def gather(self, node, oxygenation = None):
        """
        Gather all HRFs in the tree and return them as a dictionary
        Arguments:
            node (HRF object) - Node to start gathering from
            oxygenation (bool) - If True, gather only oxygenated HRFs, if False
            gather only deoxygenated HRFs, if None gather all HRFs

        Returns:
            hrfs (dict) - Dictionary of all HRFs in the tree
        """
        # Issue 3.8: empty-tree guard. Callers (notably tree.filter after
        # deleting the last node) pass node=None to indicate an empty tree
        # rooted at self.root. Return an empty dict instead of AttributeError
        # on node.left access.
        if node is None:
            return {}
        print(f"Gathering node... {node}")
        hrfs = {}
        collect = False

        if node.left:
            hrfs |= self.gather(node.left, oxygenation)
        if node.right:
            hrfs |= self.gather(node.right, oxygenation)
        if node.ch_name[:9] != "canonical": 
            # Determine if node is requested to be saved
            if oxygenation == None:
                collect = True
            elif oxygenation == node.oxygenation:
                collect = True

            if collect: # Add HRF if collectable
                hrfs |= {
                f"{'-'.join(node.ch_name.split(' '))}-{node.doi}": {
                    "hrf_mean": np.asarray(node.trace).tolist(),
                    "hrf_std": np.asarray(node.trace_std).tolist(),
                    "location": [
                        node.x,
                        node.y,
                        node.z
                    ],
                    "oxygenation":node.oxygenation,
                    "sfreq": node.sfreq,
                    "context": node.context,
                    "estimates": node.estimates,
                    "locations": node.locations,
                    "estimate_sources": getattr(node, "estimate_sources", []),
                }
        }
        return hrfs

    def to_hrf_points(self, modality_tag = "fnirs"):
        """Yield modality-agnostic HRFPoint instances for every HRF in the tree.

        Bridges the fNIRS-specific kd-tree to the modality-agnostic
        :mod:`hrfunc.spatial` layer used by spatial selection shapes,
        the GUI's 3D viz, and (in the future) a parallel fMRI HRF
        pipeline. Internal coordinates are stored in meters; HRFPoints
        live in MNI millimeters per the spatial-layer convention, so
        coordinates are converted here at the boundary.

        fNIRS-specific fields that don't fit the generic schema
        (``oxygenation``, ``ch_name``, ``doi``) ride in
        ``HRFPoint.context`` alongside the existing study/task/etc.
        metadata. Consumers that don't care about them ignore the
        extra keys; consumers that do (e.g. an HbO/HbR-grouped viz)
        read them from context.

        Skips HRFs that lack a usable 3-element location — those
        can't be placed in MNI space and have no business in a
        spatial pipeline.

        Arguments:
            modality_tag (str) - Tag used to identify the source
                pipeline. Defaults to ``"fnirs"``; reserved for a
                future fMRI tree to override.

        Yields:
            HRFPoint - One per traversable HRF node.
        """
        from .spatial.point import HRFPoint

        if self.root is None:
            return

        nodes = self.gather(self.root)
        for key, payload in nodes.items():
            loc = payload.get("location")
            if loc is None or len(loc) < 3:
                continue
            ctx = dict(payload.get("context") or {})
            # Carry the fNIRS-specific bits and the tree's own key so
            # spatial-layer consumers can route by oxygenation / look
            # the source HRF back up without re-walking the tree.
            ctx.setdefault("oxygenation", payload.get("oxygenation"))
            ctx.setdefault("hrf_key", key)
            hrf_mean = np.asarray(payload.get("hrf_mean") or [], dtype=np.float64)
            std_raw = payload.get("hrf_std")
            hrf_std = (
                np.asarray(std_raw, dtype=np.float64)
                if std_raw is not None and len(std_raw) > 0
                else None
            )
            yield HRFPoint(
                xyz_mm=(
                    float(loc[0]) * 1000.0,
                    float(loc[1]) * 1000.0,
                    float(loc[2]) * 1000.0,
                ),
                hrf_mean=hrf_mean,
                hrf_std=hrf_std,
                sfreq=float(payload.get("sfreq") or 1.0),
                context=ctx,
                modality_tag=modality_tag,
            )

    def save(self, filename = 'tree_hrfs.json'):
        hrfs = self.gather(self.root)
        # Save to a JSON file
        with open(filename, "w") as file:
            json.dump(hrfs, file, indent=4)
        return

    def split_save(self, hbo_filename, hbr_filename):
        """
        Split the tree into oxygenated and deoxygenated HRFs
        and save the outputs

        Arguments:
            hbo_filename (str) - filename to save the HbO files
            hbr_filename (str) - filename to save the HbR files
        """
        hbo_hrfs = self.gather(self.root, oxygenation = True)
        with open(hbo_filename, "w") as file:
            json.dump(hbo_hrfs, file, indent=4)

        hbr_hrfs = self.gather(self.root, oxygenation = False)
        with open(hbr_filename, "w") as file:
            json.dump(hbr_hrfs, file, indent=4)

    def traverse(self, node = None):
        """
        In-order traversal of the tree for printing purposes
        
        Arguments:
            node (HRF object) - Node to start traversal from
        """
        if node is None:
            node = self.root

        if node.left:
            self.traverse(node.left)
        if node.right:
            self.traverse(node.right)

        print(f"Node {node.ch_name}")

    def merge(self, tree, node = None):
        """
        Merge another tree into this one.

        NE-006: pre-fix called `self.insert(node)` with the source tree's
        node reference, so the inserted node in `self` still carried the
        source's left/right pointers. Subsequent kd-tree operations on
        either tree could corrupt the other. Fix: insert a fresh copy
        (HRF.copy returns a node with left=right=None and deep-copied
        payload), and recurse on the SOURCE's children so we traverse
        the full source tree rather than the (empty-children) copy.

        Arguments:
            tree (tree object) - Tree to merge into this one
            node (HRF object) - Node to start merging from
        """
        if node is None:
            node = tree.root
        # Empty source tree — nothing to merge
        if node is None:
            return

        self.insert(node.copy())

        if node.left:
            self.merge(tree, node.left)

        if node.right:
            self.merge(tree, node.right)

    def delete(self, hrf):
        """
        Delete a node from the 3D k-d tree based on spatial position.
        
        Arguments:
            hrf (HRF) - The HRF node to delete
        """
        self.root = self._delete_recursive(self.root, hrf, 0)

    def _delete_recursive(self, node, hrf, depth):
        """
        Recursive helper function to delete a node from the k-d tree.

        Uses the standard kd-tree delete algorithm: if the node-to-delete has
        a right subtree, find the axis-minimum node in it, copy that node's
        payload into the current node, then recursively delete the minimum
        from the right subtree. If it only has a left subtree, do the same
        against the left subtree and then move the subtree to the right
        (preserving the `right >= node` kd-tree invariant).

        Arguments:
            node (HRF) - Current node in the recursion
            hrf (HRF) - The HRF node to delete (matched on x, y, z)
            depth (int) - Current depth in the tree

        Returns:
            node (HRF) - Updated node after deletion, or None if removed
        """
        if node is None:
            return None

        axis = depth % 3

        if node.x == hrf.x and node.y == hrf.y and node.z == hrf.z:
            if node.right:
                min_node = self._find_min(node.right, axis, depth + 1)
                self._copy_payload(min_node, node)
                node.right = self._delete_recursive(node.right, min_node, depth + 1)
            elif node.left:
                min_node = self._find_min(node.left, axis, depth + 1)
                self._copy_payload(min_node, node)
                # Standard kd-tree trick: recurse into left, then promote the
                # mutated subtree to the right side so the `right >= node`
                # invariant on this axis still holds.
                node.right = self._delete_recursive(node.left, min_node, depth + 1)
                node.left = None
            else:
                return None  # Leaf case — remove from parent

        elif (axis == 0 and hrf.x < node.x) or (axis == 1 and hrf.y < node.y) or (axis == 2 and hrf.z < node.z):
            node.left = self._delete_recursive(node.left, hrf, depth + 1)
        else:
            node.right = self._delete_recursive(node.right, hrf, depth + 1)

        return node

    def _copy_payload(self, src, dst):
        """
        Copy all HRF payload fields from src into dst, leaving dst's
        left/right child pointers untouched. Used by the kd-tree delete
        algorithm to "move" a replacement node into a deleted node's slot
        without disturbing the surrounding tree structure.

        Notes on what is and isn't copied:
        - trace / trace_std are np.copy'd to avoid sharing the underlying
          numpy array with src (aliasing would let in-place mutations on
          one node silently affect the other).
        - context / estimates / locations are shallow-copied for the same
          reason (they're mutable dict / list containers).
        - `built` IS copied so a future `.build()` call on dst doesn't
          re-process an already-processed trace.
        - `hrf_processes` / `process_names` / `process_options` are NOT
          copied. hrf_processes contains **bound methods** whose `self`
          points to src (e.g. `src.spline_interp`); copying them into dst
          would cause dst.build() to execute methods in src's context,
          which is cross-reference corruption. dst keeps its own default
          process configuration from HRF.__init__.
        """
        dst.x = src.x
        dst.y = src.y
        dst.z = src.z
        dst.doi = src.doi
        dst.ch_name = src.ch_name
        dst.oxygenation = src.oxygenation
        dst.sfreq = src.sfreq
        dst.length = src.length
        dst.trace = np.copy(src.trace) if isinstance(src.trace, np.ndarray) else src.trace
        dst.trace_std = np.copy(src.trace_std) if isinstance(src.trace_std, np.ndarray) else src.trace_std
        dst.context = dict(src.context) if isinstance(src.context, dict) else src.context
        dst.estimates = list(src.estimates) if src.estimates is not None else []
        dst.locations = list(src.locations) if src.locations is not None else []
        dst.built = src.built

    def _find_min(self, node, axis, depth):
        """
        Find the node with the minimum value in a given dimension.
        Arguments:
            axis (int) - Dimension to find the minimum in (0 for x, 1
            for y, 2 for z)
            depth (int) - Current depth in the tree
        
        Returns:
            node (HRF) - Node with the minimum value in the specified dimension
        """
        if node is None:
            return None

        if depth % 3 == axis:
            if node.left is None:
                return node
            return self._find_min(node.left, axis, depth + 1)

        left_min = self._find_min(node.left, axis, depth + 1)
        right_min = self._find_min(node.right, axis, depth + 1)

        return min([node, left_min, right_min], key=lambda n: getattr(n, ["x", "y", "z"][axis]) if n else float('inf'))

class HRF:
    def __init__(self, doi, ch_name, duration, sfreq, trace, trace_std = None, location = None, estimates = None, locations = None, context = None, estimate_sources = None, **kwargs):
        """
        Object for storing all information apart of an estimated HRF from an fNIRS optode

        Class functions:
            self.build() - Build the HRF to fit a new sampling frequency and run through processing requested
            self.spline_interp() - Resizes the HRF to new sampling frequency using spline interpolation
            self.smooth() - Smooths the HRF trace using a gaussian filter
            self.resample() - Resampled the HRF using the estimated HRF and it's standard deviation 
            self.plot() - Plots the current HRF trace attached to the class

        Class attributes:
            trace (list of floats) - A trace of the HRF
            trace_std (list of floats) - The standard deviation of the HRF over time
            duration (float) - Duration of the HRF in seconds
            sfreq (float) - Sampling frequency of the fNIRS device that the HRF estimate was recorded from
            location (list of floats) - Location of the optode the HRF was estimated from the fNIRS device
            plot (bool) - Request for whether to plot the HRF throughout it's preprocessing
            **kwargs - Context attributes to be updated, only used by class or developers

        """
        # Add doi
        self.doi = doi

        # Clean and add channel name
        self.ch_name = standardize_name(ch_name)
        self.oxygenation = _is_oxygenated(self.ch_name)

        # Attach passed into info to class 
        self.sfreq = sfreq
        self.length = int(round(self.sfreq * duration, 0))

        # Set the HRF mean and standard deviation of the trace
        self.trace = np.asarray(trace, dtype=np.float64)
        self.trace_std = np.asarray(trace_std, dtype=np.float64)

        if location is not None: # Grab location
            self.x = location[0]
            self.y = location[1]
            self.z = location[2]
        else:
            print(f"WARNING: No location passed in, using random locations for HRF")
            self.x = -1 + random.random() 
            self.y = -1 + random.random()
            self.z = -1 + random.random()

        # 3.7: mutable default args (estimates=[], locations=[], context=[])
        # replaced with None sentinels and materialized inside the function
        # so two HRF instances don't silently share the same list/dict.
        if context:
            # NOTE: pre-fix accepted a list default `context=[]`. Keep
            # truthiness check so an empty-but-non-None value still falls
            # back to the default template.
            self.context = dict(context) if isinstance(context, dict) else {}
        else:
            self.context = {
                'method': 'toeplitz',
                'doi': doi,
                'study': None,
                'task': None,
                'conditions': None,
                'stimulus': None,
                'intensity': None,
                'duration': duration,
                'protocol': None,
                'age_range': None,
                'demographics': None
            }
        unexpected = set(kwargs) - set(self.context)
        if unexpected:
            raise ValueError(f"Unexpected contexts cannot be added: {unexpected}\n\nMake sure the contexts your searching for are within the available contexts: {','.join(self.context.keys())}")
        self.context.update({key: value for key, value in kwargs.items() if key in self.context})

        self.left = None
        self.right = None

        self.estimates = list(estimates) if estimates is not None else []
        self.locations = list(locations) if locations is not None else []
        # Provenance: a source id per estimate (e.g. the scan it came from), so
        # a saved+reloaded multi-subject montage can still report and remove a
        # specific subject's contribution. Kept parallel to ``estimates`` — pad
        # with None for estimates that predate provenance (bundled HRFs, older
        # saves) so indices always line up.
        self.estimate_sources = (
            list(estimate_sources) if estimate_sources is not None else []
        )
        if len(self.estimate_sources) < len(self.estimates):
            self.estimate_sources += [None] * (
                len(self.estimates) - len(self.estimate_sources)
            )

        # NE-003: process_options must be the same length as hrf_processes
        # and process_names so the zip in build() produces one iteration
        # per configured step. Pre-fix this was `[]`, making zip zero-
        # iterate and build() silently do nothing.
        self.hrf_processes = [self.spline_interp]
        self.process_names = ['spline_interpolate']
        self.process_options = [None]

        self.built = False

    def __repr__(self):
        """String representation of the HRF object.

        Returns:
            str: A string summarizing the HRF object.
        """
        return f"HRF: {self.doi} - {self.ch_name} \nSampling frequency: {self.sfreq}\nLocation: [{self.x}, {self.y}, {self.z}]\nTrace: {self.trace}\nTrace standrad deviation: {self.trace_std}"

    def copy(self):
        """Create a deep copy of this HRF object.

        Returns:
            HRF: A new HRF object with copies of all data including trace_std.
        """
        return HRF(
            doi=self.doi,
            ch_name=self.ch_name,
            duration=float(self.context.get('duration', 30.0)),
            sfreq=self.sfreq,
            trace=np.copy(self.trace) if isinstance(self.trace, np.ndarray) else list(self.trace) if self.trace is not None else [],
            trace_std=np.copy(self.trace_std) if isinstance(self.trace_std, np.ndarray) else list(self.trace_std) if self.trace_std is not None else None,
            location=[self.x, self.y, self.z],
            estimates=[list(e) for e in self.estimates] if self.estimates else [],
            locations=[list(loc) for loc in self.locations] if self.locations else [],
            context=dict(self.context) if self.context else {}
        )

    def build(self, new_sfreq, plot = False, show = False):
        """ Run through the processes requested for generating an hrf """
        self.target_length = new_sfreq * float(self.context['duration'])
        # 3.10: derive hrf_type from oxygenation instead of reading
        # self.type, which was never set anywhere and would AttributeError.
        hrf_type = "hbo" if self.oxygenation else "hbr"
        for process, process_name, process_option in zip(self.hrf_processes, self.process_names, self.process_options):

            if process_option == None:
                self.trace = process(self.trace)
            else:
                self.trace = process(self.trace, process_option)

            if plot: # Plot the processing step results
                title = f"HRF - {process_name}"
                filename = f"plots/{'-'.join(process_name.split(' ')).lower()}_{hrf_type}_hrf_results.png"
                self.plot(title, filename, show)
        self.built = True

    def update_centroid(self):
        """
        Update the centroid of the HRF based on the locations provided
        """
        # Format locations as a numpy array
        numpy_locations = np.array(self.locations)

        # Calculate centroid
        centroid = numpy_locations.mean(axis = 0)
        
        # Update class variables
        self.x, self.y, self.z = centroid[0], centroid[1], centroid[2]

    def spline_interp(self, trace = None):
        """
        Use spline interpolation to resample the HRF to a new size that fits
        self.target_length. Accepts an optional trace argument so it can be
        used as a pipeline step in build() (which calls
        `self.trace = process(self.trace)` on each configured process).

        Arguments:
            trace (array-like) - Source trace to resample. Defaults to
                self.trace if omitted.

        Returns:
            resampled_trace (np.ndarray) - Trace resampled to target_length
        """
        if trace is None:
            trace = self.trace
        # Original list
        hrf_indices = np.linspace(0, len(trace) - 1, len(trace))

        # Create a spline interpolation function
        spline = interp1d(hrf_indices, trace, kind='cubic')
        new_indices = np.linspace(0, len(trace) - 1, int(self.target_length))

        # Resampled list
        return spline(new_indices)

    def smooth(self, a):
        """
        Function that uses a gaussian filter to smooth the HRF trace.

        Function attributes:
            a (float) - Sigma value used in gaussian filter to dictate how much the HRF is smoothed
        """
        print(f'Smoothing HRF trace with Gaussian filter (sigma = {a})...')
        # NE-004: pre-fix called self.gaussian_filter1d which was never
        # imported or defined. Use the module-level scipy import.
        self.trace = gaussian_filter1d(self.trace, a)

        
    def normalize(self):
        """
        Function to normalize the trace between 0 and 1, useful for machine learning
        """
        self.trace = (self.trace - np.min(self.trace)) / (np.max(self.trace) - np.min(self.trace))

    def scale(self):
        """
        Function to scale around 1 using L2 normalization
        """
        self.trace /= np.linalg.norm(self.trace)
    
    def resample(self, std_seed = 0.0):
        """
        This resample function is an experimental resampling method for fNIRS (and potentially fMRI)
        for generating a new sample for machine learning and artificial intelligence training. The 
        general idea is to shift the HRF trace slightly within a confidence interval before deconvolving
        to generate multiple resampled fNIRS samples.

        Function attributes:
            std_seed (float) - Standard deviation seed between -3 and 3 to resample from the HRF trace deviation

        Returns:
            resampled_trace (list of floats) - A new resampled HRF trace
        """
        if self.trace_std == None:
            raise ValueError(f"HRF does not have a trace deviation attached to it")
        # Resample trace
        return [mean + (std_seed * std) for mean, std in zip(self.trace, self.trace_std)]

    def plot(self, plot_path = None, show_legend = True, show = True):
        """
        Function to plot the current HRF in seconds.

        Parameters:
            plot_path (str, optional): Path to save the plot. If provided, saves the figure.
            show_legend (bool): Whether to show the HRF legend
            show (bool): Whether to display the figure (default True)
        """
        hrf_mean = self.trace
        hrf_std = self.trace_std
        time = np.arange(len(hrf_mean)) / self.sfreq  # Convert samples to seconds

        plt.figure(figsize=(8, 4))
        plt.plot(time, hrf_mean, label='Mean HRF', color='blue')
        plt.fill_between(time, hrf_mean - hrf_std, hrf_mean + hrf_std, color='blue', alpha=0.3, label='±1 SD')

        plt.xlabel('Time (s)')
        plt.ylabel('HRF amplitude')
        plt.title(f'Estimated HRF for {self.ch_name} with Standard Deviation')
        plt.grid(True)

        # Cleaner x-axis ticks based on time
        plt.xticks(np.arange(0, max(time) + 0.3, 2))  # e.g., every 2 seconds

        if show_legend:
            plt.legend()

        plt.tight_layout()

        if plot_path is not None:
            plt.savefig(plot_path)

        if show:
            plt.show()
            