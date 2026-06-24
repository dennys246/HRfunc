"""
Targeted unit tests for feat/progress-callbacks (v1.3.0 GUI foundation).

Adds an optional `progress_callback(current_index, total_channels, channel_name)`
parameter to `montage.estimate_hrf` and `montage.estimate_activity`. The callback
fires once at the start of each channel iteration so GUI/batch tooling can render
progress without polling.

Tests verify the wiring without exercising the full pipeline (which requires a
real fNIRS Raw fixture). Behavior under real data is covered by integration
tests in test_estimation.py (currently disabled at module level — KI-033).
"""

import inspect

import pytest


# ---------------------------------------------------------------------------
# Signature contracts: parameter exists with default None
# ---------------------------------------------------------------------------

class TestEstimateHrfSignature:
    def test_progress_callback_is_a_parameter(self):
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_hrf)
        assert 'progress_callback' in sig.parameters

    def test_progress_callback_defaults_to_none(self):
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_hrf)
        assert sig.parameters['progress_callback'].default is None

    def test_progress_callback_is_keyword_compatible(self):
        """Must be passable as a keyword arg so callers don't need to know
        the position of every other parameter."""
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_hrf)
        param = sig.parameters['progress_callback']
        assert param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )


class TestEstimateActivitySignature:
    def test_progress_callback_is_a_parameter(self):
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_activity)
        assert 'progress_callback' in sig.parameters

    def test_progress_callback_defaults_to_none(self):
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_activity)
        assert sig.parameters['progress_callback'].default is None

    def test_progress_callback_is_keyword_compatible(self):
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_activity)
        param = sig.parameters['progress_callback']
        assert param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )


# ---------------------------------------------------------------------------
# Source-pattern contracts: the callback is invoked inside the per-channel loop
# with the expected (index, total, name) signature
# ---------------------------------------------------------------------------

class TestEstimateHrfCallbackWiring:
    def test_loop_uses_enumerate(self):
        """Without enumerate(), there is no per-iteration index to pass to
        the callback. Regression guard: a future refactor that drops the
        index would silently break progress reporting."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_hrf)
        assert 'for i, (fnirs_signal, channel) in enumerate(' in source

    def test_total_channels_computed_before_loop(self):
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_hrf)
        assert 'total_channels = len(nirx_obj.info[\'chs\'])' in source

    def test_callback_invoked_with_index_total_standardized_name(self):
        """Name passed to callback must be the standardized form (matching
        self.channels keys) so GUI code can use a single naming convention
        across both estimate_hrf and estimate_activity callbacks."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_hrf)
        assert "progress_callback(i, total_channels, standardize_name(channel['ch_name']))" in source

    def test_callback_guarded_by_none_check(self):
        """None default must short-circuit so existing callers see no behavior
        change. Bare `progress_callback(...)` would raise TypeError on None."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_hrf)
        assert 'if progress_callback is not None:' in source


class TestEstimateActivityCallbackWiring:
    def test_loop_uses_enumerate_over_channel_snapshot(self):
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        assert 'all_channels = list(self.channels.keys())' in source
        assert 'for i, ch_name in enumerate(all_channels):' in source

    def test_total_channels_computed_before_loop(self):
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        assert 'total_channels = len(all_channels)' in source

    def test_callback_invoked_with_index_total_name(self):
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        assert 'progress_callback(i, total_channels, ch_name)' in source

    def test_callback_guarded_by_none_check(self):
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        assert 'if progress_callback is not None:' in source

    def test_callback_fires_before_global_skip(self):
        """A 'global' channel entry is skipped via `continue` but the callback
        must still fire for it — total_channels counts every entry in
        self.channels so the caller sees a monotonic 0..N-1 progression.
        Regression guard: if the callback moves below the continue, the
        progress bar will appear to stall on global entries."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        # Locate both lines and assert the callback appears first
        callback_idx = source.index('progress_callback(i, total_channels, ch_name)')
        global_skip_idx = source.index("if 'global' in ch_name: continue")
        assert callback_idx < global_skip_idx, (
            "progress_callback must fire before the 'global' skip so "
            "total_channels and the current_index stay consistent"
        )


# ---------------------------------------------------------------------------
# Backwards-compatibility contracts: existing call patterns still work
# ---------------------------------------------------------------------------

class TestBackwardsCompatibility:
    """progress_callback is purely additive. Existing kwargs must still bind."""

    def test_estimate_hrf_existing_kwargs_unchanged(self):
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_hrf)
        for name in ('nirx_obj', 'events', 'duration', 'lmbda',
                     'edge_expansion', 'preprocess'):
            assert name in sig.parameters

    def test_estimate_activity_existing_kwargs_unchanged(self):
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_activity)
        for name in ('nirx_obj', 'lmbda', 'hrf_model', 'preprocess',
                     'cond_thresh', 'timeout'):
            assert name in sig.parameters

    def test_estimate_hrf_progress_callback_keeps_positional_slot(self):
        """Backwards-compat contract: the historical positional parameters
        through ``progress_callback`` keep their exact order, so old positional
        calls like ``estimate_hrf(nirx, events, 30.0, 1e-3, 0.15, True, cb)``
        bind the same names.

        New parameters are APPENDED AFTER ``progress_callback`` (e.g. timeout,
        source_id) -- that is the correct, non-breaking way to extend the
        signature. So assert the leading prefix is intact (nothing inserted
        BEFORE progress_callback), NOT that progress_callback is literally the
        last parameter (which would force new params in front of it and break
        positional binding)."""
        from hrfunc.hrfunc import montage
        params = list(inspect.signature(montage.estimate_hrf).parameters)
        expected_prefix = [
            'self', 'nirx_obj', 'events', 'duration', 'lmbda',
            'edge_expansion', 'preprocess', 'progress_callback',
        ]
        assert params[:len(expected_prefix)] == expected_prefix

    def test_estimate_activity_progress_callback_keeps_positional_slot(self):
        from hrfunc.hrfunc import montage
        params = list(inspect.signature(montage.estimate_activity).parameters)
        expected_prefix = [
            'self', 'nirx_obj', 'lmbda', 'hrf_model', 'preprocess',
            'cond_thresh', 'timeout', 'progress_callback',
        ]
        assert params[:len(expected_prefix)] == expected_prefix
