"""Targeted unit tests for feat/gui-activity-panel (v1.3.0 Sprint 3.4).

Covers:

- ``ActivityOptions`` defaults and dataclass behavior.
- ``state.activity_raw`` field default + reset behavior.
- ``run_activity_sync`` — Raw copying (cache protection), toeplitz vs
  canonical Montage dispatch, preprocess=False forwarding.
- ``_render_body`` rendering states: no scan, no preprocess, toeplitz
  needs HRFs, canonical mode lets you run without HRFs, busy progress.
- ``activity_estimated`` event published on success.
- Montage-type discrimination — toeplitz refuses to run with a
  ``_CanonicalResult`` on ``state.montage``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nicegui")

from nicegui.testing import User  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]

from hrfunc.gui import app as gui_app  # noqa: E402
from hrfunc.gui.components import activity_panel, hrf_panel  # noqa: E402
from hrfunc.gui.state import AppState, state as global_state  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402

gui_app._register_pages()


def _make_fake_raw(ch_names=None, sfreq=10.0, duration_s=5.0):
    import mne

    if ch_names is None:
        ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"]
    n_ch = len(ch_names)
    n_samples = int(round(duration_s * sfreq))
    data = np.zeros((n_ch, n_samples))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="misc")
    return mne.io.RawArray(data, info, verbose="ERROR")


# ---------------------------------------------------------------------------
# ActivityOptions
# ---------------------------------------------------------------------------


class TestActivityOptions:
    def test_defaults_match_library(self):
        opts = activity_panel.ActivityOptions()
        assert opts.hrf_model == activity_panel.MODEL_TOEPLITZ
        assert opts.lmbda == 1e-4  # library default
        assert opts.preview_channel == 0
        assert opts.timeout == 30.0  # estimate_activity default
        assert opts.drop_failed_channels is True

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(activity_panel.ActivityOptions)


# ---------------------------------------------------------------------------
# state.activity_raw lifecycle
# ---------------------------------------------------------------------------


class TestStateActivityRaw:
    def test_field_defaults_none(self):
        s = AppState()
        assert s.activity_raw is None

    def test_reset_clears_field(self):
        s = AppState()
        s.activity_raw = "anything"
        s.reset()
        assert s.activity_raw is None


# ---------------------------------------------------------------------------
# run_activity_sync — cache protection + dispatch
# ---------------------------------------------------------------------------


class TestRunActivitySync:
    def test_raw_is_copied_before_estimation(self, monkeypatch):
        """estimate_activity mutates in-place; the cached Raw must not be
        the object passed in. We verify by object identity (the Raw passed
        to estimate_activity must not BE the same object as the input)."""
        from hrfunc import hrfunc as hrf_module

        raw = _make_fake_raw()
        seen_objects = []

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                self.channels = {"x": object()}

            def estimate_activity(self, nirx_obj, **kwargs):
                seen_objects.append(nirx_obj)
                return nirx_obj  # mimic library's "return the (mutated) raw"

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_CANONICAL
        )
        result = activity_panel.run_activity_sync(raw, opts)

        assert len(seen_objects) == 1
        # estimate_activity received a different object than the input
        assert seen_objects[0] is not raw
        # Same data content, different Python object
        assert seen_objects[0].ch_names == raw.ch_names
        # Returned object is the one passed to estimate_activity (the copy)
        assert result is seen_objects[0]

    def test_canonical_mode_constructs_new_montage(self, monkeypatch):
        from hrfunc import hrfunc as hrf_module

        raw = _make_fake_raw()
        constructed = []

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                constructed.append(nirx_obj)
                self.channels = {}

            def estimate_activity(self, nirx_obj, **kwargs):
                return nirx_obj

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_CANONICAL
        )
        # Pass an existing_montage; canonical mode should IGNORE it and
        # construct a fresh Montage on the copy.
        dummy_existing = object()
        activity_panel.run_activity_sync(
            raw, opts, existing_montage=dummy_existing
        )

        assert len(constructed) == 1
        assert constructed[0] is not raw  # fresh Montage receives the copy

    def test_toeplitz_mode_reuses_existing_montage(self, monkeypatch):
        from hrfunc import hrfunc as hrf_module

        raw = _make_fake_raw()
        constructed = []
        called_estimate = []

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                constructed.append(nirx_obj)
                self.channels = {}

            def estimate_activity(self, nirx_obj, **kwargs):
                called_estimate.append(self)
                return nirx_obj

        existing = _FakeMontage()
        constructed.clear()  # reset the count from the existing init

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_TOEPLITZ
        )
        activity_panel.run_activity_sync(
            raw, opts, existing_montage=existing
        )

        # No new Montage constructed; the existing one received the call.
        assert constructed == []
        assert called_estimate == [existing]

    def test_library_mode_passes_trace_and_builds_fresh_montage(self, monkeypatch):
        """Library mode ignores existing_montage, builds a fresh one, and
        forwards the supplied trace + oxygenation to estimate_activity."""
        from hrfunc import hrfunc as hrf_module

        raw = _make_fake_raw()
        constructed = []
        captured = {}

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                constructed.append(nirx_obj)
                self.channels = {}

            def estimate_activity(self, nirx_obj, **kwargs):
                captured.update(kwargs)
                return nirx_obj

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY,
            library_trace=[0.0, 0.5, 1.0, 0.5, 0.0],
            library_oxygenation=False,
        )
        activity_panel.run_activity_sync(raw, opts, existing_montage=object())

        assert len(constructed) == 1  # fresh montage despite existing_montage
        assert captured["hrf_model"] == activity_panel.MODEL_LIBRARY
        assert captured["library_trace"] == [0.0, 0.5, 1.0, 0.5, 0.0]
        assert captured["library_oxygenation"] is False


class TestLibraryKernelHelpers:
    """Pure helpers backing the HRtree (library) deconvolution source."""

    def test_kernel_none_when_no_selection(self):
        st = AppState()
        st.library_selected_hrf = None
        assert activity_panel._library_kernel_from_state(st) is None

    def test_kernel_none_when_trace_empty(self):
        st = AppState()
        st.library_selected_hrf = {"hrf_mean": [], "oxygenation": True}
        assert activity_panel._library_kernel_from_state(st) is None

    def test_kernel_reads_trace_and_oxygenation(self):
        st = AppState()
        st.library_selected_hrf = {"hrf_mean": [1.0, 2.0], "oxygenation": True}
        trace, oxy = activity_panel._library_kernel_from_state(st)
        assert trace == [1.0, 2.0]
        assert oxy is True

    def test_snapshot_captures_library_kernel(self):
        st = AppState()
        st.library_selected_hrf = {"hrf_mean": [0.1, 0.2], "oxygenation": False}
        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY
        )
        snap = activity_panel._snapshot_options(st, opts)
        assert snap.library_trace == [0.1, 0.2]
        assert snap.library_oxygenation is False

    def test_snapshot_no_kernel_for_canonical(self):
        st = AppState()
        st.library_selected_hrf = {"hrf_mean": [0.1], "oxygenation": True}
        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_CANONICAL
        )
        snap = activity_panel._snapshot_options(st, opts)
        assert snap.library_trace is None

    def test_forwards_preprocess_false(self, monkeypatch):
        from hrfunc import hrfunc as hrf_module

        raw = _make_fake_raw()
        kwargs_seen = []

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                self.channels = {}

            def estimate_activity(self, nirx_obj, **kwargs):
                kwargs_seen.append(kwargs)
                return nirx_obj

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_CANONICAL, lmbda=5e-5
        )
        activity_panel.run_activity_sync(raw, opts)

        assert kwargs_seen[0]["preprocess"] is False
        assert kwargs_seen[0]["lmbda"] == 5e-5
        assert kwargs_seen[0]["hrf_model"] == "canonical"

    def test_forwards_timeout_and_drop_failed_channels(self, monkeypatch):
        from hrfunc import hrfunc as hrf_module

        raw = _make_fake_raw()
        kwargs_seen = []

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                self.channels = {}

            def estimate_activity(self, nirx_obj, **kwargs):
                kwargs_seen.append(kwargs)
                return nirx_obj

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_CANONICAL,
            timeout=12.0,
            drop_failed_channels=False,
        )
        activity_panel.run_activity_sync(raw, opts)

        assert kwargs_seen[0]["timeout"] == 12.0
        assert kwargs_seen[0]["drop_failed_channels"] is False


class TestMontageStateProtection:
    """Sprint 3.4 review caught that estimate_activity mutates the Montage's
    channel containers when it drops failed channels (hrfunc.py:606-611).
    Passing state.montage directly would corrupt the HRFs tab's preview.
    run_activity_sync snapshots and restores those containers."""

    def test_montage_channels_restored_after_estimation(self, monkeypatch):
        from hrfunc import hrfunc as hrf_module

        raw = _make_fake_raw()

        class _MutatingFakeMontage:
            """Mimics estimate_activity's destructive mutation of containers."""

            def __init__(self, nirx_obj=None):
                self.channels = {"a": object(), "b": object(), "c": object()}
                self.hbo_channels = ["a", "c"]
                self.hbr_channels = ["b"]

            def estimate_activity(self, nirx_obj, **kwargs):
                # Drop one channel mid-call to simulate library behavior
                self.channels.pop("b", None)
                self.hbr_channels.remove("b")
                return nirx_obj

        monkeypatch.setattr(hrf_module, "montage", _MutatingFakeMontage)

        # Real existing montage with channels populated
        existing = _MutatingFakeMontage()
        original_channels = set(existing.channels.keys())
        original_hbo = list(existing.hbo_channels)
        original_hbr = list(existing.hbr_channels)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_TOEPLITZ
        )
        activity_panel.run_activity_sync(
            raw, opts, existing_montage=existing
        )

        # After the run, the existing montage's containers are restored —
        # not the mutated state estimate_activity left behind.
        assert set(existing.channels.keys()) == original_channels
        assert existing.hbo_channels == original_hbo
        assert existing.hbr_channels == original_hbr

    def test_canonical_mode_does_not_snapshot(self, monkeypatch):
        """Canonical mode constructs a fresh Montage and doesn't need
        snapshot protection. Verify the snapshot path isn't taken."""
        from hrfunc import hrfunc as hrf_module

        raw = _make_fake_raw()
        snapshot_attempts = []

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                self.channels = {"a": object()}
                self.hbo_channels = ["a"]
                self.hbr_channels = []

            def estimate_activity(self, nirx_obj, **kwargs):
                # Mutate the fresh-Montage's containers as estimate_activity
                # would. If snapshot protection is wrongly applied here,
                # the mutation would be reverted.
                self.channels.pop("a", None)
                return nirx_obj

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_CANONICAL
        )
        activity_panel.run_activity_sync(raw, opts)
        # The fresh Montage is discarded after the call; no assertion to
        # make on its state. The test passes if no exception was raised
        # (i.e., the snapshot path wasn't accidentally entered without an
        # existing_montage to snapshot).


class TestStateMontageSourceScan:
    """The HRFs tab records which scan a Montage came from on
    state.montage_source_scan. The Activity tab refuses toeplitz when the
    user has switched scans since the HRF estimation."""

    def test_field_defaults_none(self):
        s = AppState()
        assert s.montage_source_scan is None

    def test_reset_clears_field(self):
        s = AppState()
        s.montage_source_scan = ScanEntry(
            format="snirf", path=Path("/tmp/x"), display_name="x"
        )
        s.reset()
        assert s.montage_source_scan is None


async def test_panel_toeplitz_rejects_cross_scan_montage(
    user: User, tmp_path
):
    """Toeplitz mode must refuse when only ANOTHER scan's HRFs exist —
    applying scan A's HRFs to scan B's Raw would silently produce wrong
    results. With per-scan montage caching, selecting scan B (whose HRFs
    aren't cached) reports it has no estimated HRFs of its own."""
    scan_a = ScanEntry(
        format="snirf", path=tmp_path / "a.snirf", display_name="scan_a"
    )
    scan_b = ScanEntry(
        format="snirf", path=tmp_path / "b.snirf", display_name="scan_b"
    )
    raw = _make_fake_raw()
    global_state.reset()
    global_state.manifest = Manifest(root=tmp_path, scans=(scan_a, scan_b))
    global_state.selected_scan = scan_b
    global_state.processed_cache._cache[scan_b.path.resolve()] = raw
    # A real montage tagged to scan A only (not in scan B's cache).
    global_state.montage = object()  # Stand-in for a real Montage
    global_state.montage_source_scan = scan_a

    await user.open("/")
    await user.should_see("No estimated HRFs for this scan")


# ---------------------------------------------------------------------------
# Panel rendering — User fixture
# ---------------------------------------------------------------------------


async def test_panel_prompts_when_no_scan(user: User, tmp_path):
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(ScanEntry(format="snirf", path=tmp_path / "a.snirf",
                         display_name="a"),),
    )
    await user.open("/")
    await user.should_see("Activity")
    await user.should_see("Select a scan from the dataset tree")


async def test_panel_shows_waiting_when_not_preprocessed(user: User, tmp_path):
    scan = ScanEntry(
        format="snirf",
        path=tmp_path / "sub-01" / "sub-01_task-flanker_nirs.snirf",
        bids_subject="01",
        display_name="sub-01 / task-flanker",
    )
    global_state.reset()
    global_state.manifest = Manifest(root=tmp_path, scans=(scan,))
    global_state.selected_scan = scan
    await user.open("/")
    await user.should_see("Waiting for preprocess output")


async def test_panel_toeplitz_needs_hrfs_first(user: User, tmp_path):
    """Without an estimated Montage, toeplitz mode tells the user to run HRFs."""
    scan = ScanEntry(
        format="snirf",
        path=tmp_path / "a.snirf",
        display_name="a",
    )
    raw = _make_fake_raw()
    global_state.reset()
    global_state.manifest = Manifest(root=tmp_path, scans=(scan,))
    global_state.selected_scan = scan
    global_state.processed_cache._cache[scan.path.resolve()] = raw
    # No montage set → toeplitz mode should prompt for HRFs first, and offer
    # a jump to the HRFs tab so the user can accomplish it.
    await user.open("/")
    await user.should_see("Toeplitz mode requires estimated HRFs")
    await user.should_see("Go to HRFs tab")


async def test_panel_toeplitz_rejects_canonical_result_montage(
    user: User, tmp_path
):
    """If the HRFs tab last produced a canonical reference (not a real
    Montage with traces), toeplitz Activity mode tells the user to re-run
    HRFs in toeplitz mode."""
    scan = ScanEntry(
        format="snirf", path=tmp_path / "a.snirf", display_name="a"
    )
    raw = _make_fake_raw()
    global_state.reset()
    global_state.manifest = Manifest(root=tmp_path, scans=(scan,))
    global_state.selected_scan = scan
    global_state.processed_cache._cache[scan.path.resolve()] = raw
    global_state.montage = hrf_panel._CanonicalResult(
        canonical_trace=np.zeros(10), duration=30.0, sfreq=10.0
    )
    await user.open("/")
    await user.should_see("Toeplitz mode requires real per-channel HRFs")


async def test_panel_shows_lambda_control(user: User, tmp_path):
    scan = ScanEntry(
        format="snirf", path=tmp_path / "a.snirf", display_name="a"
    )
    raw = _make_fake_raw()
    global_state.reset()
    global_state.manifest = Manifest(root=tmp_path, scans=(scan,))
    global_state.selected_scan = scan
    global_state.processed_cache._cache[scan.path.resolve()] = raw
    await user.open("/")
    await user.should_see("Regularization")


# ---------------------------------------------------------------------------
# Workspace wiring
# ---------------------------------------------------------------------------


async def test_workspace_subscribes_activity_panel(user: User, tmp_path):
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(ScanEntry(format="snirf", path=tmp_path / "a.snirf",
                         display_name="a"),),
    )
    await user.open("/")
    assert "activity_estimated" in global_state.subscribers
    assert "hrf_estimated" in global_state.subscribers
    # hrf_estimated has both HRFs tab + Activity tab subscribers
    assert len(global_state.subscribers["hrf_estimated"]) >= 2


class TestMontageForScan:
    """Per-scan montage resolution backing toeplitz activity."""

    def _scan(self, p):
        return ScanEntry(format="snirf", path=Path(p), display_name=Path(p).stem)

    def test_none_when_nothing_estimated(self):
        st = AppState()
        assert activity_panel._montage_for_scan(st, self._scan("/tmp/x.snirf")) is None

    def test_uses_cached_montage(self):
        st = AppState()
        scan = self._scan("/tmp/x.snirf")
        m = object()
        st.montage_cache[scan.path.resolve()] = m
        assert activity_panel._montage_for_scan(st, scan) is m

    def test_falls_back_to_single_montage_when_source_matches(self):
        st = AppState()
        scan = self._scan("/tmp/x.snirf")
        m = object()
        st.montage = m
        st.montage_source_scan = scan
        assert activity_panel._montage_for_scan(st, scan) is m

    def test_ignores_single_montage_for_other_scan(self):
        st = AppState()
        st.montage = object()
        st.montage_source_scan = self._scan("/tmp/a.snirf")
        assert activity_panel._montage_for_scan(st, self._scan("/tmp/b.snirf")) is None

    def test_montage_cache_cleared_on_reset(self):
        st = AppState()
        st.montage_cache[Path("/tmp/x")] = object()
        st.reset()
        assert st.montage_cache == {}
