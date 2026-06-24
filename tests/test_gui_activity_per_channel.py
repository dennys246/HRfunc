"""Activity tab — per-channel HRtree deconvolution wiring + UI.

Covers ``_compute_library_traces`` (per-scan matching), ``run_activity_sync``
forwarding the per-channel map to ``estimate_activity``, and the right-column
coverage/assignment UI.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nicegui")
pytest.importorskip("mne")

import mne  # noqa: E402
from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from hrfunc.gui.components import activity_panel, hrtree_match  # noqa: E402
from hrfunc.gui.state import AppState  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]


def _raw():
    raw = mne.io.RawArray(
        np.zeros((2, 10)),
        mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], 10.0, "hbo"),
        verbose="ERROR",
    )
    for i, ch in enumerate(raw.info["chs"]):
        ch["loc"][:3] = [0.05 + i * 0.001, 0.0, 0.0]
    return raw


def _match_result(covered=True):
    m = hrtree_match.ChannelMatch(
        "s1_d1_hbo", True, matched=covered,
        trace=[0.1, 0.9, 0.3] if covered else None,
        source="hbo:roi" if covered else None,
        distance_mm=4.0 if covered else None,
    )
    u = hrtree_match.ChannelMatch("s1_d1_hbr", False, matched=False)
    return hrtree_match.MatchResult(
        matches=[m, u], n_candidate_hrfs=2, n_rois=1,
        strategy="individual", radius_mm=20.0,
    )


# ---------------------------------------------------------------------------
# _compute_library_traces
# ---------------------------------------------------------------------------


class TestComputeLibraryTraces:
    def test_none_when_not_library(self):
        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_TOEPLITZ, library_per_channel=True,
        )
        assert activity_panel._compute_library_traces(AppState(), _raw(), opts) is None

    def test_none_when_per_channel_off(self):
        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY, library_per_channel=False,
        )
        assert activity_panel._compute_library_traces(AppState(), _raw(), opts) is None

    def test_returns_map_in_per_channel_mode(self, monkeypatch):
        monkeypatch.setattr(
            hrtree_match, "match_channels_to_hrtree",
            lambda *a, **k: _match_result(covered=True),
        )
        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY, library_per_channel=True,
        )
        traces = activity_panel._compute_library_traces(AppState(), _raw(), opts)
        assert traces == {"s1_d1_hbo": [0.1, 0.9, 0.3]}

    def test_none_when_no_matches(self, monkeypatch):
        monkeypatch.setattr(
            hrtree_match, "match_channels_to_hrtree",
            lambda *a, **k: _match_result(covered=False),
        )
        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY, library_per_channel=True,
        )
        assert activity_panel._compute_library_traces(AppState(), _raw(), opts) is None


# ---------------------------------------------------------------------------
# run_activity_sync forwards the per-channel map
# ---------------------------------------------------------------------------


class TestRunActivitySyncPerChannel:
    def test_passes_library_traces_and_uncovered(self, monkeypatch):
        from hrfunc import hrfunc as hrf_module

        captured = {}

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                self.channels = {}

            def estimate_activity(self, nirx_obj, **kwargs):
                captured.update(kwargs)
                return nirx_obj

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY,
            library_traces={"s1_d1_hbo": [0.1, 0.2]},
            library_uncovered="canonical",
            library_trace=[9.9],  # must be ignored in favour of the map
        )
        activity_panel.run_activity_sync(_raw(), opts)

        assert captured["library_traces"] == {"s1_d1_hbo": [0.1, 0.2]}
        assert captured["library_uncovered"] == "canonical"
        assert "library_trace" not in captured

    def test_single_kernel_when_no_map(self, monkeypatch):
        from hrfunc import hrfunc as hrf_module

        captured = {}

        class _FakeMontage:
            def __init__(self, nirx_obj=None):
                self.channels = {}

            def estimate_activity(self, nirx_obj, **kwargs):
                captured.update(kwargs)
                return nirx_obj

        monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY,
            library_trace=[0.0, 1.0, 0.0],
            library_oxygenation=True,
        )
        activity_panel.run_activity_sync(_raw(), opts)

        assert captured["library_trace"] == [0.0, 1.0, 0.0]
        assert "library_traces" not in captured


# ---------------------------------------------------------------------------
# UI — coverage counts + per-channel assignment column
# ---------------------------------------------------------------------------


def _project(state, tmp_path):
    scan = ScanEntry(format="snirf", path=tmp_path / "a.snirf", display_name="A")
    state.manifest = Manifest(root=tmp_path, scans=(scan,))
    state.selected_scan = scan
    raw = _raw()
    state.processed_cache._cache[scan.path.resolve()] = raw
    state.processed_deconvolved.add(scan.path.resolve())
    return scan


class TestPerChannelUI:
    @pytest.mark.asyncio
    async def test_coverage_counts_and_assignment_render(
        self, user: User, tmp_path, monkeypatch
    ):
        state = AppState()
        _project(state, tmp_path)
        state.activity_options = None
        # Pretend the user has 1 visible ROI and force a known match result.
        monkeypatch.setattr(activity_panel, "_visible_roi_count", lambda _s: 1)
        monkeypatch.setattr(
            hrtree_match, "match_channels_to_hrtree",
            lambda *a, **k: _match_result(covered=True),
        )

        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY, library_per_channel=True,
        )

        @ui.page("/_apc")
        def _p() -> None:
            activity_panel._render_library_source_status(
                state, state.selected_scan, opts, bulk_mode=False
            )

        await user.open("/_apc")
        await user.should_see("1/2 channels covered")
        await user.should_see("Per-channel HRF assignment")
        await user.should_see("uncovered")  # the HbR channel

    @pytest.mark.asyncio
    async def test_skip_warns_about_dropped_channels(
        self, user: User, tmp_path, monkeypatch
    ):
        state = AppState()
        _project(state, tmp_path)
        monkeypatch.setattr(activity_panel, "_visible_roi_count", lambda _s: 1)
        monkeypatch.setattr(
            hrtree_match, "match_channels_to_hrtree",
            lambda *a, **k: _match_result(covered=True),
        )
        opts = activity_panel.ActivityOptions(
            hrf_model=activity_panel.MODEL_LIBRARY,
            library_per_channel=True, library_uncovered="skip",
        )

        @ui.page("/_apc_skip")
        def _p() -> None:
            activity_panel._render_library_source_status(
                state, state.selected_scan, opts, bulk_mode=False
            )

        await user.open("/_apc_skip")
        await user.should_see("DROPPED")


# ---------------------------------------------------------------------------
# Group-montage mode (Phase 3): deconvolve every scan with the pooled HRFs
# ---------------------------------------------------------------------------


class TestActivityGroupMode:
    def test_group_count_skips_canonical(self):
        from pathlib import Path
        state = AppState()
        state.montage_cache[Path("/a")] = object()
        state.montage_cache[Path("/b")] = object()
        state.montage_cache[Path("/c")] = activity_panel._CanonicalResult(
            canonical_trace=np.zeros(3), duration=1.0, sfreq=10.0
        )
        assert activity_panel._group_subject_count(state) == 2

    def test_montage_for_scan_returns_group_when_enabled(self, tmp_path):
        scan = ScanEntry(format="snirf", path=tmp_path / "x.snirf")
        state = AppState()
        group = object()  # stand-in non-canonical montage
        state.project_montage = group
        state.activity_use_group_hrfs = True
        # Scan was never individually estimated, yet group mode supplies HRFs.
        assert activity_panel._montage_for_scan(state, scan) is group

    def test_group_off_uses_scans_own_montage(self, tmp_path):
        scan = ScanEntry(format="snirf", path=tmp_path / "x.snirf")
        state = AppState()
        own = object()
        state.montage_cache[scan.path.resolve()] = own
        state.project_montage = object()
        state.activity_use_group_hrfs = False
        assert activity_panel._montage_for_scan(state, scan) is own

    @pytest.mark.asyncio
    async def test_group_toggle_renders_when_two_subjects(self, user, tmp_path):
        from pathlib import Path
        state = AppState()
        state.montage_cache[Path("/a")] = object()
        state.montage_cache[Path("/b")] = object()
        state.project_montage = object()

        @ui.page("/_agroup")
        def _p() -> None:
            activity_panel._render_estimated_source_status(state, None, False)

        await user.open("/_agroup")
        await user.should_see("Group HRFs (2)")
