"""Regression tests for the GUI review bug sweep (fix/gui-review-bug-sweep).

Each test pins one defect found in the deep GUI review so it can't silently
come back:

1. ``AppState.set_manifest`` must clear ALL per-project scan/estimation state
   (not just the nav caches) so a project switch can't leak the previous
   project's montages into the next project's group HRF pool.
2. ``RawCache(maxsize=None)`` is unbounded -- the Activity cache relies on it
   so a bulk "Save all" sees every deconvolution.
3. ``_build_project_montage`` drops per-scan ``global_*`` aggregate nodes so
   the group's between-subject HRF isn't double-counted with a
   "global of globals".
4. ``_nanmean_or_none`` drops NaN as well as None so an all-NaN channel set
   returns None instead of a NaN/RuntimeWarning.

The ROI-save oxygenation-purity fix is covered in test_gui_hrtree_roi_detail
via ``_roi_average_oxygenations`` -- the save path now reuses that exact
helper, so the existing TestRoiAverageOxygenationPure also guards the save.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nicegui")

from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402
from hrfunc.io.raw_cache import RawCache  # noqa: E402


# ---------------------------------------------------------------------------
# 1. set_manifest clears per-project state (no cross-project leak)
# ---------------------------------------------------------------------------


class TestSetManifestClearsProjectState:
    def test_switch_drops_montage_and_estimation_state(self):
        from hrfunc.gui.state import AppState

        st = AppState()
        p = Path("/tmp/projectA/scan.snirf").resolve()
        # Simulate a fully-worked project A.
        st.montage_cache[p] = object()
        st.project_montage = object()
        st.montage = object()
        st.montage_source_scan = object()
        st.activity_source_scan = object()
        st.activity_raw = object()
        st.activity_cache.put(p, object())
        st.processed_deconvolved.add(p)
        st.quality_metrics[p] = {"raw": {}}
        st.project_group_excluded.add(p)
        st.events_rows = [object()]
        st.events_source_scan = object()

        # Switch to a different project.
        st.set_manifest(Manifest(root=Path("/tmp/projectB"), scans=()))

        # Every per-project field must be cleared -- otherwise project B's
        # group HRF would pool project A's subjects.
        assert st.montage_cache == {}
        assert st.project_montage is None
        assert st.montage is None
        assert st.montage_source_scan is None
        assert st.activity_source_scan is None
        assert st.activity_raw is None
        assert len(st.activity_cache) == 0
        assert st.processed_deconvolved == set()
        assert st.quality_metrics == {}
        assert st.project_group_excluded == set()
        assert st.events_rows is None
        assert st.events_source_scan is None

    def test_reset_and_set_manifest_clear_the_same_project_fields(self):
        # The two paths share ``_clear_project_data`` so they can't drift.
        from hrfunc.gui.state import AppState

        def _fill(st: AppState) -> None:
            st.montage_cache[Path("/tmp/x").resolve()] = object()
            st.project_montage = object()
            st.processed_deconvolved.add(Path("/tmp/x").resolve())
            st.quality_metrics[Path("/tmp/x").resolve()] = {}

        a, b = AppState(), AppState()
        _fill(a)
        _fill(b)
        a.reset()
        b.set_manifest(Manifest(root=Path("/tmp/b"), scans=()))
        for st in (a, b):
            assert st.montage_cache == {}
            assert st.project_montage is None
            assert st.processed_deconvolved == set()
            assert st.quality_metrics == {}

    def test_reidempotent_set_manifest_keeps_state(self):
        # Re-setting the SAME manifest is a no-op and must not wipe state.
        from hrfunc.gui.state import AppState

        st = AppState()
        man = Manifest(root=Path("/tmp/p"), scans=())
        st.set_manifest(man)
        st.montage_cache[Path("/tmp/p/s").resolve()] = object()
        st.set_manifest(man)  # same object -> guarded no-op
        assert len(st.montage_cache) == 1


# ---------------------------------------------------------------------------
# 2. RawCache(maxsize=None) is unbounded
# ---------------------------------------------------------------------------


class TestUnboundedRawCache:
    def test_none_maxsize_never_evicts(self):
        c = RawCache(maxsize=None)
        for i in range(64):
            c.put(Path(f"/tmp/s{i}.snirf"), object())
        assert len(c) == 64  # all retained; LRU(3) would have kept only 3

    def test_int_maxsize_still_evicts(self):
        c = RawCache(maxsize=3)
        for i in range(10):
            c.put(Path(f"/tmp/s{i}.snirf"), object())
        assert len(c) == 3

    def test_zero_or_negative_still_rejected(self):
        with pytest.raises(ValueError):
            RawCache(maxsize=0)
        with pytest.raises(ValueError):
            RawCache(maxsize=-1)


# ---------------------------------------------------------------------------
# 3. _build_project_montage drops per-scan global_* nodes
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(self, estimates):
        self.estimates = [list(e) for e in estimates]
        self.estimate_sources = []


class _FakeMontage:
    def __init__(self, channels):
        self.channels = channels
        self.regen_called = 0

    def generate_distribution(self):
        # No-op stand-in: lets the test inspect the pooled channels dict
        # without the real (global-rebuilding) distribution pass.
        self.regen_called += 1


class TestBuildProjectMontageDropsGlobals:
    def test_globals_not_pooled_and_real_channels_pooled(self):
        from hrfunc.gui.components import hrf_panel

        def _subject(val):
            return _FakeMontage({
                "S1_D1 hbo": _FakeNode([[val, val, val]]),
                # Per-scan generate_distribution synthesises these WITH a
                # single estimate (the subject's own grand mean).
                "global_hbo": _FakeNode([[9.0, 9.0, 9.0]]),
                "global_hbr": _FakeNode([[8.0, 8.0, 8.0]]),
            })

        sourced = [("subjA", _subject(1.0)), ("subjB", _subject(3.0))]
        group = hrf_panel._build_project_montage(sourced)

        assert group is not None
        # Globals must be stripped so the final generate_distribution rebuilds
        # them from the real channels only (no "global of globals").
        assert "global_hbo" not in group.channels
        assert "global_hbr" not in group.channels
        # The real channel pools BOTH subjects' estimates, tagged by source.
        node = group.channels["S1_D1 hbo"]
        assert len(node.estimates) == 2
        assert node.estimate_sources == ["subjA", "subjB"]
        assert group.regen_called == 1


# ---------------------------------------------------------------------------
# 4. _nanmean_or_none drops NaN as well as None
# ---------------------------------------------------------------------------


class TestNanmeanOrNone:
    def _fn(self):
        pytest.importorskip("mne")
        from hrfunc.gui.components import quality_panel

        return quality_panel._nanmean_or_none

    def test_all_nan_returns_none(self):
        assert self._fn()({"a": float("nan"), "b": float("nan")}) is None

    def test_all_none_returns_none(self):
        assert self._fn()({"a": None, "b": None}) is None

    def test_empty_or_none_returns_none(self):
        fn = self._fn()
        assert fn({}) is None
        assert fn(None) is None

    def test_mixes_drops_nan_and_none(self):
        fn = self._fn()
        out = fn({"a": 2.0, "b": float("nan"), "c": None, "d": 4.0})
        assert out == pytest.approx(3.0)
