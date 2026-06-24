"""HRFs tab — project-wide GROUP montage (between-subject mean + std).

A single scan's montage has one estimate per channel, so ``trace_std`` (the
across-subject std) is 0. Pooling every estimated scan's montage into one group
montage and re-distributing gives a real between-subject std band. Phase 1 does
this in the GUI layer from the per-scan ``montage_cache`` (no core change).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from hrfunc.gui.components import hrf_panel  # noqa: E402
from hrfunc.gui.state import AppState  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]


class _FakeNode:
    def __init__(self, estimate):
        self.estimates = [list(estimate)] if estimate is not None else []
        self.trace = None
        self.trace_std = None
        self.sfreq = 10.0


class _FakeMontage:
    """Duck-typed montage: just what _build_project_montage touches."""

    def __init__(self, ch_to_estimate):
        self.channels = {c: _FakeNode(e) for c, e in ch_to_estimate.items()}

    def generate_distribution(self):
        for n in self.channels.values():
            if n.estimates:
                n.trace = np.mean(n.estimates, axis=0)
                n.trace_std = np.std(n.estimates, axis=0)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


class TestBuildProjectMontage:
    def test_pools_estimates_and_makes_std_nonzero(self):
        m1 = _FakeMontage({"s1_d1_hbo": [1.0, 2.0, 3.0]})
        m2 = _FakeMontage({"s1_d1_hbo": [3.0, 4.0, 5.0]})
        group = hrf_panel._build_project_montage([("subjA", m1), ("subjB", m2)])
        node = group.channels["s1_d1_hbo"]
        assert len(node.estimates) == 2          # both subjects pooled
        assert np.allclose(node.trace, [2.0, 3.0, 4.0])   # between-subject mean
        assert np.all(node.trace_std > 0)        # real between-subject std
        # Provenance: each pooled estimate tagged by its source scan.
        assert node.estimate_sources == ["subjA", "subjB"]

    def test_unions_channels_across_subjects(self):
        m1 = _FakeMontage({"s1_d1_hbo": [1.0, 1.0]})
        m2 = _FakeMontage({"s1_d1_hbo": [3.0, 3.0], "s2_d1_hbo": [9.0, 9.0]})
        group = hrf_panel._build_project_montage([("a", m1), ("b", m2)])
        # Channel only subject 2 had still appears (with its one estimate).
        assert "s2_d1_hbo" in group.channels
        assert len(group.channels["s2_d1_hbo"].estimates) == 1
        assert group.channels["s2_d1_hbo"].estimate_sources == ["b"]

    def test_none_when_empty_or_canonical_only(self):
        assert hrf_panel._build_project_montage([]) is None
        canon = hrf_panel._CanonicalResult(
            canonical_trace=np.zeros(5), duration=1.0, sfreq=10.0
        )
        assert hrf_panel._build_project_montage([("c", canon)]) is None

    def test_does_not_mutate_source_montages(self):
        m1 = _FakeMontage({"s1_d1_hbo": [1.0, 2.0]})
        m2 = _FakeMontage({"s1_d1_hbo": [3.0, 4.0]})
        hrf_panel._build_project_montage([("a", m1), ("b", m2)])
        # Sources keep their single estimate (pooling worked on a copy).
        assert len(m1.channels["s1_d1_hbo"].estimates) == 1
        assert len(m2.channels["s1_d1_hbo"].estimates) == 1


class TestRebuildFromCache:
    def test_rebuild_reads_cache_and_skips_reestimated_duplicates(self):
        state = AppState()
        from pathlib import Path
        # Two scans cached; re-estimating scan A replaces its cache entry,
        # so the rebuilt group always has exactly one estimate per scan.
        state.montage_cache[Path("/a")] = _FakeMontage({"c": [1.0, 1.0]})
        state.montage_cache[Path("/b")] = _FakeMontage({"c": [3.0, 3.0]})
        hrf_panel._rebuild_project_montage(state)
        assert len(state.project_montage.channels["c"].estimates) == 2
        # "Re-estimate" scan A -> overwrite its cache entry, rebuild.
        state.montage_cache[Path("/a")] = _FakeMontage({"c": [2.0, 2.0]})
        hrf_panel._rebuild_project_montage(state)
        assert len(state.project_montage.channels["c"].estimates) == 2  # not 3


# ---------------------------------------------------------------------------
# Group preview rendering
# ---------------------------------------------------------------------------


class TestGroupPreview:
    @pytest.mark.asyncio
    async def test_group_view_shows_between_subject_band(self, user: User):
        state = AppState()
        from pathlib import Path
        state.montage_cache[Path("/a")] = _FakeMontage(
            {"s1_d1_hbo": [0.0, 2.0, 5.0, 2.0, 0.0]}
        )
        state.montage_cache[Path("/b")] = _FakeMontage(
            {"s1_d1_hbo": [0.0, 1.0, 4.0, 3.0, 1.0]}
        )
        hrf_panel._rebuild_project_montage(state)
        # Rely on the default (group view shown automatically when available).
        assert AppState().hrf_preview_group is True
        opts = hrf_panel.EstimationOptions()

        @ui.page("/_group")
        def _p() -> None:
            hrf_panel._render_preview_column(
                state, None, opts, bulk_mode=False
            )

        await user.open("/_group")
        await user.should_see("Group (2 subjects)")        # the toggle
        await user.should_see("between-subject ± std")     # group header
        await user.should_see("Save group HRFs")           # save action
        # The single-scan "no band" note must NOT appear (std is real now).
        await user.should_not_see("No ±std band shown")


class TestGroupSubjects:
    def test_subject_names_map_to_manifest(self, tmp_path):
        from hrfunc.io.manifest import Manifest, ScanEntry

        state = AppState()
        s1 = ScanEntry(format="snirf", path=tmp_path / "a.snirf",
                       display_name="subj-A")
        s2 = ScanEntry(format="snirf", path=tmp_path / "b.snirf",
                       display_name="subj-B")
        state.manifest = Manifest(root=tmp_path, scans=(s1, s2))
        state.montage_cache[s1.path.resolve()] = _FakeMontage({"c": [1.0]})
        state.montage_cache[s2.path.resolve()] = _FakeMontage({"c": [2.0]})
        assert hrf_panel._group_subject_names(state) == ["subj-A", "subj-B"]

    def test_canonical_entries_excluded_from_names(self, tmp_path):
        from hrfunc.io.manifest import Manifest, ScanEntry

        state = AppState()
        s1 = ScanEntry(format="snirf", path=tmp_path / "a.snirf",
                       display_name="subj-A")
        state.manifest = Manifest(root=tmp_path, scans=(s1,))
        state.montage_cache[s1.path.resolve()] = _FakeMontage({"c": [1.0]})
        state.montage_cache[(tmp_path / "canon.snirf").resolve()] = (
            hrf_panel._CanonicalResult(
                canonical_trace=np.zeros(3), duration=1.0, sfreq=10.0
            )
        )
        assert hrf_panel._group_subject_names(state) == ["subj-A"]


class TestGroupExclusion:
    def test_excluding_a_subject_drops_it_from_the_pool(self):
        from pathlib import Path
        state = AppState()
        state.montage_cache[Path("/a")] = _FakeMontage({"c": [1.0, 1.0]})
        state.montage_cache[Path("/b")] = _FakeMontage({"c": [3.0, 3.0]})
        state.montage_cache[Path("/c")] = _FakeMontage({"c": [5.0, 5.0]})
        hrf_panel._rebuild_project_montage(state)
        assert len(state.project_montage.channels["c"].estimates) == 3

        state.project_group_excluded.add(Path("/b"))
        hrf_panel._rebuild_project_montage(state)
        node = state.project_montage.channels["c"]
        assert len(node.estimates) == 2
        assert "/b" not in node.estimate_sources   # provenance reflects removal
