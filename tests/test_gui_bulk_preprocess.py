"""Bulk preprocess affordance — shared helper + HRFs / Activity integration.

Background: ticking scans selects them for a bulk RUN, but the per-scan
config + preprocess controls used to be gated on ``selected_scan`` (a
row-click). With only checkboxes ticked the HRFs tab showed "Pick one of
the checked scans…" and no preprocess button, which read as "the checkbox
didn't fully select the file." Both panels' bulk runs already preprocess
each scan on demand (``ensure_deconvolved_raw``), so the fix is a shared
bulk affordance — readiness summary + one-click "Preprocess all checked" —
that renders on tick alone.

These tests lock:
- the shared ``scan_is_deconvolved`` readiness gate,
- ``render_preprocess_all_checked`` showing the right state (not-ready →
  button; all-ready → confirmation),
- HRFs bulk mode rendering global params + the affordance on tick-only
  (no "Pick one of the checked scans" prompt),
- Activity bulk mode rendering the affordance on tick-only.
"""
from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from hrfunc.gui.components import preprocess_panel  # noqa: E402
from hrfunc.gui.state import AppState  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]


def _manifest(tmp_path):
    scans = (
        ScanEntry(format="snirf",
                  path=tmp_path / "sub-01" / "a.snirf",
                  bids_subject="01", display_name="SCANALPHA"),
        ScanEntry(format="snirf",
                  path=tmp_path / "sub-02" / "b.snirf",
                  bids_subject="02", display_name="SCANBETA"),
    )
    return Manifest(root=tmp_path, scans=scans)


def _mark_deconvolved(state: AppState, scan: ScanEntry) -> None:
    """Make ``scan_is_deconvolved`` true without a real Raw: stub the cache
    entry (RawCache.__contains__ only checks the resolved path key) and record
    the deconvolution flag."""
    key = scan.path.resolve()
    state.processed_cache._cache[key] = object()
    state.processed_deconvolved.add(key)


def _tick(user: User, *paths) -> None:
    user.find(ui.tree).trigger("update:ticked", [str(p) for p in paths])


# ---------------------------------------------------------------------------
# Shared readiness gate
# ---------------------------------------------------------------------------


class TestScanIsDeconvolved:
    def test_false_when_uncached(self, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        assert not preprocess_panel.scan_is_deconvolved(
            state, state.manifest.scans[0]
        )

    def test_false_when_cached_but_glm_only(self, tmp_path):
        """In processed_cache but NOT processed_deconvolved (GLM mode) → not
        ready for HRF/activity estimation."""
        state = AppState()
        state.manifest = _manifest(tmp_path)
        scan = state.manifest.scans[0]
        state.processed_cache._cache[scan.path.resolve()] = object()
        assert not preprocess_panel.scan_is_deconvolved(state, scan)

    def test_true_when_cached_and_deconvolved(self, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        scan = state.manifest.scans[0]
        _mark_deconvolved(state, scan)
        assert preprocess_panel.scan_is_deconvolved(state, scan)


# ---------------------------------------------------------------------------
# render_preprocess_all_checked — direct render states
# ---------------------------------------------------------------------------


class TestRenderPreprocessAllChecked:
    @pytest.mark.asyncio
    async def test_not_ready_shows_button(self, user: User, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        scans = list(state.manifest.scans)

        @ui.page("/_ppc_not_ready")
        def _p() -> None:
            preprocess_panel.render_preprocess_all_checked(state, scans)

        await user.open("/_ppc_not_ready")
        await user.should_see("Preprocess all checked")
        await user.should_see("2 of 2 checked scans aren't")

    @pytest.mark.asyncio
    async def test_all_ready_shows_confirmation_no_button(
        self, user: User, tmp_path
    ):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        scans = list(state.manifest.scans)
        for s in scans:
            _mark_deconvolved(state, s)

        @ui.page("/_ppc_ready")
        def _p() -> None:
            preprocess_panel.render_preprocess_all_checked(state, scans)

        await user.open("/_ppc_ready")
        await user.should_see("ready")
        await user.should_not_see("Preprocess all checked")

    @pytest.mark.asyncio
    async def test_partial_counts_only_not_ready(self, user: User, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        scans = list(state.manifest.scans)
        _mark_deconvolved(state, scans[0])  # one ready, one not

        @ui.page("/_ppc_partial")
        def _p() -> None:
            preprocess_panel.render_preprocess_all_checked(state, scans)

        await user.open("/_ppc_partial")
        await user.should_see("1 of 2 checked scans aren't")
        await user.should_see("Preprocess all checked")


# ---------------------------------------------------------------------------
# Panel integration: tick-only (no row click) engages the affordance
# ---------------------------------------------------------------------------


class TestHrfBulkAffordance:
    @pytest.mark.asyncio
    async def test_tick_only_shows_affordance_and_params(
        self, user: User, tmp_path
    ):
        from hrfunc.gui.components import dataset_tree, hrf_panel

        state = AppState()
        state.manifest = _manifest(tmp_path)
        a, b = state.manifest.scans

        @ui.page("/_hrf_bulk")
        def _p() -> None:
            with ui.row():
                dataset_tree.render(state)
                hrf_panel.render(state)

        await user.open("/_hrf_bulk")
        _tick(user, a.path, b.path)

        assert state.selected_scan is None  # no row click
        await user.should_see("Bulk run on 2 checked scan")
        await user.should_see("Preprocess all checked")
        # Global params now render in bulk too (were row-click-only before).
        await user.should_see("Regularization (lambda)")
        await user.should_see("Duration (seconds)")
        # The old dead-end prompt is gone.
        await user.should_not_see("Pick one of the checked scans")


class TestActivityBulkAffordance:
    @pytest.mark.asyncio
    async def test_tick_only_shows_affordance(self, user: User, tmp_path):
        from hrfunc.gui.components import activity_panel, dataset_tree

        state = AppState()
        state.manifest = _manifest(tmp_path)
        a, b = state.manifest.scans

        @ui.page("/_act_bulk")
        def _p() -> None:
            with ui.row():
                dataset_tree.render(state)
                activity_panel.render(state)

        await user.open("/_act_bulk")
        _tick(user, a.path, b.path)

        assert state.selected_scan is None
        await user.should_see("Bulk run on 2 checked scan")
        await user.should_see("Preprocess all checked")
