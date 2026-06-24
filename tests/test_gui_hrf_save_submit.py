"""HRFs tab — Save button + submission section placement.

Layout change: the Save button moved next to the Estimate button (popping in
only when there's a real montage to save), and the submission form moved
directly below Estimate and is now ALWAYS present — including the no-scan
empty state — so a user can share an existing HRF JSON without estimating
first.

Helpers are rendered in isolation (rather than through the full shell)
because the submission form also appears on the Export tab, so a whole-page
assertion couldn't tell the two placements apart; and because the right-
column preview can't render against a stub montage.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from hrfunc.gui.components import hrf_panel  # noqa: E402
from hrfunc.gui.state import AppState  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]


def _scan(tmp_path):
    return ScanEntry(format="snirf", path=tmp_path / "a.snirf",
                     display_name="SCANALPHA")


# ---------------------------------------------------------------------------
# Submission section — always present, including the empty state
# ---------------------------------------------------------------------------


class TestSubmissionAlwaysPresent:
    @pytest.mark.asyncio
    async def test_present_in_no_scan_empty_state(self, user: User, tmp_path):
        state = AppState()
        state.manifest = Manifest(root=tmp_path, scans=(_scan(tmp_path),))
        # No selected scan, nothing ticked -> empty state.

        @ui.page("/_hrf_empty")
        def _p() -> None:
            hrf_panel.render(state)

        await user.open("/_hrf_empty")
        await user.should_see("Select a scan to start estimating HRFs")
        # The submission form is rendered even with no scan selected.
        await user.should_see("Submit HRFs to the HRtree")


# ---------------------------------------------------------------------------
# Save button — pops in beside Estimate only when there's a montage to save
# ---------------------------------------------------------------------------


def _run_row_page(state: AppState, scan, route: str) -> None:
    opts = hrf_panel.EstimationOptions()

    @ui.page(route)
    def _p() -> None:
        hrf_panel._render_run_row(state, scan, [], opts)


class TestSaveButtonBesideEstimate:
    @pytest.mark.asyncio
    async def test_hidden_without_montage(self, user: User, tmp_path):
        state = AppState()
        scan = _scan(tmp_path)
        state.selected_scan = scan
        assert state.montage is None
        _run_row_page(state, scan, "/_rr_none")

        await user.open("/_rr_none")
        await user.should_see("Estimate HRFs")
        await user.should_not_see("Save")

    @pytest.mark.asyncio
    async def test_shown_with_real_montage(self, user: User, tmp_path):
        state = AppState()
        scan = _scan(tmp_path)
        state.selected_scan = scan
        # A non-None, non-canonical montage stub is enough to surface Save
        # (the click handler resolves the real montage lazily).
        state.montage = object()
        _run_row_page(state, scan, "/_rr_real")

        await user.open("/_rr_real")
        await user.should_see("Estimate HRFs")
        await user.should_see("Save")

    @pytest.mark.asyncio
    async def test_hidden_for_canonical_result(self, user: User, tmp_path):
        state = AppState()
        scan = _scan(tmp_path)
        state.selected_scan = scan
        # Canonical mode has no per-channel estimates to save.
        state.montage = hrf_panel._CanonicalResult(
            canonical_trace=np.zeros(10), duration=1.0, sfreq=10.0
        )
        _run_row_page(state, scan, "/_rr_canon")

        await user.open("/_rr_canon")
        await user.should_not_see("Save")

    @pytest.mark.asyncio
    async def test_hidden_while_busy(self, user: User, tmp_path):
        state = AppState()
        scan = _scan(tmp_path)
        state.selected_scan = scan
        state.montage = object()
        state.busy = True  # mid-run: Save shouldn't appear yet
        _run_row_page(state, scan, "/_rr_busy")

        await user.open("/_rr_busy")
        await user.should_not_see("Save")
