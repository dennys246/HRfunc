"""HRtree detail pane — lead with the active ROI's HRF when an ROI is selected.

The detail right column used to always show the single clicked HRF (with the
ROI average tucked below). Now, when the user SELECTS an ROI, it leads with
the ROI's aggregate HRF instead. ``_showing_active_roi`` is the gate that
distinguishes "an ROI is selected" from "a lone HRF was clicked".
"""
from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from hrfunc.gui.components import hrtree_panel  # noqa: E402
from hrfunc.gui.state import AppState, ROISlot  # noqa: E402


class TestShowingActiveRoi:
    def test_false_when_no_rois(self):
        state = AppState()
        assert hrtree_panel._showing_active_roi(state) is False

    def test_true_when_selection_is_active_anchor(self):
        state = AppState()
        anchor = {"_key": "hbo:roi1", "oxygenation": True, "hrf_mean": [1.0]}
        state.cluster_rois = [ROISlot(name="ROI 1", anchor=anchor)]
        state.cluster_active_index = 0
        # Selecting an ROI re-seeds the selection to its anchor (same object).
        state.library_selected_hrf = anchor
        assert hrtree_panel._showing_active_roi(state) is True

    def test_true_when_anchorless_roi_and_no_selection(self):
        state = AppState()
        # Shape-only ROI with no anchor; selecting it leaves selection None.
        state.cluster_rois = [ROISlot(name="ROI 1", anchor=None)]
        state.cluster_active_index = 0
        state.library_selected_hrf = None
        assert hrtree_panel._showing_active_roi(state) is True

    def test_false_when_inspecting_a_different_hrf(self):
        state = AppState()
        anchor = {"_key": "hbo:roi1", "oxygenation": True, "hrf_mean": [1.0]}
        state.cluster_rois = [ROISlot(name="ROI 1", anchor=anchor)]
        state.cluster_active_index = 0
        # User clicked a DIFFERENT individual HRF (a fresh dict, not the anchor).
        state.library_selected_hrf = {
            "_key": "hbo:other", "oxygenation": True, "hrf_mean": [2.0]
        }
        assert hrtree_panel._showing_active_roi(state) is False


class TestRenderActiveRoiDetailGuard:
    def test_returns_false_when_no_active_roi(self):
        # No rendering context needed: bails before any ui.* call.
        state = AppState()
        assert hrtree_panel._render_active_roi_detail(state) is False


class TestRoiAverageOxygenationPure:
    """ROI averages must NEVER pool HbO + HbR together (inverse responses)."""

    def test_indeterminate_returns_both_separately(self):
        st = AppState()
        st.library_oxygenation = "both"
        st.library_selected_hrf = None
        assert hrtree_panel._roi_average_oxygenations(st) == [True, False]

    def test_anchor_oxygenation_determines_single(self):
        st = AppState()
        st.library_oxygenation = "both"
        st.library_selected_hrf = {"oxygenation": True}
        assert hrtree_panel._roi_average_oxygenations(st) == [True]
        st.library_selected_hrf = {"oxygenation": False}
        assert hrtree_panel._roi_average_oxygenations(st) == [False]

    def test_panel_filter_determines_single(self):
        st = AppState()
        st.library_selected_hrf = None
        st.library_oxygenation = "hbo"
        assert hrtree_panel._roi_average_oxygenations(st) == [True]
        st.library_oxygenation = "hbr"
        assert hrtree_panel._roi_average_oxygenations(st) == [False]
