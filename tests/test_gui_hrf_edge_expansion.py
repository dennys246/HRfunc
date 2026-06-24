"""HRFs tab — edge-expansion control + edge-instability QC.

The HRFs tab exposes ``edge_expansion`` (default 0.2) and forwards it to
``montage.estimate_hrf``. After estimation it flags channels whose HRF is much
noisier at the window edges than its center (a sign edge_expansion is too
small) and nudges the user to raise it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from hrfunc.gui.components import hrf_panel  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]


def _montage(channels):
    return SimpleNamespace(channels=channels)


# ---------------------------------------------------------------------------
# Option default + wiring to estimate_hrf
# ---------------------------------------------------------------------------


def test_default_edge_expansion_is_point_two():
    assert hrf_panel.EstimationOptions().edge_expansion == 0.2


def test_run_toeplitz_forwards_edge_expansion(monkeypatch):
    import mne
    from hrfunc import hrfunc as hrf_module

    captured = {}

    class _FakeMontage:
        def __init__(self, nirx_obj=None):
            self.channels = {}

        def estimate_hrf(self, raw, **kwargs):
            captured.update(kwargs)

        def generate_distribution(self):
            pass

    monkeypatch.setattr(hrf_module, "montage", _FakeMontage)

    raw = mne.io.RawArray(
        np.zeros((1, 50)),
        mne.create_info(["S1_D1 hbo"], 10.0, "hbo"),
        verbose="ERROR",
    )
    opts = hrf_panel.EstimationOptions(
        edge_expansion=0.35, selected_events=("a",)
    )
    impulse = [0] * 50
    impulse[10] = 1

    hrf_panel.run_toeplitz_sync(raw, opts, None, None, impulse)
    assert captured["edge_expansion"] == 0.35


# ---------------------------------------------------------------------------
# Edge-instability heuristic
# ---------------------------------------------------------------------------


# Heuristic reads node.trace (NOT trace_std, which is across-subject = 0 for a
# single-scan estimate). n=20, edge frac 0.15 -> 3 samples each end.
# Noisy edges: edges oscillate hard (high local std), center is a gentle bump
# (low local std). Clean: the response sits in the center (high std), edges flat.
_NOISY_EDGES = np.array(
    [6.0, -6.0, 6.0]
    + [0.0, 0.4, 0.8, 1.0, 1.2, 1.0, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    + [6.0, -6.0, 6.0]
)
_CLEAN = np.array(
    [0.0, 0.0, 0.05]
    + [0.0, 2.0, 5.0, 8.0, 10.0, 8.0, 5.0, 2.0, 0.0, -1.0, -2.0, -1.0, 0.0, 0.0]
    + [0.05, 0.0, 0.0]
)


class TestEdgeUnstableChannels:
    def test_flags_channel_with_noisy_edges(self):
        m = _montage({
            "s1_d1_hbo": SimpleNamespace(trace=_NOISY_EDGES),
            "s2_d1_hbo": SimpleNamespace(trace=_CLEAN),
        })
        flagged = hrf_panel._edge_unstable_channels(m)
        assert "s1_d1_hbo" in flagged   # noisy edges > 2× center
        assert "s2_d1_hbo" not in flagged  # response in center, flat edges

    def test_ratio_threshold_is_respected(self):
        m = _montage({"s1_d1_hbo": SimpleNamespace(trace=_NOISY_EDGES)})
        # A high ratio makes the same channel pass (not flagged).
        assert hrf_panel._edge_unstable_channels(m, ratio=50.0) == []
        # The default ratio flags it.
        assert "s1_d1_hbo" in hrf_panel._edge_unstable_channels(m)

    def test_skips_missing_short_and_global(self):
        m = _montage({
            "no_trace": SimpleNamespace(trace=None),
            "too_short": SimpleNamespace(trace=np.array([1.0, 9.0])),
            "global_hbo": SimpleNamespace(trace=_NOISY_EDGES),
        })
        # None / too-short skipped; 'global' channels excluded.
        assert hrf_panel._edge_unstable_channels(m) == []

    def test_flat_center_not_flagged(self):
        # Center std is 0 (constant) -> can't judge -> skipped, no crash.
        flat_center = np.array([5.0, -5.0, 5.0] + [1.0] * 14 + [5.0, -5.0, 5.0])
        m = _montage({"s1_d1_hbo": SimpleNamespace(trace=flat_center)})
        assert hrf_panel._edge_unstable_channels(m) == []


# ---------------------------------------------------------------------------
# QC callout rendering
# ---------------------------------------------------------------------------


class TestEdgeQcRender:
    @pytest.mark.asyncio
    async def test_warns_when_channels_flagged(self, user: User):
        m = _montage({"s1_d1_hbo": SimpleNamespace(trace=_NOISY_EDGES)})
        opts = hrf_panel.EstimationOptions(edge_expansion=0.2)

        @ui.page("/_edge_qc")
        def _p() -> None:
            hrf_panel._render_edge_qc(m, opts)

        await user.open("/_edge_qc")
        await user.should_see("fluctuate more at the edges")
        await user.should_see("Edge expansion")

    @pytest.mark.asyncio
    async def test_silent_when_all_stable(self, user: User):
        m = _montage({"s1_d1_hbo": SimpleNamespace(trace=_CLEAN)})
        opts = hrf_panel.EstimationOptions()

        @ui.page("/_edge_qc_ok")
        def _p() -> None:
            hrf_panel._render_edge_qc(m, opts)
            ui.label("sentinel-no-warning")

        await user.open("/_edge_qc_ok")
        await user.should_see("sentinel-no-warning")
        await user.should_not_see("fluctuate more at the edges")

    @pytest.mark.asyncio
    async def test_sensitivity_is_configurable(self, user: User):
        """A high flag-ratio silences the warning for the same montage."""
        m = _montage({"s1_d1_hbo": SimpleNamespace(trace=_NOISY_EDGES)})
        opts = hrf_panel.EstimationOptions(edge_std_ratio=50.0)

        @ui.page("/_edge_qc_ratio")
        def _p() -> None:
            hrf_panel._render_edge_qc(m, opts)
            ui.label("sentinel-ratio")

        await user.open("/_edge_qc_ratio")
        await user.should_see("sentinel-ratio")
        await user.should_not_see("fluctuate more at the edges")


# ---------------------------------------------------------------------------
# Single-scan std-band note
# ---------------------------------------------------------------------------


class TestStdBandNote:
    def test_all_zero_std_detected(self):
        z = SimpleNamespace(trace=_CLEAN, trace_std=np.zeros(_CLEAN.size))
        assert hrf_panel._std_is_all_zero(_montage({"s1_d1_hbo": z})) is True

    def test_nonzero_std_detected(self):
        nz = SimpleNamespace(trace=_CLEAN, trace_std=np.abs(_CLEAN) + 0.1)
        assert hrf_panel._std_is_all_zero(_montage({"s1_d1_hbo": nz})) is False

    @pytest.mark.asyncio
    async def test_gallery_notes_absent_band_for_single_scan(self, user: User):
        node = SimpleNamespace(
            trace=_CLEAN, trace_std=np.zeros(_CLEAN.size), sfreq=10.0,
        )
        m = _montage({"s1_d1_hbo": node})

        @ui.page("/_std_note")
        def _p() -> None:
            # The gallery only reads montage data, so state is unused here.
            hrf_panel._render_toeplitz_gallery(None, m)

        await user.open("/_std_note")
        await user.should_see("No ±std band shown")
