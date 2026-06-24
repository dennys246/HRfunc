"""Core estimate_activity — per-channel HRtree (library_traces) deconvolution.

Library mode gained a per-channel map (``library_traces``: standardized
channel name -> that channel's own HRtree trace) plus ``library_uncovered``
('skip' drops channels with no mapped trace so coverage is honest;
'canonical' falls back to the SPM canonical HRF for them). The single-kernel
``library_trace`` path is unchanged when ``library_traces`` is None.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Top-of-function validation (no nirx_obj / configure needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def bare_montage():
    from hrfunc.hrfunc import montage
    return montage()


class TestLibraryValidation:
    def test_invalid_uncovered_raises(self, bare_montage):
        with pytest.raises(ValueError, match="library_uncovered"):
            bare_montage.estimate_activity(
                None, hrf_model="library",
                library_traces={"s1_d1_hbo": [1.0, 2.0]},
                library_uncovered="bogus",
            )

    def test_library_requires_trace_or_map(self, bare_montage):
        with pytest.raises(ValueError, match="library_trace"):
            bare_montage.estimate_activity(None, hrf_model="library")

    def test_empty_traces_map_raises(self, bare_montage):
        with pytest.raises(ValueError, match="at least one channel"):
            bare_montage.estimate_activity(
                None, hrf_model="library", library_traces={},
            )


# ---------------------------------------------------------------------------
# Behavioural — per-channel routing on a synthetic raw
# ---------------------------------------------------------------------------


def _raw_4ch():
    import mne

    ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"]
    data = np.random.default_rng(0).standard_normal((4, 200)) * 1e-6
    raw = mne.io.RawArray(
        data, mne.create_info(ch_names, 10.0, "hbo"), verbose="ERROR"
    )
    for i, ch in enumerate(raw.info["chs"]):
        ch["loc"][:3] = [i * 0.01, 0.0, 0.0]
    return raw


def _fresh_montage(raw):
    from hrfunc.hrfunc import montage
    return montage(nirx_obj=raw)


# Cover only the S1 pair; S2 is left uncovered.
_KERNEL = list(np.exp(-np.linspace(0, 3, 40)))
_COVER_S1 = {"s1_d1_hbo": _KERNEL, "s1_d1_hbr": _KERNEL}


@pytest.mark.integration
class TestPerChannelDeconvolution:
    def test_skip_drops_uncovered_channels(self):
        raw = _raw_4ch()
        out = _fresh_montage(raw).estimate_activity(
            raw.copy(), hrf_model="library", library_traces=_COVER_S1,
            library_uncovered="skip", preprocess=False, timeout=10,
        )
        assert out is not None
        # Only the covered S1 pair survives; uncovered S2 pair is dropped.
        assert set(out.ch_names) == {"S1_D1 hbo", "S1_D1 hbr"}

    def test_canonical_keeps_all_channels(self):
        raw = _raw_4ch()
        out = _fresh_montage(raw).estimate_activity(
            raw.copy(), hrf_model="library", library_traces=_COVER_S1,
            library_uncovered="canonical", preprocess=False, timeout=10,
        )
        assert out is not None
        # Uncovered S2 pair falls back to canonical -> all 4 retained.
        assert set(out.ch_names) == {
            "S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"
        }

    def test_covered_channel_signal_changes(self):
        """Deconvolution actually transforms covered channels (not a no-op)."""
        raw = _raw_4ch()
        before = raw.copy().get_data(picks=["S1_D1 hbo"])
        out = _fresh_montage(raw).estimate_activity(
            raw.copy(), hrf_model="library", library_traces=_COVER_S1,
            library_uncovered="skip", preprocess=False, timeout=10,
        )
        after = out.get_data(picks=["S1_D1 hbo"])
        assert not np.allclose(before, after)

    def test_single_kernel_path_unchanged_when_no_map(self):
        """library_traces=None keeps the back-compat single-kernel behaviour:
        every channel deconvolved, none dropped."""
        raw = _raw_4ch()
        out = _fresh_montage(raw).estimate_activity(
            raw.copy(), hrf_model="library", library_trace=_KERNEL,
            library_oxygenation=True, preprocess=False, timeout=10,
        )
        assert out is not None
        assert set(out.ch_names) == {
            "S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"
        }
