"""Core HRF estimate provenance — estimate_sources keyed per scan.

Each estimate carries a ``source_id`` so a multi-subject (group) montage can
report and remove a specific subject's contribution, including after a
save/load round-trip. Re-estimating the same source replaces (not duplicates)
its estimate.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pytest

pytest.importorskip("mne")
import mne  # noqa: E402

from hrfunc.hrtree import HRF  # noqa: E402


def _silence(fn, *a, **k):
    out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*a, **k)
    finally:
        sys.stdout = out


# ---------------------------------------------------------------------------
# HRF node storage
# ---------------------------------------------------------------------------


class TestHrfNodeSources:
    def test_sources_stored(self):
        n = HRF("doi", "S1_D1 hbo", 10.0, 10.0, [0.0, 1.0, 2.0],
                estimates=[[1.0, 2.0]], estimate_sources=["subjA"])
        assert n.estimate_sources == ["subjA"]

    def test_sources_pad_to_estimate_length(self):
        # Estimates predating provenance (e.g. bundled HRFs) -> pad with None.
        n = HRF("doi", "S1_D1 hbo", 10.0, 10.0, [0.0, 1.0],
                estimates=[[1.0], [2.0], [3.0]], estimate_sources=["subjA"])
        assert n.estimate_sources == ["subjA", None, None]

    def test_default_empty(self):
        n = HRF("doi", "S1_D1 hbo", 10.0, 10.0, [0.0, 1.0])
        assert n.estimate_sources == []


# ---------------------------------------------------------------------------
# Integration: estimate_hrf provenance + remove_source + save/load
# ---------------------------------------------------------------------------


def _raw(seed=0):
    raw = mne.io.RawArray(
        np.random.default_rng(seed).standard_normal((2, 200)) * 1e-6,
        mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], 10.0, "hbo"),
        verbose="ERROR",
    )
    for i, ch in enumerate(raw.info["chs"]):
        ch["loc"][:3] = [i * 0.01, 0.0, 0.0]
    return raw


def _events(n=200):
    e = [0] * n
    for t in (20, 80, 140):
        e[t] = 1
    return e


def _two_subject_montage():
    from hrfunc.hrfunc import montage

    raw_a, raw_b = _raw(seed=0), _raw(seed=1)  # distinct subjects
    m = _silence(montage, nirx_obj=raw_a)
    _silence(m.estimate_hrf, raw_a, _events(), duration=10.0,
             source_id="subjA", preprocess=False)
    _silence(m.estimate_hrf, raw_b, _events(), duration=10.0,
             source_id="subjB", preprocess=False)
    return m


@pytest.mark.integration
class TestEstimateHrfProvenance:
    def test_reestimating_same_source_replaces(self):
        from hrfunc.hrfunc import montage

        raw = _raw()
        m = _silence(montage, nirx_obj=raw)
        _silence(m.estimate_hrf, raw, _events(), duration=10.0,
                 source_id="subjA", preprocess=False)
        _silence(m.estimate_hrf, raw, _events(), duration=10.0,
                 source_id="subjA", preprocess=False)  # same source again
        node = m.channels["s1_d1_hbo"]
        assert len(node.estimates) == 1                 # not double-counted
        assert node.estimate_sources == ["subjA"]

    def test_different_sources_accumulate(self):
        node = _two_subject_montage().channels["s1_d1_hbo"]
        assert len(node.estimates) == 2
        assert node.estimate_sources == ["subjA", "subjB"]


@pytest.mark.integration
class TestRemoveSource:
    def test_remove_drops_subject_and_regenerates(self):
        m = _two_subject_montage()
        _silence(m.generate_distribution)
        before = np.array(m.channels["s1_d1_hbo"].trace_std)
        assert np.any(before > 0)                       # 2 subjects -> real std

        removed = m.remove_source("subjA")
        assert removed >= 1
        node = m.channels["s1_d1_hbo"]
        assert node.estimate_sources == ["subjB"]
        assert len(node.estimates) == 1
        # Down to one subject -> std collapses to 0 after the regenerate.
        assert np.allclose(node.trace_std, 0.0)


@pytest.mark.integration
class TestSaveLoadProvenance:
    def test_round_trip_preserves_sources(self, tmp_path):
        from hrfunc.hrfunc import load_montage

        m = _two_subject_montage()
        _silence(m.generate_distribution)
        path = tmp_path / "group_hrfs.json"
        _silence(m.save, str(path))

        loaded = _silence(load_montage, str(path), rich=True)
        all_sources = [
            s for node in loaded.channels.values()
            for s in getattr(node, "estimate_sources", [])
        ]
        assert "subjA" in all_sources
        assert "subjB" in all_sources
