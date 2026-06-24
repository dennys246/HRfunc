"""Channel -> HRtree HRF matching for per-channel Activity deconvolution."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mne")
import mne  # noqa: E402

from hrfunc.gui.components import hrtree_match  # noqa: E402
from hrfunc.gui.state import AppState  # noqa: E402


def _raw(locs):
    """4 fNIRS channels (S1/S2 × hbo/hbr) at the given source-pair locations
    (meters). ``locs`` is {'S1': (x,y,z), 'S2': (x,y,z)}."""
    ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"]
    raw = mne.io.RawArray(
        np.zeros((4, 10)), mne.create_info(ch_names, 10.0, "hbo"),
        verbose="ERROR",
    )
    pair = {0: "S1", 1: "S1", 2: "S2", 3: "S2"}
    for i, ch in enumerate(raw.info["chs"]):
        ch["loc"][:3] = locs[pair[i]]
    return raw


def _hrf(loc, oxygenation, trace=(0.0, 0.5, 1.0, 0.5)):
    return {"location": list(loc), "hrf_mean": list(trace),
            "oxygenation": oxygenation}


@pytest.fixture
def patched(monkeypatch):
    """Stub the HRtree data sources so only the matching geometry is tested."""
    from hrfunc.gui.components import hrtree_panel

    state = AppState()

    # One HbO and one HbR HRF 5 mm from the S1 source pair (which sits at
    # 0.05 m below — origin is treated as "unplaced", so keep it non-zero).
    hrfs = {
        "hbo:near_s1": _hrf((0.055, 0.0, 0.0), True, (0.1, 0.9, 0.3)),
        "hbr:near_s1": _hrf((0.055, 0.0, 0.0), False, (0.2, 0.8, 0.2)),
    }
    monkeypatch.setattr(hrtree_panel, "gather_library_hrfs",
                        lambda _st: hrfs)
    monkeypatch.setattr(
        hrtree_panel, "_visible_roi_keys",
        lambda _st, _matched: (set(hrfs.keys()), [("slot", "shape")]),
    )
    return state


class TestIndividualStrategy:
    def test_near_channels_covered_far_uncovered(self, patched):
        # S1 at origin (5 mm from the HRFs), S2 100 mm away.
        raw = _raw({"S1": (0.05, 0.0, 0.0), "S2": (0.15, 0.0, 0.0)})
        res = hrtree_match.match_channels_to_hrtree(
            patched, raw, strategy="individual", radius_mm=20.0,
        )
        covered = {m.ch_name for m in res.covered}
        uncovered = {m.ch_name for m in res.uncovered}
        assert covered == {"s1_d1_hbo", "s1_d1_hbr"}
        assert uncovered == {"s2_d1_hbo", "s2_d1_hbr"}
        assert res.n_candidate_hrfs == 2

    def test_oxygenation_respected_in_trace(self, patched):
        raw = _raw({"S1": (0.05, 0.0, 0.0), "S2": (0.15, 0.0, 0.0)})
        res = hrtree_match.match_channels_to_hrtree(
            patched, raw, strategy="individual", radius_mm=20.0,
        )
        traces = res.library_traces()
        # HbO channel got the HbO HRF's trace, HbR got the HbR's.
        assert traces["s1_d1_hbo"] == [0.1, 0.9, 0.3]
        assert traces["s1_d1_hbr"] == [0.2, 0.8, 0.2]

    def test_radius_controls_coverage(self, patched):
        raw = _raw({"S1": (0.05, 0.0, 0.0), "S2": (0.15, 0.0, 0.0)})
        # 100 mm radius now reaches S2 (95 mm away).
        res = hrtree_match.match_channels_to_hrtree(
            patched, raw, strategy="individual", radius_mm=100.0,
        )
        assert {m.ch_name for m in res.covered} == {
            "s1_d1_hbo", "s1_d1_hbr", "s2_d1_hbo", "s2_d1_hbr"
        }

    def test_channel_without_location_is_uncovered(self, patched):
        raw = _raw({"S1": (0.0, 0.0, 0.0), "S2": (0.0, 0.0, 0.0)})
        # Zero out S2's location entirely -> unmatched on geometry grounds.
        for ch in raw.info["chs"]:
            if "S2" in ch["ch_name"]:
                ch["loc"][:3] = [0.0, 0.0, 0.0]
        res = hrtree_match.match_channels_to_hrtree(
            patched, raw, strategy="individual", radius_mm=500.0,
        )
        uncovered = {m.ch_name for m in res.uncovered}
        assert "s2_d1_hbo" in uncovered and "s2_d1_hbr" in uncovered

    def test_library_traces_only_covered(self, patched):
        raw = _raw({"S1": (0.05, 0.0, 0.0), "S2": (0.15, 0.0, 0.0)})
        res = hrtree_match.match_channels_to_hrtree(
            patched, raw, strategy="individual", radius_mm=20.0,
        )
        assert set(res.library_traces().keys()) == {"s1_d1_hbo", "s1_d1_hbr"}


class TestRoiMeanStrategy:
    def test_roi_mean_assigns_roi_trace(self, monkeypatch):
        from hrfunc.gui.components import hrtree_panel

        state = AppState()
        hrfs = {
            "hbo:a": _hrf((0.0, 0.0, 0.0), True),
            "hbo:b": _hrf((0.01, 0.0, 0.0), True),
        }
        monkeypatch.setattr(hrtree_panel, "gather_library_hrfs",
                            lambda _st: hrfs)

        class _Slot:
            name = "ROI 1"
            anchor = {"oxygenation": True, "hrf_mean": [9.0, 9.0]}
            painted = set()

        monkeypatch.setattr(hrtree_panel, "_visible_shapes",
                            lambda _st: [(_Slot(), "shape")])
        monkeypatch.setattr(hrtree_panel, "_alignment_for_shape",
                            lambda _st, _sh: None)
        monkeypatch.setattr(hrtree_panel, "compute_roi_keys_by_shape",
                            lambda *a, **k: set(hrfs.keys()))
        # Pretend subject-weighted averaging isn't available -> anchor fallback.
        monkeypatch.setattr(hrtree_panel, "compute_roi_average",
                            lambda *a, **k: None)

        raw = _raw({"S1": (0.005, 0.0, 0.0), "S2": (0.5, 0.0, 0.0)})
        res = hrtree_match.match_channels_to_hrtree(
            state, raw, strategy="roi_mean", radius_mm=50.0,
        )
        # S1 hbo is within 50 mm of the ROI centroid (~5 mm); gets anchor trace.
        traces = res.library_traces()
        assert traces.get("s1_d1_hbo") == [9.0, 9.0]
        assert res.n_rois == 1

    def test_roi_mean_is_oxygenation_pure(self, monkeypatch):
        """An ROI spanning both haemoglobins yields a SEPARATE mean per
        oxygenation; HbO channels get the HbO mean, HbR channels the HbR mean
        — never a mix (HbO/HbR are inverses; pooling cancels them)."""
        from hrfunc.gui.components import hrtree_panel

        state = AppState()
        hrfs = {
            "hbo:a": _hrf((0.05, 0.0, 0.0), True, (1.0, 1.0)),
            "hbr:a": _hrf((0.05, 0.0, 0.0), False, (2.0, 2.0)),
        }
        monkeypatch.setattr(hrtree_panel, "gather_library_hrfs",
                            lambda _st: hrfs)

        class _Slot:
            name = "ROI 1"
            anchor = None
            painted = set()

        monkeypatch.setattr(hrtree_panel, "_visible_shapes",
                            lambda _st: [(_Slot(), "shape")])
        monkeypatch.setattr(hrtree_panel, "_alignment_for_shape",
                            lambda _st, _sh: None)

        def _keys(_all, _shape, _painted, oxygenation_filter=None, **_k):
            if oxygenation_filter is True:
                return {"hbo:a"}
            if oxygenation_filter is False:
                return {"hbr:a"}
            return set(hrfs.keys())

        monkeypatch.setattr(hrtree_panel, "compute_roi_keys_by_shape", _keys)

        def _avg(all_hrfs, keys):
            traces = [all_hrfs[k]["hrf_mean"] for k in keys]
            arr = np.mean(np.asarray(traces, dtype=float), axis=0)
            return arr, None, len(traces), len(traces)

        monkeypatch.setattr(hrtree_panel, "compute_roi_average", _avg)

        raw = _raw({"S1": (0.05, 0.0, 0.0), "S2": (0.5, 0.0, 0.0)})
        res = hrtree_match.match_channels_to_hrtree(
            state, raw, strategy="roi_mean", radius_mm=50.0,
        )
        traces = res.library_traces()
        assert traces["s1_d1_hbo"] == [1.0, 1.0]   # HbO-only mean
        assert traces["s1_d1_hbr"] == [2.0, 2.0]   # HbR-only mean, not mixed


class TestFiltersScopeCandidates:
    def test_oxygenation_filter_excludes_other_haemoglobin(self, patched):
        """'HbR only' drops the HbO HRF from the candidate set, so the HbO
        channel becomes uncovered — the oxygenation choice scopes matching."""
        patched.library_oxygenation = "hbr"
        raw = _raw({"S1": (0.05, 0.0, 0.0), "S2": (0.5, 0.0, 0.0)})
        res = hrtree_match.match_channels_to_hrtree(
            patched, raw, strategy="individual", radius_mm=20.0,
        )
        covered = {m.ch_name for m in res.covered}
        assert "s1_d1_hbo" not in covered
        assert "s1_d1_hbr" in covered

    def test_context_filter_excludes_nonmatching_hrfs(self, monkeypatch):
        """A context filter (e.g. task) constrains the candidate HRFs exactly
        as it constrains the Cluster view — Filter and Cluster are
        complementary."""
        from hrfunc.gui.components import hrtree_panel

        state = AppState()
        state.library_filter = {"task": "flanker"}
        hrfs = {
            "hbo:rest": {
                **_hrf((0.055, 0.0, 0.0), True),
                "context": {"task": "rest"},  # does NOT match 'flanker'
            },
        }
        monkeypatch.setattr(hrtree_panel, "gather_library_hrfs",
                            lambda _st: hrfs)
        monkeypatch.setattr(
            hrtree_panel, "_visible_roi_keys",
            lambda _st, _m: (set(hrfs.keys()), [("slot", "shape")]),
        )
        raw = _raw({"S1": (0.05, 0.0, 0.0), "S2": (0.5, 0.0, 0.0)})
        res = hrtree_match.match_channels_to_hrtree(
            state, raw, strategy="individual", radius_mm=50.0,
        )
        assert res.n_candidate_hrfs == 0
        assert all(not m.matched for m in res.matches)
