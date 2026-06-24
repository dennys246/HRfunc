"""Regression tests for the second (UX / reality-vs-expectation) review sweep.

Covers the verified correctness + guard fixes:

1. Single-scan activity Save is gated on the in-memory result belonging to the
   CURRENT scan (``_has_current_result``) so it can't write scan A's data under
   scan B's name.
2. ``_group_subject_count`` honours ``project_group_excluded`` so the Activity
   readout / ≥2 gate matches the pooled group montage.
5. ``inspect_payload_quality`` flags degenerate / under-powered HRF payloads
   before submission.
13e. ``run_bulk_in_background`` honours the cooperative cancel flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nicegui")

from hrfunc.gui import submission, workers  # noqa: E402
from hrfunc.gui.components import activity_panel  # noqa: E402
from hrfunc.gui.state import AppState  # noqa: E402
from hrfunc.io.manifest import ScanEntry  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]


def _scan(p: Path) -> ScanEntry:
    return ScanEntry(format="snirf", path=p)


# ---------------------------------------------------------------------------
# 1. _has_current_result — single-scan Save source-scan gate
# ---------------------------------------------------------------------------


class TestHasCurrentResult:
    def test_false_when_source_scan_differs(self, tmp_path):
        st = AppState()
        st.activity_raw = object()
        st.activity_source_scan = _scan(tmp_path / "a.snirf")
        assert activity_panel._has_current_result(st, _scan(tmp_path / "b.snirf")) is False

    def test_true_when_source_scan_matches(self, tmp_path):
        st = AppState()
        st.activity_raw = object()
        st.activity_source_scan = _scan(tmp_path / "a.snirf")
        # A different ScanEntry instance with the same path still matches.
        assert activity_panel._has_current_result(st, _scan(tmp_path / "a.snirf")) is True

    def test_false_when_no_result(self, tmp_path):
        st = AppState()
        st.activity_source_scan = _scan(tmp_path / "a.snirf")
        assert activity_panel._has_current_result(st, _scan(tmp_path / "a.snirf")) is False

    def test_false_when_no_source_scan(self, tmp_path):
        st = AppState()
        st.activity_raw = object()  # result with no recorded source scan
        assert activity_panel._has_current_result(st, _scan(tmp_path / "a.snirf")) is False


# ---------------------------------------------------------------------------
# 2. _group_subject_count honours project_group_excluded + skips canonical
# ---------------------------------------------------------------------------


class TestGroupSubjectCount:
    def test_excludes_removed_subject(self):
        st = AppState()
        st.montage_cache[Path("/a")] = object()
        st.montage_cache[Path("/b")] = object()
        assert activity_panel._group_subject_count(st) == 2
        st.project_group_excluded.add(Path("/a"))
        assert activity_panel._group_subject_count(st) == 1

    def test_skips_canonical_entries(self):
        st = AppState()
        st.montage_cache[Path("/a")] = object()
        st.montage_cache[Path("/c")] = activity_panel._CanonicalResult(
            canonical_trace=np.zeros(3), duration=1.0, sfreq=10.0
        )
        assert activity_panel._group_subject_count(st) == 1


# ---------------------------------------------------------------------------
# 5. inspect_payload_quality
# ---------------------------------------------------------------------------


def _write(tmp_path, payload) -> Path:
    p = tmp_path / "montage.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestInspectPayloadQuality:
    def test_clean_payload_has_no_warnings(self, tmp_path):
        payload = {
            "S1_D1-hbo-doi": {
                "hrf_mean": [0.1, 0.9, 0.3],
                "sfreq": 7.81,
                "estimate_sources": ["s1", "s2"],
                "estimates": [[0.1, 0.8, 0.3], [0.1, 1.0, 0.3]],
            }
        }
        assert submission.inspect_payload_quality(_write(tmp_path, payload)) == []

    def test_flags_flat_trace(self, tmp_path):
        payload = {
            "S1_D1-hbo-doi": {
                "hrf_mean": [0.0, 0.0, 0.0],
                "sfreq": 7.81,
                "estimates": [[0, 0, 0], [0, 0, 0]],
            }
        }
        warns = submission.inspect_payload_quality(_write(tmp_path, payload))
        assert any("flat" in w for w in warns)

    def test_flags_missing_sfreq(self, tmp_path):
        payload = {
            "S1_D1-hbo-doi": {
                "hrf_mean": [0.1, 0.9, 0.3],
                "sfreq": None,
                "estimates": [[0.1, 0.9, 0.3], [0.1, 0.8, 0.3]],
            }
        }
        warns = submission.inspect_payload_quality(_write(tmp_path, payload))
        assert any("sampling frequency" in w for w in warns)

    def test_flags_single_estimate(self, tmp_path):
        payload = {
            "S1_D1-hbo-doi": {
                "hrf_mean": [0.1, 0.9, 0.3],
                "sfreq": 7.81,
                "estimates": [[0.1, 0.9, 0.3]],  # only one subject
            }
        }
        warns = submission.inspect_payload_quality(_write(tmp_path, payload))
        assert any("fewer than 2" in w for w in warns)

    def test_roi_list_shape_is_walked(self, tmp_path):
        # build_roi_entry produces a list of entries — the walker must find them.
        payload = [
            {"name": "ROI 1", "hrf_mean": [0.0, 0.0], "sfreq": 7.81,
             "estimates": [[0, 0], [0, 0]]},
        ]
        warns = submission.inspect_payload_quality(_write(tmp_path, payload))
        assert any("flat" in w for w in warns)

    def test_non_hrf_file_warns(self, tmp_path):
        warns = submission.inspect_payload_quality(_write(tmp_path, {"foo": "bar"}))
        assert warns and "HRF" in warns[0]

    def test_unreadable_file_returns_empty(self, tmp_path):
        p = tmp_path / "nope.json"
        p.write_text("{ not json", encoding="utf-8")
        assert submission.inspect_payload_quality(p) == []


# ---------------------------------------------------------------------------
# 13e. run_bulk_in_background cooperative cancel
# ---------------------------------------------------------------------------


class TestBulkCancel:
    @pytest.mark.asyncio
    async def test_cancel_stops_before_next_scan(self, tmp_path):
        st = AppState()
        scans = [_scan(tmp_path / f"s{i}.snirf") for i in range(4)]
        processed: list = []

        def build_call(scan):
            def _do(scan=scan):
                processed.append(scan)
                # Request cancel while the FIRST scan is being processed.
                if len(processed) == 1:
                    st.cancel_requested = True
                return scan
            return (_do, (), {})

        successes, failures = await workers.run_bulk_in_background(
            st, scans, build_call, label="test"
        )

        # Only the first scan ran; the rest were cancelled, not executed.
        assert processed == [scans[0]]
        assert successes == [scans[0]]
        cancelled = [r for _, r in failures if "cancelled" in r]
        assert len(cancelled) == 3
        # Flag is reset for the next run.
        assert st.cancel_requested is False

    @pytest.mark.asyncio
    async def test_no_cancel_runs_all(self, tmp_path):
        st = AppState()
        scans = [_scan(tmp_path / f"s{i}.snirf") for i in range(3)]
        ran: list = []

        def build_call(scan):
            return ((lambda scan=scan: ran.append(scan)), (), {})

        successes, failures = await workers.run_bulk_in_background(
            st, scans, build_call, label="test"
        )
        assert len(ran) == 3
        assert len(successes) == 3
        assert failures == []


# ---------------------------------------------------------------------------
# 13b. Mass-save destination resolution (colocated / chosen folder / collisions)
# ---------------------------------------------------------------------------


class TestResolveSaveTargets:
    def test_colocated_writes_next_to_each_source(self, tmp_path):
        a = _scan(tmp_path / "sub-01" / "a.snirf")
        b = _scan(tmp_path / "sub-02" / "b.snirf")
        out = activity_panel._resolve_save_targets(
            [a, b], None, "_deconvolved", ".snirf"
        )
        assert out[a] == tmp_path / "sub-01" / "a_deconvolved.snirf"
        assert out[b] == tmp_path / "sub-02" / "b_deconvolved.snirf"

    def test_chosen_folder_is_flat(self, tmp_path):
        dest = tmp_path / "out"
        a = _scan(tmp_path / "sub-01" / "a.snirf")
        b = _scan(tmp_path / "sub-02" / "b.snirf")
        out = activity_panel._resolve_save_targets(
            [a, b], dest, "_deconvolved", ".snirf"
        )
        assert out[a] == dest / "a_deconvolved.snirf"
        assert out[b] == dest / "b_deconvolved.snirf"

    def test_flat_folder_disambiguates_colliding_stems(self, tmp_path):
        dest = tmp_path / "out"
        # Two different sources with the SAME stem would collide in one folder.
        a = _scan(tmp_path / "s1" / "scan.snirf")
        b = _scan(tmp_path / "s2" / "scan.snirf")
        out = activity_panel._resolve_save_targets(
            [a, b], dest, "_deconvolved", ".snirf"
        )
        assert out[a] == dest / "scan_deconvolved.snirf"
        assert out[b] == dest / "scan_deconvolved_2.snirf"  # disambiguated
        assert out[a] != out[b]

    def test_colocated_same_stem_different_folders_no_collision(self, tmp_path):
        # Colocated keeps them in separate folders, so no disambiguation needed.
        a = _scan(tmp_path / "s1" / "scan.snirf")
        b = _scan(tmp_path / "s2" / "scan.snirf")
        out = activity_panel._resolve_save_targets(
            [a, b], None, "_deconvolved", ".snirf"
        )
        assert out[a] == tmp_path / "s1" / "scan_deconvolved.snirf"
        assert out[b] == tmp_path / "s2" / "scan_deconvolved.snirf"
