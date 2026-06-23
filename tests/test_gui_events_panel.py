"""Unit tests for the HRFs-tab events plumbing (feat/hrf-events-upload).

Exercises the pure decision helpers in ``hrf_panel`` that resolve which
events drive a scan's estimation — the regression-prone logic behind
upload / auto-match / apply-to-all / impulse-vs-onset. These don't need a
NiceGUI render context (no ``user`` fixture); they operate on a fresh
AppState + lightweight ScanEntry stubs.

Covers:
- ``_file_applies_to_scan``: none / source-scan / apply-all / impulse.
- ``_apply_parsed_events``: loads onset rows vs impulse vector correctly.
- ``_event_labels_for_scan`` / ``_event_counts_for_scan`` for a loaded file.
- ``_maybe_automatch_events``: BIDS-strict load, opt-out via
  ``events_no_automatch``, and "found nothing" memoization.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nicegui")

from hrfunc.gui.components import hrf_panel  # noqa: E402
from hrfunc.gui.components.hrf_panel import EstimationOptions  # noqa: E402
from hrfunc.gui.events_io import parse_events_file  # noqa: E402
from hrfunc.gui.state import AppState  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402


def _scan(tmp_path: Path, name: str) -> ScanEntry:
    return ScanEntry(format="snirf", path=tmp_path / name, display_name=name)


# ---------------------------------------------------------------------------
# _file_applies_to_scan
# ---------------------------------------------------------------------------


def test_file_applies_none_loaded(tmp_path):
    st = AppState()
    assert hrf_panel._file_applies_to_scan(st, _scan(tmp_path, "a.snirf")) is False


def test_file_applies_only_to_source_scan(tmp_path):
    st = AppState()
    a, b = _scan(tmp_path, "a.snirf"), _scan(tmp_path, "b.snirf")
    st.events_rows = [object()]  # non-None marks "loaded"
    st.events_source_scan = a
    assert hrf_panel._file_applies_to_scan(st, a) is True
    assert hrf_panel._file_applies_to_scan(st, b) is False


def test_file_applies_all_when_apply_all(tmp_path):
    st = AppState()
    a, b = _scan(tmp_path, "a.snirf"), _scan(tmp_path, "b.snirf")
    st.events_rows = [object()]
    st.events_source_scan = a
    st.events_apply_all = True
    assert hrf_panel._file_applies_to_scan(st, b) is True


def test_file_applies_with_impulse(tmp_path):
    st = AppState()
    a = _scan(tmp_path, "a.snirf")
    st.events_impulse = [0, 1, 0]
    st.events_source_scan = a
    assert hrf_panel._file_applies_to_scan(st, a) is True


# ---------------------------------------------------------------------------
# _apply_parsed_events (onset vs impulse)
# ---------------------------------------------------------------------------


def test_apply_parsed_onset_rows(tmp_path):
    st, opts = AppState(), EstimationOptions()
    a = _scan(tmp_path, "a.snirf")
    f = tmp_path / "ev.csv"
    f.write_text("onset,label\n1,go\n2,stop\n")
    parsed = parse_events_file(f)
    hrf_panel._apply_parsed_events(st, opts, a, parsed, is_auto=False)
    assert st.events_rows is not None and st.events_impulse is None
    assert st.events_source_scan is a
    assert set(opts.selected_events) == {"go", "stop"}


def test_apply_parsed_impulse(tmp_path):
    st, opts = AppState(), EstimationOptions()
    a = _scan(tmp_path, "a.snirf")
    f = tmp_path / "events.txt"
    f.write_text("0\n1\n0\n1\n")
    parsed = parse_events_file(f)
    hrf_panel._apply_parsed_events(st, opts, a, parsed, is_auto=True)
    assert st.events_impulse == [0, 1, 0, 1]
    assert st.events_rows is None
    assert st.events_is_automatched is True
    assert opts.selected_events == ("events",)


# ---------------------------------------------------------------------------
# label / count helpers for a loaded file
# ---------------------------------------------------------------------------


def test_labels_and_counts_for_impulse(tmp_path):
    st = AppState()
    a = _scan(tmp_path, "a.snirf")
    st.events_impulse = [0, 1, 0, 1, 1]
    st.events_source_scan = a
    assert hrf_panel._event_labels_for_scan(st, a, None) == ["events"]
    assert hrf_panel._event_counts_for_scan(st, a, None) == {"events": 3}


def test_labels_and_counts_for_onset_rows(tmp_path):
    st, opts = AppState(), EstimationOptions()
    a = _scan(tmp_path, "a.snirf")
    f = tmp_path / "ev.csv"
    f.write_text("onset,label\n1,go\n2,go\n3,stop\n")
    hrf_panel._apply_parsed_events(st, opts, a, parse_events_file(f), is_auto=False)
    assert hrf_panel._event_labels_for_scan(st, a, None) == ["go", "stop"]
    assert hrf_panel._event_counts_for_scan(st, a, None) == {"go": 2, "stop": 1}


# ---------------------------------------------------------------------------
# _maybe_automatch_events
# ---------------------------------------------------------------------------


def test_automatch_bids_strict(tmp_path):
    scan_path = tmp_path / "sub-01_task-x_nirs.snirf"
    scan_path.write_text("x")
    (tmp_path / "sub-01_task-x_events.tsv").write_text(
        "onset\tduration\ttrial_type\n1\t0\tgo\n"
    )
    scan = ScanEntry(format="snirf", path=scan_path, display_name="sub-01")
    st, opts = AppState(), EstimationOptions()
    st.manifest = Manifest(root=tmp_path, scans=(scan,))
    hrf_panel._maybe_automatch_events(st, scan, opts)
    assert st.events_rows is not None
    assert st.events_is_automatched is True
    assert st.events_source_label == "sub-01_task-x_events.tsv"


def test_automatch_respects_opt_out(tmp_path):
    scan_path = tmp_path / "sub-01_task-x_nirs.snirf"
    scan_path.write_text("x")
    (tmp_path / "sub-01_task-x_events.tsv").write_text(
        "onset\tduration\ttrial_type\n1\t0\tgo\n"
    )
    scan = ScanEntry(format="snirf", path=scan_path, display_name="sub-01")
    st, opts = AppState(), EstimationOptions()
    st.manifest = Manifest(root=tmp_path, scans=(scan,))
    st.events_no_automatch.add(scan_path.resolve())
    hrf_panel._maybe_automatch_events(st, scan, opts)
    assert st.events_rows is None  # opted out -> nothing loaded


def test_automatch_memoizes_when_nothing_found(tmp_path):
    scan_path = tmp_path / "lonely.snirf"
    scan_path.write_text("x")
    # two scans in folder, no events file -> no match, and remembered
    other = tmp_path / "other.snirf"
    other.write_text("x")
    scan = ScanEntry(format="snirf", path=scan_path, display_name="lonely")
    other_entry = ScanEntry(format="snirf", path=other, display_name="other")
    st, opts = AppState(), EstimationOptions()
    st.manifest = Manifest(root=tmp_path, scans=(scan, other_entry))
    hrf_panel._maybe_automatch_events(st, scan, opts)
    assert st.events_rows is None and st.events_impulse is None
    assert scan_path.resolve() in st.events_no_automatch


# ---------------------------------------------------------------------------
# _resolve_bulk_events — per-scan events for a bulk HRF run
# ---------------------------------------------------------------------------


class _FakeAnnotations:
    """Minimal stand-in for mne Annotations: iterable of dict-likes + len."""

    def __init__(self, descriptions):
        self._items = [{"description": d, "onset": 1.0} for d in descriptions]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class _FakeRaw:
    def __init__(self, descriptions=None):
        self.annotations = (
            _FakeAnnotations(descriptions) if descriptions is not None else None
        )


def test_resolve_bulk_events_manual_applies(tmp_path):
    # An uploaded file set to apply-to-all drives every scan; the user's
    # label selection is preserved across the batch.
    scan = _scan(tmp_path, "sub-01_nirs.snirf")
    st = AppState()
    sentinel_rows = [object()]
    st.events_rows = sentinel_rows
    st.events_apply_all = True
    snapshot = EstimationOptions(selected_events=("go",))
    rows, impulse, selected = hrf_panel._resolve_bulk_events(
        st, scan, _FakeRaw(["ignored"]), snapshot
    )
    assert rows is sentinel_rows
    assert impulse is None
    assert selected == ("go",)


def test_resolve_bulk_events_collocated_sidecar(tmp_path):
    # No manual file -> discover this scan's own BIDS sidecar and use all of
    # its conditions (not the UI scan's selection).
    scan_path = tmp_path / "sub-02_task-x_nirs.snirf"
    scan_path.write_text("x")
    (tmp_path / "sub-02_task-x_events.tsv").write_text(
        "onset\tduration\ttrial_type\n1\t0\tgo\n2\t0\tstop\n"
    )
    scan = ScanEntry(format="snirf", path=scan_path, display_name="sub-02")
    st = AppState()
    st.manifest = Manifest(root=tmp_path, scans=(scan,))
    snapshot = EstimationOptions(selected_events=("unrelated",))
    rows, impulse, selected = hrf_panel._resolve_bulk_events(
        st, scan, _FakeRaw(None), snapshot
    )
    assert rows is not None and impulse is None
    assert set(selected) == {"go", "stop"}


def test_resolve_bulk_events_annotation_fallback(tmp_path):
    # No manual file, no sidecar -> fall back to the scan's own annotations,
    # selecting every description.
    scan_path = tmp_path / "plain.snirf"
    scan_path.write_text("x")
    scan = ScanEntry(format="snirf", path=scan_path, display_name="plain")
    st = AppState()
    st.manifest = Manifest(root=tmp_path, scans=(scan,))
    snapshot = EstimationOptions(selected_events=())
    rows, impulse, selected = hrf_panel._resolve_bulk_events(
        st, scan, _FakeRaw(["stim_b", "stim_a"]), snapshot
    )
    assert rows is None and impulse is None
    assert selected == ("stim_a", "stim_b")  # sorted, de-duplicated


def test_resolve_bulk_events_none_anywhere(tmp_path):
    # No manual file, no sidecar, no annotations -> empty tuple so the caller
    # fails the scan with a clear reason instead of estimating on nothing.
    scan_path = tmp_path / "empty.snirf"
    scan_path.write_text("x")
    scan = ScanEntry(format="snirf", path=scan_path, display_name="empty")
    st = AppState()
    st.manifest = Manifest(root=tmp_path, scans=(scan,))
    snapshot = EstimationOptions(selected_events=())
    rows, impulse, selected = hrf_panel._resolve_bulk_events(
        st, scan, _FakeRaw(None), snapshot
    )
    assert rows is None and impulse is None and selected == ()


# ---------------------------------------------------------------------------
# EstimationOptions — configurable per-channel timeout
# ---------------------------------------------------------------------------


def test_estimation_options_timeout_default():
    opts = EstimationOptions()
    assert opts.timeout == 30.0  # mirrors montage.estimate_hrf default


def test_estimation_options_timeout_override():
    opts = EstimationOptions(timeout=5.0)
    assert opts.timeout == 5.0
