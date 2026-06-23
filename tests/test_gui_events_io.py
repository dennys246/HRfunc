"""Unit tests for ``hrfunc.gui.events_io``.

Covers:
- format auto-detection (BIDS events.tsv vs simple onset,label).
- ``needs_simple_confirm`` only set for the simple shape.
- impulse construction (onset -> sample, out-of-range dropped).
- coverage_report counts (past-end, edge window, usable).
- parse errors on empty / non-numeric junk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hrfunc.gui.events_io import (
    EventsParseError,
    build_impulse_from_rows,
    build_impulse_from_vector,
    coverage_report,
    coverage_report_vector,
    discover_collocated_events,
    find_event_files,
    parse_events_file,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_parse_bids_tsv(tmp_path):
    f = _write(
        tmp_path, "sub-01_events.tsv",
        "onset\tduration\ttrial_type\n5.0\t1.0\tcongruent\n12.5\t1.0\tincongruent\n",
    )
    parsed = parse_events_file(f)
    assert parsed.fmt == "bids"
    assert parsed.needs_simple_confirm is False
    assert parsed.labels() == ["congruent", "incongruent"]
    assert len(parsed.rows) == 2
    assert parsed.rows[0].onset == 5.0


def test_parse_simple_csv_with_header(tmp_path):
    f = _write(tmp_path, "ev.csv", "onset,label\n3,stim\n9,rest\n")
    parsed = parse_events_file(f)
    assert parsed.fmt == "simple"
    assert parsed.needs_simple_confirm is True
    assert parsed.labels() == ["rest", "stim"]


def test_parse_simple_csv_headerless(tmp_path):
    f = _write(tmp_path, "ev.csv", "2.0,a\n4.0,b\n")
    parsed = parse_events_file(f)
    assert parsed.fmt == "simple"
    assert parsed.needs_simple_confirm is True
    assert parsed.labels() == ["a", "b"]


def test_parse_onset_duration_defaults_label_and_warns(tmp_path):
    f = _write(tmp_path, "d.tsv", "onset\tduration\n1\t0\n2\t0\n")
    parsed = parse_events_file(f)
    assert parsed.fmt == "bids"  # duration column marks it structured
    assert parsed.labels() == ["event"]
    assert parsed.warnings  # warned about missing label column


def test_parse_na_label_normalized(tmp_path):
    f = _write(
        tmp_path, "na.tsv",
        "onset\tduration\ttrial_type\n1\t0\tn/a\n2\t0\tgo\n",
    )
    assert parse_events_file(f).labels() == ["event", "go"]


def test_parse_skips_non_numeric_onset_rows(tmp_path):
    f = _write(tmp_path, "ev.csv", "onset,label\n3,stim\nbad,rest\n9,stim\n")
    parsed = parse_events_file(f)
    assert len(parsed.rows) == 2
    assert any("Skipped" in w for w in parsed.warnings)


def test_parse_empty_raises(tmp_path):
    with pytest.raises(EventsParseError):
        parse_events_file(_write(tmp_path, "empty.csv", "\n\n"))


def test_parse_garbage_raises(tmp_path):
    with pytest.raises(EventsParseError):
        parse_events_file(_write(tmp_path, "bad.csv", "hello world\nfoo bar\n"))


def test_build_impulse_places_and_drops(tmp_path):
    f = _write(
        tmp_path, "e.tsv",
        "onset\tduration\ttrial_type\n5\t1\tA\n12.5\t1\tA\n20\t1\tB\n",
    )
    rows = parse_events_file(f).rows
    # sfreq=10 Hz, 100 samples = 10 s scan. Only the 5 s onset lands.
    imp = build_impulse_from_rows(rows, ["A", "B"], sfreq=10.0, n_samples=100)
    assert int(imp.sum()) == 1
    assert imp[50] == 1


def test_build_impulse_label_filter(tmp_path):
    f = _write(tmp_path, "e.csv", "onset,label\n1,A\n2,B\n")
    rows = parse_events_file(f).rows
    imp = build_impulse_from_rows(rows, ["A"], sfreq=10.0, n_samples=100)
    assert int(imp.sum()) == 1
    assert imp[10] == 1  # only the A onset at 1 s


def test_parse_impulse_vector(tmp_path):
    # per-sample 0/1 design vector (the tests/data/sNIRF_formatted/events.txt shape)
    lines = "\n".join(["0"] * 5 + ["1"] + ["0"] * 3 + ["1"]) + "\n"
    parsed = parse_events_file(_write(tmp_path, "events.txt", lines))
    assert parsed.fmt == "impulse"
    assert parsed.needs_simple_confirm is False
    assert parsed.labels() == ["events"]
    assert parsed.impulse is not None
    assert len(parsed.impulse) == 10
    assert sum(parsed.impulse) == 2


def test_impulse_not_confused_with_onsets(tmp_path):
    # values beyond 0/1 -> onset interpretation, NOT impulse
    parsed = parse_events_file(_write(tmp_path, "e.txt", "5\n12\n20\n"))
    assert parsed.fmt == "simple"
    assert parsed.impulse is None


def test_build_impulse_from_vector_truncates_and_pads(tmp_path):
    vec = [0, 1, 0, 1, 0, 1]
    # scan shorter than vector -> truncated (last 1 at index 5 dropped)
    short = build_impulse_from_vector(vec, 4)
    assert int(short.sum()) == 2 and len(short) == 4
    # scan longer -> zero-padded
    long = build_impulse_from_vector(vec, 10)
    assert int(long.sum()) == 3 and len(long) == 10


def test_coverage_report_vector_flags_overrun(tmp_path):
    vec = [0] * 10
    vec[2] = 1   # in edge (if edge=5)
    vec[7] = 1   # placed
    vec[12:13] = []  # noop
    vec += [0, 1]   # index 11 -> past end of an 11-sample scan
    cov = coverage_report_vector(vec, sfreq=10.0, n_samples=11, edge_samples=5)
    assert cov.in_edge == 1
    assert cov.placed == 1
    assert cov.past_end == 1


def test_find_event_files_glob_and_substring(tmp_path):
    (tmp_path / "events.txt").write_text("0\n1\n")
    (tmp_path / "sub-01_events.tsv").write_text("onset\n1\n")
    (tmp_path / "notes.md").write_text("x")
    sub = tmp_path / ".venv"
    sub.mkdir()
    (sub / "events.txt").write_text("0\n")  # in a skipped dir
    names = lambda pat: sorted(p.name for p in find_event_files(tmp_path, pat))
    # glob
    assert names("*events*") == ["events.txt", "sub-01_events.tsv"]
    # bare term -> substring
    assert names("events") == ["events.txt", "sub-01_events.tsv"]
    # extension glob
    assert names("*.tsv") == ["sub-01_events.tsv"]
    # empty -> all event-extension files (md excluded)
    assert "notes.md" not in names("")
    # .venv is skipped (only one events.txt, the top-level one)
    assert names("*events*").count("events.txt") == 1


def test_find_event_files_no_match(tmp_path):
    (tmp_path / "events.txt").write_text("0\n")
    assert find_event_files(tmp_path, "zzz*") == []


def test_discover_bids_strict(tmp_path):
    scan = tmp_path / "sub-01_task-flanker_nirs.snirf"
    scan.write_text("x")
    ev = tmp_path / "sub-01_task-flanker_events.tsv"
    ev.write_text("onset\tduration\ttrial_type\n1\t0\tgo\n")
    # a decoy events file that should NOT win over the BIDS-entity match
    (tmp_path / "other_events.tsv").write_text("onset\n2\n")
    assert discover_collocated_events(scan, [scan]) == ev


def test_discover_lone_file_fallback(tmp_path):
    scan = tmp_path / "recording.snirf"
    scan.write_text("x")
    ev = tmp_path / "onsets.csv"
    ev.write_text("onset,label\n1,go\n")
    assert discover_collocated_events(scan, [scan]) == ev


def test_discover_ambiguous_returns_none(tmp_path):
    scan = tmp_path / "recording.snirf"
    scan.write_text("x")
    (tmp_path / "a.csv").write_text("onset\n1\n")
    (tmp_path / "b.tsv").write_text("onset\n2\n")
    # two event files, not BIDS -> ambiguous -> no match
    assert discover_collocated_events(scan, [scan]) is None


def test_discover_multiple_scans_blocks_lone_fallback(tmp_path):
    scan_a = tmp_path / "a.snirf"
    scan_a.write_text("x")
    scan_b = tmp_path / "b.snirf"
    scan_b.write_text("x")
    (tmp_path / "onsets.csv").write_text("onset\n1\n")
    # lone event file but two scans in the folder -> don't guess
    assert discover_collocated_events(scan_a, [scan_a, scan_b]) is None


def test_discover_txt_requires_event_like_name(tmp_path):
    scan = tmp_path / "recording.snirf"
    scan.write_text("x")
    (tmp_path / "README.txt").write_text("hello")
    # a plain .txt that isn't event/onset-named is ignored -> no match
    assert discover_collocated_events(scan, [scan]) is None
    (tmp_path / "events.txt").write_text("1\n2\n")
    assert discover_collocated_events(scan, [scan]).name == "events.txt"


def test_coverage_report_counts(tmp_path):
    f = _write(
        tmp_path, "e.tsv",
        "onset\tduration\ttrial_type\n0.5\t1\tA\n5\t1\tA\n20\t1\tA\n",
    )
    rows = parse_events_file(f).rows
    # sfreq=10, 100 samples (10 s), edge window = first 10 samples (1 s).
    cov = coverage_report(rows, ["A"], sfreq=10.0, n_samples=100, edge_samples=10)
    assert cov.past_end == 1   # 20 s onset
    assert cov.in_edge == 1    # 0.5 s onset (sample 5 < 10)
    assert cov.placed == 1     # 5 s onset
    assert cov.max_onset_s == 20.0
    assert cov.scan_seconds == 10.0
