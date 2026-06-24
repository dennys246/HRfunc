"""Parsing + impulse construction for user-supplied event files (HRFs tab).

Many SNIRF / NIRx recordings reach the GUI with no usable MNE annotations,
so the HRFs tab's event picker is empty and toeplitz estimation can't run.
This module lets the user supply events from a file instead. Two shapes are
accepted and auto-detected:

- **BIDS ``events.tsv``** — a header row containing ``onset`` plus a label
  column (``trial_type`` / ``value`` / ``condition``) and usually
  ``duration``. Tab- or comma-separated; onset / duration in seconds. This
  is the fNIRS/BIDS standard and carries explicit condition labels.
- **Simple ``onset,label``** — two columns (onset seconds, label string),
  header optional. Detected as a fallback. Because it has no BIDS
  provenance, :func:`parse_events_file` flags it with
  ``needs_simple_confirm`` so the UI can warn before using it.

Everything here is ONSET-based: an event at time ``t`` becomes a ``1`` at
sample ``round(t * sfreq)``. That's what lets one events table be applied
across scans of differing length / sample rate -- the impulse series is
rebuilt per scan from that scan's own ``sfreq`` and sample count
(:func:`build_impulse_from_rows`), exactly mirroring the annotation path's
``build_events_array``.

Pure / UI-free so it's unit-testable without NiceGUI (same split as
``submission.py``).
"""

from __future__ import annotations

import csv
import fnmatch
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)

# Directories the events file-finder never descends into.
_SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "node_modules", ".ipynb_checkpoints"}


# Columns whose presence marks a file as a proper BIDS events table (so it's
# used without the simple-format confirmation). ``label`` is deliberately NOT
# here -- a bare ``onset,label`` file is the *simple* shape.
_BIDS_LABEL_COLUMNS = ("trial_type", "value", "condition", "stim_type")
# Where to read each event's label, in order of preference. Includes the
# simple ``label`` column as a last resort.
_LABEL_COLUMNS = _BIDS_LABEL_COLUMNS + ("label",)
# Label used when a structured file has onsets but no recognizable label
# column (every event then shares this single condition).
_DEFAULT_LABEL = "event"
# BIDS encodes "no value" as the literal string "n/a"; normalize it so it
# doesn't masquerade as a real condition name.
_NA_TOKENS = {"n/a", "na", "nan", ""}


class EventsParseError(ValueError):
    """Raised when a file can't be read as either supported events shape."""


@dataclass
class EventRow:
    """One parsed event: an onset (seconds), a condition label, duration."""

    onset: float
    label: str
    duration: float = 0.0


@dataclass
class EventsParse:
    """Result of :func:`parse_events_file`.

    - ``rows``: parsed events (already onset-sorted, bad rows dropped).
    - ``fmt``: ``"bids"`` or ``"simple"``.
    - ``needs_simple_confirm``: True when the file only matched the simple
      ``onset,label`` shape -- the UI should warn + confirm before use.
    - ``warnings``: non-fatal parse notes (skipped rows, missing label
      column, etc.) to surface to the user.
    - ``source_name``: the file's display name.
    """

    rows: List[EventRow]
    fmt: str
    needs_simple_confirm: bool
    warnings: List[str] = field(default_factory=list)
    source_name: str = ""
    # Set for the "impulse" format: a per-sample 0/1 design vector (one value
    # per timepoint), the shape estimate_hrf consumes directly. Indexed by
    # SAMPLE, not time, so it's applied per scan by sample index (and the
    # coverage check flags a scan shorter than the vector). ``rows`` is empty
    # in this case; the single synthetic condition is labelled "events".
    impulse: Optional[List[int]] = None

    def labels(self) -> List[str]:
        """Sorted unique condition labels (or ["events"] for an impulse vector)."""
        if self.impulse is not None:
            return ["events"]
        return sorted({r.label for r in self.rows})


def _detect_delimiter(sample: str) -> str:
    """Tab if the first non-empty line has one, else comma.

    BIDS ``events.tsv`` is tab-separated; the simple fallback is usually a
    CSV. Sniffing the first populated line is enough -- we don't need
    csv.Sniffer's heuristics (which misfire on 2-column files).
    """
    for line in sample.splitlines():
        if line.strip():
            return "\t" if "\t" in line else ","
    return ","


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def _clean_label(raw: str) -> str:
    token = (raw or "").strip()
    if token.lower() in _NA_TOKENS:
        return _DEFAULT_LABEL
    return token


def parse_events_file(path: Path) -> EventsParse:
    """Parse an events file as BIDS ``events.tsv`` or simple ``onset,label``.

    Detection:
      1. Header present (a cell equals ``onset``, case-insensitive) AND a
         recognized label column or a ``duration`` column -> ``bids``.
      2. Header present with ``onset`` but no label/duration column, OR no
         header but the first column parses as a float -> ``simple``
         (``needs_simple_confirm=True``).
      3. Otherwise -> :class:`EventsParseError`.

    Rows whose onset isn't numeric are skipped (recorded in ``warnings``).
    Raises :class:`EventsParseError` on an empty / unreadable file or when
    no rows survive parsing.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")  # tolerate a BOM
    except Exception as exc:  # noqa: BLE001
        raise EventsParseError(f"Could not read {path.name}: {exc}") from exc

    delimiter = _detect_delimiter(text)
    table = [
        row for row in csv.reader(io.StringIO(text), delimiter=delimiter)
        if any(cell.strip() for cell in row)  # drop blank lines
    ]
    if not table:
        raise EventsParseError(f"{path.name} is empty.")

    header = [cell.strip().lower() for cell in table[0]]
    has_header = "onset" in header

    warnings: List[str] = []

    if has_header:
        onset_idx = header.index("onset")
        label_idx: Optional[int] = next(
            (header.index(c) for c in _LABEL_COLUMNS if c in header), None
        )
        duration_idx = header.index("duration") if "duration" in header else None
        # BIDS proper is marked by a trial_type/value/condition column or a
        # duration column. A header of just onset(+label) is the simple shape
        # and should trigger the confirm.
        is_bids = (
            any(c in header for c in _BIDS_LABEL_COLUMNS)
            or duration_idx is not None
        )
        if label_idx is None:
            # Structured but unlabeled: maybe a 2nd column is an implicit
            # label (simple), else everything shares the default label.
            if len(header) >= 2 and not is_bids:
                label_idx = 1 if onset_idx != 1 else 0
            else:
                warnings.append(
                    "No trial_type/value/condition column found; all events "
                    f"labeled '{_DEFAULT_LABEL}'."
                )
        fmt = "bids" if is_bids else "simple"
        data_rows = table[1:]
    else:
        # No header. Accept only if the first column is numeric.
        if not table[0] or not _is_float(table[0][0]):
            raise EventsParseError(
                f"{path.name} has no 'onset' header and its first column "
                "isn't numeric -- can't read it as onset,label events."
            )
        # Per-sample impulse vector: a single column whose values are all 0/1
        # (one value per timepoint), not onset times. Detect and return it as
        # the "impulse" format so it's applied by sample index rather than
        # misread as onsets-at-t=0/1.
        #
        # NOTE: a single all-0/1 column is inherently ambiguous -- it could be
        # a tiny onset list whose onset TIMES are literally 0 s / 1 s. The
        # library resolves that ambiguity in favour of "impulse" by design
        # (per-sample 0/1 design vectors are the common case; the demo
        # events.txt is one such 1999-sample vector), and the events_io tests
        # pin this for short vectors too. Onset lists with any value beyond
        # 0/1 fall through to the onset path below, which is the realistic
        # disambiguator. Do NOT add a length threshold here without updating
        # those tests -- it would reclassify legitimate short impulse vectors.
        single_col_vals = [
            r[0].strip() for r in table if r and r[0].strip()
        ]
        is_single_col = all(len(r) == 1 for r in table) and single_col_vals
        if is_single_col and all(
            _is_float(v) and float(v) in (0.0, 1.0) for v in single_col_vals
        ):
            impulse = [int(float(v)) for v in single_col_vals]
            impulse_warnings = []
            if sum(impulse) == 0:
                impulse_warnings.append(
                    "Impulse vector contains no events (all zeros)."
                )
            return EventsParse(
                rows=[],
                fmt="impulse",
                needs_simple_confirm=False,
                warnings=impulse_warnings,
                source_name=path.name,
                impulse=impulse,
            )
        onset_idx = 0
        label_idx = 1 if len(table[0]) >= 2 else None
        duration_idx = None
        fmt = "simple"
        data_rows = table

    rows: List[EventRow] = []
    for i, raw_row in enumerate(data_rows):
        if onset_idx >= len(raw_row) or not _is_float(raw_row[onset_idx]):
            warnings.append(f"Skipped row {i + 1}: onset not numeric.")
            continue
        onset = float(raw_row[onset_idx])
        label = (
            _clean_label(raw_row[label_idx])
            if label_idx is not None and label_idx < len(raw_row)
            else _DEFAULT_LABEL
        )
        duration = 0.0
        if duration_idx is not None and duration_idx < len(raw_row):
            if _is_float(raw_row[duration_idx]):
                duration = float(raw_row[duration_idx])
        rows.append(EventRow(onset=onset, label=label, duration=duration))

    if not rows:
        raise EventsParseError(
            f"{path.name} parsed to zero usable events."
        )

    rows.sort(key=lambda r: r.onset)
    return EventsParse(
        rows=rows,
        fmt=fmt,
        needs_simple_confirm=(fmt == "simple"),
        warnings=warnings,
        source_name=path.name,
    )


def build_impulse_from_rows(
    rows: Sequence[EventRow],
    labels: Iterable[str],
    sfreq: float,
    n_samples: int,
) -> "np.ndarray":
    """Build a 0/1 impulse series of length ``n_samples`` from event rows.

    Mirrors ``hrf_panel.build_events_array`` but sources onsets from a parsed
    file rather than MNE annotations. An event whose label is in ``labels``
    becomes a ``1`` at ``round(onset * sfreq)``; onsets outside
    ``[0, n_samples)`` are dropped (surfaced separately via
    :func:`coverage_report`). Built per scan, so one table fits any length.
    """
    label_set = set(labels)
    out = np.zeros(int(n_samples), dtype=np.int64)
    for row in rows:
        if row.label not in label_set:
            continue
        sample = int(round(row.onset * sfreq))
        if 0 <= sample < n_samples:
            out[sample] = 1
    return out


def build_impulse_from_vector(
    impulse: Sequence[int], n_samples: int
) -> "np.ndarray":
    """Fit a per-sample impulse vector to a scan's length.

    The vector is sample-indexed, so it's applied position-for-position:
    truncated if longer than the scan, zero-padded if shorter. (A length
    mismatch means the vector was made for a different-length recording —
    :func:`coverage_report_vector` surfaces that for the Estimate warning.)
    """
    out = np.zeros(int(n_samples), dtype=np.int64)
    m = min(len(impulse), int(n_samples))
    for i in range(m):
        if impulse[i]:
            out[i] = 1
    return out


_EVENT_DISCOVERY_EXTS = (".tsv", ".csv", ".txt")


def find_event_files(
    root, pattern: str = "", limit: int = 300
) -> List[Path]:
    """Recursively find candidate event files under ``root`` (the find window).

    The glob ``pattern`` is matched case-insensitively against each file NAME
    (``fnmatch``). A bare term with no wildcard is treated as a substring
    (wrapped to ``*term*``). An empty pattern lists all event-extension files
    (.tsv/.csv/.txt). Noise dirs (.venv, __pycache__, .git, …) and hidden
    dirs are skipped. Returns sorted paths capped at ``limit``; never raises.
    """
    pat = (pattern or "").strip().lower()
    if pat and not any(ch in pat for ch in "*?["):
        pat = f"*{pat}*"  # bare term -> substring match
    results: List[Path] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            ]
            for name in filenames:
                low = name.lower()
                if pat:
                    if not fnmatch.fnmatch(low, pat):
                        continue
                elif Path(low).suffix not in _EVENT_DISCOVERY_EXTS:
                    continue
                results.append(Path(dirpath) / name)
                if len(results) >= limit:
                    return sorted(results)
    except OSError:
        pass
    return sorted(results)


def _bids_events_prefix(scan_path: Path) -> Optional[str]:
    """BIDS entity prefix for a scan, e.g. ``sub-01_task-x_nirs`` -> ``sub-01_task-x``.

    Returns None when the name isn't BIDS-like (must start with ``sub-`` and
    contain a ``_``). Used to look for a sibling ``<prefix>_events.tsv``.
    """
    stem = scan_path.stem if scan_path.suffix else scan_path.name
    if not stem.startswith("sub-") or "_" not in stem:
        return None
    head, _, _last = stem.rpartition("_")
    return head or None


def discover_collocated_events(
    scan_path: Path, all_scan_paths: Sequence[Path] = ()
) -> Optional[Path]:
    """Find an events file collocated with ``scan_path``.

    Strategy (matches the locked UX decision):
      1. **BIDS-strict** — a ``<entities>_events.tsv`` in the scan's folder
         sharing the scan's BIDS entity prefix.
      2. **Lone-file fallback** — if the scan's folder holds exactly one event
         file (``.tsv`` / ``.csv``, or a ``.txt`` named like events/onsets) and
         at most this one scan, use it.

    Returns the matched Path, or None. Never raises (filesystem errors -> None)
    so it's safe to call from a render path. Pure aside from the directory
    read, so it's unit-testable with real temp files.
    """
    try:
        folder = scan_path.parent
        if not folder.is_dir():
            return None

        prefix = _bids_events_prefix(scan_path)
        if prefix:
            bids_candidate = folder / f"{prefix}_events.tsv"
            if bids_candidate.is_file():
                return bids_candidate

        def _is_event_file(p: Path) -> bool:
            suffix = p.suffix.lower()
            if suffix in (".tsv", ".csv"):
                return True
            if suffix == ".txt":
                low = p.name.lower()
                return "event" in low or "onset" in low
            return False

        event_files = [
            p for p in folder.iterdir() if p.is_file() and _is_event_file(p)
        ]
        folder_resolved = folder.resolve()
        scans_here = [
            p for p in all_scan_paths
            if Path(p).resolve().parent == folder_resolved
        ]
        if len(event_files) == 1 and len(scans_here) <= 1:
            return event_files[0]
        return None
    except OSError:
        return None


@dataclass
class Coverage:
    """How a selected event set lands against one scan's time axis."""

    placed: int          # events that land inside the usable window
    past_end: int        # onsets at/after the scan end (dropped)
    in_edge: int         # onsets inside the dropped edge-expansion window
    max_onset_s: float   # latest selected onset, seconds
    scan_seconds: float  # scan duration, seconds


def coverage_report(
    rows: Sequence[EventRow],
    labels: Iterable[str],
    sfreq: float,
    n_samples: int,
    edge_samples: int = 0,
) -> Coverage:
    """Count how selected events land against a scan (for the Estimate check).

    ``edge_samples`` is the toeplitz edge-expansion window at the start of
    the scan (``round(edge_expansion * duration * sfreq)`` in the core); the
    library shifts onsets back into it and drops anything that would precede
    t=0, so events there don't contribute and are worth flagging.
    """
    label_set = set(labels)
    placed = past_end = in_edge = 0
    max_onset_s = 0.0
    for row in rows:
        if row.label not in label_set:
            continue
        max_onset_s = max(max_onset_s, row.onset)
        sample = int(round(row.onset * sfreq))
        if sample < 0 or sample >= n_samples:
            past_end += 1
        elif sample < edge_samples:
            in_edge += 1
        else:
            placed += 1
    scan_seconds = (n_samples / sfreq) if sfreq > 0 else 0.0
    return Coverage(
        placed=placed,
        past_end=past_end,
        in_edge=in_edge,
        max_onset_s=max_onset_s,
        scan_seconds=scan_seconds,
    )


def coverage_report_vector(
    impulse: Sequence[int],
    sfreq: float,
    n_samples: int,
    edge_samples: int = 0,
) -> Coverage:
    """Coverage for a per-sample impulse vector against a scan.

    The vector is sample-indexed: events at index >= n_samples fall past the
    scan end (the vector is longer than this recording), and events at index
    < edge_samples sit in the dropped edge window.
    """
    placed = past_end = in_edge = 0
    last_event_idx = 0
    for i, value in enumerate(impulse):
        if not value:
            continue
        last_event_idx = i
        if i >= n_samples:
            past_end += 1
        elif i < edge_samples:
            in_edge += 1
        else:
            placed += 1
    scan_seconds = (n_samples / sfreq) if sfreq > 0 else 0.0
    max_onset_s = (last_event_idx / sfreq) if sfreq > 0 else 0.0
    return Coverage(
        placed=placed,
        past_end=past_end,
        in_edge=in_edge,
        max_onset_s=max_onset_s,
        scan_seconds=scan_seconds,
    )
