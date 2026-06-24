"""HRFs tab content — estimate HRFs from the preprocessed scan + events.

Sprint 3.3 ships:
- A "Toeplitz" mode wired to ``montage.estimate_hrf``: user picks one or more
  annotation descriptions (e.g. ``"stim_a"``, ``"task-flanker"``) from the
  scan, the panel builds an impulse-series array, dispatches estimation
  via ``workers.run_in_background``, and renders a progress bar driven by
  the ``progress_callback``.
- A "Canonical" mode that bypasses estimation entirely and renders the
  SPM-style double-gamma HRF (the same shape the library uses as a
  reference in ``correlate_canonical``). No events needed.
- Controls: model radio (toeplitz/canonical), event picker (multi-select
  toggles), lambda slider on log scale (1e-5..1e-1, default 1e-3),
  duration field (default 30.0 s).

Events can come from the scan's own MNE annotations OR an uploaded file
(``events_io``: BIDS ``events.tsv`` or a simple ``onset,label`` CSV). An
uploaded file overrides annotations for the scan it was loaded against, and
for every scan when "apply to all checked scans" is set (shared paradigms).
Onsets are converted to a per-scan impulse series, so one table fits scans
of differing length. A wildcard box mass-selects labels by glob, and the
Estimate button runs a coverage pre-flight that warns when a scan is too
short to contain the events.
- Result preview: matplotlib base64 PNG showing all estimated HRF traces
  overlaid by channel (toeplitz mode) or the canonical HRF (canonical mode).

The panel reads from ``state.processed_cache`` for toeplitz mode (per
Sprint 3.2 contract: preprocess in 3.2, estimate in 3.3). Canonical mode
needs only ``state.raw_cache`` (no preprocessing required to generate the
canonical shape).

The panel subscribes to ``scan_selected``, ``scan_loaded``,
``preprocess_done``, and ``hrf_estimated`` so it re-renders when any
upstream tab changes state.

Scientific notes
----------------

- ``estimate_hrf`` runs with ``preprocess=False`` here because the Raw
  comes from ``processed_cache`` (already preprocessed in 3.2). Passing
  ``preprocess=True`` would silently re-run the canonical preprocess on
  preprocessed data, which is wrong.
- The canonical double-gamma uses ``scipy.stats.gamma.pdf`` at peaks 6 s
  and 16 s (with a 1/6 undershoot weight) — identical to the library's
  ``correlate_canonical`` implementation at hrfunc.py:756-761.
- The lambda slider is log-scale: the displayed value is ``10 ** raw``
  where ``raw`` is the slider's actual ``-5..-1`` integer. The library's
  default is ``1e-3``.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from nicegui import background_tasks, ui

from ..state import AppState
from ..events_io import (
    EventsParseError,
    build_impulse_from_rows,
    build_impulse_from_vector,
    coverage_report,
    coverage_report_vector,
    discover_collocated_events,
    find_event_files,
    parse_events_file,
)
from ..workers import (
    capture_client,
    client_scope,
    make_progress_callback,
    notify_if_alive,
    render_bulk_cancel_button,
    run_bulk_in_background,
    run_in_background,
    summarize_failures,
)
from ...io.manifest import ScanEntry

# GUI default edge-expansion fraction passed to montage.estimate_hrf. The
# core library default is 0.15; the GUI defaults a touch higher (0.2) because
# a slightly wider pad reduces edge-instability in practice, and exposes it as
# a user control. Used both as the estimate_hrf kwarg and to estimate the
# dropped start-of-scan window for the coverage warning so it matches what the
# core actually discards for the user's chosen value.
DEFAULT_EDGE_EXPANSION = 0.2

# Edge-instability QC defaults: flag a channel when its HRF trace's local std
# over the outer DEFAULT_EDGE_STD_FRAC (each end) exceeds DEFAULT_EDGE_STD_RATIO
# × the center std. Both are exposed as user controls.
DEFAULT_EDGE_STD_FRAC = 0.15
DEFAULT_EDGE_STD_RATIO = 2.0


def _file_applies_to_scan(state: AppState, scan: Optional[ScanEntry]) -> bool:
    """True when an uploaded events file should drive ``scan``'s estimation.

    The file overrides embedded annotations for the scan it was uploaded
    against (``events_source_scan``) always, and for every scan when
    ``events_apply_all`` is set (shared paradigm across the batch).
    """
    if (state.events_rows is None and state.events_impulse is None) or scan is None:
        return False
    if state.events_apply_all:
        return True
    src = state.events_source_scan
    return src is not None and src.path == scan.path


def _event_labels_for_scan(
    state: AppState, scan: Optional[ScanEntry], raw
) -> List[str]:
    """Available event labels for a scan — file labels when the file applies,
    else the scan's own annotation descriptions. An impulse vector has the
    single synthetic label "events"."""
    if _file_applies_to_scan(state, scan):
        if state.events_impulse is not None:
            return ["events"]
        return sorted({row.label for row in state.events_rows})
    return sorted_unique_annotation_descriptions(raw)


def _resolve_bulk_events(state: AppState, scan: ScanEntry, raw, snapshot):
    """Resolve the events a *bulk* HRF run should use for one scan.

    A bulk run can't lean on the single global events slot (it only holds one
    scan's worth) or on the UI's single label selection — the single-scan path
    fills those by auto-matching on selection, but checking a batch never
    triggers that. So resolve each scan independently, mirroring the
    single-scan precedence:

      1. the manually-loaded file when it applies (apply-to-all paradigm, or
         this is the upload's source scan) — keep the user's label selection;
      2. else a collocated sidecar discovered for THIS scan (BIDS-strict then
         lone-file) — use all of its conditions;
      3. else the scan's own embedded annotations — use every description.

    Returns ``(event_rows, event_impulse, selected_events)``. When all three
    sources are empty the tuple is ``(None, None, ())`` so the caller can fail
    the scan with a clear "no events found" reason instead of silently
    estimating on an all-zero impulse.
    """
    if _file_applies_to_scan(state, scan):
        return state.events_rows, state.events_impulse, snapshot.selected_events

    all_paths = [s.path for s in state.manifest.scans] if state.manifest else []
    found = discover_collocated_events(scan.path, all_paths)
    if found is not None:
        try:
            parsed = parse_events_file(found)
        except EventsParseError:
            parsed = None
        if parsed is not None:
            if parsed.impulse is not None:
                return None, list(parsed.impulse), ("events",)
            return parsed.rows, None, tuple(parsed.labels())

    # Fall back to this scan's own annotations (all descriptions).
    return None, None, tuple(sorted_unique_annotation_descriptions(raw))


def _maybe_automatch_events(
    state: AppState, scan: Optional[ScanEntry], opts: EstimationOptions
) -> None:
    """Auto-match a collocated events file to ``scan`` on selection.

    Fires when no loaded events cover this scan and the user hasn't opted it
    out (manual upload / clear, or a prior empty discovery). BIDS-strict then
    lone-file (see ``discover_collocated_events``). Loads silently with an
    "auto-matched" badge the user can clear or override. A successful match is
    NOT added to ``events_no_automatch`` so returning to the scan re-matches
    it even after the single global slot was replaced by another scan.
    """
    if scan is None or _file_applies_to_scan(state, scan):
        return
    key = scan.path.resolve()
    if key in state.events_no_automatch:
        return
    all_paths = [s.path for s in state.manifest.scans] if state.manifest else []
    found = discover_collocated_events(scan.path, all_paths)
    if found is None:
        state.events_no_automatch.add(key)  # nothing here; don't re-scan
        return
    try:
        parsed = parse_events_file(found)
    except EventsParseError:
        state.events_no_automatch.add(key)  # broken file; don't retry
        return
    _apply_parsed_events(state, opts, scan, parsed, is_auto=True)
    state.events_apply_all = False
    # Verbose: announce the auto-match (this fires once per scan — the match
    # is memoized — so it isn't spammy). The not-found case is conveyed by
    # the status banner ("scan annotations · none loaded from file").
    #
    # The toast is incidental FEEDBACK — the auto-match data work above is the
    # real behaviour. ``ui.notify`` requires an active UI slot, and this
    # function can run outside one (a unit test, or any non-render caller),
    # where it would raise "slot stack ... is empty" and abort the match that
    # already succeeded. Make the toast best-effort so it can never break the
    # data flow.
    try:
        ui.notify(
            f"Auto-matched events: {found.name} ({_events_summary(parsed)}).",
            type="positive",
        )
    except RuntimeError:
        logger.debug("auto-match toast skipped: no active UI slot")


def _events_summary(parsed) -> str:
    """One-line description of a parsed events file (format + counts)."""
    if parsed.impulse is not None:
        return f"impulse vector, {int(sum(parsed.impulse))} events"
    return (
        f"{parsed.fmt}, {len(parsed.rows)} onsets, "
        f"{len(parsed.labels())} condition(s)"
    )


def _apply_parsed_events(
    state: AppState,
    opts: EstimationOptions,
    scan: Optional[ScanEntry],
    parsed,
    *,
    is_auto: bool,
) -> None:
    """Load an ``EventsParse`` into state (handles onset rows vs impulse vector).

    Sets the source fields and seeds the selection. Caller owns
    ``events_apply_all`` and ``events_no_automatch`` policy.
    """
    if parsed.impulse is not None:
        state.events_impulse = list(parsed.impulse)
        state.events_rows = None
    else:
        state.events_rows = parsed.rows
        state.events_impulse = None
    state.events_format = parsed.fmt
    state.events_source_label = parsed.source_name
    state.events_source_scan = scan
    state.events_is_automatched = is_auto
    opts.selected_events = tuple(parsed.labels())


def _event_counts_for_scan(state: AppState, scan, raw) -> dict:
    """Map each event label to its occurrence count, for the picker display.

    Counts come from the uploaded file when it applies to ``scan``, else
    from the scan's annotations. Surfacing counts makes a non-descriptive
    marker (e.g. a numeric SNIRF trigger labelled "1.0") read as an event
    with N occurrences rather than a mystery checkbox.
    """
    from collections import Counter

    if _file_applies_to_scan(state, scan):
        if state.events_impulse is not None:
            return {"events": int(sum(state.events_impulse))}
        return dict(Counter(row.label for row in state.events_rows))
    counts: dict = {}
    annotations = getattr(raw, "annotations", None)
    if annotations is not None:
        for ann in annotations:
            description = str(ann["description"])
            if description:
                counts[description] = counts.get(description, 0) + 1
    return counts

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


MODEL_TOEPLITZ = "toeplitz"
MODEL_CANONICAL = "canonical"
DEFAULT_DURATION = 30.0
DEFAULT_LMBDA = 1e-3
LOG_LMBDA_MIN = -5
LOG_LMBDA_MAX = -1
# Per-channel solve timeout (s); mirrors montage.estimate_hrf's default.
DEFAULT_TIMEOUT = 30.0


@dataclass
class EstimationOptions:
    """User-controlled options for the HRFs tab.

    Snapshotted at Estimate-click time so the background task sees a stable
    view. Defaults mirror ``montage.estimate_hrf`` library defaults.
    """

    model: str = MODEL_TOEPLITZ
    lmbda: float = DEFAULT_LMBDA
    duration: float = DEFAULT_DURATION
    selected_events: Tuple[str, ...] = field(default_factory=tuple)
    # Per-channel lstsq solve timeout (seconds), forwarded to
    # montage.estimate_hrf. A channel that exceeds it is skipped.
    timeout: float = DEFAULT_TIMEOUT
    # Edge-expansion fraction forwarded to montage.estimate_hrf: each event
    # onset is shifted back by ``edge_expansion * duration`` seconds so the
    # estimation window captures the pre-onset baseline / ramp. A wider pad
    # reduces edge instability but drops more early-scan events.
    edge_expansion: float = DEFAULT_EDGE_EXPANSION
    # Edge-instability QC sensitivity (post-estimation check): a channel is
    # flagged when its HRF trace's local std over the outer ``edge_std_frac``
    # (each end) exceeds ``edge_std_ratio`` × the center std. Both are
    # user-tunable so the check can be made stricter / looser.
    edge_std_frac: float = DEFAULT_EDGE_STD_FRAC
    edge_std_ratio: float = DEFAULT_EDGE_STD_RATIO


def render(state: AppState) -> None:
    """Render the HRFs tab body inside the current NiceGUI context.

    Subscribes a refreshable body to scan/preprocess/HRF-result events so
    the panel reacts to upstream tab changes without rebuilding the whole
    workspace.
    """
    opts = EstimationOptions()

    @ui.refreshable
    def _body() -> None:
        _render_body(state, opts)

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)
    state.subscribe("preprocess_done", _refresh)
    # Rebuild the pooled project (group) montage BEFORE the body re-renders,
    # so the group preview reflects the scan that just finished. Registered
    # ahead of the _refresh subscription below so it runs first.
    state.subscribe("hrf_estimated", lambda *_: _rebuild_project_montage(state))
    state.subscribe("hrf_estimated", _refresh)
    # Dedicated event for gallery channel-pick refreshes — fires only the
    # HRFs-tab body re-render, avoiding the 6-subscriber re-render
    # cascade that republishing hrf_estimated would cause.
    state.subscribe("hrf_selection_changed", _refresh)
    # React to busy transitions so the inline "Preprocess now" / estimate
    # spinner appears immediately on start and clears on finish (the poll
    # timer below only fires while estimation_progress is set).
    state.subscribe("busy_changed", _refresh)
    # Recompute bulk mode when the dataset-tree checked set changes, so
    # ticking scans immediately makes them eligible to estimate.
    state.subscribe("checked_changed", _refresh)

    # Progress polling timer — refreshes the body every 0.5 s WHILE an
    # estimation is in flight, so the progress bar advances. The
    # progress_callback fires from a worker thread (run_in_executor) and
    # cannot safely refresh NiceGUI elements from there, so we poll from
    # the main loop instead. Cheap no-op when state.busy is False.
    def _poll_progress() -> None:
        if state.busy and state.estimation_progress is not None:
            _body.refresh()

    ui.timer(0.5, _poll_progress)


def _resolve_checked_scans(state: AppState) -> List[ScanEntry]:
    """Resolve ``state.checked_scan_paths`` to ScanEntries in manifest order.

    PR #55a helper. Same shape / semantics as the preprocess panel's
    helper -- duplicated here intentionally to keep panel modules
    self-contained (no cross-panel imports beyond the worker layer).
    """
    if state.manifest is None or not state.checked_scan_paths:
        return []
    return [
        scan for scan in state.manifest.scans
        if scan.path.resolve() in state.checked_scan_paths
    ]


def _render_body(state: AppState, opts: EstimationOptions) -> None:
    """Render the HRFs body against the current state.

    Module-level so tests can call it directly inside a synthetic NiceGUI
    context without going through the refreshable wrapper.
    """
    scan = state.selected_scan
    checked_scans = _resolve_checked_scans(state)
    bulk_mode = len(checked_scans) >= 1

    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("HRFs").classes("text-2xl font-semibold")

        if scan is None and not bulk_mode:
            # Disclaimer callout: nothing can be estimated until a scan is
            # picked, so make the next step obvious rather than a faint line.
            with ui.row().classes(
                "w-full items-center gap-3 p-4 rounded border "
                "border-amber-500/40 bg-amber-500/10"
            ):
                ui.icon("arrow_back", size="1.5rem").classes("text-amber-400")
                with ui.column().classes("gap-0"):
                    ui.label(
                        "Select a scan to start estimating HRFs"
                    ).classes("text-sm font-medium text-amber-300")
                    ui.label(
                        "Pick a scan from the dataset tree on the left, or "
                        "tick several scans there for a bulk run."
                    ).classes("text-xs opacity-70")
            # Submission stays available with no scan selected so users who
            # only want to share an existing HRF JSON they have on disk can.
            _render_submission_section(state)
            return

        # Two-column workspace: estimation controls on the left, the
        # per-channel HRF visualization in its own column on the right
        # (mirrors the HRtree tab's viz | detail split so the channel
        # accordion gets horizontal room instead of stacking under the
        # controls).
        with ui.row().classes("w-full gap-6 items-start no-wrap"):
            with ui.column().classes("flex-1 min-w-0 gap-4"):
                if bulk_mode:
                    ui.label(
                        f"Bulk run on {len(checked_scans)} checked scan"
                        f"{'s' if len(checked_scans) != 1 else ''}."
                    ).classes("text-sm font-mono opacity-70")
                    if opts.model == MODEL_TOEPLITZ:
                        ui.label(
                            "Each scan auto-discovers its own events at run "
                            "time: an applied uploaded file → a collocated "
                            "sidecar (BIDS or lone file) → the scan's embedded "
                            "annotations. Scans with no events of any kind are "
                            "reported as failures with the reason."
                        ).classes("text-xs opacity-60 italic")
                elif scan is not None:
                    ui.label(scan.display_name or scan.path.name).classes(
                        "text-sm font-mono opacity-70"
                    )

                # ── Model selector + controls. The controls remain anchored
                # on ``selected_scan`` for lambda / duration / (single-scan)
                # event picking. In bulk the events are NOT taken from the UI
                # scan's selection -- each checked scan resolves its own
                # events via ``_resolve_bulk_events`` (uploaded-file →
                # collocated sidecar → annotations), so a batch of scans with
                # per-scan sidecars all estimate correctly instead of
                # inheriting one scan's labels.
                with ui.card().classes("w-full"):
                    ui.label("Estimation").classes(
                        "text-xs uppercase opacity-60 tracking-wide"
                    )
                    _render_model_radio(opts, _refresh_body_for(state))
                    if opts.model == MODEL_TOEPLITZ:
                        if scan is not None:
                            _render_toeplitz_controls(state, opts, scan)
                        else:
                            # Bulk: scans ticked, no row selected. Show the
                            # global lambda / duration / timeout controls so
                            # the batch can be tuned, plus a preprocess-
                            # readiness summary + "Preprocess all checked"
                            # button. Per-scan events are auto-discovered at
                            # run time (see the bulk note above), so there's
                            # no event picker in this branch.
                            from .preprocess_panel import (
                                render_preprocess_all_checked,
                            )
                            _render_toeplitz_global_params(opts)
                            render_preprocess_all_checked(
                                state, checked_scans
                            )
                    else:
                        _render_canonical_note()

                # ── Run button (+ Save when ready) + progress / error display
                _render_run_row(state, scan, checked_scans, opts)

                # ── Submission — directly below Estimate, always present so an
                # existing HRF JSON can be shared without estimating first.
                _render_submission_section(state)

            # ── Right column: per-channel HRF visualization
            with ui.column().classes("flex-1 min-w-0 gap-3"):
                _render_preview_column(state, scan, opts, bulk_mode)


def _project_montages(state: AppState) -> list:
    """Real per-scan montages in the group pool (skips canonical + excluded)."""
    return [
        m for path, m in state.montage_cache.items()
        if m is not None and not isinstance(m, _CanonicalResult)
        and path not in state.project_group_excluded
    ]


def _project_subject_count(state: AppState) -> int:
    return len(_project_montages(state))


def _build_project_montage(sourced_montages: list):
    """Pool per-scan montages into one GROUP montage, tagging provenance.

    ``sourced_montages`` is a list of ``(source_id, montage)`` pairs. Each group
    channel collects EVERY contributing subject's estimate (tagging each with
    its ``source_id`` in ``estimate_sources``), then ``generate_distribution``
    re-means/re-stds across them — so ``trace_std`` becomes the genuine
    between-subject variability (a single scan's montage has one estimate → std
    0). Tagging lets a saved+reloaded group montage still report / remove a
    specific subject (see ``montage.remove_source``). Returns None when nothing
    poolable. Duck-typed (``.channels`` of nodes with ``.estimates`` /
    ``.estimate_sources`` + a ``.generate_distribution()``) so tests can pass
    lightweight fakes.
    """
    import copy

    real = [
        (sid, m) for sid, m in sourced_montages
        if m is not None and not isinstance(m, _CanonicalResult)
    ]
    if not real:
        return None

    group = copy.deepcopy(real[0][1])
    # Strip any per-scan 'global_*' aggregate nodes the base carries in.
    # Each subject montage already ran generate_distribution, which
    # synthesises global_hbo/global_hbr channels that hold a single estimate
    # (that subject's own grand mean). If those survive into the pool, the
    # final generate_distribution sees their non-empty ``estimates`` and
    # treats them as real channels -- folding a "global of globals" back
    # into the group mean and deflating the between-subject std. Drop them
    # and let the final generate_distribution rebuild the globals cleanly
    # from the real channels only.
    for ch in [c for c in group.channels if "global" in c]:
        del group.channels[ch]
    # Union of channels: graft in any channel present in a later subject but
    # absent from the base (heterogeneous montage layouts across subjects).
    for _sid, m in real[1:]:
        for ch, node in getattr(m, "channels", {}).items():
            if "global" in ch:
                continue  # globals are rebuilt below, never pooled
            if ch not in group.channels:
                group.channels[ch] = copy.deepcopy(node)
    # Pool every subject's estimate into each channel, tagged by source.
    for ch, gnode in group.channels.items():
        pooled, sources = [], []
        for sid, m in real:
            other = getattr(m, "channels", {}).get(ch)
            if other is not None and getattr(other, "estimates", None):
                for est in other.estimates:
                    pooled.append(est)
                    sources.append(sid)
        gnode.estimates = pooled
        gnode.estimate_sources = sources
    try:
        group.generate_distribution()
    except Exception as exc:  # noqa: BLE001 — never break the panel on a pool error
        logger.warning("project montage generate_distribution failed: %s", exc)
        return None
    return group


def _sourced_project_montages(state: AppState) -> list:
    """``(source_id, montage)`` pairs for the group pool — source id is the
    scan's resolved path string (stable, unique, survives manifest rebuilds)."""
    return [
        (str(path), m) for path, m in state.montage_cache.items()
        if m is not None and not isinstance(m, _CanonicalResult)
        and path not in state.project_group_excluded
    ]


def _rebuild_project_montage(state: AppState) -> None:
    """Refresh ``state.project_montage`` from the current per-scan cache."""
    state.project_montage = _build_project_montage(
        _sourced_project_montages(state)
    )


def _render_preview_view_toggle(state: AppState, n_subjects: int) -> None:
    """Radio toggle between the selected scan's HRFs and the group montage."""
    def _set(event) -> None:
        state.hrf_preview_group = bool(event.value)
        state.publish("hrf_selection_changed")

    ui.radio(
        {False: "This scan", True: f"Group ({n_subjects} subjects)"},
        value=state.hrf_preview_group,
        on_change=_set,
    ).props("inline dense").classes("text-xs")


def _group_subject_names(state: AppState) -> list:
    """Display names of the scans contributing to the group montage.

    Maps the (non-canonical) ``montage_cache`` keys back to manifest scans;
    falls back to the path stem when a key isn't in the current manifest.
    """
    contributing = {
        path for path, m in state.montage_cache.items()
        if m is not None and not isinstance(m, _CanonicalResult)
        and path not in state.project_group_excluded
    }
    by_path = {}
    if state.manifest is not None:
        by_path = {s.path.resolve(): s for s in state.manifest.scans}
    names = []
    for path in contributing:
        scan = by_path.get(path)
        names.append(
            (scan.display_name or scan.path.name) if scan is not None
            else path.stem
        )
    return sorted(names)


async def _save_group_montage(state: AppState) -> None:
    """Save the pooled group montage to JSON (reuses the Export save path)."""
    import asyncio

    montage = state.project_montage
    if montage is None:
        state.last_error = "No group montage to save."
        return
    from .export_panel import _pick_save_path, save_montage_sync

    path = await _pick_save_path(
        suggested="group_hrfs.json", title="Save group HRFs"
    )
    if path is None:
        return
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, save_montage_sync, montage, path
        )
        ui.notify(f"Saved group HRFs: {path.name}", type="positive")
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"Save failed: {type(exc).__name__}: {exc}"
        logger.exception("group montage save failed: %s", exc)
        ui.notify(state.last_error, type="negative")


def _group_subject_entries(state: AppState) -> list:
    """``(path, name, excluded)`` for every non-canonical cached scan."""
    by_path = {}
    if state.manifest is not None:
        by_path = {s.path.resolve(): s for s in state.manifest.scans}
    out = []
    for path, m in state.montage_cache.items():
        if m is None or isinstance(m, _CanonicalResult):
            continue
        scan = by_path.get(path)
        name = (scan.display_name or scan.path.name) if scan else path.stem
        out.append((path, name, path in state.project_group_excluded))
    return sorted(out, key=lambda e: e[1])


def _set_group_excluded(state: AppState, path, excluded: bool) -> None:
    if excluded:
        state.project_group_excluded.add(path)
    else:
        state.project_group_excluded.discard(path)
    _rebuild_project_montage(state)
    state.publish("hrf_selection_changed")


def _render_group_subjects(state: AppState) -> None:
    """Contributing-subjects readout, per-subject remove/restore, + Save."""
    entries = _group_subject_entries(state)
    included = [e for e in entries if not e[2]]
    excluded = [e for e in entries if e[2]]

    ui.label(f"Contributing scans ({len(included)})").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )
    with ui.row().classes("items-center gap-1 flex-wrap"):
        for path, name, _ in included:
            with ui.row().classes(
                "items-center gap-1 px-2 py-0.5 rounded bg-slate-700/40"
            ):
                ui.label(name).classes("text-xs")
                # Keep at least 2 subjects so it stays a group (and the view +
                # restore controls remain reachable).
                if len(included) > 2:
                    ui.button(
                        icon="close",
                        on_click=lambda p=path: _set_group_excluded(
                            state, p, True
                        ),
                    ).props("flat dense round size=xs").tooltip(
                        "Remove this subject from the group"
                    )

    if excluded:
        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.label(
                "Excluded: " + ", ".join(e[1] for e in excluded)
            ).classes("text-xs opacity-50")
            ui.button(
                "Restore all",
                on_click=lambda: _set_group_excluded_restore_all(state),
            ).props("flat dense size=xs color=primary")

    async def _on_save() -> None:
        await _save_group_montage(state)

    with ui.row().classes("items-center gap-2"):
        ui.button(
            "Save group HRFs", icon="save", on_click=_on_save,
        ).props("flat dense color=primary")
        ui.label(
            "Saves the pooled multi-subject HRFs (mean + std) as a "
            "tree.load_hrfs JSON — the submission form below can share it."
        ).classes("text-xs opacity-50")


def _set_group_excluded_restore_all(state: AppState) -> None:
    state.project_group_excluded.clear()
    _rebuild_project_montage(state)
    state.publish("hrf_selection_changed")


def _render_preview_column(
    state: AppState,
    scan: Optional[ScanEntry],
    opts: EstimationOptions,
    bulk_mode: bool,
) -> None:
    """Right-hand column: the per-channel HRF visualization.

    Kept in its own column (mirroring the HRtree detail pane) so the channel
    accordion has horizontal room and the estimation controls on the left stay
    compact. Shows a placeholder until a montage exists. The Save button and
    submission form live in the left column (next to / below Estimate).

    Bulk mode overwrites ``state.montage`` scan-by-scan, so during a run the
    preview would just flash whichever scan landed last -- we gate on
    ``not state.busy`` so it only appears once the batch FINISHES, and point
    the preview at the scan that actually produced the montage
    (``montage_source_scan``) rather than the UI selection.
    """
    ui.label("HRF preview").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )

    if state.busy:
        ui.label("Estimating…").classes("text-sm opacity-60")
        return

    # Group view: once ≥2 scans have HRFs, offer a toggle between the selected
    # scan's HRFs and the pooled project (group) montage, whose ±std band is
    # the genuine between-subject variability.
    n_subjects = _project_subject_count(state)
    if n_subjects >= 2 and state.project_montage is not None:
        _render_preview_view_toggle(state, n_subjects)
        if state.hrf_preview_group:
            ui.label(
                f"Group HRF — {n_subjects} subjects (between-subject ± std)."
            ).classes("text-sm font-medium")
            _render_group_subjects(state)
            _render_edge_qc(state.project_montage, opts)
            _render_toeplitz_gallery(state, state.project_montage)
            return

    # Upstream nudge: estimated-HRF Neural Activity deconvolution needs a
    # GROUP montage (≥2 subjects). After a single subject, surface that here
    # so the user isn't blindsided by a disabled Run on the Activity tab.
    if (
        n_subjects == 1
        and state.montage is not None
        and not isinstance(state.montage, _CanonicalResult)
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("groups", size="sm").classes("text-blue-600")
            ui.label(
                "1 subject estimated. Estimate HRFs for at least one more "
                "subject to build the GROUP HRFs that Neural Activity "
                "deconvolution uses — a single subject has no between-subject "
                "variability. (Canonical / HRtree sources work with one scan.)"
            ).classes("text-xs text-blue-800")

    preview_scan = scan if not bulk_mode else state.montage_source_scan
    if state.montage is None or preview_scan is None:
        ui.label(
            "Estimate HRFs to see the per-channel results here."
        ).classes("text-sm opacity-60")
        return

    if bulk_mode:
        shown = preview_scan.display_name or preview_scan.path.name
        ui.label(
            f"Showing HRFs for the last-estimated scan: {shown}. "
            "A bulk run keeps only the most recent scan's HRFs in memory — "
            "save each scan's HRFs (the Save button by Estimate, or mass-"
            "save), or select a single scan to view or re-estimate it."
        ).classes("text-xs opacity-70 italic")

    _render_hrf_preview(state, preview_scan, opts)


def _render_save_button(
    state: AppState, scan: Optional[ScanEntry], bulk_mode: bool
) -> None:
    """Save-HRFs button, shown beside Estimate only when there's a real
    (toeplitz) montage to save — same pop-in-when-ready behaviour it had in
    the preview column, just relocated next to the Estimate button.

    Bulk mode points the save at the scan that produced the in-memory montage
    (``montage_source_scan``), matching the preview. Canonical results aren't
    saveable here (no per-channel estimates), so the button stays hidden.
    """
    if state.busy:
        return
    save_scan = scan if not bulk_mode else state.montage_source_scan
    if (
        state.montage is None
        or save_scan is None
        or isinstance(state.montage, _CanonicalResult)
    ):
        return

    async def _on_save_hrfs(_scan=save_scan) -> None:
        from .export_panel import _save_montage
        await _save_montage(state, _scan)

    ui.button(
        "Save",
        icon="save",
        on_click=_on_save_hrfs,
    ).props("color=primary").tooltip(
        "Saves the per-channel estimated HRFs as a tree.load_hrfs JSON file."
    )


def _render_submission_section(state: AppState) -> None:
    """HRF submission form — same form the Export tab renders, surfaced right
    below the Estimate button.

    Always rendered (not gated on an in-memory montage) so a user who just
    wants to share an existing HRF JSON they have on disk can do it here
    without first estimating. The file picker defaults to
    ``state.last_saved_roi_path`` (set by a prior save / the Cluster sub-tab);
    otherwise they pick the JSON manually.
    """
    with ui.card().classes("w-full"):
        from ..submission import render_submission_panel
        render_submission_panel(
            state,
            default_path=state.last_saved_roi_path,
        )


def _refresh_body_for(state: AppState):
    """Return a callable that re-publishes scan_selected to refresh subscribers.

    Used by the model-radio change handler to trigger a re-render of the
    panel body when the user flips between toeplitz and canonical (the
    rendered controls differ between modes).
    """
    def _trigger() -> None:
        state.publish("scan_selected", state.selected_scan)
    return _trigger


def _render_model_radio(
    opts: EstimationOptions, on_change_refresh
) -> None:
    def _set(value: str) -> None:
        opts.model = value
        on_change_refresh()

    ui.radio(
        [MODEL_TOEPLITZ, MODEL_CANONICAL],
        value=opts.model,
        on_change=lambda e: _set(e.value),
    ).props("inline")


def _preprocess_now(state: AppState, scan: ScanEntry) -> None:
    """Run the DECONVOLUTION preprocessing pipeline on ``scan`` from the HRFs
    tab.

    Reuses the Preprocess tab's ``run_pipeline_sync`` so the output is
    byte-identical to running it there (one source of truth for the
    pipeline). HRF estimation requires deconvolution-mode preprocessing, so
    this forces ``deconvolution=True`` and records it in
    ``processed_deconvolved``. On completion the result lands in
    ``processed_cache`` and ``preprocess_done`` fires, refreshing the panel.
    """
    if state.busy:
        return
    if scan not in state.raw_cache:
        state.last_error = "Raw not loaded yet; wait for the scan to load."
        return
    # Lazy import: keep the module-load import graph free of a cross-panel
    # dependency, and guarantee the same pipeline as the Preprocess tab.
    from .preprocess_panel import PreprocessOptions, run_pipeline_sync

    snapshot = PreprocessOptions(deconvolution=True)  # required for HRF est.
    raw = state.raw_cache.get(scan)

    async def _on_done(result) -> None:
        if result is None:
            return
        state.processed_cache._cache[scan.path.resolve()] = result
        # Honor the LRU bound (RawCache.put() centralizes this once PR #68
        # lands; evict inline here as the other cache writers do today).
        while len(state.processed_cache._cache) > state.processed_cache.maxsize:
            state.processed_cache._cache.popitem(last=False)
        state.processed_deconvolved.add(scan.path.resolve())
        state.publish("preprocess_done", scan)

    ui.notify("Preprocessing…", type="info")
    background_tasks.create(
        run_in_background(
            state, run_pipeline_sync, raw, snapshot, on_done=_on_done
        )
    )


def _render_preprocess_now(state: AppState, scan: ScanEntry) -> None:
    """Render the 'not preprocessed yet -> Preprocess now' affordance."""
    if state.busy:
        with ui.row().classes("items-center gap-2"):
            ui.spinner(size="sm")
            ui.label("Preprocessing…").classes("text-sm opacity-70")
        return
    ui.label(
        "This scan isn't preprocessed yet — HRF estimation runs on the "
        "preprocessed signal."
    ).classes("text-sm opacity-70")
    loaded = scan in state.raw_cache
    with ui.row().classes("items-center gap-2"):
        ui.button(
            "Preprocess now",
            icon="play_arrow",
            on_click=lambda: _preprocess_now(state, scan),
        ).props(f"color=primary {'disable' if not loaded else ''}")
        ui.label(
            "Uses default settings — for custom options use the Preprocess tab."
        ).classes("text-xs opacity-60")
    if not loaded:
        ui.label(
            "Waiting for the scan to finish loading…"
        ).classes("text-xs opacity-60")


def _render_needs_deconvolution(state: AppState, scan: ScanEntry) -> None:
    """Shown when a scan is preprocessed in GLM/haemoglobin mode: HRF
    estimation requires the deconvolution pipeline, so block + offer a
    one-click re-preprocess in deconvolution mode."""
    if state.busy:
        with ui.row().classes("items-center gap-2"):
            ui.spinner(size="sm")
            ui.label("Preprocessing…").classes("text-sm opacity-70")
        return
    ui.label(
        "This scan was preprocessed for haemoglobin (GLM mode), but HRF "
        "estimation requires the deconvolution pipeline."
    ).classes("text-sm text-amber-400")
    ui.button(
        "Re-preprocess with deconvolution",
        icon="play_arrow",
        on_click=lambda: _preprocess_now(state, scan),
    ).props("color=primary")


def _set_events_selection(
    state: AppState, opts: EstimationOptions, selection: Tuple[str, ...]
) -> None:
    """Set the selected-events tuple and refresh the picker."""
    opts.selected_events = selection
    state.publish("hrf_selection_changed")


def _set_apply_all(state: AppState, value: bool) -> None:
    state.events_apply_all = value
    state.publish("hrf_selection_changed")


def _clear_events_file(state: AppState, opts: EstimationOptions) -> None:
    """Drop the loaded events file and revert to scan annotations.

    The scan the file belonged to is added to ``events_no_automatch`` so the
    auto-matcher doesn't immediately re-load it on the next render — clearing
    is an explicit "use annotations for this scan" choice.
    """
    if state.events_source_scan is not None:
        state.events_no_automatch.add(state.events_source_scan.path.resolve())
    state.events_rows = None
    state.events_impulse = None
    state.events_format = None
    state.events_source_label = None
    state.events_apply_all = False
    state.events_source_scan = None
    state.events_is_automatched = False
    opts.selected_events = ()  # re-seeds from annotations on next render
    state.publish("hrf_selection_changed")


def _render_events_source(
    state: AppState, opts: EstimationOptions, scan: ScanEntry
) -> None:
    """Compact events-source row: a loud status of what's loaded + a single
    "Upload / find events…" button that opens the find-window dialog where
    all the loading options live."""
    loaded = state.events_rows is not None or state.events_impulse is not None

    if loaded:
        _render_events_status(state, scan)
    else:
        ui.label(
            "Events source: this scan's own annotations (none loaded from "
            "file)."
        ).classes("text-xs opacity-60")

    async def _open() -> None:
        # Surface any failure as a toast — a silent exception here is exactly
        # the "button does nothing" symptom.
        try:
            await _open_events_dialog(state, opts, scan)
        except Exception as exc:  # noqa: BLE001
            logger.exception("events dialog failed to open: %s", exc)
            ui.notify(
                f"Couldn't open events dialog: {type(exc).__name__}: {exc}",
                type="negative",
            )

    with ui.row().classes("items-center gap-2"):
        ui.button(
            "Upload / find events…", icon="upload_file", on_click=_open
        ).props("flat dense")
        if loaded:
            ui.checkbox(
                "Apply to all checked scans",
                value=state.events_apply_all,
                on_change=lambda e: _set_apply_all(state, bool(e.value)),
            ).props("dense").classes("text-xs")
            ui.button(
                "Use annotations instead",
                on_click=lambda: _clear_events_file(state, opts),
            ).props("flat dense")


async def _open_events_dialog(
    state: AppState, opts: EstimationOptions, scan: Optional[ScanEntry]
) -> None:
    """Find-window dialog for loading events.

    A glob box searches the loaded project folder for event files and
    highlights matches; selecting one shows a format/count preview so a
    wrong file is obvious before loading. "Browse…" falls back to the OS
    picker. On Load the file is parsed and applied (with the apply-to-all
    choice). All file-loading options live here so the HRFs page stays tidy.
    """
    root = state.manifest.root if state.manifest is not None else None
    sel = {
        "pattern": "",
        "results": find_event_files(root, "") if root else [],
        "chosen": None,
        "parsed": None,
        "error": None,
        "apply_all": state.events_apply_all,
    }

    def _choose(path: Path) -> None:
        sel["chosen"] = path
        try:
            sel["parsed"] = parse_events_file(path)
            sel["error"] = None
        except EventsParseError as exc:
            sel["parsed"] = None
            sel["error"] = str(exc)
        _results.refresh()
        _preview.refresh()

    async def _search(_e=None) -> None:
        # find_event_files walks the whole project tree; run it off the event
        # loop so a large dataset doesn't freeze the UI during the search.
        if root:
            import asyncio

            loop = asyncio.get_event_loop()
            sel["results"] = await loop.run_in_executor(
                None, find_event_files, root, sel["pattern"]
            )
        else:
            sel["results"] = []
        _results.refresh()

    with ui.dialog() as dialog, ui.card().classes("w-[640px] max-w-full gap-2"):
        ui.label("Load events").classes("text-lg font-semibold")
        ui.label(
            f"Searching project: {root}" if root else
            "No project loaded — use Browse to pick a file."
        ).classes("text-xs opacity-60 break-all")

        with ui.row().classes("w-full items-center gap-2"):
            ui.input(
                placeholder="filter, e.g. events*  or  *.tsv  (blank = all)",
                on_change=lambda e: sel.__setitem__("pattern", e.value or ""),
            ).props("dense outlined").classes("flex-1").on(
                # Pass the coroutine directly — a sync lambda wrapping the
                # async _search would return an un-awaited coroutine (no-op).
                "keydown.enter", _search
            )
            ui.button("Search", icon="search", on_click=_search).props("flat dense")

            async def _browse() -> None:
                from .dataset_picker import pick_file
                path = await pick_file(
                    file_types=["Events (*.tsv;*.csv;*.txt)", "All files (*.*)"],
                )
                if path is not None:
                    _choose(path)

            ui.button("Browse…", icon="folder_open", on_click=_browse).props(
                "flat dense"
            )

        @ui.refreshable
        def _results() -> None:
            if not sel["results"]:
                ui.label(
                    "No matching files. Adjust the filter and Search, or Browse."
                ).classes("text-sm opacity-60")
                return
            with ui.column().classes("w-full max-h-60 overflow-auto gap-0"):
                for path in sel["results"]:
                    rel = (
                        str(path.relative_to(root)) if root
                        and root in path.parents else path.name
                    )
                    is_chosen = sel["chosen"] == path
                    row = ui.row().classes(
                        "w-full items-center gap-2 px-2 py-1 rounded cursor-pointer "
                        + ("bg-indigo-700/40" if is_chosen else "hover:bg-slate-700/40")
                    )
                    with row:
                        ui.icon("description").classes("text-indigo-300 text-sm")
                        ui.label(rel).classes("text-xs font-mono")
                    row.on("click", lambda _e=None, p=path: _choose(p))

        _results()

        @ui.refreshable
        def _preview() -> None:
            if sel["error"]:
                ui.label(f"⚠ Can't read: {sel['error']}").classes(
                    "text-xs text-red-400"
                )
                return
            parsed = sel["parsed"]
            if parsed is None:
                return
            if parsed.impulse is not None:
                ui.label(
                    f"✓ {sel['chosen'].name}: IMPULSE vector · "
                    f"{len(parsed.impulse)} samples · "
                    f"{int(sum(parsed.impulse))} events."
                ).classes("text-xs text-green-400")
            else:
                ui.label(
                    f"✓ {sel['chosen'].name}: {parsed.fmt.upper()} · "
                    f"{len(parsed.rows)} onsets · "
                    f"{len(parsed.labels())} condition(s)."
                ).classes("text-xs text-green-400")
                if parsed.needs_simple_confirm:
                    ui.label(
                        "Note: simple onset,label shape (not BIDS events.tsv)."
                    ).classes("text-xs opacity-60")

        _preview()

        ui.checkbox(
            "Apply to all checked scans",
            value=sel["apply_all"],
            on_change=lambda e: sel.__setitem__("apply_all", bool(e.value)),
        ).props("dense").classes("text-xs")

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
            ui.button(
                "Load",
                on_click=lambda: dialog.submit(
                    sel["chosen"] if sel["parsed"] is not None else None
                ),
            ).props("color=primary")

    dialog.open()
    chosen = await dialog
    if chosen is None:
        return
    try:
        parsed = parse_events_file(chosen)
    except EventsParseError as exc:
        ui.notify(f"Couldn't read events: {exc}", type="negative")
        return
    _apply_parsed_events(state, opts, scan, parsed, is_auto=False)
    state.events_apply_all = bool(sel["apply_all"])
    if scan is not None:
        state.events_no_automatch.add(scan.path.resolve())
    for warning in parsed.warnings:
        ui.notify(warning, type="warning")
    ui.notify(
        f"Loaded events from {chosen.name} ({_events_summary(parsed)}).",
        type="positive",
    )
    state.publish("hrf_selection_changed")


def _render_events_status(state: AppState, scan: Optional[ScanEntry]) -> None:
    """LOUD banner describing the loaded events: file, format, how it loaded,
    counts, and (for impulse vectors) whether the length matches this scan."""
    fmt = (state.events_format or "").upper()
    how = "auto-matched" if state.events_is_automatched else "uploaded"
    with ui.column().classes(
        "w-full gap-0.5 px-3 py-2 rounded-md "
        "bg-indigo-950/40 border border-indigo-700/60"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon(
                "auto_awesome" if state.events_is_automatched else "description"
            ).classes("text-indigo-300")
            ui.label(
                f"Events {how}: {state.events_source_label or '(file)'}"
            ).classes("text-sm font-medium")
            ui.badge(fmt or "?").props("color=indigo")

        # Format-specific provenance line.
        if state.events_impulse is not None:
            n_samp = len(state.events_impulse)
            n_ev = int(sum(state.events_impulse))
            ui.label(
                f"Per-sample impulse vector · {n_samp} samples · {n_ev} event "
                "onset(s) · applied by sample index."
            ).classes("text-xs opacity-80")
            # Length-match check against the current scan — the key "is this
            # the right vector for this scan?" signal.
            raw = (
                state.processed_cache.get(scan)
                if scan is not None and scan in state.processed_cache
                else (
                    state.raw_cache.get(scan)
                    if scan is not None and scan in state.raw_cache
                    else None
                )
            )
            if raw is not None:
                scan_n = int(raw.n_times)
                if scan_n == n_samp:
                    ui.label(
                        f"✓ Length matches this scan ({scan_n} samples)."
                    ).classes("text-xs text-green-400")
                elif n_samp > scan_n:
                    ui.label(
                        f"⚠ Vector ({n_samp}) is LONGER than this scan "
                        f"({scan_n}) — trailing {n_samp - scan_n} samples will "
                        "be dropped."
                    ).classes("text-xs text-amber-400")
                else:
                    ui.label(
                        f"⚠ Vector ({n_samp}) is SHORTER than this scan "
                        f"({scan_n}) — the tail has no events."
                    ).classes("text-xs text-amber-400")
        else:
            rows = state.events_rows or []
            labels = sorted({r.label for r in rows})
            shown = ", ".join(labels[:6]) + ("…" if len(labels) > 6 else "")
            ui.label(
                f"Onset events · {len(rows)} onset(s) · {len(labels)} "
                f"condition(s): {shown}"
            ).classes("text-xs opacity-80")

        scope = (
            "ALL checked scans" if state.events_apply_all else "this scan only"
        )
        ui.label(f"Applies to: {scope}").classes("text-xs opacity-60")


async def _confirm_simple_events(parsed) -> bool:
    """Warn that a file matched only the simple onset,label shape; confirm use."""
    with ui.dialog() as dialog, ui.card():
        ui.label("Simple events file").classes("text-base font-semibold")
        ui.label(
            f"'{parsed.source_name}' looks like a simple onset,label file, "
            "not a BIDS events.tsv (no trial_type / duration columns). "
            f"Continue using it? Detected {len(parsed.rows)} events across "
            f"{len(parsed.labels())} label(s)."
        ).classes("text-sm opacity-80")
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button(
                "Use it", on_click=lambda: dialog.submit(True)
            ).props("color=primary")
    dialog.open()
    return bool(await dialog)


async def _confirm_coverage(issues) -> bool:
    """Confirm dialog listing scans whose events fall outside the usable window."""
    with ui.dialog() as dialog, ui.card().classes("max-w-xl"):
        ui.label("Some events fall outside the scan").classes(
            "text-base font-semibold"
        )
        ui.label(
            "These scans are shorter than the events extend, or have events "
            "inside the dropped edge-expansion window. Out-of-range events "
            "won't contribute to the estimate. Proceed anyway?"
        ).classes("text-sm opacity-80")
        for scan, cov in issues:
            name = scan.display_name or scan.path.name
            parts = []
            if cov.past_end:
                parts.append(f"{cov.past_end} past end")
            if cov.in_edge:
                parts.append(f"{cov.in_edge} in edge window")
            ui.label(
                f"• {name}: {cov.scan_seconds:.0f}s scan, events to "
                f"{cov.max_onset_s:.0f}s — {', '.join(parts)}; "
                f"{cov.placed} usable."
            ).classes("text-xs font-mono opacity-80")
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button(
                "Estimate anyway", on_click=lambda: dialog.submit(True)
            ).props("color=primary")
    dialog.open()
    return bool(await dialog)


async def _estimate_clicked(
    state: AppState,
    selected: Optional[ScanEntry],
    checked: List[ScanEntry],
    opts: EstimationOptions,
) -> None:
    """Estimate-button handler: coverage pre-flight, then dispatch.

    For toeplitz scans driven by an uploaded events file, check that the
    scan is long enough to contain the events (and flag the dropped edge
    window). If any scan drops events, confirm before running. Annotation-
    sourced scans keep the library's lenient drop behavior.
    """
    issues = []
    if opts.model == MODEL_TOEPLITZ:
        scans = list(checked) if checked else (
            [selected] if selected is not None else []
        )
        for scan in scans:
            if not _file_applies_to_scan(state, scan):
                continue
            if scan not in state.processed_cache:
                continue
            raw = state.processed_cache.get(scan)
            sfreq = float(raw.info["sfreq"])
            n_samples = int(raw.n_times)
            edge_samples = int(round(opts.edge_expansion * opts.duration * sfreq))
            if state.events_impulse is not None:
                cov = coverage_report_vector(
                    state.events_impulse, sfreq, n_samples, edge_samples,
                )
            else:
                cov = coverage_report(
                    state.events_rows, opts.selected_events,
                    sfreq, n_samples, edge_samples,
                )
            if cov.past_end or cov.in_edge:
                issues.append((scan, cov))
    if issues and not await _confirm_coverage(issues):
        return
    _run_dispatch(state, selected, checked, opts)


def _render_toeplitz_controls(
    state: AppState, opts: EstimationOptions, scan: ScanEntry
) -> None:
    """Event picker + lmbda slider + duration field for toeplitz mode."""
    # Try to auto-match a collocated events file for this scan before we read
    # the event source (no-op if one already applies or the user opted out).
    _maybe_automatch_events(state, scan, opts)
    raw = state.processed_cache.get(scan) if scan in state.processed_cache else None

    # ── Event picker
    ui.label("Events").classes("text-xs uppercase opacity-60 tracking-wide")
    if raw is None:
        # Not preprocessed yet -- offer a one-click "Preprocess now" right
        # here instead of sending the user to the Preprocess tab.
        _render_preprocess_now(state, scan)
        return

    if scan.path.resolve() not in state.processed_deconvolved:
        # Preprocessed, but with the GLM/haemoglobin pipeline — HRF
        # estimation requires deconvolution-mode preprocessing. Hard-block
        # and offer a one-click re-preprocess.
        _render_needs_deconvolution(state, scan)
        return

    # Source row: upload an events file (overrides annotations) or use the
    # scan's own annotations.
    _render_events_source(state, opts, scan)

    event_names = _event_labels_for_scan(state, scan, raw)
    if not event_names:
        ui.label(
            "No events in this scan. Upload an events file above to supply "
            "trial onsets, then pick which to estimate."
        ).classes("text-sm opacity-60")
    else:
        # Seed (or re-seed) the selection to all labels when it's empty or
        # entirely stale (e.g. after switching scans or swapping the event
        # source). A deliberate subset that still overlaps is left alone.
        if not (set(opts.selected_events) & set(event_names)):
            opts.selected_events = tuple(event_names)

        # Condition selection: which event labels to estimate. (File finding
        # / globbing lives in the Upload dialog now — this is just picking
        # conditions among the loaded events.)
        ui.label("Conditions to estimate").classes(
            "text-xs uppercase opacity-50 tracking-wide"
        )
        with ui.row().classes("items-center gap-2"):
            ui.button(
                "All",
                on_click=lambda: _set_events_selection(
                    state, opts, tuple(sorted(event_names))
                ),
            ).props("flat dense")
            ui.button(
                "None",
                on_click=lambda: _set_events_selection(state, opts, ()),
            ).props("flat dense")

        selected_set = set(opts.selected_events)
        counts = _event_counts_for_scan(state, scan, raw)

        def _toggle(name: str, checked: bool) -> None:
            new_selection = set(opts.selected_events)
            if checked:
                new_selection.add(name)
            else:
                new_selection.discard(name)
            opts.selected_events = tuple(sorted(new_selection))

        with ui.row().classes("gap-2 flex-wrap"):
            for name in event_names:
                count = counts.get(name, 0)
                # Show the occurrence count so a bare numeric marker reads as
                # an event ("1.0 (42)"), not a mystery checkbox. Selection
                # still keys off the raw label, not the displayed text.
                ui.checkbox(
                    f"{name}  ({count})" if count else name,
                    value=name in selected_set,
                    on_change=(lambda e, n=name: _toggle(n, bool(e.value))),
                )

    _render_toeplitz_global_params(opts)


def _render_toeplitz_global_params(opts: EstimationOptions) -> None:
    """Lambda / duration / timeout controls + edge advisory for toeplitz mode.

    These are all ``opts``-global (no per-scan dependency), so they render in
    both the single-scan path (inside ``_render_toeplitz_controls``) and the
    bulk path (no row selected, scans ticked) — letting a bulk run be tuned
    without first clicking a scan.
    """
    # ── Lambda slider (log scale)
    ui.label("Regularization (lambda)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )
    initial_log = int(round(np.log10(opts.lmbda))) if opts.lmbda > 0 else -3
    initial_log = max(LOG_LMBDA_MIN, min(LOG_LMBDA_MAX, initial_log))

    lmbda_display = ui.label(f"lambda = {opts.lmbda:.0e}").classes(
        "text-sm font-mono opacity-80"
    )

    def _on_lmbda_change(event) -> None:
        try:
            log_val = int(event.value)
        except (TypeError, ValueError):
            return  # ignore a None/NaN slider payload rather than crash
        opts.lmbda = float(10 ** log_val)
        lmbda_display.set_text(f"lambda = {opts.lmbda:.0e}")

    ui.slider(
        min=LOG_LMBDA_MIN,
        max=LOG_LMBDA_MAX,
        step=1,
        value=initial_log,
        on_change=_on_lmbda_change,
    )

    # ── Duration
    ui.label("Duration (seconds)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )
    ui.number(
        value=opts.duration,
        min=1.0,
        max=120.0,
        step=1.0,
        format="%.1f",
        on_change=lambda e: setattr(opts, "duration", float(e.value or DEFAULT_DURATION)),
    )

    # ── Per-channel solve timeout (forwarded to estimate_hrf). A channel
    # whose solve exceeds this is skipped, not the whole scan.
    ui.label("Per-channel timeout (seconds)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )

    def _on_timeout(event) -> None:
        try:
            opts.timeout = max(1.0, float(event.value))
        except (TypeError, ValueError):
            opts.timeout = DEFAULT_TIMEOUT

    ui.number(
        value=opts.timeout,
        min=1.0,
        step=5.0,
        format="%.0f",
        on_change=_on_timeout,
    ).tooltip(
        "Seconds to wait for one channel's solve before skipping it."
    )

    # ── Edge expansion (forwarded to estimate_hrf). Each onset is shifted
    # back by ``edge_expansion * duration`` s so the window captures the
    # pre-onset baseline; widen it if the estimated HRFs are noisy/unstable
    # at their edges (the QC check below flags that).
    ui.label("Edge expansion (fraction of duration)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )

    def _on_edge_expansion(event) -> None:
        try:
            opts.edge_expansion = max(0.0, min(1.0, float(event.value)))
        except (TypeError, ValueError):
            opts.edge_expansion = DEFAULT_EDGE_EXPANSION

    ui.number(
        value=opts.edge_expansion,
        min=0.0, max=1.0, step=0.05,
        format="%.2f",
        on_change=_on_edge_expansion,
    ).tooltip(
        "Pads each event's estimation window back by this fraction of the "
        "duration. Higher = more stable HRF edges, but events in the first "
        "edge_expansion × duration seconds are dropped."
    )

    # ── Edge-expansion advisory: estimate_hrf shifts every event onset back
    # by edge_expansion*duration seconds and silently drops events that would
    # fall before t=0, so events in the first ``edge_seconds`` are lost.
    edge_seconds = opts.edge_expansion * opts.duration
    ui.label(
        f"Note: events in the first ~{edge_seconds:.1f} s of the scan are "
        f"dropped by the toeplitz edge-expansion window "
        f"({opts.edge_expansion:.2f} × duration). Consider this when "
        "designing trial-onset timing."
    ).classes("text-xs opacity-60 italic")

    # ── Edge-noise QC sensitivity (drives the post-estimation warning).
    ui.label("Edge-noise QC sensitivity").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )

    def _on_qc_frac(event) -> None:
        try:
            opts.edge_std_frac = max(0.01, min(0.49, float(event.value)))
        except (TypeError, ValueError):
            opts.edge_std_frac = DEFAULT_EDGE_STD_FRAC

    def _on_qc_ratio(event) -> None:
        try:
            opts.edge_std_ratio = max(1.0, float(event.value))
        except (TypeError, ValueError):
            opts.edge_std_ratio = DEFAULT_EDGE_STD_RATIO

    with ui.row().classes("w-full items-center gap-3 no-wrap"):
        ui.number(
            "Edge window (frac)",
            value=opts.edge_std_frac,
            min=0.01, max=0.49, step=0.05,
            format="%.2f",
            on_change=_on_qc_frac,
        ).props("dense").classes("flex-1").tooltip(
            "Fraction of the HRF (each end) treated as 'edge' when comparing "
            "its local noise to the center."
        )
        ui.number(
            "Flag ratio (×)",
            value=opts.edge_std_ratio,
            min=1.0, max=10.0, step=0.5,
            format="%.1f",
            on_change=_on_qc_ratio,
        ).props("dense").classes("flex-1").tooltip(
            "Warn when a channel's edge std exceeds this multiple of its "
            "center std. Lower = stricter."
        )


def _render_canonical_note() -> None:
    ui.label(
        "Canonical mode renders the SPM-style double-gamma HRF (peak at "
        "~6 s, undershoot at ~16 s) — a fixed reference shape, not "
        "data-driven. Click Generate to display."
    ).classes("text-sm opacity-70")
    # Set expectations: canonical results are a reference shape only. They are
    # NOT cached as subject montages, so they do not build the project GROUP
    # montage that Neural Activity's "Estimated HRFs" source uses. Switch to
    # toeplitz mode to contribute a subject.
    ui.label(
        "Note: canonical HRFs are a reference shape only — they don't count "
        "as a subject toward the group montage. Use toeplitz mode to "
        "estimate data-driven HRFs that build the group."
    ).classes("text-xs opacity-60 italic")


def _render_run_row(
    state: AppState,
    scan: Optional[ScanEntry],
    checked: List[ScanEntry],
    opts: EstimationOptions,
) -> None:
    """Render the Run button row + progress / error display.

    PR #55a: when ``checked`` is non-empty the button label and dispatch
    switch into bulk mode (iterate every checked scan sequentially). The
    preflight checks for single-scan mode (processed_cache + selected
    events) don't gate the bulk button because each scan's gate is
    evaluated inside the bulk worker -- scans that fail their per-scan
    gate are skipped, not the whole run.
    """
    bulk_mode = bool(checked)
    if bulk_mode:
        verb = (
            "Estimate HRFs" if opts.model == MODEL_TOEPLITZ
            else "Generate canonical HRFs"
        )
        run_label = (
            f"{verb} for {len(checked)} scan"
            f"{'s' if len(checked) != 1 else ''}"
        )
        can_run = not state.busy
    elif opts.model == MODEL_TOEPLITZ:
        can_run = (
            scan is not None
            and scan in state.processed_cache
            and scan.path.resolve() in state.processed_deconvolved
            and bool(opts.selected_events)
            and not state.busy
        )
        run_label = "Estimate HRFs"
    else:
        can_run = scan is not None and not state.busy
        run_label = "Generate canonical HRF"

    with ui.row().classes("items-center gap-3"):
        # Async handler (passed directly, NOT wrapped in a sync lambda, which
        # would silently no-op) so the coverage pre-flight dialog can await.
        async def _on_estimate() -> None:
            await _estimate_clicked(state, scan, checked, opts)

        ui.button(
            run_label,
            on_click=_on_estimate,
        ).props(f"color=primary {'disable' if not can_run else ''}")
        # Save sits immediately to the right of Estimate, popping in only once
        # there's a real montage to save (see _render_save_button).
        _render_save_button(state, scan, bulk_mode)
        if state.busy:
            _render_busy_progress(state)
        elif (
            not bulk_mode
            and opts.model == MODEL_TOEPLITZ
            and scan is not None
            and scan not in state.processed_cache
        ):
            ui.label("Waiting for preprocess output…").classes(
                "text-sm opacity-60"
            )
        elif (
            not bulk_mode
            and opts.model == MODEL_TOEPLITZ
            and not opts.selected_events
        ):
            ui.label("Pick at least one event to estimate.").classes(
                "text-sm opacity-60"
            )

    if state.last_error and not state.busy:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error_outline").classes("text-red-400")
            ui.label(state.last_error).classes("text-sm text-red-400")


def _render_busy_progress(state: AppState) -> None:
    """Two-layer progress: bulk (scan i/N) + within-scan (channel i/N).

    The within-scan ``estimation_progress`` is reset between scans by
    the bulk worker so each scan's channel counter starts fresh.
    """
    bulk = state.bulk_progress
    if bulk is not None:
        idx, total, scan = bulk
        scan_label = scan.display_name or scan.path.name
        with ui.column().classes("gap-1 flex-grow"):
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="sm")
                ui.label(
                    f"Scan {idx + 1}/{total}: {scan_label}"
                ).classes("text-sm opacity-80")
            bulk_fraction = (idx + 1) / max(total, 1)
            ui.linear_progress(value=bulk_fraction).classes("w-64")
            prog = state.estimation_progress
            if prog is not None:
                current, total_ch, name = prog
                fraction = (current + 1) / max(total_ch, 1)
                ui.label(
                    f"  Channel {current + 1}/{total_ch}: {name}"
                ).classes("text-xs opacity-60")
                ui.linear_progress(value=fraction).classes("w-64")
            render_bulk_cancel_button(state)
        return

    prog = state.estimation_progress
    if prog is not None:
        current, total, name = prog
        fraction = (current + 1) / max(total, 1)
        with ui.column().classes("gap-1 flex-grow"):
            ui.label(
                f"Channel {current + 1}/{total}: {name}"
            ).classes("text-xs opacity-70")
            ui.linear_progress(value=fraction).classes("w-64")
    else:
        with ui.row().classes("items-center gap-2"):
            ui.spinner(size="sm")
            ui.label("Working…").classes("text-sm opacity-70")


def _run_dispatch(
    state: AppState,
    selected: Optional[ScanEntry],
    checked: List[ScanEntry],
    opts: EstimationOptions,
) -> None:
    """Route Estimate / Generate to single or bulk.

    PR #55a: when scans are checked in the dataset tree, iterate every
    one sequentially via the bulk worker. Otherwise fall back to the
    legacy single-scan path against ``selected``.
    """
    if checked:
        _run_bulk(state, checked, opts)
    elif selected is not None:
        _run(state, selected, opts)


def _run_bulk(
    state: AppState,
    scans: List[ScanEntry],
    opts: EstimationOptions,
) -> None:
    """Iterate the estimate / canonical call across each checked scan.

    Continue-on-error: a per-scan failure (e.g. no events matched,
    preprocess output missing, library exception) is logged to
    ``last_error``, counted in the failure list, and the run continues.
    The final toast summarises N successes / M failures.

    Toeplitz mode: each scan needs its own processed Raw in
    ``processed_cache`` and a non-empty event intersection with
    ``opts.selected_events``. Scans that fail either gate are skipped
    rather than crashing the whole run -- the per-scan call returns
    ``None`` from ``run_toeplitz_sync`` and the bulk worker treats that
    as success-with-empty-result (the on_each_done early-returns and
    nothing lands in state.montage for that scan).

    Canonical mode: every checked scan just needs a Raw (raw or
    processed cache) to size the output, which is loaded on demand
    inside the worker.
    """
    if state.busy:
        return

    snapshot = EstimationOptions(
        model=opts.model,
        lmbda=opts.lmbda,
        duration=opts.duration,
        selected_events=opts.selected_events,
        timeout=opts.timeout,
        edge_expansion=opts.edge_expansion,
        edge_std_frac=opts.edge_std_frac,
        edge_std_ratio=opts.edge_std_ratio,
    )

    def _build(scan: ScanEntry):
        if snapshot.model == MODEL_TOEPLITZ:
            progress_cb = make_progress_callback(state)

            def _pp_and_estimate(scan=scan, progress_cb=progress_cb):
                # Preprocess on demand (deconvolution) if this scan isn't
                # already cached+deconvolved, so the bulk run triggers the
                # preprocessing each scan needs instead of skipping it.
                # ensure_deconvolved_raw raises with a specific reason on
                # failure; let it propagate so the worker records why.
                from .preprocess_panel import ensure_deconvolved_raw
                raw = ensure_deconvolved_raw(state, scan)
                # Discover this scan's own events (manual file → collocated
                # sidecar → embedded annotations). The single global events
                # slot only covers one scan, so a batch must resolve each
                # scan independently or every other scan estimates on empty
                # events ("no usable event-locked responses").
                event_rows, event_impulse, selected = _resolve_bulk_events(
                    state, scan, raw, snapshot
                )
                if event_rows is None and event_impulse is None and not selected:
                    raise RuntimeError(
                        "no events found for this scan — no events file "
                        "collocated with it (BIDS sidecar or lone file) and "
                        "the scan has no embedded annotations; load/glob an "
                        "events file, or check that sidecars sit beside the "
                        "scan"
                    )
                scan_opts = EstimationOptions(
                    model=snapshot.model,
                    lmbda=snapshot.lmbda,
                    duration=snapshot.duration,
                    selected_events=selected,
                    timeout=snapshot.timeout,
                    edge_expansion=snapshot.edge_expansion,
                    edge_std_frac=snapshot.edge_std_frac,
                    edge_std_ratio=snapshot.edge_std_ratio,
                )
                return run_toeplitz_sync(
                    raw, scan_opts, progress_cb, event_rows, event_impulse
                )

            return (_pp_and_estimate, (), {})
        # Canonical: prefer processed_cache, fall back to raw_cache,
        # otherwise load on demand inside the worker thread.
        def _run_canonical():
            if scan in state.processed_cache:
                source_raw = state.processed_cache.get(scan)
            elif scan in state.raw_cache:
                source_raw = state.raw_cache.get(scan)
            else:
                source_raw = state.raw_cache.get(scan)
            return run_canonical_sync(source_raw, snapshot)
        return (_run_canonical, (), {})

    async def _on_each_done(scan: ScanEntry, result) -> None:
        if result is None:
            # Raise so an empty estimate is reported as a failure (with a
            # reason) instead of silently counting as a success.
            raise RuntimeError(
                "HRF estimation produced no result — no channels had usable "
                "event-locked responses (check events coverage and channels)"
            )
        state.montage = result
        state.montage_source_scan = scan
        # Cache per scan so toeplitz activity can use each scan's own HRFs
        # (montage only holds the most-recent estimate). Canonical results
        # aren't real per-channel montages, so they're never cached.
        if not isinstance(result, _CanonicalResult):
            state.montage_cache[scan.path.resolve()] = result
        state.publish("hrf_estimated", scan)

    async def _bulk() -> None:
        # Re-enter the captured client so refreshes work from this detached
        # task (no slot context otherwise).
        with client_scope(client):
            bulk_result = await run_bulk_in_background(
                state, scans, _build,
                on_each_done=_on_each_done,
                label="estimate_hrf",
            )
        if bulk_result is None:
            return
        successes, failures = bulk_result
        n_ok, n_fail = len(successes), len(failures)
        verb = (
            "Estimated HRFs for"
            if snapshot.model == MODEL_TOEPLITZ
            else "Generated canonical HRFs for"
        )
        summary = f"{verb} {n_ok}/{n_ok + n_fail} scan(s)."
        if failures:
            summary += f" Failed/skipped: {summarize_failures(failures)}"
        # Guarded: the page client may have been deleted during a long run.
        notify_if_alive(
            client, summary,
            type="positive" if n_fail == 0 else "warning",
            multi_line=True,
            close_button=True,
        )

    client = capture_client()
    background_tasks.create(_bulk())


def _run(
    state: AppState, scan: ScanEntry, opts: EstimationOptions
) -> None:
    """Click handler for Estimate / Generate (single-scan path).

    Snapshots options, dispatches the appropriate sync worker through
    ``workers.run_in_background``. On success, stashes the resulting
    Montage on ``state.montage`` and publishes ``hrf_estimated``.
    """
    if state.busy:
        return

    snapshot = EstimationOptions(
        model=opts.model,
        lmbda=opts.lmbda,
        duration=opts.duration,
        selected_events=opts.selected_events,
        timeout=opts.timeout,
        edge_expansion=opts.edge_expansion,
        edge_std_frac=opts.edge_std_frac,
        edge_std_ratio=opts.edge_std_ratio,
    )

    if snapshot.model == MODEL_TOEPLITZ:
        if scan not in state.processed_cache:
            state.last_error = "Preprocess the scan first."
            return
        if scan.path.resolve() not in state.processed_deconvolved:
            state.last_error = (
                "HRF estimation requires deconvolution-mode preprocessing — "
                "re-preprocess this scan with deconvolution."
            )
            return
        if not snapshot.selected_events:
            state.last_error = "Pick at least one event."
            return
        raw = state.processed_cache.get(scan)
        progress_cb = make_progress_callback(state)
        applies = _file_applies_to_scan(state, scan)
        event_rows = state.events_rows if applies else None
        event_impulse = state.events_impulse if applies else None
        sync_call = (
            run_toeplitz_sync, raw, snapshot, progress_cb, event_rows,
            event_impulse,
        )
    else:
        # Canonical mode doesn't need a processed Raw — only the raw_cache
        # entry to get sfreq/channel count for shaping the output.
        if scan not in state.raw_cache and scan not in state.processed_cache:
            state.last_error = "Load the scan first."
            return
        source_raw = (
            state.processed_cache.get(scan)
            if scan in state.processed_cache
            else state.raw_cache.get(scan)
        )
        sync_call = (
            run_canonical_sync, source_raw, snapshot
        )

    async def _on_done(result) -> None:
        if result is None:
            # Estimation yielded nothing. Surface feedback rather than leaving
            # the previous scan's montage on screen as if this one succeeded.
            # Preserve a worker-recorded exception message if present.
            if not state.last_error:
                state.last_error = (
                    "HRF estimation produced no output for this scan — check "
                    "the events selection and that the scan is "
                    "deconvolution-preprocessed."
                )
            return
        state.montage = result
        # Track which scan produced this montage so the Activity tab can
        # refuse a toeplitz run when the user switches scans mid-flow.
        state.montage_source_scan = scan
        # Cache per scan so toeplitz activity can use each scan's own HRFs.
        if not isinstance(result, _CanonicalResult):
            state.montage_cache[scan.path.resolve()] = result
        state.publish("hrf_estimated", scan)

    background_tasks.create(
        run_in_background(state, *sync_call, on_done=_on_done)
    )


def run_toeplitz_sync(
    raw: "mne.io.BaseRaw",
    opts: EstimationOptions,
    progress_callback=None,
    event_rows=None,
    event_impulse=None,
):
    """Run montage.estimate_hrf against a preprocessed Raw and return Montage.

    Events come from, in priority order: an uploaded per-sample impulse
    vector (``event_impulse``, applied by sample index), an uploaded onset
    table (``event_rows`` — ``events_io.EventRow`` list, converted to a 0/1
    impulse at this scan's sfreq), else the Raw's MNE annotations. Returns
    None if no event samples land inside the scan (nothing to estimate).
    ``estimate_hrf`` runs with ``preprocess=False`` (input already
    preprocessed). Module-level so tests can call it directly.
    """
    from ...hrfunc import montage as Montage

    n_samples = int(raw.n_times)
    if event_impulse is not None:
        events = build_impulse_from_vector(event_impulse, n_samples)
    elif event_rows is not None:
        sfreq = float(raw.info["sfreq"])
        events = build_impulse_from_rows(
            event_rows, opts.selected_events, sfreq, n_samples
        )
    else:
        events = build_events_array(raw, opts.selected_events)
    if events is None or not events.any():
        logger.warning(
            "run_toeplitz_sync: no event samples matched the selected "
            "descriptions; refusing to estimate on empty events."
        )
        return None

    m = Montage(nirx_obj=raw)
    m.estimate_hrf(
        raw,
        events=events.tolist(),
        duration=opts.duration,
        lmbda=opts.lmbda,
        edge_expansion=opts.edge_expansion,
        preprocess=False,
        progress_callback=progress_callback,
        timeout=opts.timeout,
    )
    # estimate_hrf only appends to optode.estimates; it does NOT populate
    # optode.trace (which is what the preview reads). generate_distribution
    # computes trace = mean(estimates) per channel. Without this call, the
    # HRF preview would render an empty plot after a successful estimation.
    m.generate_distribution()
    return m


def run_canonical_sync(
    raw: "mne.io.BaseRaw",
    opts: EstimationOptions,
):
    """Build a canonical SPM-style double-gamma HRF.

    Returns a lightweight object with a ``.canonical_trace`` numpy array
    and ``.duration`` / ``.sfreq`` fields. Not a real Montage — canonical
    mode doesn't go through estimate_hrf, so the per-channel structure
    isn't relevant. Sprint 3.4 (Activity) will not consume this; it has
    its own canonical path via estimate_activity(hrf_model='canonical').
    """
    sfreq = float(raw.info["sfreq"])
    trace = canonical_double_gamma(opts.duration, sfreq)
    return _CanonicalResult(
        canonical_trace=trace, duration=opts.duration, sfreq=sfreq
    )


@dataclass
class _CanonicalResult:
    """Holder for canonical-mode output.

    Deliberately not a Montage — canonical mode skips estimation entirely
    and returns a single reference HRF shape. Stored on ``state.montage``
    via duck-typing; the HRFs tab is the only consumer.
    """

    canonical_trace: np.ndarray
    duration: float
    sfreq: float


def canonical_double_gamma(duration: float, sfreq: float) -> np.ndarray:
    """SPM-style double-gamma canonical HRF.

    Standard SPM canonical: a gamma with peak at ~6 s minus a smaller
    gamma with peak at ~16 s, normalized so the positive peak is 1.0.

    The argument to ``gamma.pdf`` is in seconds, not sample indices.
    This deliberately differs from the library's ``correlate_canonical``
    (hrfunc.py:756-761) which passes raw sample indices — fine when the
    correlation is point-wise but produces an sfreq-dependent peak
    location for visualization. The GUI's canonical preview is meant to
    be the "true" SPM canonical, so it operates in seconds.
    """
    import scipy.stats

    n_samples = max(int(round(duration * sfreq)), 2)
    t_seconds = np.arange(n_samples) / sfreq
    peak1 = scipy.stats.gamma.pdf(t_seconds, 6)
    peak2 = scipy.stats.gamma.pdf(t_seconds, 16) / 6.0
    hrf = peak1 - peak2
    peak = np.max(hrf)
    if peak > 0:
        hrf = hrf / peak
    return hrf


def build_events_array(
    raw: "mne.io.BaseRaw",
    selected_descriptions: Tuple[str, ...],
) -> Optional[np.ndarray]:
    """Convert MNE annotations to a 0/1 impulse series of length n_samples.

    ``estimate_hrf`` consumes a flat list where each sample is 0 or 1; an
    event onset at time ``t`` becomes ``1`` at sample index
    ``round(t * sfreq)``. Annotations whose description is not in
    ``selected_descriptions`` are ignored. Out-of-range onsets are
    dropped silently with a logger.warning so a corrupt annotations table
    doesn't crash the estimation.

    Returns None if there are no annotations to convert at all (so the
    caller can distinguish "nothing selected" from "selection matched but
    fell outside the scan window").
    """
    annotations = raw.annotations
    if annotations is None or len(annotations) == 0:
        return None

    sfreq = float(raw.info["sfreq"])
    n_samples = raw.n_times
    selected_set = set(selected_descriptions)

    out = np.zeros(n_samples, dtype=np.int64)
    for ann in annotations:
        if str(ann["description"]) not in selected_set:
            continue
        sample = int(round(float(ann["onset"]) * sfreq))
        if 0 <= sample < n_samples:
            out[sample] = 1
        else:
            logger.warning(
                "build_events_array: dropping annotation at onset %.3fs "
                "(sample %d) — outside scan window 0..%d",
                float(ann["onset"]), sample, n_samples,
            )
    return out


def sorted_unique_annotation_descriptions(raw: "mne.io.BaseRaw") -> List[str]:
    """Distinct annotation description strings, sorted alphabetically.

    Empty list if the Raw has no annotations or all descriptions are empty.
    """
    annotations = raw.annotations
    if annotations is None or len(annotations) == 0:
        return []
    seen = sorted({str(ann["description"]) for ann in annotations})
    return [s for s in seen if s]


# ---------------------------------------------------------------------------
# Result preview rendering
# ---------------------------------------------------------------------------


def _render_hrf_preview(
    state: AppState, scan: ScanEntry, opts: EstimationOptions
) -> None:
    """Render the most-recent estimation result.

    Canonical mode: single SPM-style line plot.
    Toeplitz mode (Sprint 5.1): clickable channel grid of mini-plots,
    plus a per-channel detail panel below when the user picks a channel.
    """
    result = state.montage
    if result is None:
        return
    if isinstance(result, _CanonicalResult):
        png = _render_canonical_preview_png(result)
        if png is None:
            ui.label("Preview unavailable.").classes("text-sm opacity-60")
            return
        ui.image(png).classes("max-w-3xl")
        return

    # Toeplitz montages are scan-specific (the library matches by channel
    # name, not scan identity), so a montage estimated from scan A must not
    # be plotted under scan B's header. Selecting a new scan does NOT clear
    # state.montage, so guard the gallery the same way the Activity panel
    # guards its run — by comparing montage_source_scan to the selected scan.
    source = state.montage_source_scan
    if source is None or source.path != scan.path:
        source_name = (
            source.display_name or source.path.name
            if source is not None else "another scan"
        )
        ui.label(
            f"These HRFs were estimated from {source_name}. "
            "Re-run estimation on this scan to preview them here."
        ).classes("text-sm opacity-60")
        return

    _render_edge_qc(result, opts)
    _render_toeplitz_gallery(state, result)


def _edge_unstable_channels(
    montage,
    *,
    edge_frac: float = DEFAULT_EDGE_STD_FRAC,
    ratio: float = DEFAULT_EDGE_STD_RATIO,
) -> list:
    """Channel names whose HRF TRACE wobbles more at its edges than its center.

    Compares the local standard deviation of the estimated HRF ``trace`` over
    its outer ``edge_frac`` (each end) to the std of its center. A clean HRF
    has its response in the center (high local std) and a flat baseline at the
    edges (low local std), so ``edge_std > ratio × center_std`` flags edges
    that fluctuate more than the response itself — a sign the estimation
    window is too tight (raise ``edge_expansion``).

    NB: this reads ``node.trace`` (the estimate itself), NOT ``node.trace_std``
    — that field is the ACROSS-SUBJECT std and is all-zero for a single-scan
    estimate, so it can't drive a per-scan QC. Channels without a usable trace,
    too-short traces, or a flat center (std 0) are skipped. Pure + module-level
    so tests can call it directly.
    """
    import numpy as np

    flagged: list = []
    channels = getattr(montage, "channels", {})
    for ch_name, node in channels.items():
        if "global" in str(ch_name):
            continue
        trace = getattr(node, "trace", None)
        if trace is None:
            continue
        arr = np.asarray(trace, dtype=float)
        n = arr.size
        if n < 6 or not np.all(np.isfinite(arr)):
            continue
        k = max(1, int(round(n * edge_frac)))
        if n <= 2 * k:
            continue
        edge_std = float(np.std(np.concatenate([arr[:k], arr[-k:]])))
        center_std = float(np.std(arr[k:-k]))
        if center_std > 0 and edge_std > ratio * center_std:
            flagged.append(ch_name)
    return flagged


def _render_edge_qc(montage, opts: EstimationOptions) -> None:
    """Warn when estimated HRFs fluctuate more at their edges than their center.

    Surfaces an amber callout (with the offending channels) suggesting a larger
    edge_expansion when enough channels trip the configurable edge-noise ratio.
    """
    flagged = _edge_unstable_channels(
        montage,
        edge_frac=opts.edge_std_frac,
        ratio=opts.edge_std_ratio,
    )
    if not flagged:
        return
    n = len(flagged)
    preview = ", ".join(str(c) for c in flagged[:6])
    if n > 6:
        preview += f", +{n - 6} more"
    with ui.row().classes(
        "w-full items-start gap-2 p-3 rounded border "
        "border-amber-500/40 bg-amber-500/10"
    ):
        ui.icon("warning", size="1.25rem").classes("text-amber-400 shrink-0")
        with ui.column().classes("gap-0"):
            ui.label(
                f"{n} channel{'s' if n != 1 else ''} fluctuate more at the "
                f"edges than the center (edge std > {opts.edge_std_ratio:g}× "
                "center)."
            ).classes("text-sm font-medium text-amber-300")
            ui.label(
                f"This often means the estimation window is too tight — try "
                f"raising Edge expansion (currently {opts.edge_expansion:.2f}) "
                "and re-estimating."
            ).classes("text-xs opacity-70")
            ui.label(preview).classes("text-xs font-mono opacity-50")


def _render_toeplitz_gallery(state: AppState, montage) -> None:
    """Per-channel HRF results as an accordion of dropdowns.

    One expandable row per channel (the channel name as the header); the
    FIRST channel is open by default so the user immediately sees an
    estimated HRF. Each channel's plot (trace + ±1 std shading) renders
    lazily the first time its dropdown is opened, so a large montage doesn't
    pay for every plot up front.
    """
    channels = _gather_channel_traces(montage)
    if not channels:
        ui.label("No channel HRFs available.").classes("text-sm opacity-60")
        return

    ui.label(
        f"{len(channels)} channel HRF{'s' if len(channels) != 1 else ''} — "
        "open a channel to view its estimated response."
    ).classes("text-xs opacity-60")

    # Explain the absent ±std band so it's not read as a bug: trace_std is the
    # ACROSS-SUBJECT std (needs ≥2 pooled estimates), so a single-scan estimate
    # has none. Only shown when every channel's std is empty/zero.
    if _std_is_all_zero(montage):
        ui.label(
            "No ±std band shown — the band is across-subject variance, which "
            "needs ≥2 pooled estimates; a single-scan estimate has none. Pool "
            "scans or use the HRtree library to get a variability band."
        ).classes("text-xs opacity-50 italic")

    with ui.column().classes("w-full gap-1 max-w-2xl"):
        for index, (ch_name, node) in enumerate(channels.items()):
            _render_channel_dropdown(node, ch_name, open_default=(index == 0))


def _std_is_all_zero(montage) -> bool:
    """True when every channel's ``trace_std`` is empty / all-zero.

    That's the single-scan case: ``trace_std`` is the across-subject std
    (``np.std`` over the per-subject estimates), so one estimate yields zeros.
    Used to caption the preview rather than leave an invisible band unexplained.
    """
    import numpy as np

    for node in getattr(montage, "channels", {}).values():
        std = getattr(node, "trace_std", None)
        if std is None:
            continue
        arr = np.asarray(std, dtype=float)
        if arr.size and np.any(np.isfinite(arr) & (np.abs(arr) > 0)):
            return False
    return True


def _render_channel_dropdown(node, ch_name: str, open_default: bool) -> None:
    """One channel's collapsible HRF panel; plot rendered lazily on open."""
    expansion = ui.expansion(ch_name, value=open_default).classes(
        "w-full border border-slate-700 rounded"
    ).props("dense")
    with expansion:
        container = ui.column().classes("w-full p-2")

    done = {"rendered": False}

    def _fill() -> None:
        if done["rendered"]:
            return
        done["rendered"] = True
        with container:
            png = _render_detail_hrf_png(node, ch_name)
            if png is not None:
                ui.image(png).classes("w-full max-w-2xl")
            else:
                ui.label(
                    "Plot unavailable for this channel."
                ).classes("text-xs opacity-60")

    if open_default:
        _fill()
    # Render on first expand; collapsing doesn't tear it down (cheap to keep).
    expansion.on_value_change(lambda e: _fill() if e.value else None)


def _on_channel_click(state: AppState, ch_name: str) -> None:
    """Update the selected channel and publish the focused-refresh event.

    Uses ``hrf_selection_changed`` (HRFs-tab-only) rather than
    ``hrf_estimated`` (all 6 tab subscribers) so a click in the gallery
    doesn't cause every other panel to re-render. The payload is the
    new channel name so future subscribers can act on it without a
    state lookup.
    """
    state.hrf_selected_channel = ch_name
    state.publish("hrf_selection_changed", ch_name)


def _gather_channel_traces(montage):
    """Pull (ch_name → node) out of a Montage's channels, skipping empties.

    Module-level so tests can call it directly. Filters out channels whose
    ``trace`` attribute is missing, empty, or all-zeros — those would
    render as blank plots and confuse the gallery UX.
    """
    import numpy as np

    out = {}
    channels = getattr(montage, "channels", {})
    for ch_name, node in channels.items():
        trace = getattr(node, "trace", None)
        if trace is None:
            continue
        try:
            arr = np.asarray(trace)
            if arr.size == 0 or not np.any(np.abs(arr) > 0):
                continue
        except Exception:  # noqa: BLE001
            continue
        out[ch_name] = node
    return out


def _render_mini_hrf_png(node) -> Optional[str]:
    """Render a tiny PNG for one channel's HRF — used in the gallery grid."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable: %s", exc)
        return None

    trace = getattr(node, "trace", None)
    if trace is None or len(trace) == 0:
        return None

    fig = None
    try:
        fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.0))
        ax.plot(trace, lw=0.8, color="#6366f1")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.1)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("mini HRF render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def _render_detail_hrf_png(node, ch_name: str) -> Optional[str]:
    """Render a full-size HRF plot for the currently-selected channel.

    Shows the trace plus ±1 standard-deviation shading when ``trace_std``
    is available. Time axis in seconds (computed from the node's sfreq).
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable: %s", exc)
        return None

    trace = getattr(node, "trace", None)
    if trace is None or len(trace) == 0:
        return None
    sfreq = float(getattr(node, "sfreq", 1.0) or 1.0)
    if sfreq <= 0:
        sfreq = 1.0

    fig = None
    try:
        t = np.arange(len(trace)) / sfreq
        std = getattr(node, "trace_std", None)
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        ax.plot(t, trace, lw=1.4, color="#6366f1", label=ch_name)
        if std is not None and len(std) == len(trace):
            arr = np.asarray(trace)
            std_arr = np.asarray(std)
            ax.fill_between(
                t, arr - std_arr, arr + std_arr,
                alpha=0.18, color="#6366f1",
                label="±1 std",
            )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (a.u.)")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("detail HRF render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def _render_canonical_preview_png(result: "_CanonicalResult") -> Optional[str]:
    """Render the canonical HRF as a single line plot."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable: %s", exc)
        return None

    fig = None
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        t = np.arange(len(result.canonical_trace)) / result.sfreq
        ax.plot(t, result.canonical_trace, lw=1.5)
        ax.set_title("Canonical HRF (double-gamma)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (peak = 1.0)")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("canonical preview render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
