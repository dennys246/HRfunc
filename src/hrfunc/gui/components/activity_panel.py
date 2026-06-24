"""Activity tab content — deconvolve neural activity from preprocessed scans.

Sprint 3.4 wires the Activity tab to ``montage.estimate_activity``. The
library function takes a preprocessed Raw + a configured Montage and
returns a Raw with neural-activity values in place of haemoglobin values
(channel-wise Toeplitz deconvolution using either the user's estimated
HRFs from the HRFs tab or a canonical reference HRF).

Two modes:

- **Toeplitz** (default): uses the per-channel HRFs estimated in the HRFs
  tab (``state.montage``). Requires that the user has already run
  estimate_hrf successfully — the panel surfaces a "Estimate HRFs first"
  prompt otherwise. Cannot be selected if ``state.montage`` is a
  ``_CanonicalResult`` (from canonical-mode HRFs tab) — the toeplitz
  activity path needs real per-channel HRF traces.
- **Canonical**: each channel is deconvolved with the SPM-style canonical
  HRF from the bundled HRF database. Does not require prior estimate_hrf.
  Constructs a fresh Montage configured to the scan.

Lambda defaults to ``1e-4`` (library default for estimate_activity, which
is one decade smaller than estimate_hrf's ``1e-3``).

Result preview: matplotlib base64 PNG in the lens.plot_nirx style —
overlays the preprocessed (convolved haemoglobin) signal in red dashed
against the deconvolved (neural activity) signal in solid blue, with
event-marker vertical lines. Single-channel display, user picks via a
dropdown.

Cache protection: ``estimate_activity`` mutates the input Raw in place
(``apply_function`` + ``drop_channels``). To avoid corrupting
``state.processed_cache``, the panel passes a ``raw.copy()`` to
``estimate_activity`` and stores the result on ``state.activity_raw``.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from nicegui import background_tasks, ui

from ..state import AppState
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
from .hrf_panel import _CanonicalResult

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


MODEL_TOEPLITZ = "toeplitz"
MODEL_CANONICAL = "canonical"
MODEL_LIBRARY = "library"
DEFAULT_LMBDA = 1e-4
LOG_LMBDA_MIN = -6
LOG_LMBDA_MAX = -1
# Per-channel solve timeout (s); mirrors montage.estimate_activity's default.
DEFAULT_TIMEOUT = 30.0

# Human-readable names for the HRF sources estimate_activity supports.
# Drives the right-column "HRF source" selector and the left-column readout.
SOURCE_LABELS = {
    MODEL_TOEPLITZ: "Estimated HRFs (from the HRFs tab)",
    MODEL_CANONICAL: "Canonical reference HRF (library)",
    MODEL_LIBRARY: "HRtree HRF (selected in the HRtree tab)",
}


def _library_kernel_from_state(state: AppState):
    """The (trace, oxygenation) of the HRtree HRF selected for deconvolution.

    Reads ``state.library_selected_hrf`` (the HRF the user clicked in the
    HRtree tab). Returns ``(trace_list, oxygenation_bool)`` or ``None`` when no
    HRF is selected / it has no usable mean trace.
    """
    hrf = getattr(state, "library_selected_hrf", None)
    if not hrf:
        return None
    trace = hrf.get("hrf_mean") or []
    if not len(trace):
        return None
    return list(trace), bool(hrf.get("oxygenation"))


def _snapshot_options(state: AppState, opts: "ActivityOptions") -> "ActivityOptions":
    """Snapshot ActivityOptions for a background run, capturing the HRtree
    kernel when in library mode so the run sees a stable trace."""
    kernel = (
        _library_kernel_from_state(state)
        if opts.hrf_model == MODEL_LIBRARY else None
    )
    return ActivityOptions(
        hrf_model=opts.hrf_model,
        lmbda=opts.lmbda,
        preview_channel=opts.preview_channel,
        timeout=opts.timeout,
        drop_failed_channels=opts.drop_failed_channels,
        library_trace=kernel[0] if kernel else None,
        library_oxygenation=kernel[1] if kernel else True,
        # Per-channel matching settings travel with the snapshot; the actual
        # ``library_traces`` map is computed per scan at run time (it depends
        # on each scan's channel geometry), not captured here.
        library_per_channel=opts.library_per_channel,
        library_strategy=opts.library_strategy,
        library_radius_mm=opts.library_radius_mm,
        library_uncovered=opts.library_uncovered,
    )


@dataclass
class ActivityOptions:
    """User-controlled options for the Activity tab.

    Snapshotted at Run-click time so the background task sees a stable view.
    Defaults mirror ``montage.estimate_activity`` library defaults.
    """

    hrf_model: str = MODEL_TOEPLITZ
    lmbda: float = DEFAULT_LMBDA
    preview_channel: int = 0
    # Per-channel lstsq solve timeout (seconds) and whether a failed channel
    # is dropped (scan continues) vs aborting the whole scan. Forwarded to
    # montage.estimate_activity.
    timeout: float = DEFAULT_TIMEOUT
    drop_failed_channels: bool = True
    # Snapshot of the HRtree kernel for hrf_model == MODEL_LIBRARY, captured at
    # Run-click so a background run sees a stable trace even if the HRtree
    # selection changes mid-run. None in other modes.
    library_trace: Optional[list] = None
    library_oxygenation: bool = True
    # Per-channel HRtree mapping (hrf_model == MODEL_LIBRARY). When
    # ``library_per_channel`` is True and the user's ROIs yield candidates,
    # each channel is deconvolved with its OWN spatially-matched HRF instead
    # of one shared kernel. ``library_strategy`` picks individual-HRF vs
    # ROI-mean matching; ``library_radius_mm`` is the max match distance;
    # ``library_uncovered`` is 'skip' (drop unmatched channels) or 'canonical'.
    # ``library_traces`` is the computed {ch_name: trace} map, filled per scan
    # at run time (None until then; differs per scan so it is NOT snapshotted
    # at the panel level).
    library_per_channel: bool = True
    library_strategy: str = "individual"
    library_radius_mm: float = 20.0
    library_uncovered: str = "skip"
    library_traces: Optional[dict] = None


def _compute_library_traces(
    state: AppState, raw, opts: "ActivityOptions"
) -> Optional[dict]:
    """Per-channel HRtree ``{ch_name: trace}`` map for ONE scan's raw, or None.

    Returns None — telling the run path to fall back to the single shared
    kernel — when not in per-channel library mode, when the spatial match
    raises, or when no channel found a same-oxygenation HRF within range.
    Computed per scan because each scan's channel geometry differs.
    """
    if opts.hrf_model != MODEL_LIBRARY or not opts.library_per_channel:
        return None
    try:
        from .hrtree_match import match_channels_to_hrtree
        res = match_channels_to_hrtree(
            state, raw,
            strategy=opts.library_strategy,
            radius_mm=opts.library_radius_mm,
        )
    except Exception as exc:  # noqa: BLE001 — never break a run on a match error
        logger.warning("HRtree per-channel match failed: %s", exc)
        return None
    traces = res.library_traces()
    return traces or None


def render(state: AppState) -> None:
    """Render the Activity tab inside the current NiceGUI context.

    Subscribes a refreshable body to scan / preprocess / HRF / activity
    events. A ``ui.timer(0.5)`` polls ``state.busy`` and refreshes during
    long estimations to drive the progress bar (same pattern as Sprint 3.3).
    """
    opts = ActivityOptions()

    @ui.refreshable
    def _body() -> None:
        _render_body(state, opts)

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)
    state.subscribe("preprocess_done", _refresh)
    state.subscribe("hrf_estimated", _refresh)
    state.subscribe("activity_estimated", _refresh)
    # Recompute bulk mode when the dataset-tree checked set changes.
    state.subscribe("checked_changed", _refresh)
    # Update the "HRtree HRF" source status when the user picks an HRF in the
    # HRtree tab (so the kernel readout / Run gating refresh live).
    state.subscribe("hrtree_selection_changed", _refresh)

    def _poll_progress() -> None:
        if state.busy and state.estimation_progress is not None:
            _body.refresh()

    ui.timer(0.5, _poll_progress)


def _render_body(state: AppState, opts: ActivityOptions) -> None:
    """Render the Activity body against current state.

    Module-level so tests can call it directly inside a synthetic NiceGUI
    context without going through the refreshable wrapper.
    """
    scan = state.selected_scan
    checked_scans = _resolve_checked_scans(state)
    bulk_mode = len(checked_scans) >= 1

    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("Activity").classes("text-2xl font-semibold")

        if scan is None and not bulk_mode:
            ui.label(
                "Select a scan from the dataset tree, or tick scans "
                "for a bulk run."
            ).classes("text-sm opacity-60")
            return

        # Two-column workspace: deconvolution controls on the left, the HRF
        # source picker + preview in its own column on the right (mirrors the
        # HRFs / HRtree tabs). The right column makes it explicit WHICH HRFs
        # the deconvolution will use, and links to the HRFs tab when the
        # chosen source isn't ready yet.
        with ui.row().classes("w-full gap-6 items-start no-wrap"):
            with ui.column().classes("flex-1 min-w-0 gap-4"):
                if bulk_mode:
                    ui.label(
                        f"Bulk run on {len(checked_scans)} checked scan"
                        f"{'s' if len(checked_scans) != 1 else ''}."
                    ).classes("text-sm font-mono opacity-70")
                    # Bulk preprocess readiness + one-click "Preprocess all
                    # checked" (the deconvolution the bulk run needs). The
                    # bulk run preprocesses on demand too, so this is a
                    # readout + convenience, not a prerequisite.
                    from .preprocess_panel import render_preprocess_all_checked
                    render_preprocess_all_checked(state, checked_scans)
                elif scan is not None:
                    ui.label(scan.display_name or scan.path.name).classes(
                        "text-sm font-mono opacity-70"
                    )

                # Shared filename options for the Save action; created here so
                # the inputs (inside the Deconvolution card) and the Save
                # button (in the run row) read/write the same dict.
                naming = {"postfix": "_deconvolved", "ext": ".snirf"}

                # ── Parameter controls. HRF-source selection lives in the
                # right column; here we show the current source as a readout
                # plus the regularization control, then the save options as a
                # subsection at the bottom.
                with ui.card().classes("w-full"):
                    ui.label("Deconvolution").classes(
                        "text-xs uppercase opacity-60 tracking-wide"
                    )
                    ui.label(
                        f"HRF source: {SOURCE_LABELS[opts.hrf_model]}"
                    ).classes("text-sm opacity-70")
                    ui.label("Choose the source in the panel on the right.").classes(
                        "text-xs opacity-50"
                    )
                    _render_lmbda_slider(opts)
                    _render_deconv_controls(opts)
                    # Save-output subsection (filename postfix + format).
                    _render_save_options(state, scan, checked_scans, naming)

                # ── Run row: Estimate + Save on the same horizontal level,
                # then progress / errors.
                _render_run_row(state, scan, checked_scans, opts, naming)

                # ── Deconvolved preview (single-scan only -- bulk overwrites
                # activity_raw). Gate on activity_source_scan: activity_raw is
                # a single global slot NOT cleared when the selected scan
                # changes, so without this check the preview would overlay scan
                # A's deconvolution against scan B's preprocessed Raw (channel
                # names overlap across a shared montage, so it renders silently
                # rather than erroring).
                activity_matches = (
                    scan is not None
                    and state.activity_source_scan is not None
                    and state.activity_source_scan.path == scan.path
                )
                if (
                    state.activity_raw is not None
                    and activity_matches
                    and not bulk_mode
                ):
                    ui.separator()
                    ui.label("Deconvolved preview").classes(
                        "text-xs uppercase opacity-60 tracking-wide"
                    )
                    _render_preview(state, scan, opts)

            # ── Right column: HRF source picker + status + visualization
            with ui.column().classes("flex-1 min-w-0 gap-3"):
                _render_hrf_source_column(state, scan, opts, bulk_mode)


def _has_current_result(state: AppState, scan: Optional[ScanEntry]) -> bool:
    """True only when the in-memory deconvolution belongs to THIS scan.

    ``activity_raw`` is a single global slot that is NOT cleared on scan
    change, so gating Save on it alone lets the user write scan A's
    deconvolution under scan B's filename (with a dropped-channel count
    computed against B). Match the same ``activity_source_scan`` predicate the
    preview overlay already uses so Save only offers the current scan's result.
    """
    return (
        state.activity_raw is not None
        and scan is not None
        and state.activity_source_scan is not None
        and state.activity_source_scan.path == scan.path
    )


def _can_save(
    state: AppState,
    scan: Optional[ScanEntry],
    checked_scans: List[ScanEntry],
) -> bool:
    """Whether a Save action is available: an in-memory result for the current
    scan, or one or more checked scans to estimate-and-save in bulk."""
    has_current = _has_current_result(state, scan)
    can_mass = len(checked_scans) >= 1
    return has_current or can_mass


def _render_save_options(
    state: AppState,
    scan: Optional[ScanEntry],
    checked_scans: List[ScanEntry],
    naming: dict,
) -> None:
    """Filename postfix + format inputs for the Save action.

    Rendered as a subsection at the bottom of the Deconvolution card; the
    Save button itself lives in the run row beside Estimate. Shares ``naming``
    with that button so the chosen postfix / format reach the save call.
    """
    if not _can_save(state, scan, checked_scans):
        return
    ui.separator()
    ui.label("Save deconvolved output").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )
    with ui.row().classes("items-center gap-3 flex-wrap"):
        ui.input(
            label="Filename postfix",
            value=naming["postfix"],
            on_change=lambda e: naming.__setitem__("postfix", e.value or ""),
        ).props("dense outlined").classes("w-44")
        ui.select(
            {".snirf": "SNIRF (.snirf)", ".fif": "FIF (.fif)"},
            value=naming["ext"],
            label="Format",
            on_change=lambda e: naming.__setitem__("ext", e.value or ".snirf"),
        ).props("dense outlined").classes("w-40")


def _render_save_button(
    state: AppState,
    scan: Optional[ScanEntry],
    checked_scans: List[ScanEntry],
    naming: dict,
) -> None:
    """The Save button (rendered beside Estimate in the run row).

    Save is INDEPENDENT of Estimate — it writes results that already exist, it
    never re-estimates. With scans checked it's a batch: save every checked
    scan that has been estimated (``state.activity_cache``) to a folder;
    otherwise it saves the single in-memory deconvolved result for the current
    scan. When checked scans exist but none have been estimated yet, the button
    is disabled with a hint. Filename postfix / format come from ``naming``.
    """
    has_current = _has_current_result(state, scan)
    bulk_mode = len(checked_scans) >= 1

    if bulk_mode:
        cached = [s for s in checked_scans if s in state.activity_cache]
        if not cached:
            ui.button("Save", icon="save").props(
                "outline color=primary disable"
            ).tooltip(
                "Estimate the checked scans first — Save does not re-estimate."
            )
            return

        async def _save() -> None:
            await _mass_save_activity(state, cached, naming)

        n = len(cached)
        ui.button(
            "Save", icon="save", on_click=_save,
        ).props(
            f"outline color=primary {'disable' if state.busy else ''}"
        ).tooltip(
            f"Saves {n} already-estimated scan"
            f"{'s' if n != 1 else ''} to a folder (no re-estimation)."
        )
        return

    if has_current:
        async def _save() -> None:
            from .export_panel import _save_activity
            await _save_activity(state, scan, naming)

        ui.button(
            "Save", icon="save", on_click=_save,
        ).props("outline color=primary").tooltip(
            "Saves the deconvolved current scan."
        )


def _resolve_checked_scans(state: AppState) -> List[ScanEntry]:
    """Resolve ``state.checked_scan_paths`` to ScanEntries in manifest order.

    PR #55a helper. Same shape as the preprocess/hrf panel helpers --
    duplicated intentionally to keep panel modules self-contained.
    """
    if state.manifest is None or not state.checked_scan_paths:
        return []
    return [
        scan for scan in state.manifest.scans
        if scan.path.resolve() in state.checked_scan_paths
    ]


def _render_hrf_source_column(
    state: AppState,
    scan: Optional[ScanEntry],
    opts: ActivityOptions,
    bulk_mode: bool,
) -> None:
    """Right column: pick the HRF source, show its readiness, preview it.

    ``estimate_activity`` supports three sources:

    - **Estimated HRFs** (toeplitz): the per-channel HRFs from the HRFs tab
      (``state.montage``). If none are in memory / they're for another scan,
      the column says so and offers a "Go to HRFs tab" jump.
    - **HRtree HRF** (library): a single HRF selected in the HRtree tab, used
      as the kernel for every channel. Offers a "Go to HRtree tab" jump when
      nothing is selected.
    - **Canonical reference** (canonical): a fixed SPM double-gamma from the
      bundled library — always available, no estimation required.

    The selector sits at the top; the body below reflects the choice so the
    user can always see which HRFs the deconvolution will actually use.
    """
    ui.label("HRF source").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )

    def _set_source(value: str) -> None:
        opts.hrf_model = value
        # Re-render the body so the readout (left) and status (right) track
        # the new source.
        state.publish("scan_selected", state.selected_scan)

    ui.radio(
        SOURCE_LABELS,
        value=opts.hrf_model,
        on_change=lambda e: _set_source(e.value),
    ).props("dense").classes("w-full")

    ui.separator()

    if opts.hrf_model == MODEL_TOEPLITZ:
        _render_estimated_source_status(state, scan, bulk_mode)
    elif opts.hrf_model == MODEL_LIBRARY:
        _render_library_source_status(state, scan, opts, bulk_mode)
    else:
        _render_canonical_source_status(state, scan)


def _go_to_hrfs_button(state: AppState) -> None:
    """A "Go to HRFs tab" jump button (publishes ``navigate_estimate``)."""
    with ui.row().classes("items-center gap-2"):
        ui.icon("arrow_forward").classes("text-primary")
        ui.button(
            "Go to HRFs tab to estimate",
            icon="show_chart",
            on_click=lambda: state.publish("navigate_estimate"),
        ).props("flat dense color=primary")


def _go_to_hrtree_button(state: AppState) -> None:
    """A "Go to HRtree tab" jump button (publishes ``navigate_hrtree``)."""
    with ui.row().classes("items-center gap-2"):
        ui.icon("arrow_forward").classes("text-primary")
        ui.button(
            "Go to HRtree tab to select",
            icon="account_tree",
            on_click=lambda: state.publish("navigate_hrtree"),
        ).props("flat dense color=primary")


def _visible_roi_count(state: AppState) -> int:
    """Number of visible ROIs the user has built in the HRtree (0 on error)."""
    try:
        from .hrtree_panel import _visible_shapes
        return len(_visible_shapes(state))
    except Exception:  # noqa: BLE001
        return 0


def _render_library_source_status(
    state: AppState,
    scan: Optional[ScanEntry],
    opts: "ActivityOptions",
    bulk_mode: bool,
) -> None:
    """Status + preview for the "HRtree HRF" (library) source.

    Two sub-modes:
    - **Per-channel** (default when the user has built ROIs): each scan channel
      is matched to its own HRtree HRF; shows coverage counts + a per-channel
      assignment column.
    - **Single HRF**: one selected HRF applied to every channel (the original
      behaviour), used when there are no ROIs.
    """
    n_rois = _visible_roi_count(state)
    has_single = _library_kernel_from_state(state) is not None

    # Nothing to work with at all -> point the user at the HRtree tab.
    if n_rois == 0 and not has_single:
        ui.label("No HRtree HRFs selected yet.").classes("text-sm opacity-70")
        ui.label(
            "Open the HRtree tab and build ROIs from a montage (per-channel "
            "matching), or click a single HRF to use as one shared kernel."
        ).classes("text-xs opacity-60")
        _go_to_hrtree_button(state)
        return

    # Mode toggle (only meaningful once ROIs exist).
    if n_rois > 0:
        def _set_mode(value) -> None:
            opts.library_per_channel = bool(value)
            state.publish("scan_selected", state.selected_scan)

        ui.radio(
            {True: "Per-channel (from ROIs)", False: "Single HRF"},
            value=opts.library_per_channel,
            on_change=lambda e: _set_mode(e.value),
        ).props("inline dense").classes("text-sm")

    if opts.library_per_channel and n_rois > 0:
        _render_library_per_channel_status(state, scan, opts, bulk_mode, n_rois)
    else:
        _render_library_single_status(state)


def _render_library_per_channel_status(
    state: AppState,
    scan: Optional[ScanEntry],
    opts: "ActivityOptions",
    bulk_mode: bool,
    n_rois: int,
) -> None:
    """Per-channel HRtree matching: controls + coverage counts + assignment."""
    from .hrtree_match import (
        STRATEGY_INDIVIDUAL,
        STRATEGY_ROI_MEAN,
        match_channels_to_hrtree,
    )

    def _rerender() -> None:
        state.publish("scan_selected", state.selected_scan)

    # ── Matching strategy (the user picks between the two approaches).
    def _set_strategy(value) -> None:
        opts.library_strategy = value
        _rerender()

    ui.radio(
        {STRATEGY_INDIVIDUAL: "Nearest HRF", STRATEGY_ROI_MEAN: "ROI mean"},
        value=opts.library_strategy,
        on_change=lambda e: _set_strategy(e.value),
    ).props("inline dense").classes("text-xs")

    # ── Match radius (mm).
    def _set_radius(value) -> None:
        try:
            opts.library_radius_mm = max(1.0, float(value))
        except (TypeError, ValueError):
            opts.library_radius_mm = 20.0
        _rerender()

    ui.number(
        "Match radius (mm)",
        value=opts.library_radius_mm,
        min=1.0, max=200.0, step=1.0, format="%.0f",
        on_change=lambda e: _set_radius(e.value),
    ).props("dense").classes("w-40")
    # Scientific framing: a channel should match an HRF from the SAME
    # functional region. ~10-25 mm keeps matches local; a wide radius pools
    # HRFs across distinct cortical regions (and across the head->MNI
    # coord-frame offset), blurring region-specific responses.
    if opts.library_radius_mm > 30.0:
        ui.label(
            f"⚠ {opts.library_radius_mm:.0f} mm is wide — it may match HRFs "
            "from functionally distinct regions. ~10–25 mm is typical."
        ).classes("text-xs text-amber-700")
    else:
        ui.label(
            "Typical: ~10–25 mm (keeps matches within a functional region)."
        ).classes("text-xs opacity-50")

    # ── Uncovered-channel handling (lives here in the right column, per the
    # request). Default 'skip' fails loudly; user can switch to canonical.
    def _set_uncovered(value) -> None:
        opts.library_uncovered = value
        _rerender()

    ui.radio(
        {"skip": "Skip uncovered", "canonical": "Canonical fallback"},
        value=opts.library_uncovered,
        on_change=lambda e: _set_uncovered(e.value),
    ).props("inline dense").classes("text-xs")
    _go_to_hrtree_button(state)

    # ── Coverage needs a single, preprocessed scan to match against. In bulk
    # mode each scan matches its own geometry at run time, so just summarise.
    preview_scan = scan if not bulk_mode else None
    if preview_scan is None or preview_scan not in state.processed_cache:
        ui.label(
            f"{n_rois} ROI{'s' if n_rois != 1 else ''} selected. Coverage is "
            "computed per scan at run time — select a single preprocessed scan "
            "to preview which channels match."
        ).classes("text-sm opacity-70")
        return

    raw = state.processed_cache.get(preview_scan)
    try:
        res = match_channels_to_hrtree(
            state, raw,
            strategy=opts.library_strategy,
            radius_mm=opts.library_radius_mm,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HRtree coverage preview failed: %s", exc)
        ui.label("Coverage preview unavailable.").classes("text-sm opacity-60")
        return

    n_cov = len(res.covered)
    n_unc = len(res.uncovered)
    total = len(res.matches)
    # ── Headline counts at the top.
    ui.label(
        f"{n_cov}/{total} channels covered · {n_unc} uncovered · "
        f"{res.n_candidate_hrfs} HRF{'s' if res.n_candidate_hrfs != 1 else ''} "
        f"across {res.n_rois} ROI{'s' if res.n_rois != 1 else ''}"
    ).classes("text-sm font-medium")
    if n_unc:
        if opts.library_uncovered == "skip":
            ui.label(
                f"{n_unc} uncovered channel{'s' if n_unc != 1 else ''} will be "
                "DROPPED from the output (switch to canonical to keep them)."
            ).classes("text-xs text-amber-400")
        else:
            ui.label(
                f"{n_unc} uncovered channel{'s' if n_unc != 1 else ''} will use "
                "the canonical HRF."
            ).classes("text-xs opacity-60")

    _render_match_gallery(res)


def _render_match_gallery(res) -> None:
    """Per-channel assignment column: each channel + its matched HRF or
    'uncovered' (mirrors the per-channel HRF columns on the other tabs)."""
    from types import SimpleNamespace

    ui.label("Per-channel HRF assignment").classes(
        "text-xs uppercase opacity-60 tracking-wide mt-2"
    )
    with ui.column().classes(
        "w-full gap-1 max-h-96 overflow-auto border border-slate-700 "
        "rounded p-2"
    ):
        for m in res.matches:
            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                if m.matched:
                    try:
                        from .hrf_panel import _render_mini_hrf_png
                        png = _render_mini_hrf_png(
                            SimpleNamespace(trace=m.trace)
                        )
                    except Exception:  # noqa: BLE001
                        png = None
                    if png is not None:
                        ui.image(png).style("width:3rem;height:2rem;")
                    ui.label(m.ch_name).classes("text-xs font-mono")
                    dist = f"  ({m.distance_mm:.0f} mm)" if m.distance_mm is not None else ""
                    ui.label(f"→ {m.source}{dist}").classes(
                        "text-xs opacity-60 truncate"
                    )
                else:
                    ui.icon("block").classes("text-amber-500")
                    ui.label(m.ch_name).classes("text-xs font-mono opacity-70")
                    ui.label("uncovered").classes("text-xs text-amber-500")


def _render_library_single_status(state: AppState) -> None:
    """Single-kernel HRtree source: one selected HRF for every channel."""
    hrf = getattr(state, "library_selected_hrf", None)
    kernel = _library_kernel_from_state(state)

    if kernel is None:
        ui.label(
            "No HRtree HRF selected yet."
        ).classes("text-sm opacity-70")
        ui.label(
            "Open the HRtree tab and click an HRF in the 3D view; it will be "
            "used as the deconvolution kernel for every channel."
        ).classes("text-xs opacity-60")
        _go_to_hrtree_button(state)
        return

    trace, oxy = kernel
    key = hrf.get("_key", "") if isinstance(hrf, dict) else ""
    ui.label(
        f"Using the HRtree HRF: {key or '(selected HRF)'}"
    ).classes("text-sm opacity-70")
    ui.label(
        f"{'HbO' if oxy else 'HbR'} source · applied to every channel "
        "(sign-flipped for the opposite oxygenation)."
    ).classes("text-xs opacity-60")
    _go_to_hrtree_button(state)

    # Preview the selected kernel shape (reuse the canonical PNG renderer —
    # it just plots a trace at a given sfreq/duration).
    try:
        from .hrf_panel import (
            _CanonicalResult as _CR,
            _render_canonical_preview_png,
        )
        sfreq = float(hrf.get("sfreq", 0)) if isinstance(hrf, dict) else 0.0
        if sfreq <= 0:
            sfreq = 7.81
        arr = np.asarray(trace, dtype=float)
        duration = len(arr) / sfreq if sfreq else 30.0
        png = _render_canonical_preview_png(
            _CR(canonical_trace=arr, duration=duration, sfreq=sfreq)
        )
        if png is not None:
            ui.image(png).classes("max-w-md")
            ui.label(
                "Selected HRtree HRF (mean trace)."
            ).classes("text-xs opacity-50")
    except Exception as exc:  # noqa: BLE001
        logger.warning("HRtree kernel preview failed: %s", exc)


def _safe_render_gallery(state: AppState, montage) -> None:
    """Render the per-channel HRF accordion; never blank the panel on error."""
    try:
        from .hrf_panel import _render_toeplitz_gallery
        _render_toeplitz_gallery(state, montage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("activity HRF-source gallery render failed: %s", exc)
        ui.label("HRF preview unavailable.").classes("text-sm opacity-60")


def _render_estimated_source_status(
    state: AppState, scan: Optional[ScanEntry], bulk_mode: bool
) -> None:
    """Status + preview for the "Estimated HRFs" (toeplitz) source.

    Deconvolution with estimated HRFs uses the GROUP montage — the >=2-subject
    pool built on the HRFs tab — matched by channel name. Deconvolving a scan
    with its OWN single-subject HRFs is intentionally NOT offered (not
    validated/recommended in the paper). With fewer than two estimated
    subjects there's no group, so we point the user at the HRtree library or
    canonical source instead.
    """
    montage = _montage_for_scan(state, scan)  # group (>=2 subjects) or None
    if montage is not None:
        n_group = _group_subject_count(state)
        ui.label(
            f"Deconvolving every scan with the GROUP HRFs ({n_group} subjects), "
            "matched by channel name."
        ).classes("text-sm opacity-70")
        _safe_render_gallery(state, montage)
        return

    # Not enough subjects for a validated group HRF.
    ui.label(
        "Estimated-HRF deconvolution uses a GROUP HRF — the average across "
        "≥2 subjects' estimates. Deconvolving a scan with its own HRFs isn't "
        "offered (not validated in the paper)."
    ).classes("text-sm opacity-70")
    n_cached = _group_subject_count(state)
    ui.label(
        f"{n_cached} scan{'s' if n_cached != 1 else ''} estimated so far — "
        "estimate HRFs for at least 2 on the HRFs tab, or switch the source "
        "above to the HRtree library or canonical."
    ).classes("text-xs opacity-60")
    _go_to_hrfs_button(state)


def _render_canonical_source_status(
    state: AppState, scan: Optional[ScanEntry]
) -> None:
    """Status + illustrative curve for the canonical-reference source."""
    ui.label(
        "Using the canonical reference HRF — an SPM-style double-gamma shape "
        "from the bundled library. No estimation needed; works on every scan."
    ).classes("text-sm opacity-70")

    # Nudge toward the validated, subject-specific group HRFs when they exist:
    # a generic fixed shape is a downgrade from the project's own estimates.
    if _group_subject_count(state) >= 2:
        with ui.row().classes("items-center gap-2 q-mt-xs"):
            ui.icon("lightbulb", size="sm").classes("text-amber-600")
            ui.label(
                f"You have estimated GROUP HRFs from "
                f"{_group_subject_count(state)} subjects — those are "
                "subject-specific and preferred over this generic shape. "
                "Switch the source to \"Estimated HRFs\" to use them."
            ).classes("text-xs text-amber-800")

    try:
        from .hrf_panel import (
            _CanonicalResult as _CR,
            _render_canonical_preview_png,
            canonical_double_gamma,
        )
        sfreq = _representative_sfreq(state, scan)
        trace = canonical_double_gamma(30.0, sfreq)
        png = _render_canonical_preview_png(
            _CR(canonical_trace=trace, duration=30.0, sfreq=sfreq)
        )
        if png is not None:
            ui.image(png).classes("max-w-md")
            ui.label(
                "Representative shape (30 s window)."
            ).classes("text-xs opacity-50")
    except Exception as exc:  # noqa: BLE001
        logger.warning("canonical source preview failed: %s", exc)


def _representative_sfreq(
    state: AppState, scan: Optional[ScanEntry]
) -> float:
    """Best-effort sampling rate for the illustrative canonical curve.

    Prefer the scan's processed / raw sfreq when already cached (no load
    forced); fall back to a sensible fNIRS default otherwise.
    """
    if scan is not None:
        for cache in (state.processed_cache, state.raw_cache):
            try:
                if scan in cache:
                    return float(cache.get(scan).info["sfreq"])
            except Exception:  # noqa: BLE001
                continue
    return 7.81  # typical NIRx sampling rate; illustrative only


def _render_lmbda_slider(opts: ActivityOptions) -> None:
    ui.label("Regularization (lambda)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )
    initial_log = int(round(np.log10(opts.lmbda))) if opts.lmbda > 0 else -4
    initial_log = max(LOG_LMBDA_MIN, min(LOG_LMBDA_MAX, initial_log))

    lmbda_display = ui.label(f"lambda = {opts.lmbda:.0e}").classes(
        "text-sm font-mono opacity-80"
    )

    def _on_change(event) -> None:
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
        on_change=_on_change,
    )


def _render_deconv_controls(opts: ActivityOptions) -> None:
    """Per-channel timeout + drop-failed-channels controls.

    Both forward to montage.estimate_activity. The timeout bounds each
    channel's lstsq solve; "Drop failed channels" keeps the scan with its
    surviving channels when a channel errors (the default) rather than
    failing the whole scan.
    """
    with ui.row().classes("items-center gap-3 flex-wrap"):
        def _on_timeout(event) -> None:
            try:
                opts.timeout = max(1.0, float(event.value))
            except (TypeError, ValueError):
                opts.timeout = DEFAULT_TIMEOUT

        ui.number(
            label="Per-channel timeout (s)",
            value=opts.timeout,
            min=1.0,
            step=5.0,
            format="%.0f",
            on_change=_on_timeout,
        ).props("dense outlined").classes("w-48").tooltip(
            "Seconds to wait for one channel's solve before dropping it."
        )
        ui.checkbox(
            "Drop failed channels",
            value=opts.drop_failed_channels,
            on_change=lambda e: setattr(
                opts, "drop_failed_channels", bool(e.value)
            ),
        ).props("dense").tooltip(
            "On: a channel that fails is dropped and the scan keeps its good "
            "channels. Off: any channel failure fails the whole scan."
        )


def _render_run_row(
    state: AppState,
    scan: Optional[ScanEntry],
    checked: List[ScanEntry],
    opts: ActivityOptions,
    naming: dict,
) -> None:
    """Render the Estimate + Save buttons (same row) + progress / errors.

    PR #55a: dispatches single vs bulk based on the checked set. Bulk
    button is enabled whenever there's no in-flight task; per-scan gates
    (raw + montage match) are evaluated inside the worker and incompatible
    scans are skipped. The Save button sits on the same horizontal level
    (filename options come from the Deconvolution card via ``naming``).
    """
    bulk_mode = bool(checked)
    bulk_block_reason: Optional[str] = None
    if bulk_mode:
        run_label = (
            f"Estimate activity for {len(checked)} scan"
            f"{'s' if len(checked) != 1 else ''}"
        )
        can_run = not state.busy
        # Bulk toeplitz (estimated-HRF) deconvolution needs the project GROUP
        # montage. With fewer than 2 estimated subjects, _montage_for_scan
        # returns None for every scan and the whole batch would be skipped --
        # block the run with a clear reason instead of failing N scans.
        if opts.hrf_model == MODEL_TOEPLITZ and _group_subject_count(state) < 2:
            can_run = False
            bulk_block_reason = (
                "Estimated-HRF deconvolution needs GROUP HRFs from ≥2 "
                "subjects. Estimate HRFs for more scans on the HRFs tab, or "
                "switch the source to Canonical / HRtree."
            )
    else:
        # Activity deconvolution must run on deconvolution-preprocessed data
        # (not the GLM/haemoglobin pipeline), so require the scan to be in
        # processed_deconvolved as well as the cache.
        raw_ready = (
            scan is not None
            and scan in state.processed_cache
            and scan.path.resolve() in state.processed_deconvolved
        )
        if opts.hrf_model == MODEL_TOEPLITZ:
            # This scan needs its own estimated HRFs (per-scan cache).
            can_run = (
                raw_ready
                and _montage_for_scan(state, scan) is not None
                and not state.busy
            )
        elif opts.hrf_model == MODEL_LIBRARY:
            # Library mode needs an HRtree HRF selected as the kernel.
            can_run = (
                raw_ready
                and _library_kernel_from_state(state) is not None
                and not state.busy
            )
        else:
            can_run = raw_ready and not state.busy
        run_label = "Estimate activity"

    with ui.row().classes("items-center gap-3"):
        ui.button(
            run_label,
            on_click=lambda: _run_dispatch(state, scan, checked, opts),
        ).props(f"color=primary {'disable' if not can_run else ''}")

        # Save sits on the same horizontal level as Estimate.
        _render_save_button(state, scan, checked, naming)

        if bulk_block_reason and not state.busy:
            with ui.row().classes("items-center gap-2"):
                ui.icon("info").classes("text-amber-500")
                ui.label(bulk_block_reason).classes("text-sm opacity-70")

        if state.busy:
            _render_busy_progress(state)
        elif not bulk_mode and scan is not None and scan not in state.processed_cache:
            # Not preprocessed yet — link straight to the Preprocess tab via
            # the event bus (no reference to the shell's tab control needed).
            with ui.row().classes("items-center gap-2"):
                ui.icon("info").classes("text-amber-500")
                ui.label(
                    "This scan hasn't been preprocessed yet — activity is "
                    "estimated from the preprocessed signal."
                ).classes("text-sm opacity-70")
                ui.button(
                    "Go to Preprocess",
                    icon="arrow_forward",
                    on_click=lambda: state.publish("navigate_preprocess"),
                ).props("flat dense color=primary")
        elif (
            not bulk_mode and scan is not None
            and scan.path.resolve() not in state.processed_deconvolved
        ):
            # Preprocessed, but GLM/haemoglobin mode — activity needs the
            # deconvolution pipeline.
            ui.label(
                "This scan was preprocessed for haemoglobin (GLM). Neural-"
                "activity estimation needs the deconvolution pipeline — re-"
                "preprocess it in deconvolution mode (Preprocess tab)."
            ).classes("text-sm text-amber-400")

    if state.last_error and not state.busy:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error_outline").classes("text-red-400")
            ui.label(state.last_error).classes("text-sm text-red-400")


def _render_busy_progress(state: AppState) -> None:
    """Two-layer progress: bulk (scan i/N) + within-scan (channel i/N)."""
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
            ui.linear_progress(
                value=(idx + 1) / max(total, 1)
            ).classes("w-64")
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
    opts: ActivityOptions,
) -> None:
    """Route Estimate activity to single or bulk based on the checked set."""
    if checked:
        _run_bulk(state, checked, opts)
    elif selected is not None:
        _run(state, selected, opts)


def _group_subject_count(state: AppState) -> int:
    """Number of scans actually pooled into the group montage.

    Must mirror ``hrf_panel._sourced_project_montages``: count non-canonical
    cache entries EXCLUDING any the user removed from the group
    (``project_group_excluded``). Counting raw ``montage_cache`` would overstate
    the subject count after a removal and let the >=2 gate pass on a pool that
    actually holds a single subject (whose between-subject std is 0).
    """
    return sum(
        1 for path, m in state.montage_cache.items()
        if m is not None and not isinstance(m, _CanonicalResult)
        and path not in state.project_group_excluded
    )


def _montage_for_scan(state: AppState, scan: Optional[ScanEntry]):
    """The per-channel Montage for toeplitz ("Estimated HRFs") activity.

    Deconvolution with estimated HRFs uses the GROUP montage — the pooled,
    >=2-subject project montage (matched by channel name in estimate_activity).
    Deconvolving a scan with its OWN single-subject HRFs is intentionally NOT
    offered: it wasn't validated/recommended in the paper. With fewer than two
    estimated subjects there is no group, so this returns None and the user is
    pointed at the HRtree library or canonical sources instead.

    ``scan`` is accepted for call-site compatibility but unused — the group
    HRF applies to every scan by channel name.
    """
    group = state.project_montage
    if (
        group is not None
        and not isinstance(group, _CanonicalResult)
        and _group_subject_count(state) >= 2
    ):
        return group
    return None


def _toeplitz_skip_reason(state: AppState, scan: ScanEntry) -> Optional[str]:
    """Why this scan can't be deconvolved in toeplitz mode, or None if it can.

    Toeplitz activity needs this scan's own estimated per-channel HRFs. Each
    scan's montage is cached when HRFs are estimated (single or bulk), so a
    scan qualifies once it's been estimated. Returns a user-facing reason for
    the bulk worker to log when it hasn't.
    """
    if _montage_for_scan(state, scan) is not None:
        return None
    # Nothing estimated anywhere vs. estimated-but-not-this-scan: tailor the hint.
    any_estimated = bool(state.montage_cache) or (
        state.montage is not None
        and not isinstance(state.montage, _CanonicalResult)
    )
    if not any_estimated:
        return (
            "toeplitz mode needs per-channel HRFs, but none have been "
            "estimated — estimate HRFs on the HRFs tab first, or switch this "
            "run to canonical mode (works on every scan)"
        )
    return (
        "no estimated HRFs for this scan — estimate HRFs for it on the HRFs "
        "tab, or use canonical mode"
    )


def _run_bulk(
    state: AppState,
    scans: List[ScanEntry],
    opts: ActivityOptions,
) -> None:
    """Iterate ``estimate_activity`` across the checked scans (PR #55a).

    Continue-on-error per scan; per-scan preflight inside ``_build`` skips
    scans that aren't preprocessed (toeplitz + canonical) or whose
    montage doesn't match (toeplitz only). Final toast summarises
    successes / failures.

    Toeplitz mode caveat: ``state.montage`` is single-valued, so only the
    scan matching ``state.montage_source_scan`` can use toeplitz. The
    bulk run will succeed on that scan and skip every other. Canonical
    mode works universally (each scan gets a fresh canonical Montage).
    """
    if state.busy:
        return

    snapshot = _snapshot_options(state, opts)

    def _build(scan: ScanEntry):
        if snapshot.hrf_model == MODEL_TOEPLITZ:
            reason = _toeplitz_skip_reason(state, scan)
            if reason is not None:
                return reason  # intentional skip carrying its reason
            existing_montage = _montage_for_scan(state, scan)
        elif snapshot.hrf_model == MODEL_LIBRARY:
            # Per-channel mode resolves each scan's own HRtree match in the
            # worker below, so it doesn't need a single selected kernel here;
            # single-kernel mode still does.
            if not snapshot.library_per_channel and snapshot.library_trace is None:
                return (
                    "no HRtree HRF selected — pick one in the HRtree tab to "
                    "use as the deconvolution kernel"
                )
            existing_montage = None
        else:
            existing_montage = None

        progress_cb = make_progress_callback(state)

        def _pp_and_estimate(
            scan=scan, existing_montage=existing_montage, progress_cb=progress_cb,
        ):
            # Preprocess on demand (deconvolution) if needed, so the bulk run
            # triggers the preprocessing each scan needs instead of skipping.
            # ensure_deconvolved_raw raises a specific reason on failure.
            from .preprocess_panel import ensure_deconvolved_raw
            raw = ensure_deconvolved_raw(state, scan)
            scan_snapshot = snapshot
            if snapshot.hrf_model == MODEL_LIBRARY and snapshot.library_per_channel:
                # Match THIS scan's channels to the HRtree ROIs.
                import dataclasses
                traces = _compute_library_traces(state, raw, snapshot)
                scan_snapshot = dataclasses.replace(
                    snapshot, library_traces=traces
                )
                if not traces and snapshot.library_trace is None:
                    raise RuntimeError(
                        "no HRtree HRFs matched any channel of this scan within "
                        f"{snapshot.library_radius_mm:.0f} mm — widen the radius, "
                        "add ROIs, or select a single HRF as a fallback"
                    )
            return run_activity_sync(
                raw, scan_snapshot, existing_montage, progress_cb
            )

        return (_pp_and_estimate, (), {})

    async def _on_each_done(scan: ScanEntry, result) -> None:
        if result is None:
            # Raise so an empty deconvolution is reported as a failure with a
            # reason instead of being silently counted as a success.
            raise RuntimeError(
                "deconvolution produced no channels — every channel was "
                "dropped (check the scan's HRFs and preprocessing)"
            )
        state.activity_raw = result
        # Cache per scan so channel-wise 3-stage QC can read each scan's
        # deconvolution (activity_raw only holds the most-recent one). Route
        # through ``put`` rather than poking ``_cache`` directly -- the cache
        # is unbounded (maxsize=None) so nothing is evicted, and going via
        # the API keeps key normalisation in one place.
        state.activity_cache.put(scan, result)
        # Track which scan produced activity_raw so the preview won't overlay
        # one scan's deconvolution against another's preprocessed Raw.
        state.activity_source_scan = scan
        state.publish("activity_estimated", scan)

    client = capture_client()

    async def _bulk() -> None:
        with client_scope(client):
            bulk_result = await run_bulk_in_background(
                state, scans, _build,
                on_each_done=_on_each_done,
                label="estimate_activity",
            )
        if bulk_result is None:
            return
        successes, failures = bulk_result
        n_ok, n_fail = len(successes), len(failures)
        summary = (
            f"Estimated activity for {n_ok}/{n_ok + n_fail} scan(s)."
        )
        if failures:
            summary += f" Failed/skipped: {summarize_failures(failures)}"
        # Guarded: the page client may have been deleted during a long run.
        notify_if_alive(
            client, summary,
            type="positive" if n_fail == 0 else "warning",
            multi_line=True,
            close_button=True,
        )

    background_tasks.create(_bulk())


def _resolve_save_targets(scans, folder, postfix, ext) -> "dict":
    """Output path per scan for a mass activity save (pure / testable).

    ``folder=None`` → colocated next to each source file. A folder → flat
    destination, with name-collision disambiguation (``_2``, ``_3``, …) so two
    sources sharing a stem don't clobber each other in the same folder.
    """
    out: dict = {}
    seen: set = set()
    for s in scans:
        base = folder if folder is not None else s.path.parent
        stem = f"{s.path.stem}{postfix}"
        path = base / f"{stem}{ext}"
        if folder is not None:
            i = 2
            while path.resolve() in seen:
                path = base / f"{stem}_{i}{ext}"
                i += 1
        seen.add(path.resolve())
        out[s] = path
    return out


async def _mass_save_activity(
    state: AppState,
    scans: List[ScanEntry],
    naming: dict,
) -> None:
    """Save the ALREADY-estimated activity for each scan.

    A single confirmation dialog shows the filename format and lets the user
    pick the destination: colocated next to each source file (default,
    ``<stem><postfix><ext>``) or a single chosen folder — the latter so a
    READ-ONLY source directory doesn't dead-end the save. In a flat folder,
    colliding output names are disambiguated (``_2``, ``_3``, …). Save is
    independent of Estimate — it writes each scan's cached deconvolution
    (``state.activity_cache``) and never re-estimates; scans with no cached
    result are skipped with a clear "not estimated yet" reason, and a scan
    whose target would overwrite its own source file is skipped too.
    Continue-on-error with a final summary toast.
    """
    if state.busy:
        return
    from .export_panel import _save_raw

    postfix = naming.get("postfix", "_deconvolved")
    ext = naming.get("ext", ".snirf")
    n = len(scans)

    # Destination: None = colocated (next to each source file); a Path = a
    # single chosen folder (flat). The chosen-folder option lets the user save
    # off a READ-ONLY source directory (e.g. a shared/mounted dataset) instead
    # of being stuck when every colocated write fails.
    dest: dict = {"folder": None}

    def _resolve_outputs() -> "dict":
        return _resolve_save_targets(scans, dest["folder"], postfix, ext)

    @ui.refreshable
    def _dest_section() -> None:
        outputs = _resolve_outputs()
        colocated = dest["folder"] is None
        # Source-clobber only happens colocated (empty postfix + same ext).
        clobber_source = [
            s for s in scans if outputs[s].resolve() == s.path.resolve()
        ]
        existing = [
            s for s in scans
            if s not in clobber_source and outputs[s].exists()
        ]
        if colocated:
            ui.label(
                f"{n} deconvolved scan{'s' if n != 1 else ''} will be saved "
                "next to each source file (same folder)."
            ).classes("text-sm")
        else:
            ui.label(
                f"{n} deconvolved scan{'s' if n != 1 else ''} will be saved "
                "to this folder:"
            ).classes("text-sm")
            ui.label(str(dest["folder"])).classes(
                "text-xs font-mono opacity-70 break-all"
            )
        ui.label(f"Filename format:  <scan>{postfix}{ext}").classes(
            "text-xs font-mono opacity-70"
        )
        if existing:
            ui.label(
                f"⚠ {len(existing)} file(s) already exist and will be "
                "OVERWRITTEN."
            ).classes("text-xs text-amber-700")
        if clobber_source:
            ui.label(
                f"⚠ {len(clobber_source)} would write over the SOURCE scan "
                "(empty postfix + same format) — these will be SKIPPED. Set a "
                "filename postfix, or choose a different folder."
            ).classes("text-xs text-red-700")

    async def _choose_folder() -> None:
        from .dataset_picker import pick_folder

        picked = await pick_folder()
        if picked is not None:
            dest["folder"] = picked
            _dest_section.refresh()

    def _use_colocated() -> None:
        dest["folder"] = None
        _dest_section.refresh()

    with ui.dialog() as dialog, ui.card().classes("gap-2 w-[540px] max-w-full"):
        ui.label("Save deconvolved scans").classes("text-base font-semibold")
        _dest_section()
        with ui.row().classes("items-center gap-2"):
            ui.button(
                "Choose folder…", icon="folder_open", on_click=_choose_folder,
            ).props("flat dense").tooltip(
                "Save the whole batch into one folder — use this when the "
                "source folder is read-only."
            )
            ui.button(
                "Next to each source", icon="restore", on_click=_use_colocated,
            ).props("flat dense")
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button("Save", icon="save", on_click=lambda: dialog.submit(True)).props(
                "color=primary"
            )
    confirmed = await dialog
    if not confirmed:
        return

    # Freeze the destination mapping at confirm time.
    outputs = _resolve_outputs()

    def _build(scan: ScanEntry):
        # Save only — never estimate. Skip scans that haven't been run yet.
        if scan not in state.activity_cache:
            return (
                "not estimated yet — run Estimate first (Save does not "
                "re-estimate)"
            )
        result = state.activity_cache.get(scan)
        out_path = outputs[scan]
        # Never overwrite the SOURCE file (empty postfix + same extension) —
        # that would destroy the raw recording. Skip with a clear reason.
        if out_path.resolve() == scan.path.resolve():
            return (
                "would overwrite the source file — set a filename postfix to "
                "save the deconvolution separately"
            )

        def _save_only(result=result, out_path=out_path):
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                _save_raw(result, out_path)
            except Exception as exc:  # noqa: BLE001 — disk/format failure
                raise RuntimeError(
                    f"could not write {out_path.name} "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            return out_path

        return (_save_only, (), {})

    client = capture_client()

    async def _bulk() -> None:
        with client_scope(client):
            bulk_result = await run_bulk_in_background(
                state, scans, _build, label="save_activity",
            )
        if bulk_result is None:
            return
        successes, failures = bulk_result
        n_ok, n_fail = len(successes), len(failures)
        where = (
            f"to {dest['folder']}" if dest["folder"] is not None
            else "next to their source files"
        )
        summary = (
            f"Saved {n_ok}/{n_ok + n_fail} deconvolved scan(s) {where}."
        )
        if failures:
            summary += f" Failed/skipped: {summarize_failures(failures)}"
        notify_if_alive(
            client, summary,
            type="positive" if n_fail == 0 else "warning",
            multi_line=True,
            close_button=True,
        )

    background_tasks.create(_bulk())


def _run(
    state: AppState, scan: ScanEntry, opts: ActivityOptions
) -> None:
    """Click handler for Estimate activity (single-scan path).

    Validates state preconditions, snapshots options, copies the Raw
    before handing it to estimate_activity (which mutates in place),
    dispatches via run_in_background. On success, stashes the result on
    state.activity_raw and publishes ``activity_estimated``.
    """
    if state.busy:
        return
    if scan not in state.processed_cache:
        state.last_error = "Preprocess the scan first."
        return
    if scan.path.resolve() not in state.processed_deconvolved:
        state.last_error = (
            "Neural-activity estimation requires deconvolution-mode "
            "preprocessing — re-preprocess this scan with deconvolution."
        )
        return

    if opts.hrf_model == MODEL_TOEPLITZ:
        # Use this scan's own estimated HRFs (per-scan cache), so it works
        # whether the scan was estimated singly or as part of a bulk run.
        if _montage_for_scan(state, scan) is None:
            state.last_error = (
                "No estimated HRFs for this scan. Estimate HRFs for it on the "
                "HRFs tab (toeplitz mode), or switch the source to canonical."
            )
            return
    raw = state.processed_cache.get(scan)

    library_traces = None
    if opts.hrf_model == MODEL_LIBRARY:
        # Per-channel HRtree map for this scan (None => single-kernel fallback).
        library_traces = _compute_library_traces(state, raw, opts)
        if not library_traces and _library_kernel_from_state(state) is None:
            state.last_error = (
                "No HRtree HRFs to deconvolve with. Build ROIs in the HRtree "
                "for per-channel matching, or select a single HRF there first."
            )
            return

    snapshot = _snapshot_options(state, opts)
    snapshot.library_traces = library_traces

    progress_cb = make_progress_callback(state)
    existing_montage = (
        _montage_for_scan(state, scan)
        if snapshot.hrf_model == MODEL_TOEPLITZ else None
    )

    async def _on_done(result) -> None:
        if result is None:
            # The estimate produced nothing (every channel dropped, or no
            # deconvolvable data). Don't silently leave the PREVIOUS scan's
            # result on screen implying this one succeeded — surface an error.
            # Only set it if the worker didn't already record an exception
            # message, so a real traceback isn't overwritten.
            if not state.last_error:
                state.last_error = (
                    "Deconvolution produced no output for this scan — every "
                    "channel may have been dropped (check the scan's HRFs and "
                    "preprocessing)."
                )
            return
        state.activity_raw = result
        # Cache per scan so channel-wise 3-stage QC can read this scan's
        # deconvolution later (activity_raw only holds the most-recent one).
        # ``put`` on the unbounded (maxsize=None) cache: no eviction, one
        # place for key normalisation.
        state.activity_cache.put(scan, result)
        # Track which scan produced activity_raw (preview gate uses it).
        state.activity_source_scan = scan
        state.publish("activity_estimated", scan)

    background_tasks.create(
        run_in_background(
            state,
            run_activity_sync,
            raw,
            snapshot,
            existing_montage,
            progress_cb,
            on_done=_on_done,
        )
    )


def run_activity_sync(
    raw: "mne.io.BaseRaw",
    opts: ActivityOptions,
    existing_montage=None,
    progress_callback=None,
):
    """Run ``estimate_activity`` and return the deconvolved Raw.

    The input ``raw`` is copied before passing to estimate_activity so the
    cached processed Raw (state.processed_cache) is not corrupted by the
    library's in-place mutation. Returns None if all channels are dropped
    or estimate_activity returns None.

    For toeplitz mode, ``existing_montage`` MUST be a real Montage with
    per-channel HRF traces already populated (i.e., from a prior
    estimate_hrf + generate_distribution run). The function snapshots the
    Montage's channel containers (``channels`` dict + ``hbo_channels`` /
    ``hbr_channels`` lists) before estimate_activity and restores them
    afterward, so the library's drop-bad-channel behavior (hrfunc.py:606-
    611) does not corrupt ``state.montage`` for the HRFs tab. The
    returned Raw still reflects whichever channels survived deconvolution.

    For canonical and library modes, ``existing_montage`` is ignored and a
    fresh Montage is configured to the scan. Library mode deconvolves every
    channel with ``opts.library_trace`` (an HRtree selection), oriented by
    ``opts.library_oxygenation``.

    Module-level so tests can call it without dispatching through workers.
    """
    from ...hrfunc import montage as Montage

    work_raw = raw.copy()

    snapshot = None
    fresh_montage_models = (MODEL_CANONICAL, MODEL_LIBRARY)
    if opts.hrf_model in fresh_montage_models or existing_montage is None:
        m = Montage(nirx_obj=work_raw)
    else:
        m = existing_montage
        # Shallow-copy the containers the library mutates. estimate_activity
        # only pops dict keys / removes list entries — it does not mutate
        # the HRF node objects themselves — so a shallow snapshot is
        # sufficient to restore the pre-run channel set after the call.
        snapshot = {
            "channels": dict(m.channels),
            "hbo_channels": list(getattr(m, "hbo_channels", [])),
            "hbr_channels": list(getattr(m, "hbr_channels", [])),
        }

    estimate_kwargs = {
        "timeout": opts.timeout,
        "drop_failed_channels": opts.drop_failed_channels,
    }
    if opts.hrf_model == MODEL_LIBRARY:
        if opts.library_traces:
            # Per-channel HRtree map (already matched per this scan's channels).
            estimate_kwargs["library_traces"] = opts.library_traces
            estimate_kwargs["library_uncovered"] = opts.library_uncovered
        else:
            # Single-kernel fallback (one selected HRF for every channel).
            estimate_kwargs["library_trace"] = opts.library_trace
            estimate_kwargs["library_oxygenation"] = opts.library_oxygenation

    try:
        result = m.estimate_activity(
            work_raw,
            lmbda=opts.lmbda,
            hrf_model=opts.hrf_model,
            preprocess=False,
            progress_callback=progress_callback,
            **estimate_kwargs,
        )
    finally:
        if snapshot is not None:
            m.channels = snapshot["channels"]
            if hasattr(m, "hbo_channels"):
                m.hbo_channels = snapshot["hbo_channels"]
            if hasattr(m, "hbr_channels"):
                m.hbr_channels = snapshot["hbr_channels"]

    return result


# ---------------------------------------------------------------------------
# Result preview
# ---------------------------------------------------------------------------


def _render_preview(
    state: AppState, scan: ScanEntry, opts: ActivityOptions
) -> None:
    """Render the lens.plot_nirx-style preproc/deconv overlay."""
    activity = state.activity_raw
    if activity is None:
        ui.label("No activity result.").classes("text-sm opacity-60")
        return

    processed = state.processed_cache.get(scan) if scan in state.processed_cache else None
    if processed is None:
        ui.label("Original preprocessed scan unavailable for overlay.").classes(
            "text-sm opacity-60"
        )
        return

    n_channels = len(activity.ch_names)
    if n_channels == 0:
        ui.label("Activity result has no channels.").classes(
            "text-sm opacity-60"
        )
        return

    # Channel picker — bound to opts.preview_channel
    channel_options = {i: name for i, name in enumerate(activity.ch_names)}
    safe_default = min(opts.preview_channel, n_channels - 1)
    opts.preview_channel = safe_default

    def _on_pick(event) -> None:
        opts.preview_channel = int(event.value)
        state.publish("activity_estimated", scan)

    ui.select(
        options=channel_options,
        value=safe_default,
        on_change=_on_pick,
        label="Preview channel",
    ).classes("w-64")

    png = _render_overlay_png(
        processed=processed,
        deconvolved=activity,
        channel=safe_default,
    )
    if png is None:
        ui.label("Preview unavailable.").classes("text-sm opacity-60")
        return
    ui.image(png).classes("max-w-3xl")


def _render_overlay_png(
    processed: "mne.io.BaseRaw",
    deconvolved: "mne.io.BaseRaw",
    channel: int,
    length: Optional[int] = None,
) -> Optional[str]:
    """Build a matplotlib lens.plot_nirx-style overlay as base64 PNG.

    Normalizes the preprocessed signal to the deconvolved signal's range
    (so they sit on the same y-axis), plots both, and overlays event
    annotations from the deconvolved Raw as vertical lines.

    ``length`` is the number of samples to display. None → first 500
    samples (matches the library's lens.plot_nirx default).
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for overlay: %s", exc)
        return None

    if length is None:
        length = 500

    fig = None
    try:
        # Map deconvolved channel back to processed channel name in case the
        # estimation dropped channels (the indices won't align).
        deconv_name = deconvolved.ch_names[channel]
        if deconv_name in processed.ch_names:
            proc_idx = processed.ch_names.index(deconv_name)
        else:
            # The deconvolved Raw exposes a channel name the processed Raw
            # doesn't have — should not normally happen (the deconvolved
            # Raw is derived from a copy of processed). Bail out instead of
            # silently plotting the wrong channel.
            logger.warning(
                "activity overlay: channel %r exists on deconvolved Raw "
                "but not on processed Raw; refusing to render with a "
                "mismatched channel.",
                deconv_name,
            )
            return None

        proc_data = processed.get_data(picks=[proc_idx])[0]
        deconv_data = deconvolved.get_data(picks=[channel])[0]

        proc_window = proc_data[:length]
        deconv_window = deconv_data[:length]

        # Normalize proc to match deconv y-range so the overlay is readable.
        # If either window is constant (range 0), fall back to plotting raw
        # values so we don't divide by zero.
        proc_range = proc_window.max() - proc_window.min()
        deconv_range = deconv_window.max() - deconv_window.min()
        if proc_range > 0 and deconv_range > 0:
            proc_norm = (proc_window - proc_window.min()) / proc_range
            proc_scaled = (
                proc_norm * deconv_range + deconv_window.min()
            )
        else:
            proc_scaled = proc_window

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(
            proc_scaled,
            color="red",
            linestyle="--",
            lw=0.8,
            label="Preprocessed (rescaled)",
        )
        ax.plot(
            deconv_window,
            color="steelblue",
            lw=1.0,
            label="Deconvolved neural activity",
        )

        # Overlay event markers from the deconvolved Raw's annotations
        # (or from the processed Raw if the deconvolved one dropped them).
        annotations = deconvolved.annotations
        if annotations is None or len(annotations) == 0:
            annotations = processed.annotations
        if annotations is not None and len(annotations) > 0:
            sfreq = float(deconvolved.info["sfreq"])
            for ann in annotations:
                sample = int(round(float(ann["onset"]) * sfreq))
                if 0 <= sample < length:
                    ax.axvline(
                        x=sample, color="orange", lw=0.6, alpha=0.5,
                    )

        ax.set_title(f"Channel: {deconv_name}")
        ax.set_xlabel("samples")
        ax.set_ylabel("amplitude (a.u.)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("activity overlay render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
