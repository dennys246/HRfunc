"""Preprocess tab content — runs hrfunc.preprocess_fnirs on the selected scan.

Sprint 3.2 ships:
- A "Run full pipeline" button that calls ``hrfunc.preprocess_fnirs`` end-to-
  end on the cached Raw and stores the result in ``state.processed_cache``.
- Staged toggles letting the user opt out of specific steps. The toggles
  do NOT change the pipeline order (preprocess_fnirs runs OD → SCI →
  interpolate-bads → TDDR → optional polynomial detrend → Beer-Lambert →
  baseline-correct → optional filter). They control which stages to apply.
  The default toggle state mirrors the library default.
- A "Deconvolution mode" switch — drives the ``deconvolution`` kwarg on
  ``preprocess_fnirs`` (polynomial detrend on, bandpass filter off).
- A before/after channel plot rendered as a matplotlib base64 PNG, showing
  the first few channels before and after preprocessing for sanity-check.

The panel subscribes to ``scan_selected`` and ``scan_loaded`` events so it
refreshes when the user changes scan or when the Raw becomes available.
After a successful preprocess, it publishes ``preprocess_done`` so the HRFs
and Activity tabs (Sprint 3.3/3.4) can wake up.

Scientific caveat
-----------------

The staged toggles are GUI conveniences. The library's ``preprocess_fnirs``
does not currently accept skip flags for individual stages — Sprint 3.2
implements the same operations inline when toggles are set. The default
"Run full pipeline" path calls the library function untouched so users
get the canonical hrfunc pipeline. Skipping stages is for diagnostic
exploration, not for publishable analyses.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from nicegui import background_tasks, ui

from ..state import AppState
from ..workers import (
    capture_client,
    client_scope,
    notify_if_alive,
    run_bulk_in_background,
    run_in_background,
    summarize_failures,
)
from ...io.manifest import ScanEntry

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


@dataclass
class PreprocessOptions:
    """User-controlled toggles for the Preprocess tab.

    A snapshot of the panel's UI state, captured at "Run" click time so the
    background task sees a stable view. Default values mirror the library's
    canonical pipeline.

    ``apply_baseline_correct`` is only user-controllable in deconvolution
    mode. In GLM mode (``deconvolution=False``), the library always baseline-
    corrects and the GUI mirrors that — skipping baseline correction in GLM
    mode would feed unbaselined data to the bandpass filter, which is a
    well-known scientific footgun. ``run_pipeline_sync`` enforces this
    invariant defensively (in addition to the UI hiding the toggle).
    """

    deconvolution: bool = False
    apply_motion_correction: bool = True
    apply_beer_lambert: bool = True
    apply_baseline_correct: bool = True


def render(state: AppState) -> None:
    """Render the Preprocess tab body inside the current NiceGUI context.

    Subscribes to ``scan_selected`` and ``scan_loaded`` so the panel reacts
    when the user changes scans or when a Raw becomes available. The Run
    button is disabled when no scan is selected or the Raw is not yet
    loaded. ``state.busy`` further disables it (preprocess shares the busy
    gate with estimation since both are heavy CPU tasks).
    """

    # Holder for the toggles — updated by ui.switch on_change handlers and
    # consumed by the click handler when Run is pressed. The GUI defaults to
    # DECONVOLUTION mode: it's the pipeline HRF / activity estimation
    # requires, and the dominant use of this app. Users can toggle it off
    # for plain haemoglobin (GLM) preprocessing, but then the HRFs / Activity
    # tabs will ask them to re-preprocess in deconvolution mode. (The library
    # PreprocessOptions default stays GLM; this is a GUI-workflow default.)
    opts = PreprocessOptions(deconvolution=True)

    @ui.refreshable
    def _body() -> None:
        _render_body(state, opts)

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)
    state.subscribe("preprocess_done", _refresh)
    # Recompute bulk mode when the dataset-tree checked set changes.
    state.subscribe("checked_changed", _refresh)


def _record_deconvolved(state: AppState, scan, deconvolution: bool) -> None:
    """Track whether a scan's cached processed Raw came from the
    deconvolution pipeline. HRF / activity estimation gate on this — a
    GLM/hemoglobin preprocess (deconvolution=False) is removed from the set
    so estimation refuses it until re-preprocessed in deconvolution mode."""
    key = scan.path.resolve()
    if deconvolution:
        state.processed_deconvolved.add(key)
    else:
        state.processed_deconvolved.discard(key)


def ensure_deconvolved_raw(state: AppState, scan):
    """Return a deconvolution-preprocessed Raw for ``scan``, preprocessing on
    demand if one isn't already cached.

    Lets a bulk HRF / Activity run TRIGGER the preprocessing each scan needs
    instead of skipping un-preprocessed scans. Runs synchronously (call from
    a worker thread). Returns the processed Raw.

    Raises ``RuntimeError`` (never returns None) with a specific reason when
    the scan can't be made ready — the source file won't load, or the
    deconvolution pipeline yields nothing. Raising rather than returning None
    is deliberate: the bulk worker turns the exception into a per-scan failure
    with this message, so the user sees *why* a scan dropped out instead of a
    silent skip counted as success.

    Reuse rule: a scan already preprocessed in deconvolution mode is returned
    from cache as-is; a scan that's missing OR was GLM-preprocessed is
    (re)run through the deconvolution pipeline so estimation gets valid input.
    """
    key = scan.path.resolve()
    if scan in state.processed_cache and key in state.processed_deconvolved:
        return state.processed_cache.get(scan)
    try:
        raw = state.raw_cache.get(scan)
    except Exception as exc:  # noqa: BLE001 — bad/missing source file
        logger.warning("ensure_deconvolved_raw: load failed for %s: %s",
                       getattr(scan, "path", scan), exc)
        raise RuntimeError(
            f"could not load source file ({type(exc).__name__}: {exc})"
        ) from exc
    result = run_pipeline_sync(raw, PreprocessOptions(deconvolution=True))
    if result is None:
        raise RuntimeError(
            "deconvolution preprocessing produced no output — the scan may "
            "have no fNIRS channels, or every channel was dropped by the "
            "pipeline (check the source file and montage)"
        )
    state.processed_cache._cache[key] = result
    _record_deconvolved(state, scan, True)
    return result


def scan_is_deconvolved(state: AppState, scan: ScanEntry) -> bool:
    """True when ``scan`` has a deconvolution-preprocessed Raw cached.

    The same readiness gate HRF / activity estimation use: present in
    ``processed_cache`` AND recorded in ``processed_deconvolved`` (a
    GLM/haemoglobin preprocess does not count). Shared so the HRFs and
    Activity tabs report bulk readiness consistently.
    """
    return (
        scan in state.processed_cache
        and scan.path.resolve() in state.processed_deconvolved
    )


def preprocess_all_checked(state: AppState, scans: list) -> None:
    """Deconvolution-preprocess every not-yet-ready scan in ``scans`` as one
    background batch — the bulk analog of the HRFs tab's single-scan
    "Preprocess now".

    Reuses ``ensure_deconvolved_raw`` via ``run_bulk_in_background`` so the
    busy gate, per-scan progress, and continue-on-error semantics match the
    bulk estimate flows. Already-deconvolved scans are filtered out so the
    toast counts reflect real work. Shared by the HRFs and Activity tabs.
    """
    if state.busy:
        return
    targets = [s for s in scans if not scan_is_deconvolved(state, s)]
    if not targets:
        ui.notify("All checked scans are already preprocessed.", type="info")
        return

    def _build(scan: ScanEntry):
        # ensure_deconvolved_raw loads + deconvolution-preprocesses + caches +
        # records processed_deconvolved itself, raising with a specific reason
        # on failure; the worker turns that into a per-scan failure.
        return (ensure_deconvolved_raw, (state, scan), {})

    async def _on_each(scan: ScanEntry, _result) -> None:
        state.publish("preprocess_done", scan)

    async def _run() -> None:
        result = await run_bulk_in_background(
            state, targets, _build,
            on_each_done=_on_each, label="bulk preprocess",
        )
        if result is None:
            return
        successes, failures = result
        if failures:
            ui.notify(
                f"Preprocessed {len(successes)} scan(s); "
                f"{len(failures)} failed — see the latest error.",
                type="warning",
            )
        else:
            ui.notify(
                f"Preprocessed {len(successes)} scan(s).", type="positive"
            )

    ui.notify("Preprocessing checked scans…", type="info")
    background_tasks.create(_run())


def render_preprocess_all_checked(state: AppState, scans: list) -> None:
    """Bulk preprocess affordance: readiness summary + one-click run.

    Mirrors the HRFs tab's single-scan "Preprocess now" for the checked set.
    The bulk estimate already preprocesses each scan on demand, so this is a
    convenience / pre-warm plus a clear readiness readout — not a
    prerequisite. Shared by the HRFs and Activity tabs.
    """
    if state.busy:
        with ui.row().classes("items-center gap-2"):
            ui.spinner(size="sm")
            ui.label("Preprocessing…").classes("text-sm opacity-70")
        return

    n_total = len(scans)
    not_ready = [s for s in scans if not scan_is_deconvolved(state, s)]
    plural = "s" if n_total != 1 else ""
    if not not_ready:
        ui.label(
            f"All {n_total} checked scan{plural} are deconvolution-"
            "preprocessed and ready."
        ).classes("text-sm text-emerald-400")
        return

    ui.label(
        f"{len(not_ready)} of {n_total} checked scan{plural} aren't "
        "deconvolution-preprocessed yet. The bulk run preprocesses each scan "
        "automatically — or preprocess them all now."
    ).classes("text-sm opacity-70")
    with ui.row().classes("items-center gap-2"):
        ui.button(
            "Preprocess all checked",
            icon="play_arrow",
            on_click=lambda: preprocess_all_checked(state, scans),
        ).props("color=primary")
        ui.label(
            "Uses default settings — for custom options use the Preprocess tab."
        ).classes("text-xs opacity-60")


def _resolve_checked_scans(state: AppState) -> list:
    """Resolve ``state.checked_scan_paths`` to ScanEntries in manifest order.

    PR #55a helper. Returns the list of currently-checked scans, in the
    same order they appear in the manifest, so a bulk run iterates in a
    stable order. Paths that no longer match a manifest scan (e.g. after
    a rescan dropped a file) are silently skipped -- the on-disk source
    is gone, the user wouldn't expect the run to fail on it.
    """
    if state.manifest is None or not state.checked_scan_paths:
        return []
    return [
        scan for scan in state.manifest.scans
        if scan.path.resolve() in state.checked_scan_paths
    ]


def _render_body(state: AppState, opts: PreprocessOptions) -> None:
    """Render the body against the current scan + cache state.

    Extracted to module scope so tests can call it directly inside a
    synthetic NiceGUI context without going through the refreshable wrapper.
    """
    scan = state.selected_scan
    checked_scans = _resolve_checked_scans(state)
    bulk_mode = len(checked_scans) >= 1

    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("Preprocess").classes("text-2xl font-semibold")

        if scan is None and not bulk_mode:
            ui.label(
                "Select a scan from the dataset tree, or tick scans "
                "for a bulk run."
            ).classes("text-sm opacity-60")
            return

        if bulk_mode:
            ui.label(
                f"Bulk run on {len(checked_scans)} checked scan"
                f"{'s' if len(checked_scans) != 1 else ''}."
            ).classes("text-sm font-mono opacity-70")
        else:
            ui.label(scan.display_name or scan.path.name).classes(
                "text-sm font-mono opacity-70"
            )

        raw_loaded = scan is not None and scan in state.raw_cache
        already_processed = (
            scan is not None and scan in state.processed_cache
        )

        # ── Options
        with ui.card().classes("w-full"):
            ui.label("Pipeline options").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            ui.switch(
                "Deconvolution mode (polynomial detrend, skip bandpass)",
                value=opts.deconvolution,
                on_change=lambda e: setattr(opts, "deconvolution", bool(e.value)),
            )
            ui.switch(
                "Motion correction (TDDR)",
                value=opts.apply_motion_correction,
                on_change=lambda e: setattr(
                    opts, "apply_motion_correction", bool(e.value)
                ),
            )
            ui.switch(
                "Beer-Lambert conversion to haemoglobin",
                value=opts.apply_beer_lambert,
                on_change=lambda e: setattr(
                    opts, "apply_beer_lambert", bool(e.value)
                ),
            )
            # Baseline correct is only user-skippable in deconvolution mode.
            # In GLM mode, the library always applies it (see run_pipeline_sync
            # and PreprocessOptions docstring).
            if opts.deconvolution:
                ui.switch(
                    "Baseline correct",
                    value=opts.apply_baseline_correct,
                    on_change=lambda e: setattr(
                        opts, "apply_baseline_correct", bool(e.value)
                    ),
                )
            ui.label(
                "Default settings reproduce the library's canonical pipeline "
                "and are publication-ready. Non-default toggles are for "
                "diagnostic exploration only — do not use them for analyses "
                "you intend to publish."
            ).classes("text-xs opacity-60 italic")

        # ── Run button
        # PR #55a: when scans are checked, the button iterates over the
        # checked set sequentially (bulk mode). Otherwise it runs against
        # the currently-selected scan (single mode), which is the legacy
        # behaviour. Per-scan raw loading happens inside the worker so
        # an unloaded scan in the checked set doesn't block the run.
        if bulk_mode:
            run_label = (
                f"Run full pipeline on {len(checked_scans)} scan"
                f"{'s' if len(checked_scans) != 1 else ''}"
            )
            run_disabled = state.busy
        else:
            run_label = "Run full pipeline"
            run_disabled = (not raw_loaded) or state.busy

        with ui.row().classes("items-center gap-3"):
            ui.button(
                run_label,
                on_click=lambda: _run_pipeline_dispatch(
                    state, scan, checked_scans, opts
                ),
            ).props(
                f"color=primary {'disable' if run_disabled else ''}"
            )
            if not bulk_mode and not raw_loaded:
                ui.label("Waiting for scan to load…").classes(
                    "text-sm opacity-60"
                )
            elif state.busy:
                _render_busy_progress(state)

        # ── Surface the last error if there is one
        if state.last_error and not state.busy:
            with ui.row().classes("items-center gap-2"):
                ui.icon("error_outline").classes("text-red-400")
                ui.label(state.last_error).classes("text-sm text-red-400")

        # ── Before/after preview
        if already_processed:
            ui.separator()
            ui.label("Before / after").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            _render_before_after(state, scan)


def _snapshot_opts(opts: PreprocessOptions) -> PreprocessOptions:
    """Snapshot options at click-time so the closure sees stable values.

    Baseline correct is forced True in GLM mode to match library
    behaviour -- the UI hides the toggle there but a manual state
    mutation could still leave it False, so enforce defensively.
    """
    return PreprocessOptions(
        deconvolution=opts.deconvolution,
        apply_motion_correction=opts.apply_motion_correction,
        apply_beer_lambert=opts.apply_beer_lambert,
        apply_baseline_correct=(
            opts.apply_baseline_correct or not opts.deconvolution
        ),
    )


def _run_pipeline_dispatch(
    state: AppState,
    selected: Optional[ScanEntry],
    checked: list,
    opts: PreprocessOptions,
) -> None:
    """Route the Run button to single or bulk based on the checked set.

    PR #55a: when the dataset tree has scans checked, the click iterates
    over the whole checked set sequentially (continue-on-error). When
    nothing is checked, falls back to the legacy single-scan path against
    ``state.selected_scan``.
    """
    if checked:
        _run_pipeline_bulk(state, checked, opts)
    elif selected is not None:
        _run_pipeline(state, selected, opts)


def _run_pipeline(
    state: AppState, scan: ScanEntry, opts: PreprocessOptions
) -> None:
    """Click handler for the Run button (single-scan path).

    Snapshots the toggle state, then dispatches the actual preprocessing on
    the background-task helper. The helper sets ``state.busy`` so the
    HRFs/Activity tabs see "estimation pipeline is occupied" — preprocess is
    treated as part of the heavy-CPU pipeline group.
    """
    if state.busy:
        return
    if scan not in state.raw_cache:
        state.last_error = "Raw not loaded; wait for the scan to finish loading."
        return

    snapshot = _snapshot_opts(opts)

    async def _on_done(result) -> None:
        if result is None:
            return
        # run_pipeline_sync wraps the result so we can stash the processed
        # Raw under the scan path in processed_cache. put() enforces the LRU
        # bound (a direct _cache write would not). _record_deconvolved tracks
        # whether this was a deconvolution-mode preprocess (the activity gate
        # reads it).
        state.processed_cache.put(scan, result)
        _record_deconvolved(state, scan, snapshot.deconvolution)
        state.publish("preprocess_done", scan)

    background_tasks.create(
        run_in_background(
            state,
            run_pipeline_sync,
            state.raw_cache.get(scan),
            snapshot,
            on_done=_on_done,
        )
    )


def _run_pipeline_bulk(
    state: AppState,
    scans: list,
    opts: PreprocessOptions,
) -> None:
    """Iterate ``run_pipeline_sync`` across a set of checked scans (PR #55a).

    Each scan's raw is loaded on demand inside the worker thread
    (``state.raw_cache.get(scan)`` synchronously loads if missing), so
    pre-loading every checked scan isn't required. Per-scan failures land
    in ``state.last_error`` and the run continues; the final summary toast
    spells out N successes / M failures.
    """
    if state.busy:
        return

    snapshot = _snapshot_opts(opts)

    def _build(scan: ScanEntry):
        # Per-scan: load the raw (blocking inside the worker thread is
        # fine), then run the pipeline against it.
        def _run() -> Optional["mne.io.BaseRaw"]:
            raw = state.raw_cache.get(scan)
            return run_pipeline_sync(raw, snapshot)
        return (_run, (), {})

    async def _on_each_done(scan: ScanEntry, result) -> None:
        if result is None:
            # Counted as a failure (raising moves it to the failures bucket)
            # so an empty pipeline result isn't silently scored as success.
            raise RuntimeError(
                "preprocessing produced no output — no fNIRS channels "
                "survived the pipeline"
            )
        # put() enforces the LRU bound (the prior direct _cache write let a
        # bulk preprocess retain every result and grow memory unbounded).
        state.processed_cache.put(scan, result)
        _record_deconvolved(state, scan, snapshot.deconvolution)
        state.publish("preprocess_done", scan)

    client = capture_client()

    async def _bulk() -> None:
        with client_scope(client):
            bulk_result = await run_bulk_in_background(
                state,
                scans,
                _build,
                on_each_done=_on_each_done,
                label="preprocess",
            )
        if bulk_result is None:
            return
        successes, failures = bulk_result
        n_ok, n_fail = len(successes), len(failures)
        summary = f"Preprocessed {n_ok}/{n_ok + n_fail} scan(s)."
        if failures:
            summary += f" Failed: {summarize_failures(failures)}"
        # Guarded: the page client may have been deleted during a long run.
        notify_if_alive(
            client, summary,
            type="positive" if n_fail == 0 else "warning",
            multi_line=True,
            close_button=True,
        )

    background_tasks.create(_bulk())


def _render_busy_progress(state: AppState) -> None:
    """Render the spinner + per-scan bulk progress + within-scan progress.

    PR #55a: during a bulk run there are two layers of progress -- the
    outer ``bulk_progress`` (scan i/N) and the inner
    ``estimation_progress`` (channel i/N) when the per-scan worker
    pushes channel-level callbacks. Preprocess doesn't currently emit
    channel progress, but the bulk line still gives users feedback that
    the run is advancing across scans.
    """
    bulk = state.bulk_progress
    if bulk is not None:
        idx, total, scan = bulk
        scan_label = scan.display_name or scan.path.name
        with ui.column().classes("gap-1"):
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="sm")
                ui.label(
                    f"Scan {idx + 1}/{total}: {scan_label}"
                ).classes("text-sm opacity-80")
            fraction = (idx + 1) / max(total, 1)
            ui.linear_progress(value=fraction).classes("w-64")
    else:
        with ui.row().classes("items-center gap-2"):
            ui.spinner(size="sm")
            ui.label("Preprocessing…").classes("text-sm opacity-70")


def run_pipeline_sync(
    raw: "mne.io.BaseRaw", opts: PreprocessOptions
) -> Optional["mne.io.BaseRaw"]:
    """Run the preprocessing pipeline against a Raw and return the processed
    Raw (or None if all channels were flagged bad).

    All steps are taken straight from ``hrfunc.preprocess_fnirs`` so the GUI
    matches the library's canonical pipeline. The toggles in
    ``PreprocessOptions`` opt out of specific stages — the order of the
    remaining stages is preserved.

    Module-level so tests can call it without dispatching through the
    background-task helper.
    """
    # Lazy imports to keep module-import cheap.
    from itertools import compress

    import mne

    from ...hrfunc import baseline_correct, polynomial_detrend

    # Always start by loading data and converting to optical density.
    # preprocess_fnirs does this unconditionally and the entire pipeline
    # downstream depends on OD-space data.
    raw.load_data()
    raw_od = mne.preprocessing.nirs.optical_density(raw, verbose="ERROR")

    # Scalp coupling index → mark bad channels.
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od, verbose="ERROR")
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.95))

    if len(raw_od.info["bads"]) == len(raw_od.ch_names):
        logger.warning(
            "preprocess: every channel scored SCI<0.95 — refusing to "
            "preprocess the all-bad scan."
        )
        return None

    if raw_od.info["bads"]:
        raw_od.interpolate_bads(reset_bads=False, verbose="ERROR")

    od = (
        mne.preprocessing.nirs.tddr(raw_od, verbose="ERROR")
        if opts.apply_motion_correction
        else raw_od
    )

    # Polynomial detrend is part of the deconvolution-mode pipeline only —
    # mirrors preprocess_fnirs(scan, deconvolution=True) at line ~1072.
    if opts.deconvolution:
        od = polynomial_detrend(od, order=3)

    if opts.apply_beer_lambert:
        haemo = mne.preprocessing.nirs.beer_lambert_law(
            od.copy(), ppf=0.1
        )
    else:
        haemo = od.copy()

    if opts.apply_baseline_correct:
        haemo = baseline_correct(haemo, baseline=(None, 0.0))

    # GLM-friendly bandpass — skipped in deconvolution mode because the
    # detrend already handles slow drift.
    if not opts.deconvolution:
        haemo.filter(0.01, 0.2, verbose="ERROR")

    return haemo


def _render_before_after(state: AppState, scan: ScanEntry) -> None:
    """Render a side-by-side matplotlib PNG of the first few channels."""
    raw = state.raw_cache.get(scan) if scan in state.raw_cache else None
    processed = state.processed_cache.get(scan) if scan in state.processed_cache else None
    if raw is None or processed is None:
        ui.label("Preview unavailable.").classes("text-sm opacity-60")
        return

    png = _render_before_after_png(raw, processed)
    if png is None:
        ui.label("Could not render before/after preview.").classes(
            "text-sm opacity-60"
        )
        return
    ui.image(png).classes("max-w-3xl")


def _render_before_after_png(
    raw: "mne.io.BaseRaw",
    processed: "mne.io.BaseRaw",
    n_channels: int = 4,
) -> Optional[str]:
    """Encode a 2-row matplotlib figure as base64 PNG.

    Top row: raw signal for the first ``n_channels`` channels.
    Bottom row: processed signal for the same channel indices.

    Returns None on any matplotlib / MNE failure so the caller can render
    a fallback label instead.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for before/after: %s", exc)
        return None

    fig = None
    try:
        n = min(n_channels, len(raw.ch_names), len(processed.ch_names))
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=False)

        raw_data = raw.get_data(picks=list(range(n)))
        proc_data = processed.get_data(picks=list(range(n)))
        raw_times = raw.times
        proc_times = processed.times

        for i in range(n):
            axes[0].plot(
                raw_times, raw_data[i], lw=0.6,
                label=raw.ch_names[i],
            )
            axes[1].plot(
                proc_times, proc_data[i], lw=0.6,
                label=processed.ch_names[i],
            )
        axes[0].set_title("Before preprocessing")
        axes[1].set_title("After preprocessing")
        axes[0].set_ylabel("amplitude")
        axes[1].set_ylabel("amplitude")
        axes[1].set_xlabel("time (s)")
        axes[0].legend(loc="upper right", fontsize=6)
        axes[1].legend(loc="upper right", fontsize=6)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("before/after render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
