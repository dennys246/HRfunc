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
    make_progress_callback,
    run_bulk_in_background,
    run_in_background,
)
from ...io.manifest import ScanEntry
from .hrf_panel import _CanonicalResult

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


MODEL_TOEPLITZ = "toeplitz"
MODEL_CANONICAL = "canonical"
DEFAULT_LMBDA = 1e-4
LOG_LMBDA_MIN = -6
LOG_LMBDA_MAX = -1


@dataclass
class ActivityOptions:
    """User-controlled options for the Activity tab.

    Snapshotted at Run-click time so the background task sees a stable view.
    Defaults mirror ``montage.estimate_activity`` library defaults.
    """

    hrf_model: str = MODEL_TOEPLITZ
    lmbda: float = DEFAULT_LMBDA
    preview_channel: int = 0


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

        if bulk_mode:
            ui.label(
                f"Bulk run on {len(checked_scans)} checked scan"
                f"{'s' if len(checked_scans) != 1 else ''}."
            ).classes("text-sm font-mono opacity-70")
            if opts.hrf_model == MODEL_TOEPLITZ:
                # Toeplitz bulk needs per-scan estimated HRFs, but
                # ``state.montage`` is single-valued. The bulk worker
                # will skip scans whose montage_source_scan doesn't
                # match -- in practice that's every scan except whichever
                # one the user last ran HRFs on. Canonical bulk works
                # universally.
                ui.label(
                    "Toeplitz mode in bulk only succeeds on the scan "
                    "whose HRFs are currently in memory; other scans "
                    "will be skipped. Switch to canonical for an "
                    "every-scan deconvolution."
                ).classes("text-xs opacity-60 italic")
        elif scan is not None:
            ui.label(scan.display_name or scan.path.name).classes(
                "text-sm font-mono opacity-70"
            )

        # ── Mode + parameter controls
        with ui.card().classes("w-full"):
            ui.label("Estimation").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            _render_model_radio(state, opts)
            _render_lmbda_slider(opts)

            if opts.hrf_model == MODEL_TOEPLITZ and scan is not None:
                _render_toeplitz_requirements(state, scan)

        # ── Run row + progress + errors
        _render_run_row(state, scan, checked_scans, opts)

        # ── Preview (single-scan only -- bulk overwrites activity_raw)
        if state.activity_raw is not None and not bulk_mode:
            ui.separator()
            ui.label("Deconvolved preview").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            _render_preview(state, scan, opts)


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


def _render_model_radio(state: AppState, opts: ActivityOptions) -> None:
    def _set(value: str) -> None:
        opts.hrf_model = value
        # Trigger a body re-render so toeplitz-requirements text appears/
        # disappears with the selection.
        state.publish("scan_selected", state.selected_scan)

    ui.radio(
        [MODEL_TOEPLITZ, MODEL_CANONICAL],
        value=opts.hrf_model,
        on_change=lambda e: _set(e.value),
    ).props("inline")


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
        log_val = int(event.value)
        opts.lmbda = float(10 ** log_val)
        lmbda_display.set_text(f"lambda = {opts.lmbda:.0e}")

    ui.slider(
        min=LOG_LMBDA_MIN,
        max=LOG_LMBDA_MAX,
        step=1,
        value=initial_log,
        on_change=_on_change,
    )


def _render_toeplitz_requirements(state: AppState, scan: ScanEntry) -> None:
    """Surface the toeplitz-mode dependency on a real estimated Montage.

    Three blocking conditions:
    - No Montage yet (HRFs tab not run).
    - Montage is a _CanonicalResult (HRFs tab last ran in canonical mode,
      which doesn't produce per-channel traces toeplitz needs).
    - Montage was produced for a different scan than the one currently
      selected (Sprint 3.4 review: applying scan A's HRFs to scan B's
      Raw silently produces wrong results because the library matches
      by channel name, not scan identity).
    """
    montage = state.montage
    if montage is None:
        ui.label(
            "Toeplitz mode requires estimated HRFs from this scan. Run the "
            "HRFs tab in toeplitz mode first."
        ).classes("text-sm opacity-70")
    elif isinstance(montage, _CanonicalResult):
        ui.label(
            "Toeplitz mode requires real per-channel HRFs, but the HRFs tab "
            "last produced a canonical reference shape. Re-run the HRFs tab "
            "in toeplitz mode or switch the model selector below to "
            "canonical."
        ).classes("text-sm opacity-70")
    elif (
        state.montage_source_scan is None
        or state.montage_source_scan.path != scan.path
    ):
        source_label = (
            state.montage_source_scan.display_name
            or state.montage_source_scan.path.name
            if state.montage_source_scan is not None
            else "another scan"
        )
        ui.label(
            f"The current HRFs were estimated from {source_label}, not this "
            f"scan. Re-run the HRFs tab on this scan in toeplitz mode, or "
            f"switch the model selector below to canonical."
        ).classes("text-sm opacity-70")
    else:
        ui.label(
            "Using HRFs estimated in the HRFs tab."
        ).classes("text-sm opacity-60")


def _render_run_row(
    state: AppState,
    scan: Optional[ScanEntry],
    checked: List[ScanEntry],
    opts: ActivityOptions,
) -> None:
    """Render the Run button row + progress / error display.

    PR #55a: dispatches single vs bulk based on the checked set. Bulk
    button is enabled whenever there's no in-flight task; per-scan gates
    (raw + montage match) are evaluated inside the worker and incompatible
    scans are skipped.
    """
    bulk_mode = bool(checked)
    if bulk_mode:
        run_label = (
            f"Estimate activity for {len(checked)} scan"
            f"{'s' if len(checked) != 1 else ''}"
        )
        can_run = not state.busy
    else:
        raw_ready = scan is not None and scan in state.processed_cache
        if opts.hrf_model == MODEL_TOEPLITZ:
            montage_ready = state.montage is not None and not isinstance(
                state.montage, _CanonicalResult
            )
            scan_matches = (
                state.montage_source_scan is not None
                and scan is not None
                and state.montage_source_scan.path == scan.path
            )
            can_run = (
                raw_ready and montage_ready and scan_matches
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

        if state.busy:
            _render_busy_progress(state)
        elif not bulk_mode and scan is not None and scan not in state.processed_cache:
            # Activity deconvolution runs on the *preprocessed* signal, so
            # the Run button is disabled until the scan has been through the
            # Preprocess tab. Spell that out and link straight there rather
            # than leaving a disabled button with no explanation. The link
            # goes through the event bus (navigate_preprocess) so this panel
            # doesn't need a reference to the shell's tab control.
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

    snapshot = ActivityOptions(
        hrf_model=opts.hrf_model,
        lmbda=opts.lmbda,
        preview_channel=opts.preview_channel,
    )

    def _build(scan: ScanEntry):
        if scan not in state.processed_cache:
            return None
        if snapshot.hrf_model == MODEL_TOEPLITZ:
            if state.montage is None or isinstance(
                state.montage, _CanonicalResult
            ):
                return None
            if (
                state.montage_source_scan is None
                or state.montage_source_scan.path != scan.path
            ):
                return None
            existing_montage = state.montage
        else:
            existing_montage = None

        raw = state.processed_cache.get(scan)
        progress_cb = make_progress_callback(state)
        return (
            run_activity_sync,
            (raw, snapshot, existing_montage, progress_cb),
            {},
        )

    async def _on_each_done(scan: ScanEntry, result) -> None:
        if result is None:
            return
        state.activity_raw = result
        state.publish("activity_estimated", scan)

    async def _bulk() -> None:
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
            fail_names = ", ".join(
                s.display_name or s.path.name for s, _ in failures[:3]
            )
            if len(failures) > 3:
                fail_names += f" (+{len(failures) - 3} more)"
            summary += f" Failed/skipped: {fail_names}."
        ui.notify(
            summary, type="positive" if n_fail == 0 else "warning"
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

    if opts.hrf_model == MODEL_TOEPLITZ:
        if state.montage is None:
            state.last_error = "Estimate HRFs (toeplitz mode) first."
            return
        if isinstance(state.montage, _CanonicalResult):
            state.last_error = (
                "Toeplitz activity needs real estimated HRFs; the HRFs tab "
                "last produced a canonical reference. Re-run HRFs in "
                "toeplitz mode."
            )
            return
        if (
            state.montage_source_scan is None
            or state.montage_source_scan.path != scan.path
        ):
            state.last_error = (
                "The HRFs in memory were estimated for a different scan. "
                "Re-run the HRFs tab on this scan in toeplitz mode first."
            )
            return

    snapshot = ActivityOptions(
        hrf_model=opts.hrf_model,
        lmbda=opts.lmbda,
        preview_channel=opts.preview_channel,
    )

    raw = state.processed_cache.get(scan)
    progress_cb = make_progress_callback(state)
    existing_montage = (
        state.montage if snapshot.hrf_model == MODEL_TOEPLITZ else None
    )

    async def _on_done(result) -> None:
        if result is None:
            return
        state.activity_raw = result
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

    For canonical mode, ``existing_montage`` is ignored and a fresh Montage
    is configured to the scan.

    Module-level so tests can call it without dispatching through workers.
    """
    from ...hrfunc import montage as Montage

    work_raw = raw.copy()

    snapshot = None
    if opts.hrf_model == MODEL_CANONICAL or existing_montage is None:
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

    try:
        result = m.estimate_activity(
            work_raw,
            lmbda=opts.lmbda,
            hrf_model=opts.hrf_model,
            preprocess=False,
            progress_callback=progress_callback,
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
