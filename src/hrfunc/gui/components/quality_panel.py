"""Quality tab — signal-quality metrics for the lens-module pipeline.

Sprint 4.1 wraps the existing `hrfunc.observer.lens` algorithms in a GUI
that researchers can use to QC scans before publication. Two views:

- **Per-scan** (default): metrics for the currently-selected scan in
  whichever stages are cached. ``raw_cache`` provides the source signal
  for SCI; ``processed_cache`` provides preprocessed signal for
  skewness/kurtosis/SNR/variance; ``activity_cache`` (per scan) provides
  the deconvolved signal for the same set. Beyond the stage-summary table,
  a **channel-wise** table compares each channel across the three stages
  (raw → hemoglobin → activity) with Δ (activity − hemoglobin) and ratio
  (activity / hemoglobin) columns — the QC outcome for how much
  deconvolution changed each channel. The raw column is mapped to a
  haemoglobin channel by its source-detector prefix (raw wavelength
  channels share the S-D pair).
- **Dataset-wide aggregate**: a "Run on all scans" button kicks off a
  background task that walks every scan in the manifest, loading +
  preprocessing each (using library defaults) and computing the same
  metric set. Results aggregate into bar charts so a researcher can
  spot outliers across their study.

The panel does not call `lens.compare_subject` directly because that
function writes plots to disk. Instead the metric algorithms are
inlined here (mirroring scipy.stats.skew / kurtosis, MNE
scalp_coupling_index, and the PSD-based SNR estimator) so the same
numbers ship into the GUI without disk side effects.

Cache protection: dataset-wide runs use the existing raw_cache /
processed_cache helpers, which respect LRU(3) eviction — running over a
large manifest will only retain the last 3 raws in memory at any time.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from nicegui import background_tasks, ui

from ..state import AppState
from ..workers import run_in_background
from ...io.manifest import ScanEntry

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


STAGE_RAW = "raw"
STAGE_PREPROCESSED = "preprocessed"
STAGE_DECONVOLVED = "deconvolved"


@dataclass
class QualityMetrics:
    """Per-stage signal-quality summary.

    The ``*_mean`` fields are scalars averaged across channels (used by the
    dataset-wide aggregate plot and the stage-summary table). The
    ``*_by_channel`` dicts map channel name -> value and drive the
    channel-wise 3-stage table. None for metrics that weren't computable on
    the given stage (e.g. SCI is only meaningful on raw fNIRS data in
    optical-amplitude units; it is None for the preprocessed/deconvolved
    stages).
    """

    snr_mean: Optional[float] = None
    skew_mean: Optional[float] = None
    kurtosis_mean: Optional[float] = None
    variance_mean: Optional[float] = None
    sci_mean: Optional[float] = None
    n_channels: int = 0

    # Per-channel breakdowns (channel name -> value). None when not computed.
    snr_by_channel: Optional[Dict[str, float]] = None
    skew_by_channel: Optional[Dict[str, float]] = None
    kurtosis_by_channel: Optional[Dict[str, float]] = None
    variance_by_channel: Optional[Dict[str, float]] = None
    sci_by_channel: Optional[Dict[str, float]] = None


def render(state: AppState) -> None:
    """Render the Quality tab inside the current NiceGUI context."""
    @ui.refreshable
    def _body() -> None:
        _render_body(state)

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)
    state.subscribe("preprocess_done", _refresh)
    state.subscribe("activity_estimated", _refresh)
    state.subscribe("quality_computed", _refresh)

    def _poll_progress() -> None:
        if state.busy and state.estimation_progress is not None:
            _body.refresh()

    ui.timer(0.5, _poll_progress)


def _render_body(state: AppState) -> None:
    """Render the Quality tab body against current state.

    Module-level so tests can call it directly inside a synthetic NiceGUI
    context without going through the refreshable wrapper.
    """
    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("Quality").classes("text-2xl font-semibold")

        scan = state.selected_scan
        if scan is None and not state.quality_metrics:
            ui.label("Select a scan from the dataset tree.").classes(
                "text-sm opacity-60"
            )
            return

        # ── Per-scan view (top half) — only renders if a scan is selected
        if scan is not None:
            _render_per_scan(state, scan)

        # ── Dataset-wide aggregate (bottom half) — always available when
        # a manifest is loaded
        if state.manifest is not None and len(state.manifest.scans) > 1:
            ui.separator()
            _render_dataset_aggregate(state)


def _render_per_scan(state: AppState, scan: ScanEntry) -> None:
    """Per-scan metrics card."""
    with ui.card().classes("w-full"):
        ui.label("Current scan").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        ui.label(scan.display_name or scan.path.name).classes(
            "text-sm font-mono opacity-70"
        )

        path_key = scan.path.resolve()
        cached = state.quality_metrics.get(path_key)

        if cached is None:
            # Three preconditions for computing per-scan:
            # - raw_cache has the source (for SCI)
            # - processed_cache has the preprocessed Raw
            # - (optional) activity_raw + montage_source_scan matches for
            #   deconvolved metrics
            with ui.row().classes("items-center gap-3"):
                run_disabled = state.busy or scan not in state.processed_cache
                ui.button(
                    "Compute metrics for this scan",
                    on_click=lambda: _run_per_scan(state, scan),
                ).props(f"color=primary {'disable' if run_disabled else ''}")
                if scan not in state.processed_cache:
                    ui.label(
                        "Preprocess the scan first (Preprocess tab)."
                    ).classes("text-sm opacity-60")
                elif state.busy:
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm")
                        ui.label("Working…").classes("text-sm opacity-70")
        else:
            # Offer a recompute so users can refresh after deconvolving (the
            # cached metrics may predate the activity stage).
            with ui.row().classes("items-center gap-2"):
                ui.button(
                    "Recompute",
                    icon="refresh",
                    on_click=lambda: _run_per_scan(state, scan),
                ).props(f"flat dense {'disable' if state.busy else ''}")
                if (
                    STAGE_DECONVOLVED not in cached
                    and scan in state.activity_cache
                ):
                    ui.label(
                        "Activity available — recompute to include it."
                    ).classes("text-xs opacity-60")
            _render_metrics_table(cached)
            ui.separator()
            _render_channel_wise(state, scan, cached)


# Metrics exposed in the channel-wise table, in display order.
# (key on QualityMetrics, label). SCI is raw-only and handled separately.
_CHANNEL_METRICS = (
    ("snr_by_channel", "SNR"),
    ("variance_by_channel", "Variance"),
    ("skew_by_channel", "Skewness"),
    ("kurtosis_by_channel", "Kurtosis"),
)


def _sd_prefix(ch_name: str) -> str:
    """Source-detector prefix of a channel name ("S1_D1 hbo" -> "S1_D1").

    Used to map a haemoglobin channel back to its raw wavelength channels,
    which share the S-D pair but differ by the trailing chromophore /
    wavelength token.
    """
    return ch_name.rsplit(" ", 1)[0] if " " in ch_name else ch_name


def _raw_value_for(
    raw_by_channel: Optional[Dict[str, float]], hemo_ch: str
) -> Optional[float]:
    """Raw-stage value for a haemoglobin channel, averaged over the raw
    wavelength channels that share its source-detector pair."""
    if not raw_by_channel:
        return None
    prefix = _sd_prefix(hemo_ch)
    vals = [
        v for ch, v in raw_by_channel.items()
        if _sd_prefix(ch) == prefix and v is not None
    ]
    if not vals:
        return None
    return float(np.nanmean(vals))


def _render_channel_wise(
    state: AppState, scan: ScanEntry, stages_metrics: Dict[str, QualityMetrics]
) -> None:
    """Channel-wise QC across raw / hemoglobin / activity stages.

    Rows are keyed by the haemoglobin (preprocessed) channel names — the
    hemoglobin and activity stages align 1:1 there, so the Δ (activity −
    hemoglobin) and ratio (activity / hemoglobin) columns are exact. The raw
    column is mapped by source-detector prefix (raw wavelength channels share
    the S-D pair), shown as a reference. One tab per metric.
    """
    hemo = stages_metrics.get(STAGE_PREPROCESSED)
    raw_m = stages_metrics.get(STAGE_RAW)
    act = stages_metrics.get(STAGE_DECONVOLVED)

    ui.label("Channel-wise QC (raw → hemoglobin → activity)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )

    if hemo is None or not hemo.snr_by_channel and not hemo.variance_by_channel:
        ui.label(
            "Preprocess this scan to populate the channel-wise table."
        ).classes("text-sm opacity-60")
        return

    if act is None:
        with ui.row().classes("items-center gap-2"):
            ui.icon("info").classes("text-amber-500")
            ui.label(
                "Deconvolve this scan on the Activity tab to fill the "
                "activity / Δ / ratio columns."
            ).classes("text-sm opacity-70")

    # Channel order from the hemoglobin stage (the alignment key).
    channels = sorted(
        (hemo.variance_by_channel or hemo.snr_by_channel or {}).keys()
    )
    if not channels:
        ui.label("No channels to display.").classes("text-sm opacity-60")
        return

    with ui.tabs().props("dense").classes("w-full") as metric_tabs:
        for _key, label in _CHANNEL_METRICS:
            ui.tab(label)
    first_label = _CHANNEL_METRICS[0][1]
    with ui.tab_panels(metric_tabs, value=first_label).classes("w-full"):
        for key, label in _CHANNEL_METRICS:
            with ui.tab_panel(label).classes("p-0"):
                png = _render_channel_bar_png(
                    key, label, channels, raw_m, hemo, act
                )
                if png is not None:
                    ui.image(png).classes("w-full max-w-4xl")
                _render_channel_metric_table(key, channels, raw_m, hemo, act)

    # SCI is raw-only; surface it as a compact reference table when present.
    if raw_m is not None and raw_m.sci_by_channel:
        ui.label("Scalp coupling index (raw only)").classes(
            "text-xs uppercase opacity-50 tracking-wide pt-2"
        )
        sci_rows = [
            {"channel": ch, "sci": _format_metric(v)}
            for ch, v in sorted(raw_m.sci_by_channel.items())
        ]
        ui.table(
            columns=[
                {"name": "channel", "label": "Channel", "field": "channel", "align": "left"},
                {"name": "sci", "label": "SCI", "field": "sci", "align": "right"},
            ],
            rows=sci_rows,
            row_key="channel",
        ).classes("w-full").props("dense flat")


def _render_channel_metric_table(
    metric_key: str,
    channels: List[str],
    raw_m: Optional[QualityMetrics],
    hemo: QualityMetrics,
    act: Optional[QualityMetrics],
) -> None:
    """One metric's channel table: raw | hemoglobin | activity | Δ | ratio."""
    raw_by = getattr(raw_m, metric_key) if raw_m is not None else None
    hemo_by = getattr(hemo, metric_key) or {}
    act_by = getattr(act, metric_key) if act is not None else None

    rows = []
    for ch in channels:
        hemo_v = hemo_by.get(ch)
        act_v = act_by.get(ch) if act_by else None
        raw_v = _raw_value_for(raw_by, ch)

        delta = (
            act_v - hemo_v
            if (act_v is not None and hemo_v is not None) else None
        )
        ratio = (
            act_v / hemo_v
            if (act_v is not None and hemo_v not in (None, 0)) else None
        )
        rows.append(
            {
                "channel": ch,
                "raw": _format_metric(raw_v),
                "hemo": _format_metric(hemo_v),
                "activity": _format_metric(act_v),
                "delta": _format_metric(delta),
                "ratio": _format_metric(ratio),
            }
        )

    ui.table(
        columns=[
            {"name": "channel", "label": "Channel", "field": "channel", "align": "left", "sortable": True},
            {"name": "raw", "label": "Raw", "field": "raw", "align": "right", "sortable": True},
            {"name": "hemo", "label": "Hemoglobin", "field": "hemo", "align": "right", "sortable": True},
            {"name": "activity", "label": "Activity", "field": "activity", "align": "right", "sortable": True},
            {"name": "delta", "label": "Δ (act−hemo)", "field": "delta", "align": "right", "sortable": True},
            {"name": "ratio", "label": "Ratio (act/hemo)", "field": "ratio", "align": "right", "sortable": True},
        ],
        rows=rows,
        row_key="channel",
        pagination=0,
    ).classes("w-full").props("dense flat")


def _render_channel_bar_png(
    metric_key: str,
    label: str,
    channels: List[str],
    raw_m: Optional[QualityMetrics],
    hemo: QualityMetrics,
    act: Optional[QualityMetrics],
) -> Optional[str]:
    """Grouped bar chart of one metric per channel across the stages.

    One cluster of bars per channel — raw / hemoglobin / activity — so the
    per-channel quality and the hemoglobin → activity change are visible at a
    glance. Stages with no data (e.g. activity before deconvolution) are
    omitted. Returns a base64 PNG data URI, or None if nothing to plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for channel bars: %s", exc)
        return None

    raw_by = getattr(raw_m, metric_key) if raw_m is not None else None
    hemo_by = getattr(hemo, metric_key) or {}
    act_by = getattr(act, metric_key) if act is not None else None

    # One series per available stage (skip a stage with no usable values).
    # Fixed colors: raw=orange, hemoglobin=red, neural activity=blue.
    stage_colors = {
        "Raw": "#ff7f0e",          # orange
        "Hemoglobin": "#d62728",   # red
        "Activity": "#1f77b4",     # blue
    }
    series: List[Tuple[str, List[float]]] = []
    raw_vals = [_raw_value_for(raw_by, ch) for ch in channels]
    hemo_vals = [hemo_by.get(ch) for ch in channels]
    act_vals = [act_by.get(ch) if act_by else None for ch in channels]
    for name, vals in (
        ("Raw", raw_vals),
        ("Hemoglobin", hemo_vals),
        ("Activity", act_vals),
    ):
        if any(v is not None for v in vals):
            series.append(
                (name, [v if v is not None else np.nan for v in vals])
            )
    if not series:
        return None

    fig = None
    try:
        n = len(channels)
        n_series = len(series)
        x = np.arange(n)
        # Bars share each channel's slot; width scales with series count.
        width = 0.8 / n_series
        fig, ax = plt.subplots(1, 1, figsize=(max(8, n * 0.5), 4))
        for i, (name, vals) in enumerate(series):
            offset = (i - (n_series - 1) / 2.0) * width
            ax.bar(
                x + offset, vals, width, label=name,
                color=stage_colors.get(name),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"Per-channel {label} by stage")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("channel bar render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def _render_metrics_table(stages_metrics: Dict[str, QualityMetrics]) -> None:
    """Render a 4-row table comparing metrics across the cached stages."""
    rows = []
    for stage_name in (STAGE_RAW, STAGE_PREPROCESSED, STAGE_DECONVOLVED):
        m = stages_metrics.get(stage_name)
        if m is None:
            continue
        rows.append(
            {
                "stage": stage_name,
                "snr": _format_metric(m.snr_mean),
                "variance": _format_metric(m.variance_mean),
                "skew": _format_metric(m.skew_mean),
                "kurtosis": _format_metric(m.kurtosis_mean),
                "sci": _format_metric(m.sci_mean),
                "channels": str(m.n_channels),
            }
        )

    ui.table(
        columns=[
            {"name": "stage", "label": "Stage", "field": "stage", "align": "left"},
            {"name": "snr", "label": "SNR", "field": "snr", "align": "right"},
            {"name": "variance", "label": "Variance", "field": "variance", "align": "right"},
            {"name": "skew", "label": "Skewness", "field": "skew", "align": "right"},
            {"name": "kurtosis", "label": "Kurtosis", "field": "kurtosis", "align": "right"},
            {"name": "sci", "label": "SCI", "field": "sci", "align": "right"},
            {"name": "channels", "label": "Channels", "field": "channels", "align": "right"},
        ],
        rows=rows,
        row_key="stage",
    ).classes("w-full")


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _render_dataset_aggregate(state: AppState) -> None:
    """Dataset-wide aggregate card."""
    with ui.card().classes("w-full"):
        ui.label("Dataset-wide aggregate").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        n_scans = len(state.manifest.scans) if state.manifest else 0
        ui.label(
            f"Run metrics over all {n_scans} scans in the project. "
            "Each scan is loaded, preprocessed with library defaults, "
            "and summarized. Re-running overwrites cached results."
        ).classes("text-sm opacity-70")

        # Only enable when not busy
        can_run = not state.busy and n_scans > 0
        with ui.row().classes("items-center gap-3"):
            ui.button(
                "Run on all scans",
                on_click=lambda: _run_dataset(state),
            ).props(f"color=primary {'disable' if not can_run else ''}")
            if state.busy:
                prog = state.estimation_progress
                if prog is not None:
                    current, total, name = prog
                    fraction = (current + 1) / max(total, 1)
                    with ui.column().classes("gap-1 flex-grow"):
                        ui.label(
                            f"Scan {current + 1}/{total}: {name}"
                        ).classes("text-xs opacity-70")
                        ui.linear_progress(value=fraction).classes("w-64")
                else:
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm")
                        ui.label("Working…").classes("text-sm opacity-70")

        # If we have metrics for ≥2 scans, render the aggregate plot
        scans_with_metrics = sum(
            1 for v in state.quality_metrics.values()
            if STAGE_PREPROCESSED in v
        )
        if scans_with_metrics >= 2:
            png = _render_aggregate_png(state.quality_metrics)
            if png is not None:
                ui.image(png).classes("max-w-3xl")

    if state.last_error and not state.busy:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error_outline").classes("text-red-400")
            ui.label(state.last_error).classes("text-sm text-red-400")


# ---------------------------------------------------------------------------
# Click handlers
# ---------------------------------------------------------------------------


def _run_per_scan(state: AppState, scan: ScanEntry) -> None:
    """Compute metrics for the single selected scan."""
    if state.busy:
        return
    if scan not in state.processed_cache:
        state.last_error = "Preprocess the scan first."
        return

    raw_obj = state.raw_cache.get(scan) if scan in state.raw_cache else None
    processed_obj = state.processed_cache.get(scan)
    deconvolved_obj = None
    if scan in state.activity_cache:
        # Per-scan deconvolution cache (any scan the user has run).
        deconvolved_obj = state.activity_cache.get(scan)
    elif (
        state.activity_raw is not None
        and state.montage_source_scan is not None
        and state.montage_source_scan.path == scan.path
    ):
        # Fallback to the single most-recent result for the matching scan.
        deconvolved_obj = state.activity_raw

    async def _on_done(result) -> None:
        if result is None:
            return
        state.quality_metrics[scan.path.resolve()] = result
        state.publish("quality_computed", scan)

    background_tasks.create(
        run_in_background(
            state,
            compute_per_scan_sync,
            raw_obj,
            processed_obj,
            deconvolved_obj,
            on_done=_on_done,
        )
    )


def _run_dataset(state: AppState) -> None:
    """Walk the manifest, computing metrics for every scan."""
    if state.busy or state.manifest is None:
        return

    scans = list(state.manifest.scans)
    if not scans:
        return

    def _progress_callback(i: int, total: int, name: str) -> None:
        state.estimation_progress = (i, total, name)

    async def _on_done(results) -> None:
        # compute_dataset_sync now returns a (metrics_dict, failed_names)
        # tuple so the panel can surface a "loaded N/M; failed: ..." message.
        if results is None:
            return
        metrics, failed = results
        if metrics:
            state.quality_metrics.update(metrics)
        if failed:
            state.last_error = (
                f"Quality run: {len(metrics)}/{len(scans)} scans succeeded; "
                f"{len(failed)} failed: {', '.join(failed[:5])}"
                + (" …" if len(failed) > 5 else "")
            )
        state.publish("quality_computed", None)

    background_tasks.create(
        run_in_background(
            state,
            compute_dataset_sync,
            state.raw_cache,
            state.processed_cache,
            scans,
            _progress_callback,
            on_done=_on_done,
        )
    )


# ---------------------------------------------------------------------------
# Sync metric workers (module-level so tests can call them directly)
# ---------------------------------------------------------------------------


def compute_per_scan_sync(
    raw_obj: Optional["mne.io.BaseRaw"],
    processed_obj: Optional["mne.io.BaseRaw"],
    deconvolved_obj: Optional["mne.io.BaseRaw"],
) -> Optional[Dict[str, QualityMetrics]]:
    """Compute the per-stage metrics dict for one scan.

    Returns a dict keyed by stage name (raw / preprocessed / deconvolved)
    containing :class:`QualityMetrics`. Stages with no input Raw are
    omitted. Returns None if every input is None (nothing to compute).
    """
    out: Dict[str, QualityMetrics] = {}
    if raw_obj is not None:
        out[STAGE_RAW] = _compute_raw_metrics(raw_obj)
    if processed_obj is not None:
        out[STAGE_PREPROCESSED] = _compute_signal_metrics(processed_obj)
    if deconvolved_obj is not None:
        out[STAGE_DECONVOLVED] = _compute_signal_metrics(deconvolved_obj)
    return out or None


def compute_dataset_sync(
    raw_cache,
    processed_cache,
    scans: List[ScanEntry],
    progress_callback=None,
):
    """Walk the manifest, loading + preprocessing each scan and computing
    metrics on raw + preprocessed stages.

    Skips scans that fail to load/preprocess rather than aborting the
    whole run. ``progress_callback(i, n, scan_name)`` fires once per
    scan.

    Returns a 2-tuple ``(results, failed_names)`` where ``results`` is a
    dict keyed by ``scan.path.resolve()`` of stage-name metrics dicts and
    ``failed_names`` is a list of display names of scans that failed to
    load. The 2-tuple is None only when no scans succeeded AND no scans
    were attempted (i.e. empty input).
    """
    from ...hrfunc import preprocess_fnirs

    results: Dict[Path, Dict[str, QualityMetrics]] = {}
    failed_names: List[str] = []
    total = len(scans)
    for i, scan in enumerate(scans):
        if progress_callback is not None:
            progress_callback(
                i, total, scan.display_name or scan.path.name
            )
        try:
            raw_obj = raw_cache.get(scan)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "quality dataset run: failed to load %s — %s", scan.path, exc
            )
            failed_names.append(scan.display_name or scan.path.name)
            continue

        if scan in processed_cache:
            processed_obj = processed_cache.get(scan)
        else:
            try:
                processed_obj = preprocess_fnirs(raw_obj)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "quality dataset run: preprocess failed for %s — %s",
                    scan.path, exc,
                )
                processed_obj = None
            else:
                if processed_obj is not None:
                    # put() enforces the LRU bound in one place (was an inline
                    # direct write + manual popitem loop here).
                    processed_cache.put(scan, processed_obj)

        stages: Dict[str, QualityMetrics] = {}
        stages[STAGE_RAW] = _compute_raw_metrics(raw_obj)
        if processed_obj is not None:
            stages[STAGE_PREPROCESSED] = _compute_signal_metrics(processed_obj)
        results[scan.path.resolve()] = stages

    if not results and not failed_names:
        return None
    return results, failed_names


# ---------------------------------------------------------------------------
# Metric primitives (lens-algorithm parity)
# ---------------------------------------------------------------------------


def _compute_signal_metrics(raw: "mne.io.BaseRaw") -> QualityMetrics:
    """Compute SNR / skewness / kurtosis on a Raw assumed to be in a
    band-pass-meaningful state (preprocessed haemoglobin or deconvolved).

    SCI is left None because it only makes sense on raw cw-amplitude data.

    **Scientific note on the deconvolved stage:** the SNR estimator uses
    bands tuned for the haemodynamic-response timescale (0.03-0.1 Hz signal,
    0-0.01 and 0.1-0.5 Hz noise). On a deconvolved neural-activity signal,
    these bands are no longer aligned with the underlying physiology, so
    the *absolute* SNR number is not directly comparable to the
    preprocessed-stage SNR. The relative comparison across channels /
    scans within the deconvolved stage is still meaningful (a noisy
    channel will still rank high-noise relative to its peers).
    """
    try:
        from scipy.stats import kurtosis, skew
    except Exception as exc:  # noqa: BLE001
        logger.warning("scipy unavailable for skew/kurtosis: %s", exc)
        return QualityMetrics()

    raw.load_data()
    data = raw.get_data()
    ch_names = list(raw.ch_names)
    n_ch = data.shape[0]

    # axis=1 → per-channel summaries. lens.calc_skewness_and_kurtosis
    # accidentally uses axis=0 (over channels at a given time-point), but
    # the GUI wants per-channel-then-averaged statistics, which is what
    # the per-channel comparison plots downstream actually need.
    skew_vals = skew(data, axis=1)
    kurt_vals = kurtosis(data, axis=1)
    var_vals = np.var(data, axis=1)

    skew_by = {ch: float(v) for ch, v in zip(ch_names, skew_vals)}
    kurt_by = {ch: float(v) for ch, v in zip(ch_names, kurt_vals)}
    var_by = {ch: float(v) for ch, v in zip(ch_names, var_vals)}
    snr_by = _compute_snr_by_channel(raw)

    return QualityMetrics(
        snr_mean=_nanmean_or_none(snr_by),
        skew_mean=float(np.nanmean(skew_vals)),
        kurtosis_mean=float(np.nanmean(kurt_vals)),
        variance_mean=float(np.nanmean(var_vals)),
        sci_mean=None,
        n_channels=n_ch,
        snr_by_channel=snr_by,
        skew_by_channel=skew_by,
        kurtosis_by_channel=kurt_by,
        variance_by_channel=var_by,
    )


def _nanmean_or_none(by_channel: Optional[Dict[str, float]]) -> Optional[float]:
    """Mean of a by-channel dict's values, or None when absent/empty.

    Drops both ``None`` and ``NaN`` values: a channel can yield a real NaN
    metric (e.g. a flat/zero-variance channel), and ``np.nanmean`` over a
    list that is *all* NaN both emits a RuntimeWarning and returns NaN rather
    than the None this is meant to produce. Filtering NaN first means an
    all-NaN (or all-None) input cleanly returns None.
    """
    if not by_channel:
        return None
    vals = [
        v for v in by_channel.values()
        if v is not None and not np.isnan(v)
    ]
    if not vals:
        return None
    return float(np.mean(vals))


def _compute_raw_metrics(raw: "mne.io.BaseRaw") -> QualityMetrics:
    """Compute SCI + signal metrics on raw fNIRS data.

    SCI requires optical-amplitude data (cw_amplitude channel types in
    MNE). If the data isn't suitable, SCI returns None and the rest of
    the metrics still populate.
    """
    sci_by = _compute_sci_by_channel(raw)
    base = _compute_signal_metrics(raw)
    return QualityMetrics(
        snr_mean=base.snr_mean,
        skew_mean=base.skew_mean,
        kurtosis_mean=base.kurtosis_mean,
        variance_mean=base.variance_mean,
        sci_mean=_nanmean_or_none(sci_by),
        n_channels=base.n_channels,
        snr_by_channel=base.snr_by_channel,
        skew_by_channel=base.skew_by_channel,
        kurtosis_by_channel=base.kurtosis_by_channel,
        variance_by_channel=base.variance_by_channel,
        sci_by_channel=sci_by,
    )


def _compute_sci_by_channel(
    raw: "mne.io.BaseRaw",
) -> Optional[Dict[str, float]]:
    """Per-channel SCI (channel name -> value), None on failure.

    SCI is only defined on raw fNIRS cw_amplitude data. Wrapped in
    try/except because passing the wrong channel type raises ValueError
    in MNE and we want the rest of the metrics to render anyway. Keyed by the
    optical-density channel names (the raw wavelength channels).
    """
    try:
        import mne

        raw_copy = raw.copy().load_data()
        od = mne.preprocessing.nirs.optical_density(raw_copy, verbose="ERROR")
        sci = mne.preprocessing.nirs.scalp_coupling_index(od, verbose="ERROR")
        return {ch: float(v) for ch, v in zip(od.ch_names, sci)}
    except Exception as exc:  # noqa: BLE001
        logger.debug("SCI computation skipped: %s", exc)
        return None


def _compute_snr_by_channel(
    raw: "mne.io.BaseRaw",
) -> Optional[Dict[str, float]]:
    """PSD-based per-channel SNR per lens.calc_snr (channel name -> value).

    Returns None on any MNE / scipy failure so the calling table still
    renders the skew/kurtosis numbers.
    """
    try:
        signal_band = (0.03, 0.1)
        noise_bands = [(0.0, 0.01), (0.1, 0.5)]

        # MNE's filter() requires nyquist > each band's upper bound; the
        # signal/noise bands above need sfreq > 1.0. Cheap guard so we
        # don't crash on synthetic test rigs at 0.5 Hz etc.
        if raw.info["sfreq"] <= 1.0:
            return None

        # Bandpass the signal and each noise band on copies so the cached
        # Raw is not mutated.
        sig = raw.copy().filter(
            signal_band[0], signal_band[1],
            fir_design="firwin", verbose="ERROR",
        )
        psd_signal = sig.compute_psd(
            fmin=signal_band[0], fmax=signal_band[1], verbose="ERROR"
        )
        signal_power = psd_signal.get_data().mean(axis=-1)

        noise_powers = []
        for lo, hi in noise_bands:
            n = raw.copy().filter(
                lo if lo > 0 else 0.001, hi,
                fir_design="firwin", verbose="ERROR",
            )
            psd_n = n.compute_psd(
                fmin=lo if lo > 0 else 0.001, fmax=hi, verbose="ERROR"
            )
            noise_powers.append(psd_n.get_data().mean(axis=-1))
        noise_power = np.mean(noise_powers, axis=0)
        snr_per_channel = signal_power / np.where(
            noise_power > 0, noise_power, np.nan
        )
        return {
            ch: float(v) for ch, v in zip(raw.ch_names, snr_per_channel)
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("SNR computation skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Aggregate rendering
# ---------------------------------------------------------------------------


def _render_aggregate_png(
    quality_metrics: Dict[Path, Dict[str, QualityMetrics]],
) -> Optional[str]:
    """Render a bar chart of per-scan SNR / skewness / kurtosis on the
    preprocessed stage.

    Each scan is one cluster of three bars (one per metric). Useful for
    spotting outliers across a study at a glance.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for aggregate: %s", exc)
        return None

    items: List[Tuple[str, QualityMetrics]] = []
    for path, stages in quality_metrics.items():
        prep = stages.get(STAGE_PREPROCESSED)
        if prep is None:
            continue
        items.append((path.stem, prep))

    if not items:
        return None

    fig = None
    try:
        names = [name for name, _ in items]
        snrs = [m.snr_mean if m.snr_mean is not None else np.nan for _, m in items]
        skews = [m.skew_mean if m.skew_mean is not None else np.nan for _, m in items]
        kurts = [m.kurtosis_mean if m.kurtosis_mean is not None else np.nan for _, m in items]

        n = len(items)
        width = 0.27
        x = np.arange(n)
        fig, ax = plt.subplots(1, 1, figsize=(max(8, n * 0.6), 4))
        ax.bar(x - width, snrs, width, label="SNR")
        ax.bar(x, skews, width, label="Skewness")
        ax.bar(x + width, kurts, width, label="Kurtosis")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_title("Dataset-wide preprocessed-stage metrics")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("aggregate render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
