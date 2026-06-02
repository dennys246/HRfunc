"""Quality tab — signal-quality metrics for the lens-module pipeline.

Sprint 4.1 wraps the existing `hrfunc.observer.lens` algorithms in a GUI
that researchers can use to QC scans before publication. Two views:

- **Per-scan** (default): metrics for the currently-selected scan in
  whichever stages are cached. ``raw_cache`` provides the source signal
  for SCI; ``processed_cache`` provides preprocessed signal for
  skewness/kurtosis/SNR; ``activity_raw`` (when the source scan matches
  the selected scan) provides deconvolved signal for the same triplet.
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

    Each field is a scalar averaged across channels. None for metrics that
    weren't computable on the given stage (e.g. SCI is only meaningful on
    raw fNIRS data in optical-amplitude units; it returns None for the
    preprocessed/deconvolved stages).
    """

    snr_mean: Optional[float] = None
    skew_mean: Optional[float] = None
    kurtosis_mean: Optional[float] = None
    sci_mean: Optional[float] = None
    n_channels: int = 0


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
            _render_metrics_table(cached)
            # A cached entry computed BEFORE Activity ran holds only the
            # raw + preprocessed stages; nothing invalidates it when Activity
            # later produces a deconvolved Raw for this scan. Offer an
            # explicit recompute (the panel otherwise only shows the compute
            # button on the empty branch) so the deconvolved row can be added
            # without a full project reset.
            deconv_available = (
                STAGE_DECONVOLVED not in cached
                and state.activity_raw is not None
                and state.montage_source_scan is not None
                and state.montage_source_scan.path == scan.path
            )
            with ui.row().classes("items-center gap-3 mt-2"):
                recompute_disabled = (
                    state.busy or scan not in state.processed_cache
                )
                ui.button(
                    "Recompute metrics",
                    icon="refresh",
                    on_click=lambda: _run_per_scan(state, scan),
                ).props(f"flat dense {'disable' if recompute_disabled else ''}")
                if deconv_available:
                    ui.label(
                        "Activity result available — recompute to add the "
                        "deconvolved row."
                    ).classes("text-sm opacity-60")


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
    if (
        state.activity_raw is not None
        and state.montage_source_scan is not None
        and state.montage_source_scan.path == scan.path
    ):
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
    n_ch = data.shape[0]

    # axis=1 → per-channel summaries. lens.calc_skewness_and_kurtosis
    # accidentally uses axis=0 (over channels at a given time-point), but
    # the GUI wants per-channel-then-averaged statistics, which is what
    # the per-channel comparison plots downstream actually need.
    skew_vals = skew(data, axis=1)
    kurt_vals = kurtosis(data, axis=1)

    snr_mean = _compute_snr_safe(raw)

    return QualityMetrics(
        snr_mean=snr_mean,
        skew_mean=float(np.nanmean(skew_vals)),
        kurtosis_mean=float(np.nanmean(kurt_vals)),
        sci_mean=None,
        n_channels=n_ch,
    )


def _compute_raw_metrics(raw: "mne.io.BaseRaw") -> QualityMetrics:
    """Compute SCI + signal metrics on raw fNIRS data.

    SCI requires optical-amplitude data (cw_amplitude channel types in
    MNE). If the data isn't suitable, SCI returns None and the rest of
    the metrics still populate.
    """
    sci_mean = _compute_sci_safe(raw)
    base = _compute_signal_metrics(raw)
    return QualityMetrics(
        snr_mean=base.snr_mean,
        skew_mean=base.skew_mean,
        kurtosis_mean=base.kurtosis_mean,
        sci_mean=sci_mean,
        n_channels=base.n_channels,
    )


def _compute_sci_safe(raw: "mne.io.BaseRaw") -> Optional[float]:
    """SCI mean across channels, None on failure.

    SCI is only defined on raw fNIRS cw_amplitude data. Wrapped in
    try/except because passing the wrong channel type raises ValueError
    in MNE and we want the rest of the metrics to render anyway.
    """
    try:
        import mne

        raw_copy = raw.copy().load_data()
        od = mne.preprocessing.nirs.optical_density(raw_copy, verbose="ERROR")
        sci = mne.preprocessing.nirs.scalp_coupling_index(od, verbose="ERROR")
        return float(np.nanmean(sci))
    except Exception as exc:  # noqa: BLE001
        logger.debug("SCI computation skipped: %s", exc)
        return None


def _compute_snr_safe(raw: "mne.io.BaseRaw") -> Optional[float]:
    """PSD-based SNR per lens.calc_snr, averaged across channels.

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
        return float(np.nanmean(snr_per_channel))
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
