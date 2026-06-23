"""Export tab — save analysis artifacts to disk.

Sprint 5 final piece. Each section of the panel corresponds to one
artifact the workspace can produce and exposes a button that opens an
OS save dialog (or directory picker for multi-file exports) and writes
the data. Buttons disable themselves when the source artifact isn't
in state (e.g. "Export montage HRFs" is dim until estimate_hrf has
run).

Exporters
---------

- **Processed Raw (SNIRF/FIF)** — `state.processed_cache[selected_scan]
  .save(path)`. Output of the Preprocess tab.
- **Activity Raw (SNIRF/FIF)** — `state.activity_raw.save(path)`. Output
  of the Activity tab.
- **Montage HRFs (JSON)** — `state.montage.save(path)`. The estimated
  HRFs from the HRFs tab in their canonical on-disk form (the same
  format that `tree.load_hrfs` consumes).
- **HRF plot PNGs (folder)** — writes one PNG per channel to a chosen
  directory. Re-uses the gallery's mini-plot renderer for consistency.
- **Quality metrics (CSV)** — flattens `state.quality_metrics` to one
  row per (scan, stage) with the SNR/skew/kurtosis/SCI columns.

File dialogs use pywebview's SAVE_DIALOG / FOLDER_DIALOG via
`app.native.main_window`. Tests can monkeypatch ``_pick_save_path`` /
``_pick_folder_path`` so they don't try to open real dialogs.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io as _io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

from nicegui import app, ui

from ..state import AppState
from .hrf_panel import _CanonicalResult

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Panel render
# ---------------------------------------------------------------------------


def render(state: AppState) -> None:
    """Render the Export tab inside the current NiceGUI context."""

    @ui.refreshable
    def _body() -> None:
        _render_body(state)

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)
    state.subscribe("preprocess_done", _refresh)
    state.subscribe("hrf_estimated", _refresh)
    state.subscribe("activity_estimated", _refresh)
    state.subscribe("quality_computed", _refresh)


def _render_body(state: AppState) -> None:
    """Render the Export tab body against current state."""
    scan = state.selected_scan
    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("Export").classes("text-2xl font-semibold")
        ui.label(
            "Save analysis artifacts produced by the workspace. Each row "
            "is enabled when its source data is available — preprocess "
            "first, estimate HRFs, run Activity, or run Quality to unlock "
            "the corresponding exports."
        ).classes("text-sm opacity-70")

        if scan is None:
            ui.label("Select a scan from the dataset tree.").classes(
                "text-sm opacity-60"
            )
            return

        ui.label(scan.display_name or scan.path.name).classes(
            "text-sm font-mono opacity-70"
        )

        # ── Processed Raw
        _render_row(
            "Processed Raw",
            "Output of the Preprocess tab. Saves as SNIRF (.snirf) or FIF "
            "(.fif) — extension on the chosen filename decides the format.",
            available=(scan in state.processed_cache),
            unavailable_hint="Run the Preprocess tab first.",
            button_label="Save processed scan…",
            on_click=lambda: _save_processed(state, scan),
        )

        # ── Activity Raw. Gate on the activity result belonging to the
        # selected scan: activity_raw is a single global slot not cleared on
        # scan change, so without this the row would offer to save scan A's
        # deconvolved data under a "<scanB>_activity.snirf" name with a
        # dropped-channel count computed against scan B's baseline. Gating on
        # the match means `scan` here IS the source scan, so the filename and
        # source-count in _save_activity are correct by construction.
        activity_matches = (
            state.activity_raw is not None
            and state.activity_source_scan is not None
            and state.activity_source_scan.path == scan.path
        )
        _render_row(
            "Activity Raw",
            "Deconvolved neural-activity output of the Activity tab. Saves "
            "as SNIRF or FIF.",
            available=activity_matches,
            unavailable_hint=(
                "Run the Activity tab on this scan first."
                if state.activity_raw is None
                else "The in-memory activity result is from another scan — "
                "re-run the Activity tab on this scan."
            ),
            button_label="Save activity scan…",
            on_click=lambda: _save_activity(state, scan),
        )

        # ── Montage HRFs
        toeplitz_montage_ready = (
            state.montage is not None
            and not isinstance(state.montage, _CanonicalResult)
        )
        _render_row(
            "Montage HRFs (JSON)",
            "Per-channel estimated HRFs and their context, in the same "
            "JSON format that `hrfunc.tree.load_hrfs` consumes.",
            available=toeplitz_montage_ready,
            unavailable_hint=(
                "Run the HRFs tab in toeplitz mode first (canonical mode "
                "produces a reference shape, not estimated HRFs)."
            ),
            button_label="Save montage HRFs…",
            on_click=lambda: _save_montage(state, scan),
        )

        # ── HRF plot PNGs
        _render_row(
            "HRF plots (PNG folder)",
            "One PNG per channel — same renderer the HRFs-tab gallery "
            "uses. Saved to a chosen directory.",
            available=toeplitz_montage_ready,
            unavailable_hint="Run the HRFs tab in toeplitz mode first.",
            button_label="Save HRF plots…",
            on_click=lambda: _save_hrf_plots(state, scan),
        )

        # ── Quality metrics CSV
        quality_ready = bool(state.quality_metrics)
        _render_row(
            "Quality metrics (CSV)",
            "Flat CSV of the metrics computed by the Quality tab — one "
            "row per (scan, stage) with SNR / skewness / kurtosis / SCI "
            "columns.",
            available=quality_ready,
            unavailable_hint="Compute metrics in the Quality tab first.",
            button_label="Save quality CSV…",
            on_click=lambda: _save_quality_csv(state),
        )

        if state.last_error:
            with ui.row().classes("items-center gap-2"):
                ui.icon("error_outline").classes("text-red-400")
                ui.label(state.last_error).classes("text-sm text-red-400")

        # ── Submit to HRtree (desktop counterpart of hrfunc-web's
        # /hrf_upload form). Renders the metadata-collection panel
        # inside a card so it visually matches the other export
        # rows. The file picker inside the panel defaults to the
        # most recently saved montage path so save -> submit is a
        # two-click flow.
        ui.separator()
        with ui.card().classes("w-full"):
            from ..submission import render_submission_panel
            render_submission_panel(
                state,
                default_path=state.last_saved_roi_path,
            )


def _render_row(
    title: str,
    description: str,
    available: bool,
    unavailable_hint: str,
    button_label: str,
    on_click,
) -> None:
    """Render one export-row card."""
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("flex-grow gap-1"):
                ui.label(title).classes(
                    "text-sm uppercase tracking-wide"
                )
                ui.label(description).classes("text-xs opacity-70")
                if not available:
                    ui.label(unavailable_hint).classes(
                        "text-xs opacity-60 italic"
                    )
            with ui.column().classes("items-end gap-1"):
                ui.button(
                    button_label, on_click=on_click,
                ).props(f"color=primary {'disable' if not available else ''}")


# ---------------------------------------------------------------------------
# Click handlers (async because they await the file dialog)
# ---------------------------------------------------------------------------


async def _save_processed(state: AppState, scan) -> None:
    if scan not in state.processed_cache:
        state.last_error = "No preprocessed scan available."
        return
    path = await _pick_save_path(
        suggested=f"{scan.path.stem}_processed.snirf",
        title="Save processed scan",
    )
    if path is None:
        return
    raw = state.processed_cache.get(scan)
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, _save_raw, raw, path
        )
        ui.notify(f"Saved processed scan: {path.name}", type="positive")
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"Save failed: {type(exc).__name__}: {exc}"
        logger.exception("processed save failed: %s", exc)
        ui.notify(state.last_error, type="negative")


async def _save_activity(state: AppState, scan, naming=None) -> None:
    if state.activity_raw is None:
        state.last_error = "No activity scan available."
        return
    postfix = (naming or {}).get("postfix", "_deconvolved")
    ext = (naming or {}).get("ext", ".snirf")
    path = await _pick_save_path(
        suggested=f"{scan.path.stem}{postfix}{ext}",
        title="Save activity scan",
    )
    if path is None:
        return
    # estimate_activity drops channels whose lstsq solve failed
    # (hrfunc.py:599 + 606-611). Surface the count so the user knows
    # if the exported file has fewer channels than the source.
    saved_count = len(state.activity_raw.ch_names)
    source_count = (
        len(state.processed_cache.get(scan).ch_names)
        if scan in state.processed_cache
        else saved_count
    )
    dropped = source_count - saved_count
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, _save_raw, state.activity_raw, path
        )
        notice = f"Saved activity scan: {path.name}"
        if dropped > 0:
            notice += f" ({dropped} channel(s) dropped during deconvolution)"
        ui.notify(notice, type="positive")
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"Save failed: {type(exc).__name__}: {exc}"
        logger.exception("activity save failed: %s", exc)
        ui.notify(state.last_error, type="negative")


async def _save_montage(state: AppState, scan) -> None:
    montage = state.montage
    if montage is None or isinstance(montage, _CanonicalResult):
        state.last_error = "No estimated montage to save."
        return
    path = await _pick_save_path(
        suggested=f"{scan.path.stem}_hrfs.json",
        title="Save montage HRFs",
    )
    if path is None:
        return
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, save_montage_sync, montage, path
        )
        ui.notify(f"Saved montage: {path.name}", type="positive")
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"Save failed: {type(exc).__name__}: {exc}"
        logger.exception("montage save failed: %s", exc)
        ui.notify(state.last_error, type="negative")


async def _save_hrf_plots(state: AppState, scan) -> None:
    montage = state.montage
    if montage is None or isinstance(montage, _CanonicalResult):
        state.last_error = "No estimated montage to save."
        return
    folder = await _pick_folder_path(title="Save HRF plots to folder")
    if folder is None:
        return
    try:
        count = await asyncio.get_event_loop().run_in_executor(
            None, save_hrf_plots_sync, montage, folder, scan.path.stem
        )
        ui.notify(
            f"Saved {count} HRF plots to {folder.name}/", type="positive"
        )
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"Save failed: {type(exc).__name__}: {exc}"
        logger.exception("HRF plots save failed: %s", exc)
        ui.notify(state.last_error, type="negative")


async def _save_quality_csv(state: AppState) -> None:
    if not state.quality_metrics:
        state.last_error = "No quality metrics computed yet."
        return
    path = await _pick_save_path(
        suggested="quality_metrics.csv",
        title="Save quality metrics CSV",
    )
    if path is None:
        return
    try:
        rows = await asyncio.get_event_loop().run_in_executor(
            None, save_quality_csv_sync, state.quality_metrics, path
        )
        ui.notify(
            f"Saved {rows} rows to {path.name}", type="positive"
        )
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"Save failed: {type(exc).__name__}: {exc}"
        logger.exception("quality CSV save failed: %s", exc)
        ui.notify(state.last_error, type="negative")


# ---------------------------------------------------------------------------
# Sync workers (module-level so tests can call without async dispatch)
# ---------------------------------------------------------------------------


def _save_raw(raw: "mne.io.BaseRaw", path: Path) -> None:
    """Save an MNE Raw to SNIRF or FIF based on file extension.

    Core MNE has no SNIRF writer (``mne.export.export_raw`` only supports
    bdf / brainvision / edf / eeglab, and ``read_raw_snirf`` is read-only),
    so SNIRF is written via ``mne_nirs.io.write_raw_snirf`` (h5py opens the
    file in "w" mode, so it overwrites). FIF uses the Raw's own ``.save()``.
    Dispatch on extension.
    """
    raw.load_data()
    suffix = path.suffix.lower()
    if suffix == ".snirf":
        try:
            from mne_nirs.io import write_raw_snirf
        except Exception as exc:  # noqa: BLE001 — mne-nirs not installed
            raise RuntimeError(
                "SNIRF export needs the 'mne-nirs' package "
                f"({type(exc).__name__}: {exc}). Install mne-nirs or save as "
                ".fif instead."
            ) from exc
        write_raw_snirf(raw, str(path))
    elif suffix == ".fif":
        raw.save(str(path), overwrite=True, verbose="ERROR")
    else:
        raise ValueError(
            f"Unsupported extension {suffix!r}; use .snirf or .fif"
        )


def save_montage_sync(montage, path: Path) -> None:
    """Save a Montage's estimated HRFs to JSON via montage.save()."""
    montage.save(str(path))


def save_hrf_plots_sync(montage, folder: Path, prefix: str) -> int:
    """Write one PNG per channel into ``folder`` and return the count.

    Uses the same matplotlib renderer as the HRFs-tab gallery detail
    panel (full-size with std shading) so the saved plots match what the
    user previewed in the GUI.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"matplotlib unavailable: {exc}") from exc

    folder.mkdir(parents=True, exist_ok=True)
    channels = getattr(montage, "channels", {})
    count = 0
    for ch_name, node in channels.items():
        trace = getattr(node, "trace", None)
        if trace is None or len(trace) == 0:
            continue
        sfreq = float(getattr(node, "sfreq", 1.0) or 1.0)
        if sfreq <= 0:
            sfreq = 1.0
        t = np.arange(len(trace)) / sfreq
        std = getattr(node, "trace_std", None)

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        try:
            ax.plot(t, trace, lw=1.3, color="#6366f1", label=ch_name)
            if std is not None and len(std) == len(trace):
                arr = np.asarray(trace)
                std_arr = np.asarray(std)
                ax.fill_between(
                    t, arr - std_arr, arr + std_arr,
                    alpha=0.18, color="#6366f1", label="±1 std",
                )
            ax.set_xlabel("time (s)")
            ax.set_ylabel("amplitude (a.u.)")
            ax.set_title(f"{prefix}: {ch_name}")
            ax.legend(fontsize=8, loc="upper right")
            fig.tight_layout()
            safe_name = _safe_filename(ch_name)
            out_path = folder / f"{prefix}_{safe_name}.png"
            fig.savefig(out_path, dpi=110, bbox_inches="tight")
            count += 1
        finally:
            plt.close(fig)
    return count


def save_quality_csv_sync(quality_metrics: dict, path: Path) -> int:
    """Flatten ``state.quality_metrics`` to one row per (scan, stage).

    Returns the number of rows written (header excluded).
    """
    columns = [
        "scan_path",
        "stage",
        "n_channels",
        "snr_mean",
        "skew_mean",
        "kurtosis_mean",
        "sci_mean",
    ]
    rows: List[List[Any]] = []
    for scan_path, stages in quality_metrics.items():
        for stage_name, metrics in stages.items():
            rows.append([
                str(scan_path),
                stage_name,
                _attr(metrics, "n_channels", 0),
                _attr(metrics, "snr_mean", ""),
                _attr(metrics, "skew_mean", ""),
                _attr(metrics, "kurtosis_mean", ""),
                _attr(metrics, "sci_mean", ""),
            ])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows)


def _attr(obj, name: str, fallback: Any) -> Any:
    """Pull a possibly-missing attribute from a QualityMetrics dataclass-or-dict."""
    if obj is None:
        return fallback
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return fallback if value is None else value


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a cross-platform filename.

    Replaces any character that's not alphanumeric, dash, underscore, or
    dot with an underscore. Windows forbids ``\\ / : * ? " < > |`` and
    treats trailing dots/spaces specially; this conservative whitelist
    handles all of those without per-platform branching. Also collapses
    runs of underscores and strips leading/trailing dots so we don't
    accidentally produce hidden files on Unix.
    """
    sanitized = "".join(
        c if (c.isalnum() or c in "-_.") else "_" for c in name
    )
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("._") or "channel"


# ---------------------------------------------------------------------------
# File dialog helpers
# ---------------------------------------------------------------------------


async def _pick_save_path(suggested: str, title: str) -> Optional[Path]:
    """Open the OS save-file dialog. Returns None on cancel."""
    try:
        import webview
    except ImportError:
        ui.notify("pywebview not installed", type="negative")
        return None

    window = getattr(app, "native", None) and app.native.main_window
    if window is None:
        ui.notify(
            "Save dialog requires native window mode. "
            "Launch with `hrfunc` (not via browser).",
            type="warning",
        )
        return None

    # pywebview ≥5 returns a coroutine here — see welcome.py:_pick_folder
    # for the equivalent fix. Await the coroutine when present; older
    # versions return synchronously.
    import inspect

    from .._webview_compat import dialog_kind
    result = window.create_file_dialog(
        dialog_kind(webview, "SAVE"),
        save_filename=suggested,
    )
    if inspect.isawaitable(result):
        result = await result
    if not result:
        return None
    # pywebview returns a string for SAVE_DIALOG and a list for OPEN/FOLDER
    if isinstance(result, (list, tuple)):
        result = result[0]
    return Path(result)


async def _pick_folder_path(title: str) -> Optional[Path]:
    """Open the OS folder picker. Returns None on cancel."""
    try:
        import webview
    except ImportError:
        ui.notify("pywebview not installed", type="negative")
        return None

    window = getattr(app, "native", None) and app.native.main_window
    if window is None:
        ui.notify(
            "Folder picker requires native window mode.",
            type="warning",
        )
        return None

    import inspect

    from .._webview_compat import dialog_kind
    result = window.create_file_dialog(dialog_kind(webview, "FOLDER"))
    paths = await result if inspect.isawaitable(result) else result
    if not paths:
        return None
    return Path(paths[0])
