"""Inspect panel — scan metadata + recording details.

Ported from the legacy ``pages.workspace`` Inspect tab when v1.4
reintroduced scan inspection into the single-shell GUI. The panel
renders against the current ``state.selected_scan`` and subscribes to
``scan_selected`` + ``scan_loaded`` so dataset-tree clicks and
background Raw loads drive re-renders without rebuilding the whole tab.

Two sections:

- **Metadata** — always available from the ScanEntry (format, path,
  channel count, sampling rate, BIDS fields when present).
- **Recording** — renders once the MNE Raw is in ``state.raw_cache``.
  Three expanders: channel list, 2D probe layout PNG, events table.
  While the background load is in flight, shows a spinner placeholder.

Pure render helpers (``render_probe_png``, ``render_recording_sections``)
stay module-level so tests can call them with a synthetic Raw without
spinning up the full GUI.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING, Optional

from nicegui import ui

from ..state import AppState

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


def render(state: AppState) -> None:
    """Mount the Inspect panel against the given AppState.

    Wraps the body in a ``ui.refreshable`` and subscribes the refresher
    to ``scan_selected`` + ``scan_loaded`` so the panel updates whenever
    the dataset-tree changes the selection or a background Raw load
    completes. Subscribers stay alive across tab switches in the
    single-shell GUI (per Phase 1's no-clear-on-render contract); they
    only clear on project switch.
    """

    @ui.refreshable
    def _body() -> None:
        _render_body(state)

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)


def _render_body(state: AppState) -> None:
    """Render the Inspect body against the current ``state.selected_scan``.

    No-scan path → prompt to pick one. Scan-but-not-cached → spinner.
    Scan-cached → metadata + recording sections. Cache-clear-race →
    "Recording unavailable — cache entry was cleared" (defensive: the
    cache could be cleared between the ``__contains__`` check and the
    ``get()`` call by another callback).

    Module-level so tests can call it inside a synthetic NiceGUI
    context without going through the refreshable wrapper.
    """
    scan = state.selected_scan
    if scan is None:
        with ui.column().classes("p-6 gap-2"):
            ui.label("Inspect").classes("text-2xl font-semibold")
            ui.label("Select a scan from the dataset tree.").classes(
                "text-sm opacity-60"
            )
        return

    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label(scan.display_name or scan.path.name).classes(
            "text-2xl font-semibold"
        )

        # Metadata (always available from ScanEntry)
        ui.label("Metadata").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        _kv_row("Format", scan.format)
        _kv_row("Path", str(scan.path))
        if scan.n_channels is not None:
            _kv_row("Channels", str(scan.n_channels))
        if scan.sfreq is not None:
            _kv_row("Sampling rate", f"{scan.sfreq:.4g} Hz")
        if scan.bids_subject:
            _kv_row("BIDS subject", scan.bids_subject)
        if scan.bids_session:
            _kv_row("BIDS session", scan.bids_session)
        if scan.bids_task:
            _kv_row("BIDS task", scan.bids_task)
        if scan.bids_run:
            _kv_row("BIDS run", scan.bids_run)

        ui.separator()

        # Recording (loaded MNE Raw)
        ui.label("Recording").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        if scan in state.raw_cache:
            try:
                raw = state.raw_cache.get(scan)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Inspect: cache.get raised: %s", exc)
                ui.label(
                    "Recording unavailable — cache entry was cleared."
                ).classes("text-sm opacity-60")
                return
            render_recording_sections(raw)
        else:
            with ui.row().classes("items-center gap-3"):
                ui.spinner(size="sm")
                ui.label("Loading recording…").classes(
                    "text-sm opacity-70"
                )


def render_recording_sections(raw: "mne.io.BaseRaw") -> None:
    """Render channel list + probe layout + events for a loaded Raw."""
    ch_names = list(raw.ch_names)
    annotations = raw.annotations if raw.annotations is not None else []

    # Channel list (collapsed by default; channel counts dominate the
    # vertical real estate on dense montages otherwise).
    with ui.expansion(
        f"Channels ({len(ch_names)})",
        icon="sensors",
    ).classes("w-full"):
        with ui.column().classes(
            "max-h-64 overflow-auto gap-1 text-xs font-mono"
        ):
            for name in ch_names:
                ui.label(name).classes("opacity-80")

    # 2D probe layout — matplotlib via base64 PNG. ui.matplotlib would
    # also work, but PNG keeps the snapshot purely declarative and
    # avoids holding a Figure across the NiceGUI re-render cycle.
    with ui.expansion("Probe layout", icon="scatter_plot").classes(
        "w-full"
    ):
        probe_html = render_probe_png(raw)
        if probe_html is None:
            ui.label("Probe layout unavailable for this scan.").classes(
                "text-sm opacity-60"
            )
        else:
            ui.image(probe_html).classes("max-w-md")

    # Events / annotations
    n_events = len(annotations)
    with ui.expansion(
        f"Events ({n_events})", icon="event"
    ).classes("w-full"):
        if n_events == 0:
            ui.label("No events recorded in this scan.").classes(
                "text-sm opacity-60"
            )
        else:
            rows = [
                {
                    # Unique per-row key: two annotations at the same onset
                    # (common with simultaneous multi-condition markers) would
                    # collapse to one row under row_key="onset" since Quasar
                    # de-duplicates by key.
                    "_idx": i,
                    "description": str(ann["description"]),
                    "onset": f"{float(ann['onset']):.3f}",
                    "duration": f"{float(ann['duration']):.3f}",
                }
                for i, ann in enumerate(annotations)
            ]
            ui.table(
                columns=[
                    {
                        "name": "description",
                        "label": "Description",
                        "field": "description",
                        "align": "left",
                    },
                    {
                        "name": "onset",
                        "label": "Onset (s)",
                        "field": "onset",
                        "align": "right",
                    },
                    {
                        "name": "duration",
                        "label": "Duration (s)",
                        "field": "duration",
                        "align": "right",
                    },
                ],
                rows=rows,
                row_key="_idx",
            ).classes("w-full")


def render_probe_png(raw: "mne.io.BaseRaw") -> Optional[str]:
    """Render the probe layout to a base64-encoded PNG data URL.

    Returns None if MNE refuses to plot (no montage, no sensor positions,
    or any matplotlib failure). Callers are expected to fall back to a
    placeholder label in that case.
    """
    try:
        # Lazy imports — matplotlib import time is noticeable at GUI startup,
        # and this function is only called on a successful Raw load.
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for probe layout: %s", exc)
        return None

    fig = None
    try:
        fig = raw.plot_sensors(show=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("raw.plot_sensors failed: %s", exc)
        if fig is not None:
            plt.close(fig)
        return None

    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("probe layout encode failed: %s", exc)
        return None
    finally:
        plt.close(fig)


def _kv_row(key: str, value: str) -> None:
    with ui.row().classes("w-full gap-4"):
        ui.label(key).classes("text-xs uppercase opacity-60 w-32")
        ui.label(value).classes("text-sm break-all")
