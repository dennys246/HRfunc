"""Project picker — the upper-right dropdown for the v1.4 single-shell GUI.

Pulls together the three project-selection affordances that the legacy
welcome page exposed as cards:

- **Open folder...** — OS folder dialog via pywebview.
- **Recent** — submenu listing manifests previously cached under
  ``platformdirs.user_cache_dir("hrfunc")``.
- **Close project** — clears ``state.manifest`` and fires
  ``project_changed`` so dependent tabs blank their refreshables.

The closed-state shows the active project's name (or "No project") and
opens a Quasar ``ui.menu`` on click. Wired into the shell's toolbar in
Phase 2; for now the dropdown is testable in isolation.

Three helpers are exported for direct reuse (the legacy welcome page
imports them so we don't ship a parallel implementation):

- :func:`pick_folder` — async, the OS folder dialog.
- :func:`list_recent_manifests` — sync, recent-manifest enumeration.
- :func:`open_project_path` — async, scan + ``set_manifest`` + notify.

Phase 3 will gate Open/Close on ``state.busy`` with a tooltip; for now
the menu is always enabled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from nicegui import app, ui

from ..state import AppState
from ..workers import run_in_background
from ...io.manifest import Manifest
from ...io.scan import scan_folder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — shared with the legacy welcome page in Phase 1
# ---------------------------------------------------------------------------


async def pick_folder() -> Optional[Path]:
    """Open the native OS folder picker.

    Returns the chosen Path, or None on cancel / non-native launch. Wraps
    pywebview's :meth:`create_file_dialog` with the v1.3 picklable-enum
    shim (see ``gui._webview_compat``) and the v1.3 pywebview-≥5 async
    return-type handling.
    """
    try:
        import webview
    except ImportError:
        ui.notify("pywebview not installed", type="negative")
        return None

    window = getattr(app, "native", None) and app.native.main_window
    if window is None:
        ui.notify(
            "Folder picker requires native window mode. "
            "Launch with `hrfunc` (not via browser).",
            type="warning",
        )
        return None

    # pywebview ≥5 returns a coroutine; pywebview 6 deprecated the
    # module-level FOLDER_DIALOG constant in favor of FileDialog.FOLDER.
    # Both quirks are isolated in gui._webview_compat.dialog_kind.
    import inspect

    from .._webview_compat import dialog_kind
    result = window.create_file_dialog(dialog_kind(webview, "FOLDER"))
    if inspect.isawaitable(result):
        paths = await result
    else:
        paths = result
    if not paths:
        return None
    return Path(paths[0])


async def pick_file(
    *,
    file_types: Optional[List[str]] = None,
) -> Optional[Path]:
    """Open the native OS file-open picker.

    PR #57 helper for the Cluster sub-tab's "Add montage" button, which
    needs to read an MNE-compatible fNIRS file (SNIRF / NIRX header /
    FIF) and turn its channels into per-channel sphere ROIs. Mirrors
    :func:`pick_folder` exactly -- same pywebview-≥5 async return-type
    handling and same picklable-enum shim -- but uses the OPEN dialog
    kind so the user picks a file instead of a folder.

    ``file_types``: optional list of pywebview file-type filter strings
    (e.g. ``["SNIRF files (*.snirf)", "All files (*.*)"]``). Ignored on
    non-native launches.

    Returns the chosen Path, or None on cancel / non-native launch.
    """
    try:
        import webview
    except ImportError:
        ui.notify("pywebview not installed", type="negative")
        return None

    window = getattr(app, "native", None) and app.native.main_window
    if window is None:
        ui.notify(
            "File picker requires native window mode. "
            "Launch with `hrfunc` (not via browser).",
            type="warning",
        )
        return None

    import inspect

    from .._webview_compat import dialog_kind
    kwargs = {}
    if file_types:
        # pywebview validates each filter string against
        # ``description (*.ext1;*.ext2)`` -- semicolon-separated, NOT
        # space-separated. Catch the malformed shape here so callers
        # see the failing string rather than a stack trace from the
        # multiprocessing feeder thread (which is hard to spot
        # without a launching terminal).
        try:
            from webview.util import parse_file_type
            for ft in file_types:
                parse_file_type(ft)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "pick_file: invalid file_types entry: %s", exc
            )
            ui.notify(
                f"Invalid file filter -- contact the developer "
                f"({type(exc).__name__}: {exc})",
                type="negative",
            )
            return None
        kwargs["file_types"] = tuple(file_types)
    result = window.create_file_dialog(
        dialog_kind(webview, "OPEN"), **kwargs
    )
    if inspect.isawaitable(result):
        paths = await result
    else:
        paths = result
    if not paths:
        return None
    return Path(paths[0])


def list_recent_manifests(limit: int = 10) -> List[Manifest]:
    """Enumerate cached manifests from the XDG cache directory.

    Reads every ``manifest_*.json`` in ``platformdirs.user_cache_dir``,
    deserializes via :meth:`Manifest.from_json`, sorts by ``scanned_at``
    descending, and returns the top ``limit``. Corrupt files are silently
    skipped — same fail-safe contract as the scanner cache.
    """
    try:
        import platformdirs
    except ImportError:
        return []

    cache_dir = Path(platformdirs.user_cache_dir("hrfunc"))
    if not cache_dir.exists():
        return []

    manifests: List[Manifest] = []
    for cache_file in cache_dir.glob("manifest_*.json"):
        try:
            m = Manifest.from_json(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — fail-safe per scan.py contract
            logger.debug("Skipping recent-manifest %s: %s", cache_file, exc)
            continue
        manifests.append(m)

    manifests.sort(key=lambda m: m.scanned_at, reverse=True)
    return manifests[:limit]


async def open_project_path(state: AppState, path: Path) -> None:
    """Scan ``path`` and install the resulting manifest as the active project.

    On success: ``state.set_manifest(manifest)`` (which fires
    ``project_changed`` for dependent tabs). On failure: a UI notification
    surfaces ``state.last_error``. No navigation — the single-shell GUI
    expects the active tab to react to ``project_changed`` rather than
    being routed away.

    Files are wrapped to their parent folder for now; a future revision
    can refine the single-file flow.
    """

    async def _on_done(result: Optional[Manifest]) -> None:
        if result is None:
            ui.notify(f"Scan failed: {state.last_error}", type="negative")
            return
        state.set_manifest(result)

    if not path.exists():
        ui.notify(f"Path does not exist: {path}", type="negative")
        return
    if path.is_file():
        path = path.parent

    await run_in_background(state, scan_folder, path, on_done=_on_done)


# ---------------------------------------------------------------------------
# Dropdown UI — the v1.4 toolbar widget
# ---------------------------------------------------------------------------


def render(state: AppState) -> None:
    """Render the project picker dropdown for the shell toolbar.

    Closed-state shows ``Project: <name>`` or ``No project``. Opening
    the dropdown reveals:

    - Open folder...
    - ─────────
    - Recent  (submenu, present only when there's at least one entry)
    - ─────────
    - Close project  (present only when a project is loaded)

    The label is wrapped in a :func:`ui.refreshable` body so swapping
    ``state.manifest`` via the dropdown actions updates the display
    immediately — same mechanism the shell uses to react to
    ``project_changed`` events from any source (CLI preload, recent
    pick, etc.).
    """

    @ui.refreshable
    def _label_body() -> None:
        manifest = state.manifest
        if manifest is None:
            text = "No project"
        else:
            text = f"Project: {manifest.root.name}"
        ui.label(text).classes("text-sm font-medium")

    @ui.refreshable
    def _menu_body() -> None:
        """Render the dropdown menu items.

        Open / Close are disabled while ``state.busy`` is True — switching
        projects mid-estimate would silently land the result on the new
        project (workers capture state by reference; cooperative
        cancellation is v1.4+ scope). The disable carries a tooltip
        explaining the gate. Recent items stay enabled because picking a
        cached manifest is a project SWITCH whose hazard is the same
        as Open's, so they're treated the same way: disabled while busy.
        """
        busy = state.busy
        busy_tooltip = (
            "Finish or wait for the running task before switching projects."
        )

        open_item = ui.menu_item(
            "Open folder...",
            on_click=lambda: _on_open_folder(state, menu),
        )
        if busy:
            open_item.props("disable")
            open_item.tooltip(busy_tooltip)

        recent = list_recent_manifests()
        if recent:
            ui.separator()
            ui.label("Recent").classes(
                "text-xs uppercase opacity-50 px-3 pt-2"
            )
            for m in recent:
                recent_item = ui.menu_item(
                    f"{m.root.name}  ·  "
                    f"{m.scanned_at.strftime('%Y-%m-%d %H:%M')}",
                    on_click=lambda m=m: _on_pick_recent(state, m, menu),
                )
                if busy:
                    recent_item.props("disable")
                    recent_item.tooltip(busy_tooltip)

        if state.manifest is not None:
            ui.separator()

            # NOTE: pass the coroutine function directly (NOT a sync lambda
            # wrapping it) so NiceGUI awaits the confirm dialog -- a sync
            # lambda returning a coroutine silently no-ops.
            async def _close_clicked() -> None:
                await _on_close_project(state, menu)

            close_item = ui.menu_item(
                "Close project",
                on_click=_close_clicked,
            )
            if busy:
                close_item.props("disable")
                close_item.tooltip(busy_tooltip)

    def _refresh_label(_payload=None) -> None:
        _label_body.refresh()

    def _refresh_menu(_payload=None) -> None:
        _menu_body.refresh()

    # Subscribe before rendering so the very first event after mount
    # lands on the live refreshable (rather than racing with the
    # initial render).
    state.subscribe("project_changed", _refresh_label)
    # project_changed also affects the menu (Close item appears/hides).
    state.subscribe("project_changed", _refresh_menu)
    state.subscribe("busy_changed", _refresh_menu)

    with ui.button(icon="folder_open").props("flat color=primary"):
        _label_body()
        with ui.menu() as menu:
            _menu_body()


async def _on_open_folder(state: AppState, menu) -> None:
    menu.close()
    path = await pick_folder()
    if path is None:
        return
    await open_project_path(state, path)


def _on_pick_recent(state: AppState, manifest: Manifest, menu) -> None:
    """Load a previously-scanned manifest from the recent list.

    Sync because the manifest is already deserialized — no scan needed.
    Sets the manifest via :meth:`AppState.set_manifest` so
    ``project_changed`` subscribers blank/refresh their views.
    """
    menu.close()
    state.set_manifest(manifest)


async def _on_close_project(state: AppState, menu) -> None:
    """Clear the active project without touching cached library data.

    Uses :meth:`AppState.set_manifest` rather than :meth:`reset` so the
    bundled HRtree library and event subscribers survive — closing a
    project shouldn't kick the user out of the Library tab.

    Confirms first when closing would discard in-memory results: estimated
    HRFs (``montage_cache``) and deconvolutions (``activity_cache``) are not
    auto-persisted, and ``set_manifest(None)`` clears them.
    """
    menu.close()
    has_unsaved = bool(state.montage_cache) or len(state.activity_cache) > 0
    if has_unsaved:
        with ui.dialog() as dlg, ui.card():
            ui.label("Close project?").classes("text-lg font-bold")
            ui.label(
                "This discards the estimated HRFs and deconvolutions held in "
                "memory for this project — they are not auto-saved. Save "
                "anything you want to keep (HRFs / activity / montage) first."
            ).classes("text-sm")
            with ui.row().classes("justify-end w-full"):
                ui.button(
                    "Cancel", on_click=lambda: dlg.submit(False)
                ).props("flat")
                ui.button(
                    "Close anyway", on_click=lambda: dlg.submit(True)
                ).props("color=negative")
        if not await dlg:
            return
    state.set_manifest(None)
