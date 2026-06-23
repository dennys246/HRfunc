"""HRfunc single-shell GUI — the v1.4 root route.

Replaces the Sprint 2.2 three-route layout (welcome at ``/``, workspace
at ``/workspace``, library at ``/library``) with one tabbed shell:

  ┌────────────────────────────────────────────────────────────────────┐
  │ [icon] HR_func_   [Library | Preprocess | … | Export]   [Project ▾] │
  ├────────────────────────────────────────────────────────────────────┤
  │                                                                    │
  │                       active tab content                           │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘

Tabs (left-to-right): **Library**, **Preprocess**, **Estimate**,
**Activity**, **Quality**, **Export**. Library is the soft default
because it has intrinsic content (bundled literature HRFs) and teaches
the domain without requiring data. The other tabs show a centered empty
state when no project is loaded.

Project state:
- ``state.manifest`` is the active project (None when no project loaded).
- The dataset picker dropdown (upper right) drives loads / switches /
  closes; it publishes ``project_changed`` so tab content refreshes.
- CLI preload (``hrfunc <path>``) is consumed here and lands the user on
  the Preprocess tab once the scan completes.

What stays from Sprint 2.3+:
- Existing tab-content modules (``preprocess_panel``, ``hrf_panel``,
  ``activity_panel``, ``quality_panel``, ``export_panel``) embed as-is.
- The dataset tree (``dataset_tree.render``) is rendered on the left
  side of each data-dependent tab so the existing per-scan flow works
  unchanged.
- The legacy "Inspect" tab is dropped in v1.4 (its functionality
  partially folded into the dataset tree); a future revision can
  reintroduce a scan-detail surface if needed.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from nicegui import background_tasks, ui

from ..components import (
    activity_panel,
    brand,
    dataset_picker,
    dataset_tree,
    export_panel,
    hrf_panel,
    hrtree_panel,
    inspect_panel,
    preprocess_panel,
    quality_panel,
)
from ..state import AppState, state as global_state
from ..theme import apply_theme
from ...io.manifest import ScanEntry

logger = logging.getLogger(__name__)


# Tabs in left-to-right order matching the v1.4 plan. HRtree is leftmost
# and the soft-default landing tab; the data-dependent tabs follow in
# rough pipeline order (Preprocess → Estimate → Activity → Quality →
# Export). The legacy "Inspect" tab is intentionally absent.
#
# "HRtree" matches the paper's terminology for the spatial HRF database;
# the tooltip explains the concept for users arriving without paper
# context. The Brand wordmark inside the tab content reinforces it as a
# proper noun (Times New Roman with italicized ``tree`` suffix).
# Inspect sits between HRtree and Preprocess — a passive "look at the
# scan" step before the active "preprocess / estimate / ..." pipeline.
TAB_HRTREE = "HRtree"
TAB_INSPECT = "Inspect"
TAB_PREPROCESS = "Preprocess"
TAB_ESTIMATE = "HRFs"
TAB_ACTIVITY = "Activity"
TAB_QUALITY = "Quality"
TAB_EXPORT = "Export"

TAB_NAMES = (
    TAB_HRTREE,
    TAB_INSPECT,
    TAB_PREPROCESS,
    TAB_ESTIMATE,
    TAB_ACTIVITY,
    TAB_QUALITY,
    TAB_EXPORT,
)

# One-line tooltip shown when hovering each tab. Empty string = no
# tooltip. The HRtree tooltip is the main onboarding hook for users
# who haven't read the paper.
TAB_TOOLTIPS = {
    TAB_HRTREE: (
        "Explore the HRtree — a 3D spatial database of literature HRFs, "
        "filterable by task / DOI / demographics."
    ),
    TAB_INSPECT: "Scan metadata, channel list, probe layout, and events.",
    TAB_PREPROCESS: "Filter and clean the loaded project's fNIRS recordings.",
    TAB_ESTIMATE: "Estimate hemodynamic response functions from task events.",
    TAB_ACTIVITY: "Deconvolve HRFs to recover underlying neural activity.",
    TAB_QUALITY: "Per-scan + project-wide signal quality metrics.",
    TAB_EXPORT: "Export results back as SNIRF / workspace files.",
}


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def register() -> None:
    """Register the shell page handler at ``/``.

    Called by ``app._register_pages()``. Replaces the Sprint 2.2 welcome
    route; the welcome page module is kept around for one phase as a
    safety net (its ``register()`` call is commented out, so its
    ``@ui.page("/")`` decorator never fires).
    """

    @ui.page("/")
    def shell_page() -> None:
        _render(global_state)


def _render(state: AppState) -> None:
    """Render the shell against the given AppState.

    Split from the page handler so tests can call it with a synthetic
    state. Side effects:

    - Consumes ``state.preload_path`` (CLI ``hrfunc <path>``) by
      scheduling a background scan and selecting the Preprocess tab as
      the initial view.
    - Subscribes the shell's body to ``project_changed`` so empty-state
      / panel-content toggles happen automatically on project swap.

    Viewport-height anchor: NiceGUI's default page wrapping (q-layout
    → q-page-container → q-page → .nicegui-content) provides only
    ``min-height: 100vh`` along the chain, never an explicit pixel
    height. That means downstream ``flex-1`` + ``h-full`` resolve
    against an unbounded parent — the chain grows to fit content
    rather than constraining to the viewport, and whichever splitter
    pane has less content shows visual empty space below it. Pinning
    ``.nicegui-content`` to ``h-screen`` gives the flex tree a real
    100vh anchor to chain against. The default ``p-4`` padding on
    ``.nicegui-content`` is also stripped so the toolbar can sit
    flush at the top of the viewport.
    """
    apply_theme()
    ui.query(".nicegui-content").classes(
        "h-screen w-screen overflow-hidden"
    ).style("padding: 0; gap: 0;")

    initial_tab = TAB_HRTREE
    if state.preload_path is not None:
        path = state.preload_path
        state.preload_path = None  # consume so a future re-render is idle
        background_tasks.create(
            dataset_picker.open_project_path(state, path),
            name="shell-cli-preload",
        )
        initial_tab = TAB_PREPROCESS

    tabs_holder: dict = {}
    _render_toolbar(state, tabs_holder)
    tabs = tabs_holder["tabs"]
    _render_tab_panels(state, tabs, initial_tab)

    # Let panels request a tab switch without holding a reference to the
    # ``ui.tabs`` element (and without importing this module, which would be
    # circular). The Activity tab publishes ``navigate_preprocess`` when the
    # selected scan hasn't been preprocessed yet, linking the user straight
    # to the Preprocess tab. Keep the tab-name string solely here.
    state.subscribe(
        "navigate_preprocess", lambda *_: tabs.set_value(TAB_PREPROCESS)
    )
    # The Activity tab publishes ``navigate_estimate`` when the chosen HRF
    # source is "estimated HRFs" but none are in memory yet — linking the
    # user straight to the HRFs tab to estimate them.
    state.subscribe(
        "navigate_estimate", lambda *_: tabs.set_value(TAB_ESTIMATE)
    )
    # The Activity tab publishes ``navigate_hrtree`` when the chosen HRF
    # source is "HRtree HRF" but none is selected — linking the user to the
    # HRtree tab to pick one.
    state.subscribe(
        "navigate_hrtree", lambda *_: tabs.set_value(TAB_HRTREE)
    )


# ---------------------------------------------------------------------------
# Toolbar — wordmark + tabs + project picker
# ---------------------------------------------------------------------------


def _render_toolbar(state: AppState, tabs_holder: dict) -> None:
    """Render the persistent top toolbar.

    Layout: brand wordmark (left) · tabs (center) · project picker (right).
    ``tabs_holder`` is mutated to expose the ``ui.tabs`` instance so the
    caller can bind ``ui.tab_panels`` to it — NiceGUI's tab-panel pattern
    requires the tabs object to thread through.
    """
    with ui.row().classes(
        "w-full items-center justify-between px-6 py-3 "
        "border-b border-slate-800 gap-4"
    ):
        # Left: brand mark — the bundled executable PNG icon (same file
        # that powers the install-shortcut launcher) rather than a
        # Material-icons head-with-gear placeholder, so the in-app
        # branding matches the OS-level shortcut.
        with ui.row().classes("items-center gap-3 shrink-0"):
            icon_path = _app_icon_path()
            if icon_path is not None:
                ui.image(icon_path).style("width: 2rem; height: 2rem;")
            brand.brand("HRfunc", italic_suffix="func", size_rem=1.6)

        # Center: tabs (per-tab tooltip via Quasar's q-tooltip for the
        # onboarding hint — most useful on HRtree where the term is a
        # proper noun some users will see for the first time)
        with ui.tabs().classes("flex-1") as tabs:
            for name in TAB_NAMES:
                tab = ui.tab(name)
                tooltip_text = TAB_TOOLTIPS.get(name, "")
                if tooltip_text:
                    tab.tooltip(tooltip_text)
        tabs_holder["tabs"] = tabs

        # Right: project picker
        with ui.row().classes("items-center gap-2 shrink-0"):
            dataset_picker.render(state)


# ---------------------------------------------------------------------------
# Tab panels
# ---------------------------------------------------------------------------


def _render_tab_panels(
    state: AppState,
    tabs: ui.tabs,
    initial_tab: str,
) -> None:
    """Render the tab-panels container with one panel per tab.

    The Library panel renders unconditionally (it works without a
    project). The data-dependent panels guard on ``state.manifest`` and
    fall back to an empty state when no project is loaded.
    """
    # ``overflow-hidden`` on the tab-panels container prevents
    # page-level scrollbars from appearing when a tab's content is
    # taller than the viewport. Tabs that need internal scrolling
    # (the dataset tree side pane, the detail pane) opt into it
    # with their own ``overflow-auto`` so scrolling stays scoped to
    # the pane the user is reading.
    #
    # ``min-h-0`` is load-bearing: flex items default to
    # ``min-height: auto`` (refusing to shrink below content size),
    # which would push tab-panels taller than the viewport when one
    # tab's content overflows. ``min-h-0`` lets ``flex-1`` actually
    # constrain the height so ``overflow-hidden`` can clip.
    with ui.tab_panels(tabs, value=initial_tab).classes(
        "w-full flex-1 min-h-0 overflow-hidden"
    ):
        # ``h-full p-0`` on each tab_panel makes the inner content fill
        # the tab-panels container vertically (Quasar's default
        # tab_panel pads content and doesn't stretch — the result is
        # panels that collapse to their content's intrinsic height,
        # which leaves the bottom half of the viewport empty and the
        # plotly viz squeezed into the top half).
        with ui.tab_panel(TAB_HRTREE).classes("h-full p-0"):
            _render_hrtree_tab(state)
        with ui.tab_panel(TAB_INSPECT).classes("h-full p-0"):
            _render_data_tab(
                state,
                verb="inspect scan metadata",
                panel_render=inspect_panel.render,
            )
        with ui.tab_panel(TAB_PREPROCESS).classes("h-full p-0"):
            _render_data_tab(
                state,
                verb="preprocess",
                panel_render=preprocess_panel.render,
            )
        with ui.tab_panel(TAB_ESTIMATE).classes("h-full p-0"):
            _render_data_tab(
                state,
                verb="estimate HRFs",
                panel_render=hrf_panel.render,
            )
        with ui.tab_panel(TAB_ACTIVITY).classes("h-full p-0"):
            _render_data_tab(
                state,
                verb="estimate neural activity",
                panel_render=activity_panel.render,
            )
        with ui.tab_panel(TAB_QUALITY).classes("h-full p-0"):
            _render_data_tab(
                state,
                verb="see quality metrics",
                panel_render=quality_panel.render,
            )
        with ui.tab_panel(TAB_EXPORT).classes("h-full p-0"):
            _render_data_tab(
                state,
                verb="export results",
                panel_render=export_panel.render,
            )


def _render_hrtree_tab(state: AppState) -> None:
    """HRtree tab — bundled HRF database explorer, plus first-launch card.

    The HR_tree_ wordmark lives inside the panel above its left-side
    sub-tabs (not here in the shell tab) so the viz + detail columns
    use the full viewport height. When the user has no project loaded
    AND no recent projects in the XDG cache AND has not previously
    dismissed the welcome card, an overlay card explains the app and
    points up-right at the project picker.
    """
    hrtree_panel.render(state)

    if _should_show_welcome_card(state):
        _render_welcome_card()


def _render_data_tab(
    state: AppState,
    *,
    verb: str,
    panel_render: Callable[[AppState], None],
) -> None:
    """Render a data-dependent tab.

    With a project loaded: dataset tree on the left, the existing panel
    module (e.g. ``preprocess_panel.render``) on the right.

    Without a project: a centered empty-state prompting the user to
    select a project. The "Open folder" button opens the same OS picker
    the toolbar dropdown does, so users see one entry point with two
    surfaces.
    """

    @ui.refreshable
    def _body() -> None:
        if state.manifest is None:
            _render_empty_state(state, verb=verb)
            return
        with ui.splitter(value=22, limits=(15, 40)).classes(
            "w-full h-full"
        ) as outer:
            with outer.before:
                with ui.column().classes(
                    "w-full h-full p-3 gap-2 overflow-auto"
                ):
                    ui.label("Project").classes(
                        "text-xs uppercase opacity-60 tracking-wide"
                    )
                    ui.label(str(state.manifest.root)).classes(
                        "text-xs font-mono opacity-70 break-all"
                    )
                    dataset_tree.render(
                        state,
                        on_select_scan=lambda scan: _on_scan_selected(
                            state, scan
                        ),
                    )
            with outer.after:
                panel_render(state)

    _body()

    # Refresh when the project changes so the empty-state / data-loaded
    # branches swap automatically. The subscription survives across
    # tab switches (the shell does not clear subscribers — that's a
    # project-switch operation in Phase 3).
    state.subscribe("project_changed", lambda _m=None: _body.refresh())


def _render_empty_state(state: AppState, *, verb: str) -> None:
    """Render the centered no-project prompt for a data-dependent tab.

    The single "Open folder" button opens the same picker the toolbar
    dropdown uses. The copy is verb-customized per tab so users know
    what they'll be doing once a project is loaded.
    """
    with ui.column().classes(
        "w-full items-center justify-center mt-24 gap-4"
    ):
        ui.icon("folder_off", size="4rem").classes("opacity-40")
        ui.label(f"Select a project to {verb}.").classes(
            "text-xl opacity-80"
        )
        ui.label(
            "Open a folder of fNIRS scans (.snirf / .fif / NIRx) to begin."
        ).classes("text-sm opacity-60")

        async def _on_click() -> None:
            # Gate on busy BEFORE opening the OS dialog. During the very first
            # project scan the empty-state is still showing (manifest is None)
            # while busy is True; without this guard the click opens the
            # folder dialog and then open_project_path -> run_in_background
            # silently refuses (busy), so the click appears to do nothing.
            # The picker dropdown disables Open while busy; this second entry
            # point needs equivalent feedback. Checking here (rather than
            # disabling the button) also avoids the button staying stuck if an
            # initial scan fails — busy clears but no project_changed fires to
            # re-render the empty state.
            if state.busy:
                ui.notify(
                    "A task is still running — wait for it to finish before "
                    "opening a project.",
                    type="warning",
                )
                return
            path = await dataset_picker.pick_folder()
            if path is None:
                return
            await dataset_picker.open_project_path(state, path)

        ui.button(
            "Open folder",
            icon="folder_open",
            on_click=_on_click,
        ).props("color=primary")


def _on_scan_selected(state: AppState, scan: Optional[ScanEntry]) -> None:
    """Publish ``scan_selected`` and kick off a background Raw load if needed.

    Mirrors the workspace page's selection handler so the existing panels
    (which subscribe to ``scan_selected`` and ``scan_loaded``) behave
    unchanged inside the shell.
    """
    state.publish("scan_selected", scan)

    if scan is None or scan in state.raw_cache:
        return
    background_tasks.create(_load_scan_raw(state, scan))


async def _load_scan_raw(state: AppState, scan: ScanEntry) -> None:
    """Load ``scan`` into ``state.raw_cache`` off the event loop.

    Same contract as ``workspace._load_scan_raw``: bypasses the
    ``workers.run_in_background`` busy gate (which is reserved for the
    long estimation tasks) so rapid scan navigation isn't blocked.
    Publishes ``scan_loaded`` only when the user is still on the same
    scan when the load finishes.
    """
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, state.raw_cache.get, scan)
    except Exception as exc:  # noqa: BLE001 — surface to UI via last_error
        state.last_error = (
            f"Failed to load scan: {type(exc).__name__}: {exc}"
        )
        logger.exception("Failed to load scan %s", scan.path)
        return

    current = state.selected_scan
    if current is not None and current.path == scan.path:
        state.publish("scan_loaded", scan)


# ---------------------------------------------------------------------------
# Bundled app icon — for the toolbar wordmark
# ---------------------------------------------------------------------------


def _app_icon_path() -> Optional[str]:
    """Resolve the bundled ``executable_icon.png`` to an absolute path.

    Returns the path as a string for ``ui.image``, or ``None`` on any
    resolution failure (broken install, missing asset). Callers fall
    back to no-icon rendering rather than crashing the toolbar.
    """
    try:
        from importlib import resources

        ref = resources.files("hrfunc.assets") / "executable_icon.png"
        with resources.as_file(ref) as path:
            return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("app icon unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# One-time welcome card — first-launch onboarding overlay
# ---------------------------------------------------------------------------


def _welcome_card_marker_path() -> Path:
    """Path of the XDG marker that records "we've shown the card once."""
    try:
        import platformdirs

        cache_dir = Path(platformdirs.user_cache_dir("hrfunc"))
    except ImportError:
        cache_dir = Path.home() / ".cache" / "hrfunc"
    return cache_dir / ".welcome_card_dismissed"


def _was_welcome_card_dismissed() -> bool:
    return _welcome_card_marker_path().exists()


def _mark_welcome_card_dismissed() -> None:
    marker = _welcome_card_marker_path()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError as exc:
        # Best-effort: a failed write means the card may re-appear on
        # the next launch. Better than crashing the GUI.
        logger.warning("Failed to write welcome-card marker: %s", exc)


def _should_show_welcome_card(state: AppState) -> bool:
    """True when this is the cold-launch first-time-user condition.

    Requires all three: no project loaded, no recent projects in the
    cache, and the dismissal marker not yet set. Any one of these
    false → the user has been here before, no card.
    """
    if state.manifest is not None:
        return False
    if _was_welcome_card_dismissed():
        return False
    if dataset_picker.list_recent_manifests(limit=1):
        return False
    return True


def _render_welcome_card() -> None:
    """Render the dismissible welcome card overlay.

    Position-absolute pinned to the top-right of the Library tab content
    so it visually points at the project picker dropdown above. Uses a
    high z-index so it floats above the plotly viz.
    """

    @ui.refreshable
    def _card_body() -> None:
        if _was_welcome_card_dismissed():
            return
        with ui.card().style(
            "position: absolute; top: 1.5rem; right: 1.5rem; "
            "max-width: 22rem; z-index: 50;"
        ).classes("p-4"):
            ui.label("Welcome to HRfunc").classes(
                "text-lg font-semibold mb-2"
            )
            ui.label(
                "Browse the bundled literature HRFs here in the Library, "
                "or open a folder of fNIRS scans (top-right) to start "
                "estimating HRFs from your own data."
            ).classes("text-sm opacity-80 leading-relaxed")
            with ui.row().classes("w-full justify-end mt-3"):
                def _on_dismiss() -> None:
                    _mark_welcome_card_dismissed()
                    _card_body.refresh()

                ui.button("Got it", on_click=_on_dismiss).props(
                    "color=primary dense"
                )

    _card_body()
