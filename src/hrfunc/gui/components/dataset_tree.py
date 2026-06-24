"""Dataset tree — left-pane navigation of a Manifest.

Renders a Manifest's scans as a hierarchical tree grouped by BIDS subject /
session when the metadata is available, or by parent directory for non-BIDS
folders. Clicking a leaf node sets ``state.selected_scan``, which the
workspace's center and right panes react to.

Grouping rules:
- ``bids_subject`` present  → group under ``sub-<id>``
- ``bids_subject`` absent   → group under ``📁 <parent-dir-name>``
- ``bids_session`` present  → sub-group ``ses-<id>``
- ``bids_session`` absent   → sub-group ``(no session)``

The leaf node label uses the ScanEntry's ``display_name`` (which itself
combines BIDS components when available — see ``scan._make_display_name``).
Node IDs are the absolute scan path stringified — this makes the click
lookup O(1) given a dict from path → ScanEntry.

Search filter (Sprint 3.1):
``build_nodes`` accepts an optional ``filter_text``. When non-empty, scans
are kept only if the filter (case-insensitive) appears in ``display_name``
or the stringified path. Subjects and sessions with zero surviving scans
are pruned, so the tree only shows the relevant subtree. ``render`` wires
a ``ui.input`` above the tree that refreshes the body on change.

Bulk action UI (PR #55a):
- Tree renders with ``tick_strategy='leaf'`` so each scan has its own
  checkbox. The ticked set is mirrored into ``state.checked_scan_paths``
  (a set of resolved Paths) so it survives tab switches and re-renders.
- Subjects + sessions are auto-expanded on first render so every scan
  is immediately visible without clicking chevrons.
- A "Select all" checkbox above the search filter ticks / unticks every
  visible scan in one shot.

Preprocess, HRFs, and Activity panels read ``checked_scan_paths`` and
run their action sequentially across the whole set when it's non-empty.
Inspect, Quality, and Export render the same checkbox UI for consistency
but their handlers stay on ``state.selected_scan``.

Public API:
    build_nodes(manifest, filter_text="") -> list[dict]   - tree node structure
    all_group_node_ids(nodes) -> list[str]                - subject + session ids
    all_leaf_node_ids(nodes) -> list[str]                 - scan-leaf ids
    render(state, on_select_scan=None)                    - render the tree + wire selection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from nicegui import ui

from ..state import AppState
from ...io.manifest import Manifest, ScanEntry

logger = logging.getLogger(__name__)

_NO_SESSION_LABEL = "(no session)"

OnScanSelect = Callable[[Optional[ScanEntry]], None]


def all_group_node_ids(nodes: List[dict]) -> List[str]:
    """Collect every non-leaf node id from a ``build_nodes`` result.

    Used to pre-populate the tree's ``expanded`` prop so subjects +
    sessions render open on first paint (PR #55a). Leaf nodes (scans)
    are not expandable, so we only need subject / session ids.
    """
    out: List[str] = []
    for sub in nodes:
        out.append(sub["id"])
        for ses in sub.get("children", []):
            out.append(ses["id"])
    return out


def all_leaf_node_ids(nodes: List[dict]) -> List[str]:
    """Collect every leaf (scan) node id from a ``build_nodes`` result.

    Used by the Select-All checkbox to tick every visible scan in one
    shot (PR #55a). Filter-pruned scans are not in ``nodes`` so the
    select-all spans whatever the user currently sees.
    """
    out: List[str] = []
    for sub in nodes:
        for ses in sub.get("children", []):
            for scan in ses.get("children", []):
                out.append(scan["id"])
    return out


def _scan_matches_filter(scan: ScanEntry, filter_text: str) -> bool:
    """Case-insensitive substring match against display_name + path."""
    if not filter_text:
        return True
    needle = filter_text.lower()
    if scan.display_name and needle in scan.display_name.lower():
        return True
    return needle in str(scan.path).lower()


def build_nodes(manifest: Manifest, filter_text: str = "") -> List[dict]:
    """Convert a Manifest into the dict structure ``ui.tree`` expects.

    Returns a list of subject-level nodes. Each subject node has children
    (sessions); each session has children (scans). The scan-level nodes use
    the stringified absolute path as the ``id`` so click handlers can do a
    direct dict lookup back to the ScanEntry.

    When ``filter_text`` is non-empty, scans are filtered by case-insensitive
    substring match against their display_name and path. Subjects/sessions
    with zero surviving scans are pruned.

    Stable ordering: subjects sorted alphabetically, sessions within a
    subject sorted alphabetically, scans within a session sorted by
    display_name. Without sorting, the tree would jitter between scans
    depending on filesystem walk order.
    """
    # subject_key -> session_key -> list[ScanEntry]
    subjects: Dict[str, Dict[str, List[ScanEntry]]] = {}

    for scan in manifest.scans:
        if not _scan_matches_filter(scan, filter_text):
            continue
        sub_key = (
            f"sub-{scan.bids_subject}"
            if scan.bids_subject
            else f"📁 {scan.path.parent.name}"
        )
        ses_key = (
            f"ses-{scan.bids_session}"
            if scan.bids_session
            else _NO_SESSION_LABEL
        )
        subjects.setdefault(sub_key, {}).setdefault(ses_key, []).append(scan)

    nodes: List[dict] = []
    for sub_key in sorted(subjects):
        sub_node = {
            "id": f"subject::{sub_key}",
            "label": sub_key,
            "children": [],
        }
        for ses_key in sorted(subjects[sub_key]):
            ses_node = {
                "id": f"session::{sub_key}::{ses_key}",
                "label": ses_key,
                "children": [
                    {
                        "id": str(scan.path),
                        "label": scan.display_name or scan.path.name,
                    }
                    for scan in sorted(
                        subjects[sub_key][ses_key],
                        key=lambda s: s.display_name or str(s.path),
                    )
                ],
            }
            sub_node["children"].append(ses_node)
        nodes.append(sub_node)
    return nodes


def render(
    state: AppState,
    on_select_scan: Optional[OnScanSelect] = None,
) -> None:
    """Render the dataset tree inside the current NiceGUI context.

    Reads ``state.manifest`` to build nodes and wires the click handler to
    update ``state.selected_scan``. If ``on_select_scan`` is provided, it
    is called after the state update with the resolved ScanEntry (or
    ``None`` if the user clicked a group node) — typically used by the
    workspace to refresh the inspector panel.

    A search input above the tree filters scans by case-insensitive
    substring match against display_name or path. Filter state lives in a
    closure dict so the refreshable body always reads the latest value.

    PR #55a additions:
    - Per-scan checkboxes via ``tick_strategy='leaf'`` -- ticks mirror to
      ``state.checked_scan_paths`` (set of resolved Paths).
    - "Select all" checkbox above the filter input ticks every currently-
      visible scan; unticking it clears the same set.
    - Tree opens with all subjects + sessions expanded so every scan is
      visible without the user clicking chevrons.

    Caller is responsible for placing this inside the desired layout
    (typically the left pane of a splitter).
    """
    if state.manifest is None or not state.manifest.scans:
        ui.label("No dataset loaded.").classes("opacity-60 text-sm p-4")
        return

    # Closure-held filter state. A mutable dict (rather than a nonlocal
    # string) lets the on_change handler write without scope gymnastics.
    filter_state: Dict[str, str] = {"text": ""}

    # Path string -> ScanEntry, for O(1) lookup in the click handler.
    # Built once over the full manifest — filtering only affects which
    # nodes are *rendered*, not which paths can be resolved on click.
    path_to_scan: Dict[str, ScanEntry] = {
        str(s.path): s for s in state.manifest.scans
    }

    def _on_select(event) -> None:
        node_id = event.value
        scan = path_to_scan.get(node_id)
        # Group nodes (subject/session) have no entry in path_to_scan, so
        # scan is None — clear selection to revert the inspector to empty.
        state.selected_scan = scan
        if scan is not None:
            logger.debug("Selected scan: %s", scan.path)
        if on_select_scan is not None:
            on_select_scan(scan)

    def _on_tick(event) -> None:
        """Mirror the tree's ticked-id list into ``state.checked_scan_paths``.

        NiceGUI's tree emits the full list of currently-ticked leaf ids
        on every change (not a delta), so this handler just rebuilds the
        set from scratch. Resolved Paths are the storage form so equality
        is filesystem-stable across manifest re-walks.
        """
        ticked_ids = event.value or []
        new_set = set()
        for node_id in ticked_ids:
            scan = path_to_scan.get(node_id)
            if scan is not None:
                new_set.add(scan.path.resolve())
        state.checked_scan_paths = new_set
        # Re-render so the Select-all checkbox indeterminate / checked
        # display tracks the new set without waiting on another event.
        _tree_body.refresh()
        # Notify the action panels so they recompute bulk mode — without
        # this, ticking scans never makes them eligible for a bulk
        # Preprocess / HRF / Activity run (the panels only re-render on
        # their subscribed events, none of which fired on a tick).
        state.publish("checked_changed", state.checked_scan_paths)

    @ui.refreshable
    def _tree_body() -> None:
        nodes = build_nodes(state.manifest, filter_state["text"])
        if not nodes:
            ui.label("No scans match filter.").classes(
                "opacity-60 text-xs p-2"
            )
            return
        leaf_ids = all_leaf_node_ids(nodes)
        # PR #55a Select-all: tri-state checkbox driven by the visible
        # leaf set. ``state.checked_scan_paths`` is the source of truth,
        # but the UI checkbox needs a flat True/False -- intermediate
        # selection renders as unchecked (Quasar's q-checkbox supports
        # indeterminate but ui.checkbox doesn't expose it). Two clicks
        # off a partial state cycles all-on -> all-off, which matches
        # most users' expectation.
        visible_paths = {
            path_to_scan[node_id].path.resolve()
            for node_id in leaf_ids
            if node_id in path_to_scan
        }
        all_ticked = bool(visible_paths) and visible_paths.issubset(
            state.checked_scan_paths
        )

        def _on_select_all(event) -> None:
            if event.value:
                state.checked_scan_paths = (
                    state.checked_scan_paths | visible_paths
                )
            else:
                state.checked_scan_paths = (
                    state.checked_scan_paths - visible_paths
                )
            _tree_body.refresh()
            state.publish("checked_changed", state.checked_scan_paths)

        n_checked = sum(
            1 for p in visible_paths if p in state.checked_scan_paths
        )
        select_all_label = (
            f"Select all  ({n_checked}/{len(visible_paths)} checked)"
            if visible_paths
            else "Select all"
        )
        def _on_clear_checked() -> None:
            # Clear the ENTIRE checked set (not just the visible/filtered
            # subset) so a sticky bulk selection can be reset in one click --
            # the set is global across tabs, so this is the escape hatch from
            # an unexpected bulk mode on a later tab.
            state.checked_scan_paths = set()
            _tree_body.refresh()
            state.publish("checked_changed", state.checked_scan_paths)

        with ui.row().classes("items-center gap-2 w-full"):
            ui.checkbox(
                select_all_label,
                value=all_ticked,
                on_change=_on_select_all,
            ).props("dense").classes("text-xs")
            total_checked = len(state.checked_scan_paths)
            if total_checked:
                ui.button(
                    f"Clear ({total_checked})",
                    icon="clear",
                    on_click=_on_clear_checked,
                ).props("flat dense color=primary").classes("text-xs").tooltip(
                    "Clear the checked-scan selection used for bulk actions "
                    "(Preprocess / HRFs / Neural Activity)."
                )

        # Initial ``ticked`` reflects state.checked_scan_paths so the
        # tree re-renders with the saved selection after tab switches.
        ticked_initial = [
            node_id for node_id in leaf_ids
            if node_id in path_to_scan
            and path_to_scan[node_id].path.resolve()
            in state.checked_scan_paths
        ]
        # All subjects + sessions expanded by default so every scan is
        # visible -- bulk-action workflows are the common case now and
        # hunting through chevrons for each subject is friction.
        expanded_initial = all_group_node_ids(nodes)
        tree = ui.tree(
            nodes,
            on_select=_on_select,
            tick_strategy="leaf",
            on_tick=_on_tick,
        ).classes("w-full")
        # Apply initial expanded + ticked state via NiceGUI's
        # ``Tree.expand()`` / ``Tree.tick()`` helpers (the supported
        # path for setting initial state from Python; setting
        # ``.expanded`` directly is not exposed as a public attribute).
        if expanded_initial:
            tree.expand(expanded_initial)
        if ticked_initial:
            tree.tick(ticked_initial)

    def _on_filter_change(event) -> None:
        filter_state["text"] = event.value or ""
        _tree_body.refresh()

    ui.input(
        placeholder="Filter scans…",
        on_change=_on_filter_change,
    ).props("dense clearable").classes("w-full")
    _tree_body()

    # Cross-tab sync: each data tab (Preprocess / HRFs / Activity) mounts its
    # OWN dataset tree, but they all share ``state.checked_scan_paths``. When
    # one tab changes the checked set (tick / select-all here, or a panel that
    # clears it), the other tabs' trees must re-render their visual ticks or
    # they drift — e.g. the Activity bulk label reads "2 selected" while this
    # tab's checkboxes show none. Refresh on ``checked_changed`` so every tree
    # reflects the shared set. ``_tree_body.refresh()`` rebuilds the tree and
    # re-applies ticks from state; programmatic ``tree.tick()`` does not emit
    # ``on_tick``, so this can't loop with the tree that published the event.
    state.subscribe("checked_changed", lambda _p=None: _tree_body.refresh())


def find_scan(manifest: Optional[Manifest], path_id: str) -> Optional[ScanEntry]:
    """Look up a ScanEntry by its stringified absolute path.

    Convenience for tests and components that receive a node id from a
    tree event and need the corresponding ScanEntry.
    """
    if manifest is None:
        return None
    for scan in manifest.scans:
        if str(scan.path) == path_id:
            return scan
    return None
