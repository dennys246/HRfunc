"""HRtree panel — embeddable HRF-tree explorer.

This module is the Library/HRtree implementation extracted from the
Sprint 4 ``/library`` route so the v1.4 single-shell GUI can mount it as a
tab without dragging route-specific chrome with it. Three deliberate
differences from the legacy ``pages.library`` code it was ported from:

1. **No toolbar.** The shell renders a single brand wordmark + project
   picker; the panel renders only the filter / viz / detail panes.
2. **No ``state.subscribers.clear()``.** The legacy ``/library`` and
   ``/workspace`` route handlers cleared the subscriber list on every
   render so repeat visits didn't accumulate stale refreshable
   handles. In the single-shell model that clear-on-render is a
   footgun — it nukes other tabs' subscriptions. Subscribers are
   instead cleared only on project switch (Phase 3 work).
3. **Event prefix is ``hrtree_*``.** Legacy events were prefixed
   ``library_*``; the rename frees up the namespace for the future
   "Library / Project / Both" data-source toggle where project-side
   events live alongside.

Public API:
    render(state, *, data_source="library")
        Mount the three-pane HRtree explorer. ``data_source`` is
        plumbed for the future toggle but only ``"library"`` is wired
        in Phase 1 — passing other values currently falls back to
        library-tree rendering.

Pure helpers (``gather_library_hrfs``, ``apply_filter``,
``filter_by_oxygenation``, ``compute_roi_keys``, ``compute_roi_average``,
``build_plotly_figure``, ``load_mesh``) stay module-level so tests can
call them without a UI context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from nicegui import ui

from ...spatial.atlas import load_harvard_oxford_cortical
from ...spatial.coords import meters_to_mm
from ...spatial.shapes import AtlasRegion, Box, Shape, Sphere
from ...viz.brain_scene import (
    make_box_overlay_trace,
    make_sphere_overlay_trace,
    make_surface_trace,
)
from ...viz.meshes import MESH_CACHE as _MESH_CACHE
from ...viz.meshes import MESH_FILENAMES as _MESH_FILENAMES
from ...viz.meshes import load_brain_mesh, load_mesh
from . import brand
from ..state import AppState

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Mesh loaders were moved to :mod:`hrfunc.viz.meshes` during the v1.3
# spatial/viz compartmentalization refactor. The names are re-exported
# here so existing callers (``library.load_mesh``, ``library.load_brain_mesh``,
# ``library._MESH_CACHE``, ``library._MESH_FILENAMES``) and the v1.3.0
# test suite continue to work without import-path churn. New code should
# import from ``hrfunc.viz.meshes`` directly.
__all__ = (
    "load_mesh",
    "load_brain_mesh",
    "_MESH_CACHE",
    "_MESH_FILENAMES",
)


# Subset of context fields exposed as filter controls. The library tree
# context has ~10 fields; researchers most commonly filter on the first
# few (task, doi, study, demographics). Less-used fields (intensity,
# protocol) stay accessible via the data but don't get a control.
FILTER_FIELDS = (
    "task",
    "doi",
    "study",
    "demographics",
    "stimulus",
    "conditions",
)


# Available shape modes for the Cluster sub-tab's ROI selector.
SHAPE_SPHERE = "sphere"
SHAPE_BOX = "box"
SHAPE_ATLAS_REGION = "atlas_region"
SHAPE_MODES = (SHAPE_SPHERE, SHAPE_BOX, SHAPE_ATLAS_REGION)


def _resolve_cluster_oxygenation(state: AppState) -> Optional[bool]:
    """Pick the oxygenation filter the Cluster sub-tab should apply.

    Returns ``True`` to keep HbO only, ``False`` to keep HbR only,
    ``None`` to skip oxygenation filtering and let mixed-haemoglobin
    HRFs into the ROI.

    Precedence (matches the v1.2 anchor+radius behaviour and extends
    it for free-floating modes):

    1. If a click-anchor is set, use the anchor's oxygenation.
       Averaging mixed-haemoglobin traces is scientifically wrong --
       same rationale as the original ``compute_roi_keys`` behaviour.
    2. Else if the filter sub-tab's oxygenation is ``"hbo"`` or
       ``"hbr"``, route the binary through.
    3. Else (filter is ``"both"`` with no anchor), return None --
       the user has explicitly opted into mixed visibility, and
       silently filtering would surprise them.
    """
    anchor = state.library_selected_hrf
    if anchor is not None and anchor.get("oxygenation") is not None:
        return bool(anchor.get("oxygenation"))
    if state.library_oxygenation == "hbo":
        return True
    if state.library_oxygenation == "hbr":
        return False
    return None


def _build_current_shape(state: AppState) -> Optional[Shape]:
    """Build the spatial-layer :class:`Shape` for the current Cluster mode.

    Sphere mode:
        Centre = ``(cluster_center_*_mm)``. Radius =
        ``library_roi_radius_m * 1000``. Note that the cluster centre
        is normally seeded from a clicked HRF (see the viz pane's
        click handler), so in the legacy "click and adjust radius"
        workflow the resulting sphere matches the v1.2 anchor-based
        sphere.

    Box mode:
        Centre = ``(cluster_center_*_mm)``. Half-extents =
        ``(cluster_box_half_*_mm)``. Always free-floating.

    Returns None for unknown shape modes (defensive -- a stale state
    value from a future / legacy build shouldn't crash the render).

    Layout refactor (2026-05-16): returns None when the ROI list is
    empty -- adding a ROI is the implicit activation. Pre-refactor
    this checked ``state.cluster_roi_active`` (a removed field).
    """
    if not state.cluster_rois:
        return None
    return _build_shape_unconditional(state)


def _build_shape_unconditional(state: AppState) -> Optional[Shape]:
    """Same as :func:`_build_current_shape` but ignores the ROI-active
    toggle. The Cluster sub-tab UI needs to render shape-specific
    controls (sphere radius, atlas dropdown) regardless of whether
    the toggle is on -- otherwise turning the toggle off would
    collapse the entire UI body and disorient the user."""
    if state.cluster_shape == SHAPE_BOX:
        return Box(
            center_mm=(
                state.cluster_center_x_mm,
                state.cluster_center_y_mm,
                state.cluster_center_z_mm,
            ),
            half_extents_mm=(
                state.cluster_box_half_x_mm,
                state.cluster_box_half_y_mm,
                state.cluster_box_half_z_mm,
            ),
        )
    if state.cluster_shape == SHAPE_ATLAS_REGION:
        # Atlas mode needs both the loaded atlas and a selected region.
        # If either is missing we return None so callers fall back to
        # "no ROI yet" UI (the save button disables, the viz skips the
        # shape-membership filter).
        if not state.cluster_atlas_label:
            return None
        atlas = load_harvard_oxford_cortical()
        if atlas is None:
            return None
        try:
            return AtlasRegion(atlas, state.cluster_atlas_label)
        except ValueError:
            # Label no longer in atlas (e.g. user state from a future
            # version with a richer atlas). Treat as unselected.
            return None
    if state.cluster_shape == SHAPE_SPHERE:
        return Sphere(
            center_mm=(
                state.cluster_center_x_mm,
                state.cluster_center_y_mm,
                state.cluster_center_z_mm,
            ),
            radius_mm=float(meters_to_mm(state.library_roi_radius_m)),
        )
    return None


def _build_atlas_alignment_affine(state: AppState) -> "Optional[np.ndarray]":
    """Compose the full HRF-coord -> MNI mm affine for atlas lookups.

    Returns ``None`` when the alignment is identity (no transform
    needed) -- callers fast-path the lookup.

    The user can provide alignment two ways:

    1. A full 4x4 ``cluster_atlas_alignment_affine`` loaded from a
       JSON or .npy file via the file picker in atlas mode.
    2. Three pure-translation offsets (``cluster_atlas_offset_*_mm``)
       for users without a registered affine.

    Both compose -- offsets translate AFTER the affine. Identity
    affine + zero offsets returns None so callers know they can
    skip the transform.
    """
    import numpy as np

    ox = float(state.cluster_atlas_offset_x_mm)
    oy = float(state.cluster_atlas_offset_y_mm)
    oz = float(state.cluster_atlas_offset_z_mm)
    has_offset = ox != 0.0 or oy != 0.0 or oz != 0.0
    has_affine = state.cluster_atlas_alignment_affine is not None

    if not has_offset and not has_affine:
        return None

    if has_affine:
        affine = np.asarray(
            state.cluster_atlas_alignment_affine, dtype=np.float64
        )
        if affine.shape != (4, 4):
            # Defensive: stale state; ignore and fall back to identity.
            affine = np.eye(4, dtype=np.float64)
    else:
        affine = np.eye(4, dtype=np.float64)

    if has_offset:
        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = (ox, oy, oz)
        # Translation applied AFTER the affine ("T @ A @ point").
        affine = translation @ affine

    return affine


def _alignment_for_shape(state: AppState, shape: Optional[Shape]) -> "Optional[np.ndarray]":
    """Return the HRF -> atlas alignment affine when applicable, else None.

    Atlas mode needs the alignment because library HRFs are stored in
    MNE head coords (origin near auditory meatus) while the bundled
    atlas is in MNI mm. Sphere / Box modes don't need alignment --
    their geometry lives in the same frame as the HRF locations.
    """
    if not isinstance(shape, AtlasRegion):
        return None
    return _build_atlas_alignment_affine(state)


def _apply_alignment_to_point(
    point_mm: Tuple[float, float, float],
    alignment_affine: "Optional[np.ndarray]",
) -> Tuple[float, float, float]:
    """Map an MNE-head mm point to MNI mm through the alignment affine.

    Mirrors the per-HRF transform in ``compute_roi_keys_by_shape`` so a
    single-point lookup (the Cluster centre's "Region at centre" readout)
    lands in the same frame as the membership check. Returns the point
    unchanged when there's no affine (sphere/box mode, or atlas mode with
    no alignment configured), matching the membership path's behaviour.
    """
    if alignment_affine is None:
        return point_mm
    import numpy as np  # local: module-level numpy is TYPE_CHECKING-only

    homo = np.array(
        [point_mm[0], point_mm[1], point_mm[2], 1.0], dtype=np.float64
    )
    aligned = alignment_affine @ homo
    return (float(aligned[0]), float(aligned[1]), float(aligned[2]))


def _build_shape_for_slot(state: AppState, slot) -> Optional[Shape]:
    """Build the spatial-layer Shape for a specific ROI slot.

    Sibling to :func:`_build_shape_unconditional` but reads from the
    given slot rather than the active proxy. Used by the multi-ROI
    visibility flow to render every visible slot's overlay
    simultaneously on the viz.

    Returns ``None`` for slots whose shape can't be built (atlas mode
    without a region picked, or atlas mode when the bundled atlas
    failed to load).
    """
    if slot.shape == SHAPE_BOX:
        return Box(
            center_mm=(slot.center_x_mm, slot.center_y_mm, slot.center_z_mm),
            half_extents_mm=(
                slot.box_half_x_mm, slot.box_half_y_mm, slot.box_half_z_mm,
            ),
        )
    if slot.shape == SHAPE_ATLAS_REGION:
        if not slot.atlas_label:
            return None
        atlas = load_harvard_oxford_cortical()
        if atlas is None:
            return None
        try:
            return AtlasRegion(atlas, slot.atlas_label)
        except ValueError:
            return None
    if slot.shape == SHAPE_SPHERE:
        return Sphere(
            center_mm=(slot.center_x_mm, slot.center_y_mm, slot.center_z_mm),
            radius_mm=float(slot.radius_mm),
        )
    return None


def _visible_shapes(state: AppState) -> "List[Tuple[Any, Shape]]":
    """Collect every visible ROI's (slot, Shape) pair (PR follow-up).

    Returns an empty list when the ROI list is empty (no implicit
    activation yet) or when no ROI is both visible and has a
    buildable shape. Each visible slot whose shape can't be
    constructed (atlas mode without a region selected, etc.) is
    silently dropped so the viz still renders the others.

    Used by both the viz pane (multi-overlay rendering) and the
    Cluster sub-tab's ROI-status / save handler (iterate every
    visible slot's membership).
    """
    if not state.cluster_rois:
        return []
    pairs: List = []
    for slot in state.cluster_rois:
        if not slot.visible:
            continue
        shape = _build_shape_for_slot(state, slot)
        if shape is None:
            continue
        pairs.append((slot, shape))
    return pairs


def _visible_roi_keys(
    state: AppState,
    matched: Dict[str, Dict[str, Any]],
) -> Tuple[set, "List[Tuple[Any, Shape]]"]:
    """Compute the union of roi_keys across every visible ROI.

    Returns ``(union_keys, visible_pairs)`` so callers can reuse the
    ``[(slot, shape)]`` list for the multi-overlay viz without rebuilding.

    Each pair's membership is computed with that pair's shape's
    alignment affine (atlas mode needs it; sphere / box mode does
    not). Per-slot oxygenation filter follows
    :func:`_resolve_cluster_oxygenation` -- the slot's anchor wins
    when present, otherwise the panel-level filter applies. Painted
    keys come from the slot's own ``painted`` set.
    """
    pairs = _visible_shapes(state)
    if not pairs:
        return set(), []
    union: set = set()
    for slot, shape in pairs:
        anchor = slot.anchor
        if anchor is not None and anchor.get("oxygenation") is not None:
            oxy: Optional[bool] = bool(anchor.get("oxygenation"))
        elif state.library_oxygenation == "hbo":
            oxy = True
        elif state.library_oxygenation == "hbr":
            oxy = False
        else:
            oxy = None
        alignment = _alignment_for_shape(state, shape)
        slot_keys = compute_roi_keys_by_shape(
            matched, shape, slot.painted,
            oxygenation_filter=oxy,
            alignment_affine=alignment,
        )
        union |= set(slot_keys)
    return union, pairs


DataSource = Literal["library", "project", "both"]


def render(state: AppState, *, data_source: DataSource = "library") -> None:
    """Render the HRtree explorer panel against the given AppState.

    Lazy-loads the bundled HRF trees on first call; subsequent calls
    reuse the cached state. The three-pane layout (filter / viz / detail)
    is the same as the legacy ``/library`` page minus the toolbar.

    :param state: AppState singleton (or a synthetic one in tests).
    :param data_source: ``"library"`` shows bundled literature HRFs;
        ``"project"`` and ``"both"`` are reserved for the future
        project-HRF integration and currently fall back to library
        rendering.
    """
    # The data_source param is plumbed for forward compat; the project /
    # both code paths aren't wired yet (the panel renders library data
    # regardless). Once project HRFs are sourced from state.montage in a
    # future phase, this becomes a true switch.
    _ = data_source

    if state.library_hbo is None or state.library_hbr is None:
        _load_trees(state)

    _render_three_pane(state)


def _load_trees(state: AppState) -> None:
    """Read the bundled HRF databases into memory once.

    The trees stay on state for the lifetime of the process. Failures are
    surfaced to ``state.last_error`` but the panel still renders so users
    can see what went wrong rather than getting a blank screen.
    """
    try:
        # Lazy-import to keep the GUI import graph minimal at module load.
        from ...hrtree import tree as Tree
        from ... import __file__ as hrfunc_file

        import os
        lib_dir = os.path.join(os.path.dirname(hrfunc_file), "hrfs")
        hbo_path = os.path.join(lib_dir, "hbo_hrfs.json")
        hbr_path = os.path.join(lib_dir, "hbr_hrfs.json")
        # ``rich=True`` keeps the per-subject ``estimates`` lists on each
        # HRF node so ROI averaging can pool subject-level traces (the
        # statistically correct grand mean). The default ``rich=False``
        # strips them to save memory; for the GUI we accept the ~1-2 MB
        # cost in exchange for accurate ROI averages.
        state.library_hbo = Tree(hbo_path, rich=True)
        state.library_hbr = Tree(hbr_path, rich=True)
        logger.info(
            "Loaded library trees: HbO=%d nodes, HbR=%d nodes",
            len(state.library_hbo.gather(state.library_hbo.root)),
            len(state.library_hbr.gather(state.library_hbr.root)),
        )
    except Exception as exc:  # noqa: BLE001
        state.last_error = (
            f"Failed to load bundled HRF library: {type(exc).__name__}: {exc}"
        )
        logger.exception("library load failed: %s", exc)


# ---------------------------------------------------------------------------
# ROI shift-hover paint — JS injection
# ---------------------------------------------------------------------------


_PAINT_HOOK_HEAD_INJECTED = False


def _ensure_shift_tracker_injected() -> None:
    """Inject a global shift-key tracker into the page head (once per process).

    The tracker writes ``window._hrfShift = True/False`` on Shift down/up,
    plus a safety reset on window blur (so a Shift-down outside the window
    followed by a release-outside doesn't leave a stuck shift state).

    Module-level idempotency flag ensures we add the ``<script>`` to the
    document head only once per process. Subsequent calls are no-ops.
    """
    global _PAINT_HOOK_HEAD_INJECTED
    if _PAINT_HOOK_HEAD_INJECTED:
        return
    ui.add_head_html(
        """
<script>
  if (window._hrfShiftWired === undefined) {
    window._hrfShift = false;
    window._hrfShiftWired = true;
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Shift') window._hrfShift = true;
    });
    document.addEventListener('keyup', (e) => {
      if (e.key === 'Shift') window._hrfShift = false;
    });
    window.addEventListener('blur', () => { window._hrfShift = false; });
  }
</script>
"""
    )
    _PAINT_HOOK_HEAD_INJECTED = True


def _install_paint_hook(plot_id: int, anchor=None) -> None:
    """Hook the freshly-rendered plotly element's ``plotly_hover`` event.

    Plotly's native hover event includes the points data; what we need
    beyond that is the Shift-key state. We read it from the global
    ``window._hrfShift`` flag wired by :func:`_ensure_shift_tracker_injected`,
    then forward a custom ``roi_paint`` event up to the NiceGUI Python
    handler via the element's ``$emit``.

    A small ``ui.timer`` delay gives plotly's render cycle time to attach
    the underlying div to the DOM before we query it. The timer is parked on
    ``anchor`` (a stable element OUTSIDE the refreshable viz body) when given,
    so a rapid ``_viz_body.refresh()`` — e.g. the user clicking through HRFs
    faster than 0.5 s — can't delete the slot the pending timer lives in and
    crash it with "parent slot of the element has been deleted".
    """
    _ensure_shift_tracker_injected()

    js = f"""
const el = getElement({plot_id}).$el;
if (el && el.on && !el._hrfPaintHooked) {{
  el._hrfPaintHooked = true;
  el.on('plotly_hover', function(data) {{
    if (!window._hrfShift) return;
    if (!data || !data.points || !data.points.length) return;
    const key = data.points[0].customdata;
    if (!key) return;
    getElement({plot_id}).$emit('roi_paint', {{key: key}});
  }});
}}
"""

    cb = lambda: ui.run_javascript(js)  # noqa: E731
    if anchor is not None and not anchor.is_deleted:
        with anchor:
            ui.timer(0.5, cb, once=True)
    else:
        ui.timer(0.5, cb, once=True)


# ---------------------------------------------------------------------------
# Three-pane layout — splitters for left | center | right
# ---------------------------------------------------------------------------


def _render_three_pane(state: AppState) -> None:
    """Render left | viz | detail with two splitters.

    ``h-full`` (vs the legacy ``h-screen``) sizes the splitter to fill
    its parent container — the shell's tab-panels — instead of the full
    viewport. The shell's toolbar already consumes the top of the
    viewport, so ``h-screen`` here would push the bottom of the panel
    below the fold and cause page-level scrolling.

    The left pane hosts the HR_tree_ wordmark plus Filter / Cluster
    sub-tabs (Phase 6 redesign); the center is the 3D viz at full
    height; the right is the HRF detail card.
    """
    with ui.splitter(value=22, limits=(15, 35)).classes(
        "w-full h-full"
    ) as outer:
        with outer.before:
            _render_left_pane(state)
        with outer.after:
            with ui.splitter(value=68, limits=(40, 90)).classes(
                "w-full h-full"
            ) as inner:
                with inner.before:
                    _render_viz_pane(state)
                with inner.after:
                    _render_detail_pane(state)


# ---------------------------------------------------------------------------
# Left pane: HR_tree_ wordmark + Filter / Cluster sub-tabs
# ---------------------------------------------------------------------------


# Sub-tab labels for the left pane. ``Filter`` holds the context inputs,
# overlay toggles, oxygenation radio, and ROI radius slider. ``Cluster``
# holds actions that commit the current ROI to disk (save averaged
# trace) plus room for future clustering scripts.
SUBTAB_FILTER = "Filter"
SUBTAB_CLUSTER = "Cluster"
SUBTAB_NAMES = (SUBTAB_FILTER, SUBTAB_CLUSTER)


def _render_oxygenation_radio(state: AppState) -> None:
    """HbO / HbR / Both oxygenation filter.

    Rendered above the Filter / Cluster sub-tabs (not inside Filter) so it
    applies to BOTH sub-tabs and stays changeable while viewing either. It
    publishes ``hrtree_filter_changed`` so the viz, detail pane, ROI
    membership, and per-channel deconvolution matching all re-scope to the
    chosen oxygenation.
    """

    def _on_change(event) -> None:
        state.library_oxygenation = event.value or "both"
        state.publish("hrtree_filter_changed", state.library_filter)

    with ui.row().classes("items-center gap-2 w-full"):
        ui.label("Oxygenation").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        ui.radio(
            {"both": "Both", "hbo": "HbO only", "hbr": "HbR only"},
            value=state.library_oxygenation,
            on_change=_on_change,
        ).props("inline dense")


def _render_left_pane(state: AppState) -> None:
    """Wordmark + sub-tabs at the top, active sub-tab content below.

    The HR_tree_ Brand wordmark lives here (not in the shell) so the
    branding sits inside the panel where the user is looking. Sub-tabs
    keep filter and cluster actions docked side-by-side rather than
    competing for vertical space; switching tabs is one click.
    """
    with ui.column().classes("w-full h-full p-3 gap-2 overflow-hidden"):
        # Brand wordmark + one-line subtitle. Compact size_rem so the
        # header doesn't dominate the pane.
        with ui.row().classes("items-center gap-2 w-full"):
            brand.brand("HRtree", italic_suffix="tree", size_rem=1.3)
        ui.label(
            "3D spatial database of literature HRFs."
        ).classes("text-xs opacity-60")

        # Oxygenation filter sits ABOVE the sub-tabs so it applies to both
        # Filter and Cluster and stays changeable on either — it scopes
        # everything downstream (visible HRFs, ROI membership, per-channel
        # deconvolution matching), not just the Filter sub-tab.
        _render_oxygenation_radio(state)

        # Sub-tabs.
        with ui.tabs().props("dense").classes("w-full") as subtabs:
            for name in SUBTAB_NAMES:
                ui.tab(name)
        # ``min-h-0`` is required alongside ``flex-1`` for the
        # ``overflow-hidden`` boundary to actually clip — without it,
        # the Filter sub-tab's content height pushes the column past
        # the splitter pane and the left side ends up taller than the
        # viz on the right (causing the mismatch you'd otherwise see).
        with ui.tab_panels(subtabs, value=SUBTAB_FILTER).classes(
            "w-full flex-1 min-h-0 overflow-hidden"
        ):
            with ui.tab_panel(SUBTAB_FILTER).classes(
                "p-0 h-full overflow-auto"
            ):
                _render_filter_subtab(state)
            with ui.tab_panel(SUBTAB_CLUSTER).classes(
                "p-0 h-full overflow-auto"
            ):
                _render_cluster_subtab(state)


def _render_filter_subtab(state: AppState) -> None:
    """The Filter sub-tab -- oxygenation + context inputs.

    Filter sub-tab owns "what's visible" (oxygenation, context filters);
    the Cluster sub-tab owns "what's in the ROI" (shape, radius, paint,
    clear). PR #49 moved the radius slider + Clear ROI button to the
    Cluster sub-tab so the two sub-tabs have clearly separated
    responsibilities.

    Each FILTER_FIELDS entry becomes a text input bound to
    ``state.library_filter[field]`` -- empty string = field not
    filtered. "Apply" then refreshes the dependent viz + detail panes.
    """
    with ui.column().classes("w-full gap-3"):
        # ── Context inputs (set-once / refine-slowly). The oxygenation radio
        # moved up to the left pane (above the sub-tabs) so it applies to both
        # Filter and Cluster.
        inputs: Dict[str, Any] = {}
        for field in FILTER_FIELDS:
            initial = str(state.library_filter.get(field, ""))
            inputs[field] = ui.input(
                label=field,
                value=initial,
            ).props("dense clearable").classes("w-full")

        def _apply() -> None:
            new_filter: Dict[str, Any] = {}
            for field, widget in inputs.items():
                value = (widget.value or "").strip()
                if value:
                    new_filter[field] = value
            state.library_filter = new_filter
            state.publish("hrtree_filter_changed", new_filter)

        def _reset() -> None:
            for widget in inputs.values():
                widget.value = ""
            state.library_filter = {}
            state.publish("hrtree_filter_changed", {})

        with ui.row().classes("w-full gap-2"):
            ui.button("Apply", on_click=_apply).props("color=primary dense")
            ui.button("Reset", on_click=_reset).props("flat dense")

        # Live match count — annotates how many filtered HRFs are
        # invisible in the 3D viz for lacking a location, so users
        # aren't confused by "5 / 22 match" while the viz shows 3 points.
        @ui.refreshable
        def _count_label() -> None:
            all_hrfs = gather_library_hrfs(state)
            matched = filter_by_oxygenation(
                apply_filter(all_hrfs, state.library_filter),
                state.library_oxygenation,
            )
            visualizable = sum(
                1
                for hrf in matched.values()
                if hrf.get("location") is not None
                and len(hrf.get("location") or []) >= 3
            )
            text = f"{len(matched)} / {len(all_hrfs)} HRFs match"
            if visualizable < len(matched):
                missing = len(matched) - visualizable
                text += f" ({missing} not visualizable: missing location)"
            ui.label(text).classes("text-xs opacity-70")

        _count_label()
        state.subscribe(
            "hrtree_filter_changed", lambda _p=None: _count_label.refresh()
        )


def _render_cluster_subtab(state: AppState) -> None:
    """The Cluster sub-tab -- ROI shape selection, sizing, save action.

    Two shape modes since PR #53:

    - **Sphere** (default): the centre + radius selection from
      PR #49. Centre seeds from clicks in the viz; radius slider
      lives in this sub-tab.
    - **Atlas region**: pick a Harvard-Oxford cortical region from
      a dropdown; the ROI is every HRF whose MNI coordinate lies in
      that region's voxel mask. Region name + "Region at centre"
      readout serve as the methods-section provenance line.

    Box mode is hidden from the UI; the underlying class is still
    available for the v1.4 rotatable-box UI work (PR #52 made it
    orientation-aware).

    Contents:

    - **Shape radio**: Sphere | Atlas region.
    - **Centre inputs**: three MNI-mm number inputs. Visible in
      both modes -- they drive the atlas readout even when not
      driving membership.
    - **Radius slider** (sphere only).
    - **Region dropdown** (atlas only).
    - **Clear ROI button**: drops the anchor + painted set.
    - **MNI readout** + **Region-at-centre readout**: copy-pasteable
      methods-section provenance.
    - **Save ROI average**: writes the averaged trace + shape
      metadata to the workspace folder.
    """

    @ui.refreshable
    def _body() -> None:
        # Load the atlas lazily on first sub-tab render. The loader
        # caches per-process so repeat renders pay nothing; sphere-only
        # users still get the per-render cost (small, ~ms) but in
        # exchange the atlas readout works in sphere mode too.
        atlas = load_harvard_oxford_cortical()

        with ui.column().classes("w-full gap-3"):
            # --- Multi-ROI list (PR #55, layout refactor 2026-05-16) ---
            # Top of the sub-tab is just the list -- no separate
            # "Cluster" header (redundant with the sub-tab strip) and
            # no master "ROI active" toggle (adding a ROI IS the
            # activation now). Each row carries its own shape dropdown
            # (Sphere / Atlas region) and, when atlas mode is picked,
            # an inline region dropdown next to it.
            _render_roi_list(state, _body, atlas=atlas)

            # Empty-state guard: when no ROIs exist yet, skip every
            # control below (radius / centre / readouts / save) and
            # render a single placeholder line. Researchers get a
            # clean "I need to add a ROI" cue rather than a panel
            # full of disabled controls.
            if not state.cluster_rois:
                ui.label(
                    "Add a ROI or montage to begin."
                ).classes("text-sm opacity-60 italic mt-3")
                return

            # --- Sphere radius (sphere mode only) --------------------
            # Layout-refactor: radius bar sits directly below the ROI
            # list / shape radio so the most-used per-ROI knob is right
            # under the buttons that select the ROI. Centre coords
            # follow underneath.
            #
            # Bulk-edit semantics (layout refactor 2026-05-16): the
            # slider's on_change applies the new radius to EVERY slot
            # in ``state.bulk_edit_targets()`` (the selected set if
            # non-empty, otherwise just the active slot). The slider
            # value display tracks the active slot for readability.
            targets = state.bulk_edit_targets()
            n_targets = len(targets)
            bulk_suffix = (
                f"  ({n_targets} selected)" if n_targets > 1 else ""
            )
            if state.cluster_shape == SHAPE_SPHERE:
                ui.label(f"ROI radius{bulk_suffix}").classes(
                    "text-xs uppercase opacity-60 tracking-wide"
                )
                radius_label = ui.label(
                    f"{state.library_roi_radius_m * 100:.1f} cm"
                ).classes("text-xs font-mono opacity-80")

                def _on_radius_change(event) -> None:
                    cm = float(event.value)
                    new_radius_mm = cm * 10.0  # cm -> mm
                    for slot in state.bulk_edit_targets():
                        slot.radius_mm = new_radius_mm
                    radius_label.set_text(f"{cm:.1f} cm")
                    state.publish(
                        "hrtree_filter_changed", state.library_filter
                    )

                ui.slider(
                    min=0.5, max=10.0, step=0.1,
                    value=state.library_roi_radius_m * 100.0,
                    on_change=_on_radius_change,
                ).props("dense")

            # --- Centre inputs (MNI mm) ------------------------------
            # Always visible: drives the atlas readout in both modes,
            # and is also the sphere centre in sphere mode. Bulk-edit
            # semantics same as the radius slider above.
            ui.label(f"Centre (MNI mm){bulk_suffix}").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )

            def _make_centre_input(axis: str, slot_attr: str, proxy_attr: str):
                def _on_change(event) -> None:
                    try:
                        value = float(event.value or 0.0)
                    except (TypeError, ValueError):
                        return
                    for slot in state.bulk_edit_targets():
                        setattr(slot, slot_attr, value)
                    state.publish("hrtree_filter_changed", state.library_filter)

                return ui.number(
                    label=axis,
                    value=getattr(state, proxy_attr),
                    step=1.0,
                    format="%.1f",
                    on_change=_on_change,
                ).props("dense").classes("w-20")

            with ui.row().classes("w-full gap-2"):
                _make_centre_input("x", "center_x_mm", "cluster_center_x_mm")
                _make_centre_input("y", "center_y_mm", "cluster_center_y_mm")
                _make_centre_input("z", "center_z_mm", "cluster_center_z_mm")

            # --- Atlas alignment (atlas-active slot only) ------------
            # Region selection moved INLINE into each ROI row (see
            # _render_roi_list). The alignment section stays at the
            # panel level because alignment is a GLOBAL property of
            # the dataset's coord frame, not of any individual ROI
            # (locked decision 2026-05-14).
            if state.cluster_shape == SHAPE_ATLAS_REGION and atlas is not None:
                # --- Atlas alignment (HRF coords -> MNI mm) ----------
                # Library HRFs are stored in MNE head coords (origin
                # near auditory meatus); the atlas is in MNI mm.
                # Without alignment, every lookup falls outside the
                # atlas volume -> silently empty ROIs. Two UI paths:
                # full 4x4 affine upload (for users with a registered
                # head->MNI transform) or pure-translation offsets
                # (a one-click "shift everything"). They compose.
                _render_atlas_alignment_section(state, _body)

            # Clear-ROI lives in the ROI-list header now (left of the
            # Add ROI / Add Montage buttons), so there's no standalone
            # button here anymore.

            # --- Atlas readout ---------------------------------------
            # (The shape/MNI coords readout was removed — it just repeated the
            # centre inputs + radius slider above. The atlas Region-at-centre
            # stays as real, non-redundant provenance.)
            ui.separator()
            if atlas is not None:
                # Atlas readout is shown in BOTH modes so sphere users
                # can see "my centre sits in: Frontal Pole" without
                # switching modes -- useful navigation aid.
                #
                # The centre is stored in MNE-head mm (it's seeded from
                # clicked HRF locations, which are head-coord). atlas.region_at
                # expects MNI mm, so apply the same alignment the membership
                # check uses before the lookup -- otherwise this readout
                # reports a different (wrong) region than the ROI it sits
                # above whenever the user has set atlas offsets/affine.
                # _alignment_for_shape returns None outside atlas mode, so
                # sphere/box readouts are unchanged.
                _centre_shape = _build_current_shape(state)
                _centre_alignment = _alignment_for_shape(state, _centre_shape)
                region_at_centre = atlas.region_at(_apply_alignment_to_point(
                    (
                        state.cluster_center_x_mm,
                        state.cluster_center_y_mm,
                        state.cluster_center_z_mm,
                    ),
                    _centre_alignment,
                ))
                centre_region_text = (
                    f"Region at centre: {region_at_centre}"
                    if region_at_centre is not None
                    else "Region at centre: (outside atlas / background)"
                )
                ui.label(centre_region_text).classes(
                    "text-xs font-mono opacity-70"
                )

            # --- ROI status + Save button ----------------------------
            # The status line reflects the ACTIVE ROI (which the user
            # is currently editing in the right panel). It still
            # serves as the "is the current shape valid?" check, but
            # the save button reports across every visible slot.
            all_hrfs = gather_library_hrfs(state)
            matched = filter_by_oxygenation(
                apply_filter(all_hrfs, state.library_filter),
                state.library_oxygenation,
            )
            shape = _build_current_shape(state)
            oxy_filter = _resolve_cluster_oxygenation(state)
            alignment = _alignment_for_shape(state, shape)
            roi_keys = compute_roi_keys_by_shape(
                matched, shape, state.library_roi_painted,
                oxygenation_filter=oxy_filter,
                alignment_affine=alignment,
            )
            roi_result = compute_roi_average(matched, roi_keys)
            can_save = roi_result is not None

            # The "averaging N subjects across M channels" readout moved to the
            # detail pane (right). Only the can't-save gate stays here so a
            # disabled Save button still explains itself.
            if roi_result is None:
                ui.label(
                    "Active ROI has fewer than 2 averageable subject "
                    "estimates in the current shape. Either widen the "
                    "shape, paint more neighbours, or seed a different "
                    "centre. (Note: HRFs without per-subject estimates "
                    "are excluded.)"
                ).classes("text-xs opacity-60 italic")

            # Visibility summary for the multi-ROI case so the user
            # knows the save button reflects fewer than the list shows.
            visible_count = sum(
                1 for s in state.cluster_rois if s.visible
            )
            hidden_count = len(state.cluster_rois) - visible_count
            if hidden_count > 0:
                ui.label(
                    f"  ({hidden_count} ROI"
                    f"{'s' if hidden_count != 1 else ''} hidden -- "
                    f"will be excluded from save and viz)"
                ).classes("text-xs opacity-50 italic")

            def _on_save_roi() -> None:
                # PR #55: walk every slot in cluster_rois, build a
                # per-ROI entry for each one that has enough data to
                # average, and write a single montage.json with the
                # whole list. Slots that fail the averaging gate
                # ("fewer than 2 subject estimates") are skipped --
                # the notify at the end says how many were saved /
                # skipped so the user knows nothing was silently lost.
                _matched = filter_by_oxygenation(
                    apply_filter(gather_library_hrfs(state),
                                 state.library_filter),
                    state.library_oxygenation,
                )

                entries: list = []
                skipped: list = []
                original_index = state.cluster_active_index
                original_anchor = state.library_selected_hrf
                try:
                    for i, slot in enumerate(state.cluster_rois):
                        # Hidden slots ride the layer-toggle semantics
                        # -- excluded from BOTH viz AND save. They
                        # don't even count as "skipped" in the failure
                        # bucket since they were deliberately omitted.
                        if not slot.visible:
                            continue
                        # Make the slot active so the proxy-based
                        # helpers (_build_current_shape,
                        # _resolve_cluster_oxygenation, etc.) read its
                        # state. The save handler restores the original
                        # active index in the finally block.
                        state.cluster_active_index = i
                        state.library_selected_hrf = slot.anchor

                        _shape = _build_current_shape(state)
                        if _shape is None:
                            skipped.append((slot.name, "no shape"))
                            continue
                        _alignment = _alignment_for_shape(state, _shape)
                        from ..workspace_io import build_roi_entry

                        # Oxygenation purity: HbO and HbR are inverse
                        # responses, so averaging them together cancels and
                        # is always wrong. Reuse the SAME helper the detail
                        # pane uses (_roi_average_oxygenations) so the saved
                        # montage can never diverge from what's displayed: a
                        # determinate oxygenation saves one entry; an
                        # indeterminate "both, no anchor" slot fans out into a
                        # separate, labelled entry per haemoglobin instead of
                        # persisting one mixed (scientifically wrong) average.
                        _oxys = _roi_average_oxygenations(state)

                        _slot_saved = 0
                        for _oxy in _oxys:
                            _roi_keys = compute_roi_keys_by_shape(
                                _matched, _shape, slot.painted,
                                oxygenation_filter=_oxy,
                                alignment_affine=_alignment,
                            )
                            _result = compute_roi_average(_matched, _roi_keys)
                            if _result is None:
                                continue
                            _mean, _std, _n_subjects, _n_channels = _result
                            _sfreq = _resolve_roi_sfreq(
                                slot.anchor, _matched, _roi_keys
                            )
                            # Tag the name by haemoglobin only when the slot
                            # fanned out into both, so single-oxygenation
                            # saves keep the user's exact ROI name.
                            _entry_name = (
                                f"{slot.name} ({'HbO' if _oxy else 'HbR'})"
                                if len(_oxys) > 1 else slot.name
                            )
                            entries.append(
                                build_roi_entry(
                                    roi_keys=_roi_keys,
                                    hrf_mean=_mean,
                                    hrf_std=_std,
                                    sfreq=_sfreq,
                                    shape=_shape,
                                    anchor=slot.anchor,
                                    library_filter=state.library_filter,
                                    oxygenation_filter=_oxy,
                                    name=_entry_name,
                                )
                            )
                            _slot_saved += 1
                        if _slot_saved == 0:
                            skipped.append(
                                (slot.name, "insufficient estimates")
                            )
                            continue
                finally:
                    state.cluster_active_index = original_index
                    state.library_selected_hrf = original_anchor

                if not entries:
                    ui.notify(
                        "No ROI in the montage has enough data to "
                        "average. Widen a shape, paint more neighbours, "
                        "or seed a different centre.",
                        type="warning",
                    )
                    return

                # Alignment is a global per-scan property of the HRF
                # library's coord frame (locked decision 2026-05-14) --
                # one block at the wrapper level for all ROIs.
                from ..workspace_io import save_montage, workspace_dir
                try:
                    out_path = save_montage(
                        rois=entries,
                        alignment_offset_mm=(
                            float(state.cluster_atlas_offset_x_mm),
                            float(state.cluster_atlas_offset_y_mm),
                            float(state.cluster_atlas_offset_z_mm),
                        ),
                        alignment_affine=state.cluster_atlas_alignment_affine,
                    )
                    state.last_saved_roi_path = out_path
                    saved_msg = (
                        f"Saved montage ({len(entries)} ROI"
                        f"{'s' if len(entries) != 1 else ''}) to "
                        f"{out_path.name} ({workspace_dir()})"
                    )
                    if skipped:
                        names = ", ".join(name for name, _ in skipped)
                        saved_msg += (
                            f". Skipped {len(skipped)}: {names}"
                        )
                    ui.notify(saved_msg, type="positive")
                    _body.refresh()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("save montage failed: %s", exc)
                    ui.notify(
                        f"Save failed: {type(exc).__name__}: {exc}",
                        type="negative",
                    )

            save_label = (
                "Save ROI average"
                if len(state.cluster_rois) <= 1
                else f"Save montage ({len(state.cluster_rois)} ROIs)"
            )
            save_btn = ui.button(
                save_label,
                icon="download",
                on_click=_on_save_roi,
            ).props("color=primary dense")
            # Disable only when EVERY slot fails the averaging gate.
            # With multiple ROIs we let the user click even if the
            # active slot is invalid -- the iteration handles per-slot
            # skips and the notify spells out what was saved.
            if not can_save and len(state.cluster_rois) <= 1:
                save_btn.props("disable")

            # --- Persistent "last saved" feedback (PR #54) ----------
            # ``ui.notify`` toasts vanish in seconds; this label stays
            # visible until the next render replaces it, so users can
            # confirm the save happened even if they didn't catch the
            # toast.
            if state.last_saved_roi_path is not None:
                ui.label(
                    f"Last saved: {state.last_saved_roi_path.name}"
                ).classes("text-xs font-mono opacity-60 break-all")

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("hrtree_selection_changed", _refresh)
    state.subscribe("hrtree_filter_changed", _refresh)


def _format_shape_readout(state: AppState) -> str:
    """One-line summary of the current Cluster shape, suitable for copy-paste.

    Branches on ``state.cluster_shape`` so when the box / lasso UIs
    return in v1.4 / PR #54 this helper renders them too without
    touching the call sites. Today the box branch is unreachable
    from the GUI but stays here as the contract for the spatial-
    layer ``cluster_shape`` field.
    """
    cx, cy, cz = (
        state.cluster_center_x_mm,
        state.cluster_center_y_mm,
        state.cluster_center_z_mm,
    )
    centre = f"Centre: ({cx:.1f}, {cy:.1f}, {cz:.1f}) mm"
    if state.cluster_shape == SHAPE_BOX:
        hx = state.cluster_box_half_x_mm
        hy = state.cluster_box_half_y_mm
        hz = state.cluster_box_half_z_mm
        dims = f"Box {hx * 2:.1f}x{hy * 2:.1f}x{hz * 2:.1f} mm"
        return f"{centre}  ·  {dims}"
    if state.cluster_shape == SHAPE_ATLAS_REGION:
        region = state.cluster_atlas_label or "(no region selected)"
        return f"{centre}  ·  Atlas region: {region}"
    radius_mm = state.library_roi_radius_m * 1000.0
    return f"{centre}  ·  Sphere r={radius_mm:.1f} mm"


def _atlas_alignment_status(state: AppState) -> str:
    """Short label describing the current HRF->MNI alignment state."""
    import numpy as np
    has_affine = state.cluster_atlas_alignment_affine is not None
    has_offset = (
        state.cluster_atlas_offset_x_mm != 0.0
        or state.cluster_atlas_offset_y_mm != 0.0
        or state.cluster_atlas_offset_z_mm != 0.0
    )
    if has_affine and has_offset:
        return "Alignment: custom affine + offsets"
    if has_affine:
        affine = np.asarray(state.cluster_atlas_alignment_affine)
        if affine.shape == (4, 4) and np.allclose(affine, np.eye(4), atol=1e-9):
            return "Alignment: identity (no transform)"
        return "Alignment: custom 4x4 affine"
    if has_offset:
        ox = state.cluster_atlas_offset_x_mm
        oy = state.cluster_atlas_offset_y_mm
        oz = state.cluster_atlas_offset_z_mm
        return f"Alignment: offset ({ox:+.1f}, {oy:+.1f}, {oz:+.1f}) mm"
    return "Alignment: identity (no transform)"


def _looks_out_of_mni(hrfs: Dict[str, Dict[str, Any]]) -> bool:
    """Heuristic: sample a few HRF locations and check if they're out of MNI mm bounds.

    MNI Y axis runs roughly -100 to +80 mm. Bundled P-CAT HRFs are
    stored in MNE head coords with origin near the auditory meatus,
    so their Y values land around +60 to +110 mm -- the ``> 100`` mm
    test catches that case while letting properly-MNI HRFs pass.
    Used by the Cluster sub-tab to surface an alignment warning.
    """
    import numpy as np
    # Sample up to 8 HRFs; require >=3 to have Y > 100 mm to flag.
    over_threshold = 0
    sampled = 0
    for hrf in hrfs.values():
        loc = hrf.get("location")
        if loc is None or len(loc) < 3:
            continue
        sampled += 1
        try:
            y_mm = float(loc[1]) * 1000.0
        except (TypeError, ValueError):
            continue
        if abs(y_mm) > 100.0:
            over_threshold += 1
        if sampled >= 8:
            break
    return sampled >= 3 and over_threshold >= 3


def _render_roi_list(state: AppState, body_refreshable, *, atlas=None) -> None:
    """Render the multi-ROI list (PR #55, layout refactor 2026-05-16).

    Layout:
        +------------------------------------------------------+
        | [Clear ROI]                  [+ ROI] [+ Montage]     |
        | > ROI 1   [Sphere ▾]                  (eye)  x       |
        |   ROI 2   [Atlas  ▾] [Frontal Pole ▾] (eye)  x       |
        |   ROI 3   [Sphere ▾]                  (eye)  x       |
        +------------------------------------------------------+

    Each row carries its own shape dropdown so each ROI can be sphere
    or atlas independently (matches the per-slot storage). When a row
    is in atlas mode, a region dropdown appears inline next to the
    shape dropdown. The eye icon toggles visibility; the trash icon
    deletes the row; clicking elsewhere on the row switches the
    active index.

    ``atlas`` (optional) is the loaded Harvard-Oxford atlas; required
    for the per-row region dropdown to populate. ``None`` means the
    atlas failed to load and rows can't switch into atlas mode.

    ``body_refreshable`` is the cluster sub-tab's ``@ui.refreshable``
    body so list mutations re-render the sub-tab and the right-side
    controls pick up the new active slot.
    """

    def _refresh_all() -> None:
        body_refreshable.refresh()
        state.publish("hrtree_filter_changed", state.library_filter)

    def _on_add() -> None:
        state.add_roi()
        _refresh_all()

    async def _on_add_montage() -> None:
        """Open a file picker, load the chosen scan, and emit one
        sphere ROI per unique channel location (PR #57)."""
        from .dataset_picker import pick_file

        # pywebview's file_types syntax: extensions are joined by
        # semicolons inside the parens. Space-separated extensions
        # (the natural-looking form) fail pywebview's filter regex
        # with "is not a valid file filter".
        path = await pick_file(
            file_types=[
                "fNIRS files (*.snirf;*.fif;*.hdr)",
                "All files (*.*)",
            ],
        )
        if path is None:
            return  # user cancelled

        # Load on the event loop -- MNE's read_raw_* is sync. For
        # typical fNIRS files this is fast enough (< 1 s); larger
        # recordings could move to a worker thread later. The path
        # discrimination mirrors RawCache._load_from_path so the
        # picker accepts the same set of formats the project loader
        # already understands.
        try:
            raw = _load_raw_for_montage(path)
        except Exception as exc:  # noqa: BLE001 -- surface, don't crash
            logger.exception(
                "add montage: failed to load %s: %s", path, exc
            )
            ui.notify(
                f"Couldn't read {path.name}: {type(exc).__name__}: {exc}",
                type="negative",
            )
            return

        # Filter to the current haemoglobin so a HbO-only library
        # doesn't get HbR channel locations slipped in. ``None`` /
        # ``"both"`` means "emit a sphere per source-detector pair
        # regardless" (the dedupe still collapses the HbO/HbR pair
        # at the same xyz).
        if state.library_oxygenation == "hbo":
            oxy_filter: Optional[bool] = True
        elif state.library_oxygenation == "hbr":
            oxy_filter = False
        else:
            oxy_filter = None

        new_slots = rois_from_raw(raw, library_oxygenation=oxy_filter)
        if not new_slots:
            ui.notify(
                f"No channels with locations found in {path.name}.",
                type="warning",
            )
            return

        # When the only ROI in the list is the pristine default (sphere
        # at MNI origin, never edited), drop it before appending the
        # new montage slots. Without this, Add Montage on a fresh
        # project leaves an orphan "ROI 1" at (0, 0, 0) cluttering the
        # list above all the per-channel spheres. Conservative: a
        # single field edit (centre move, radius tweak, anchor click,
        # visibility toggle) keeps the slot in place.
        dropped_default = (
            len(state.cluster_rois) == 1
            and state.cluster_rois[0].is_pristine_default()
        )
        if dropped_default:
            state.cluster_rois.clear()

        # Selection: Add Montage clears any prior per-slot selection
        # and ticks every newly-appended slot. The intent is "this
        # montage is the user's working set" -- they likely want to
        # bulk-edit the whole montage immediately (size, position).
        # Stamp ``selected=True`` directly on the new slots (the
        # ``rois_from_raw`` helper defaulted them to ``selected=False``
        # to keep its pure-helper contract simple).
        state.set_all_selected(False)
        for slot in new_slots:
            slot.selected = True

        # Append all new slots and activate the last one so the user
        # can immediately tweak it. ``add_roi`` is the per-slot
        # appender; we bypass it here for the bulk append to avoid
        # publishing "filter_changed" N times in a row, then publish
        # once at the end via _refresh_all.
        for slot in new_slots:
            state.cluster_rois.append(slot)
        state.cluster_active_index = len(state.cluster_rois) - 1
        toast = (
            f"Added {len(new_slots)} ROI"
            f"{'s' if len(new_slots) != 1 else ''} from {path.name}."
        )
        if dropped_default:
            toast += " Replaced the unused default ROI."
        ui.notify(toast, type="positive")
        _refresh_all()

    def _on_select(index: int) -> None:
        # Re-seed the library_selected_hrf from the slot's anchor so
        # the detail pane + viz click-state mirror what the active
        # ROI was last anchored on. None is fine -- detail pane shows
        # its empty state.
        state.set_active_roi(index)
        active = state.active_roi
        state.library_selected_hrf = active.anchor if active else None
        state.publish(
            "hrtree_selection_changed",
            active.anchor.get("_key") if active and active.anchor else None,
        )
        _refresh_all()

    def _on_delete(index: int) -> None:
        # Use clear_active_roi semantics, but delete the targeted
        # index rather than always the active. Switch active there
        # first so the helper's "remove active, walk back one" logic
        # applies to the right slot. After clearing, the list may be
        # empty -- ``state.active_roi`` returns None and the detail
        # pane reverts to its empty state.
        state.set_active_roi(index)
        state.clear_active_roi()
        active = state.active_roi
        state.library_selected_hrf = active.anchor if active else None
        _refresh_all()

    def _on_shape_dropdown(index: int, value: str) -> None:
        """Per-row shape dropdown handler. Updates the slot's shape
        mode; if switching from atlas to sphere/box, the atlas_label
        is dropped so the descriptor reads cleanly."""
        if not (0 <= index < len(state.cluster_rois)):
            return
        slot = state.cluster_rois[index]
        slot.shape = value
        if value != SHAPE_ATLAS_REGION:
            slot.atlas_label = None
        _refresh_all()

    def _on_region_dropdown(index: int, value) -> None:
        """Per-row region dropdown handler. Atlas mode only."""
        if not (0 <= index < len(state.cluster_rois)):
            return
        state.cluster_rois[index].atlas_label = value or None
        _refresh_all()

    def _on_toggle_visible(index: int) -> None:
        """Flip a slot's ``visible`` flag (true layer-toggle).

        Visibility gates BOTH viz rendering AND save inclusion. The
        user can edit a hidden ROI's parameters without it appearing
        on the viz -- active state is independent of visibility.
        """
        if 0 <= index < len(state.cluster_rois):
            slot = state.cluster_rois[index]
            slot.visible = not slot.visible
            _refresh_all()

    def _on_toggle_selected(index: int, value: bool) -> None:
        """Per-row selection checkbox handler. Marks the slot as part
        of the bulk-edit set so the panel's radius / centre controls
        apply to every selected slot at once."""
        if 0 <= index < len(state.cluster_rois):
            state.cluster_rois[index].selected = bool(value)
            _refresh_all()

    def _on_clear_roi() -> None:
        """Reset the active slot (single-ROI case) or delete it
        (multi-ROI case). Sibling to ``state.clear_active_roi`` but
        also clears the library-side anchor + republishes the
        selection / filter events so dependent panes re-render."""
        state.clear_active_roi()
        state.library_selected_hrf = None
        state.publish("hrtree_selection_changed", None)
        _refresh_all()

    # All three header actions sit together on a single line for
    # vertical density. Buttons stretch with ``flex-1`` so they
    # divide the sub-panel width evenly and hug both outer edges --
    # narrower splitter panes shrink them, wider panes give them
    # more breathing room. Order reads left-to-right: bulk additive
    # (Add Montage) -> additive (Add ROI) -> destructive (Clear) --
    # the destructive action sits on the far right so it's the
    # button furthest from where users typically hover-click.
    clear_label = (
        "Clear"
        if len(state.cluster_rois) <= 1
        else "Delete"
    )
    with ui.row().classes("w-full items-center gap-1 no-wrap"):
        # PR #57: auto-create one sphere ROI per unique channel
        # location from an MNE-compatible file. The handler is
        # async (file picker + load), so it must be passed as a
        # coroutine-function -- a sync lambda wrapping the async
        # call would silently no-op (see gui-async-gotchas memory).
        ui.button(
            "Add montage",
            icon="device_hub",
            on_click=_on_add_montage,
        ).props("flat dense").classes("flex-1").tooltip(
            "Pick a SNIRF / NIRX / FIF file and add a sphere ROI "
            "per unique channel location."
        )
        ui.button(
            "Add ROI", icon="add", on_click=_on_add,
        ).props("flat dense color=primary").classes("flex-1")
        ui.button(
            clear_label, icon="clear", on_click=_on_clear_roi,
        ).props("flat dense").classes("flex-1").tooltip(
            "Reset the active ROI to defaults (when it's the only "
            "ROI) or remove it (when 2+ exist)."
        )

    # Cap the visible height so a long montage scrolls inside the
    # sub-tab instead of stretching the page. ~8 rows at ~32px each.
    # Shape dropdown options: sphere always; atlas only when the
    # bundled atlas loaded. Box stays hidden from the UI until v1.4
    # rotatable-box work.
    shape_options = {SHAPE_SPHERE: "Sphere"}
    if atlas is not None:
        shape_options[SHAPE_ATLAS_REGION] = "Atlas region"

    with ui.column().classes(
        "w-full gap-1 overflow-auto"
    ).style("max-height: 256px"):
        for i, slot in enumerate(state.cluster_rois):
            is_active = i == state.cluster_active_index
            row_classes = (
                "w-full items-center gap-2 px-2 py-1 rounded "
                + (
                    "bg-amber-50 dark:bg-amber-900/30 "
                    "border border-amber-400"
                    if is_active
                    else "hover:bg-gray-100 dark:hover:bg-gray-800"
                )
                + ("" if slot.visible else " opacity-60")
            )
            # No click handler on the outer row anymore -- that handler
            # used to call ``body_refreshable.refresh()`` which would
            # rebuild the DOM mid-click and silently cancel the
            # per-row dropdowns from opening. Selection is wired only
            # on the name label + active arrow / glyph (the user-
            # obvious "row identity" region), which keeps the
            # dropdown clicks free to expand their menus.
            with ui.row().classes(row_classes):
                # Selection checkbox -- bulk-edit toggle. Multiple
                # selected slots become a group target for the radius
                # slider + centre inputs. Auto-selected on Add ROI
                # and Add Montage so the new slots are immediately
                # ready for bulk edits.
                ui.checkbox(
                    value=slot.selected,
                    on_change=lambda e, idx=i: _on_toggle_selected(
                        idx, e.value
                    ),
                ).props("dense size=sm").tooltip(
                    "Tick to include in bulk edits (radius / centre "
                    "apply to all selected ROIs at once)."
                )
                # Active indicator + name -- clicking either selects
                # the row as active. Wrapping them in a clickable
                # element keeps the click target obvious without
                # capturing the dropdowns to its right.
                with ui.row().classes(
                    "items-center gap-2 cursor-pointer"
                ).on("click", lambda _e=None, idx=i: _on_select(idx)):
                    ui.icon(
                        "arrow_right" if is_active else "circle",
                        size="sm" if is_active else "xs",
                    ).classes(
                        "opacity-80" if is_active else "opacity-30"
                    )
                    ui.label(slot.name).classes(
                        "text-sm "
                        + ("font-semibold" if is_active else "")
                    )
                # Spacer pushes the dropdowns + action buttons to the
                # right side of the row so the layout reads
                # "checkbox / name | dropdowns | actions".
                ui.element("div").classes("flex-1")
                # Dropdowns column: shape (always) + region (atlas
                # mode only) stacked vertically on the right side of
                # the row. Region sits directly under the shape
                # dropdown rather than alongside it -- long Harvard-
                # Oxford region names ("Inferior Frontal Gyrus, pars
                # triangularis", etc.) don't fit in a same-row slot
                # without truncation, which was the "I can't see atlas
                # regions once selected" report.
                current_shape = (
                    slot.shape if slot.shape in shape_options
                    else SHAPE_SPHERE
                )
                with ui.column().classes("items-end gap-1"):
                    ui.select(
                        options=shape_options,
                        value=current_shape,
                        on_change=(
                            lambda e, idx=i: _on_shape_dropdown(idx, e.value)
                        ),
                    ).props("dense outlined options-dense").classes("w-48")
                    if slot.shape == SHAPE_ATLAS_REGION and atlas is not None:
                        ui.select(
                            options=atlas.region_names,
                            value=slot.atlas_label,
                            on_change=(
                                lambda e, idx=i: _on_region_dropdown(idx, e.value)
                            ),
                        ).props(
                            "dense outlined options-dense use-input"
                        ).classes("w-48").tooltip(
                            "Pick a Harvard-Oxford cortical region."
                        )
                # Visibility toggle (eye / eye-off). Mirrors a layers
                # panel: clicking hides the ROI from the viz AND from
                # the saved montage.json. Re-clicking restores it.
                # The button click bubbles up to the row's _on_select
                # handler -- intentional, since toggling a slot's
                # visibility usually means the user is focused on
                # that slot anyway.
                vis_icon = (
                    "visibility" if slot.visible else "visibility_off"
                )
                vis_opacity = "opacity-80" if slot.visible else "opacity-40"
                ui.button(
                    icon=vis_icon,
                    on_click=lambda _e=None, idx=i: _on_toggle_visible(idx),
                ).props("flat dense round size=sm").classes(
                    vis_opacity
                ).tooltip(
                    "Hide from viz + save"
                    if slot.visible
                    else "Show on viz + include in save"
                )
                # Stop propagation on the delete click so it doesn't
                # also fire the row's _on_select.
                ui.button(
                    icon="delete_outline",
                    on_click=lambda _e=None, idx=i: _on_delete(idx),
                ).props("flat dense round size=sm").classes("opacity-60")


# ---------------------------------------------------------------------------
# PR #57: ADD MONTAGE per-channel auto-create
# ---------------------------------------------------------------------------


# Default sphere radius (mm) for ROIs auto-created from channel locations.
# A typical fNIRS source-detector pair covers roughly 1.5 cm of cortex
# directly beneath the midpoint, so 15 mm is a sensible default. Mirrors
# the ``ROISlot.radius_mm`` default of 20 mm tuned slightly tighter
# because per-channel montages tend to be denser than free-floating ROIs.
DEFAULT_PER_CHANNEL_RADIUS_MM = 15.0


def _strip_oxygenation_suffix(ch_name: str) -> str:
    """Drop the trailing hbo/hbr/760/850 suffix from a channel name.

    Per-channel ROIs are scientifically one-per-source-detector pair
    (HbO and HbR for the same pair share the optode location), so the
    ROI name should not duplicate the haemoglobin distinction. Returns
    the input unchanged when no recognised suffix is found.

    Pattern: matches ``"... hbo"``, ``"... hbr"``, ``"... 760"``,
    ``"... 850"`` with any whitespace / underscore / hyphen separator.
    """
    import re

    # Trailing oxygenation/wavelength + optional separator before it.
    m = re.match(
        r"^(.*?)[\s_\-]*(hbo|hbr|760|850|760nm|850nm)$",
        ch_name.strip(),
        flags=re.IGNORECASE,
    )
    return m.group(1).rstrip(" _-") if m else ch_name.strip()


def _load_raw_for_montage(path: "Path"):
    """Read a NIRX / SNIRF / FIF file just for its info structure.

    PR #57 helper for the "Add montage" picker -- we only need the
    channel locations, so we preload data (cheap for SNIRF / NIRX
    headers, single fseek for FIF). The loader matches the format
    detection used by ``hrfunc.io.raw_cache._load_from_path`` so the
    picker accepts whatever the project loader does.

    Raises whatever MNE raises on a malformed file; the GUI click
    handler catches and surfaces.
    """
    import mne

    suffix = path.suffix.lower()
    if suffix == ".snirf":
        return mne.io.read_raw_snirf(str(path), verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(str(path), verbose="ERROR")
    # NIRX directories use ``*.hdr`` as the recognisable file in the
    # tree; if the user pointed at a .hdr, MNE wants the parent dir.
    if suffix == ".hdr":
        return mne.io.read_raw_nirx(str(path.parent), verbose="ERROR")
    # Folder fallback (user picked the NIRX directory directly).
    return mne.io.read_raw_nirx(str(path), verbose="ERROR")


def rois_from_raw(
    raw,
    *,
    radius_mm: float = DEFAULT_PER_CHANNEL_RADIUS_MM,
    library_oxygenation: Optional[bool] = None,
) -> "List":
    """Build a list of ROISlots, one per unique channel location.

    PR #57 helper. fNIRS channels naturally come in HbO / HbR pairs
    (or 760 nm / 850 nm pairs in OD space) that share the same
    source-detector midpoint. The naive "one ROI per channel"
    interpretation would emit duplicate spheres at every location;
    the right behaviour is "one ROI per unique optode location" so
    a 40-channel cap turns into 20 ROIs.

    Deduplication is on the rounded mm location to absorb floating-
    point jitter in the MNE info structure (channels often co-locate
    within microns rather than exactly).

    Channel names are normalised to drop the oxygenation suffix
    ("S1_D1 hbo" → "S1_D1") so each ROI carries the source-detector
    label, not a hemoglobin distinction.

    ``library_oxygenation``: when set (True for HbO, False for HbR),
    only channels matching that hemoglobin contribute to the ROI
    locations. None means "use both" and lets HbO+HbR contribute the
    same midpoint (still deduped).

    Returns a fresh list of ROISlots; the caller is responsible for
    appending them to ``state.cluster_rois``. Each slot is named
    "Montage: <ch_name>" so the multi-ROI list visually groups them.

    Module-level + pure so the per-channel logic is testable without
    the GUI.
    """
    from ..state import ROISlot
    from ...spatial.coords import meters_to_mm

    slots: List = []
    seen_keys: Dict[Tuple[int, int, int], int] = {}

    # MNE locations are in meters; we round to a small mm grid for the
    # dedupe key so micron-scale jitter between HbO/HbR doesn't make
    # them look like different channels. 0.1 mm is finer than any
    # realistic fNIRS placement uncertainty.
    DEDUPE_GRID_MM = 0.1

    for channel in raw.info["chs"]:
        ch_name = channel.get("ch_name") or ""
        if not ch_name or ch_name.lower() == "canonical":
            continue

        # Honour the haemoglobin filter when set.
        if library_oxygenation is not None:
            from ..._utils import _is_oxygenated

            try:
                is_hbo = _is_oxygenated(ch_name.lower())
            except (ValueError, LookupError):
                # Channel without an oxygenation suffix -- include it
                # only when the caller hasn't asked to filter.
                continue
            if is_hbo != library_oxygenation:
                continue

        loc = channel.get("loc")
        if loc is None or len(loc) < 3:
            continue
        x_m, y_m, z_m = float(loc[0]), float(loc[1]), float(loc[2])
        # MNE channels without a recorded location report (0, 0, 0).
        # Emitting a sphere at the origin would be meaningless and
        # pile every such channel on top of each other.
        if x_m == 0.0 and y_m == 0.0 and z_m == 0.0:
            continue

        x_mm = meters_to_mm(x_m)
        y_mm = meters_to_mm(y_m)
        z_mm = meters_to_mm(z_m)

        dedupe_key = (
            int(round(x_mm / DEDUPE_GRID_MM)),
            int(round(y_mm / DEDUPE_GRID_MM)),
            int(round(z_mm / DEDUPE_GRID_MM)),
        )
        if dedupe_key in seen_keys:
            # Same location as a previous channel (most commonly its
            # HbO or HbR partner). Skip so we emit one sphere per
            # source-detector pair.
            continue
        seen_keys[dedupe_key] = len(slots)

        display = _strip_oxygenation_suffix(ch_name)
        slots.append(
            ROISlot(
                name=f"Montage: {display}",
                shape=SHAPE_SPHERE,
                center_x_mm=x_mm,
                center_y_mm=y_mm,
                center_z_mm=z_mm,
                radius_mm=float(radius_mm),
            )
        )
    return slots


def _describe_slot(slot) -> str:
    """One-line summary of a ROISlot for the list secondary text."""
    if slot.shape == SHAPE_ATLAS_REGION:
        return slot.atlas_label or "(no region)"
    centre = f"({slot.center_x_mm:.0f}, {slot.center_y_mm:.0f}, {slot.center_z_mm:.0f})"
    if slot.shape == SHAPE_BOX:
        return (
            f"box {centre} ±"
            f"({slot.box_half_x_mm:.0f},"
            f"{slot.box_half_y_mm:.0f},"
            f"{slot.box_half_z_mm:.0f}) mm"
        )
    return f"sphere {centre} r={slot.radius_mm:.0f} mm"


def _render_atlas_alignment_section(state: AppState, body_refreshable) -> None:
    """Render the alignment controls in atlas mode.

    Three pieces:

    1. Out-of-MNI warning when HRF locations look like MNE head coords.
       Helps the user understand why atlas mode shows empty ROIs.
    2. Three offset number inputs (x / y / z mm) for users who want
       to dial in a rough translation by eye.
    3. An upload widget for a JSON 4x4 affine matrix. Cleared via a
       "reset" button.
    """
    import json as _json
    import numpy as np

    matched = filter_by_oxygenation(
        apply_filter(gather_library_hrfs(state), state.library_filter),
        state.library_oxygenation,
    )
    if _looks_out_of_mni(matched):
        with ui.row().classes("w-full items-start gap-2"):
            ui.icon("warning").classes("text-amber-500 text-sm")
            ui.label(
                "HRF locations appear to be in MNE head coords (not MNI). "
                "Atlas membership will be inaccurate until you load an "
                "alignment matrix or set offsets below."
            ).classes("text-xs opacity-80")

    ui.label("Atlas alignment (HRF coord -> MNI mm)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )

    # --- Offset inputs ---
    def _make_offset_input(axis: str, attr: str):
        def _on_change(event) -> None:
            try:
                value = float(event.value or 0.0)
            except (TypeError, ValueError):
                return
            setattr(state, attr, value)
            state.publish("hrtree_filter_changed", state.library_filter)
            body_refreshable.refresh()

        return ui.number(
            label=axis,
            value=getattr(state, attr),
            step=1.0,
            format="%.1f",
            on_change=_on_change,
        ).props("dense").classes("w-20")

    with ui.row().classes("w-full gap-2"):
        _make_offset_input("dx", "cluster_atlas_offset_x_mm")
        _make_offset_input("dy", "cluster_atlas_offset_y_mm")
        _make_offset_input("dz", "cluster_atlas_offset_z_mm")

    # --- Affine matrix upload ---
    def _on_upload(event) -> None:
        try:
            content = event.content.read()
            payload = _json.loads(content.decode("utf-8"))
            # Accept either {"affine_mm": [[...], ...]} or a bare
            # nested-list 4x4. Both make sense as user input.
            raw = (
                payload.get("affine_mm")
                if isinstance(payload, dict) and "affine_mm" in payload
                else payload
            )
            affine = np.asarray(raw, dtype=np.float64)
            if affine.shape != (4, 4):
                raise ValueError(
                    f"affine must be 4x4, got shape {affine.shape}"
                )
            state.cluster_atlas_alignment_affine = affine
            ui.notify(
                "Loaded HRF -> MNI alignment matrix.", type="positive"
            )
            state.publish("hrtree_filter_changed", state.library_filter)
            body_refreshable.refresh()
        except Exception as exc:  # noqa: BLE001
            ui.notify(
                f"Failed to load alignment: {type(exc).__name__}: {exc}",
                type="negative",
            )

    def _on_reset_alignment() -> None:
        state.cluster_atlas_alignment_affine = None
        state.cluster_atlas_offset_x_mm = 0.0
        state.cluster_atlas_offset_y_mm = 0.0
        state.cluster_atlas_offset_z_mm = 0.0
        state.publish("hrtree_filter_changed", state.library_filter)
        body_refreshable.refresh()

    with ui.row().classes("w-full gap-2 items-center"):
        ui.upload(
            label="Load alignment .json",
            on_upload=_on_upload,
            auto_upload=True,
            max_files=1,
        ).props("flat dense accept=.json").classes("w-48")
        ui.button(
            "Reset alignment",
            on_click=_on_reset_alignment,
        ).props("flat dense")

    ui.label(_atlas_alignment_status(state)).classes(
        "text-xs font-mono opacity-70"
    )


# ---------------------------------------------------------------------------
# Center pane: HRtree 3D viz
# ---------------------------------------------------------------------------


def _render_viz_pane(state: AppState) -> None:
    """The plotly 3D scatter of HRF locations.

    Refreshable so the Apply button can re-render against the filter.

    The refreshable's container (a NiceGUI ``RefreshableContainer``
    custom element) defaults to inline-ish display, which breaks the
    ``h-full`` chain — the plotly viz would otherwise shrink to its
    content's intrinsic height and leave the bottom half of the
    splitter pane empty. We apply ``w-full h-full`` to the container
    after the first call so the body fills the available height.
    """

    # Stable, zero-footprint anchor for deferred ``once=True`` timers (the
    # plotly resize hook + the on-selection viz re-centre). It lives OUTSIDE
    # ``_viz_body`` so a ``_viz_body.refresh()`` can't delete the slot a
    # pending timer is parented to — which is what raised "parent slot of the
    # element has been deleted" when clicking through HRFs quickly.
    _deferred_anchor = ui.element("div").style("display:none")
    _pending_viz_timer: dict = {"timer": None}

    @ui.refreshable
    def _viz_body() -> None:
        all_hrfs = gather_library_hrfs(state)
        matched = filter_by_oxygenation(
            apply_filter(all_hrfs, state.library_filter),
            state.library_oxygenation,
        )

        if not all_hrfs:
            with ui.column().classes("p-6 gap-2"):
                ui.label("HRtree").classes("text-2xl font-semibold")
                if state.last_error:
                    ui.label(state.last_error).classes(
                        "text-sm text-red-400"
                    )
                else:
                    ui.label(
                        "Library trees not loaded. Re-opening the HRtree tab "
                        "may help."
                    ).classes("text-sm opacity-60")
            return

        # Use a flex column so the plotly viz can claim flex-1 and the
        # overlay-toggles row sits underneath at content-height.
        with ui.column().classes("w-full h-full p-3 gap-2 flex flex-col"):
            # Compute the union of ROI keys across every VISIBLE ROI
            # so the figure highlights everything currently in scope
            # (multi-ROI visibility flow). Each visible slot's shape
            # overlay also renders independently below.
            union_roi_keys, visible_pairs = _visible_roi_keys(
                state, matched,
            )
            visible_shapes = [shape for _slot, shape in visible_pairs]
            roi_status = ""
            if union_roi_keys:
                n_visible = len(visible_pairs)
                roi_status = (
                    f"  •  ROI: {len(union_roi_keys)} highlighted "
                    f"across {n_visible} visible ROI"
                    f"{'s' if n_visible != 1 else ''}"
                )
            ui.label(
                f"{len(matched)} HRFs shown{roi_status}"
            ).classes("text-sm opacity-70")
            fig = build_plotly_figure(
                matched,
                show_brain=state.library_show_brain,
                show_scalp=state.library_show_scalp,
                show_info=state.library_show_info,
                roi_keys=union_roi_keys,
                roi_shapes=visible_shapes,
            )
            plot = ui.plotly(fig).classes("w-full flex-1 min-h-0")

            def _on_click(event) -> None:
                # NiceGUI's plotly click event delivers an args dict with a
                # 'points' list. Each point has a 'customdata' field if we
                # set it on the trace, which we use to store the HRF key.
                hrf_key = _extract_clicked_hrf_key(event)
                if hrf_key is None:
                    return
                hrf = matched.get(hrf_key) or all_hrfs.get(hrf_key)
                if hrf is None:
                    return
                # Stash the key on the dict so the detail pane can show it.
                # A plain click also resets the painted set -- the user is
                # picking a fresh anchor, so accumulated shift-hover paint
                # from a prior anchor shouldn't carry over.
                anchor_dict = {**hrf, "_key": hrf_key}
                state.library_selected_hrf = anchor_dict
                # PR #55: also stamp the anchor onto the *active* ROI
                # slot so it travels with the slot (and into the saved
                # montage). Pre-PR-#55 the anchor was a single global
                # field; now each ROI keeps its own. Since the 2026-05-16
                # layout refactor cluster_rois defaults to EMPTY, so on a
                # fresh library load active_roi is None -- guard the stamp
                # rather than dereferencing None (which aborted the handler
                # before the selection event published, so the very first
                # HRF click did nothing visible). The centre-seed proxy
                # writes and the painted-set clear below already no-op when
                # there's no active slot.
                active = state.active_roi
                if active is not None:
                    active.anchor = anchor_dict
                # Seed the Cluster sub-tab's shape centre from the clicked
                # HRF so sphere mode's behaviour matches the v1.2 "click
                # an HRF, sphere centres on it" workflow even though the
                # spatial layer now drives the centre from state. Box mode
                # uses the same centre so a click also re-centres the box.
                # HRF locations are stored in meters; spatial layer is mm.
                loc = hrf.get("location") or [0, 0, 0]
                if len(loc) >= 3:
                    state.cluster_center_x_mm = float(loc[0]) * 1000.0
                    state.cluster_center_y_mm = float(loc[1]) * 1000.0
                    state.cluster_center_z_mm = float(loc[2]) * 1000.0
                state.library_roi_painted.clear()
                state.publish("hrtree_selection_changed", hrf_key)

            def _on_paint(event) -> None:
                # Shift+hover fired our custom roi_paint event with a key
                # in event.args. Add to painted set, refresh viz + detail.
                args = getattr(event, "args", None) or {}
                key = args.get("key") if isinstance(args, dict) else None
                if not key or key not in matched:
                    return
                # Only paint HRFs that match the anchor's oxygenation so
                # the average trace doesn't mix HbO + HbR (different
                # physiological signals, scientifically wrong to average).
                anchor_inner = state.library_selected_hrf
                if anchor_inner is not None:
                    if matched[key].get("oxygenation") != anchor_inner.get("oxygenation"):
                        return
                if key in state.library_roi_painted:
                    return  # already painted, no-op
                state.library_roi_painted.add(key)
                state.publish("hrtree_selection_changed", key)

            plot.on("plotly_click", _on_click)
            plot.on("roi_paint", _on_paint)
            # Wire the JS shift-tracker + plotly_hover hook AFTER the
            # plotly element has rendered. Slight delay so the
            # underlying div is queryable in the DOM. once=True so
            # the hook isn't registered repeatedly on each refresh.
            _install_paint_hook(plot.id, _deferred_anchor)

            # MNI overlay toggles under the viz — they control what's
            # rendered above (brain mesh, scalp mesh), so visual
            # adjacency makes them easier to discover than the legacy
            # left-sidebar placement. Independent switches so users can
            # show either, both, or neither.
            def _publish_filter_change() -> None:
                state.publish("hrtree_filter_changed", state.library_filter)

            def _on_brain_toggle(event) -> None:
                state.library_show_brain = bool(event.value)
                _publish_filter_change()

            def _on_scalp_toggle(event) -> None:
                state.library_show_scalp = bool(event.value)
                _publish_filter_change()

            def _on_info_toggle(event) -> None:
                state.library_show_info = bool(event.value)
                _publish_filter_change()

            with ui.row().classes(
                "w-full items-center justify-center gap-6 shrink-0 pt-1"
            ):
                ui.switch(
                    "Show MNI brain",
                    value=state.library_show_brain,
                    on_change=_on_brain_toggle,
                ).props("dense").tooltip(
                    "Translucent fsaverage pial cortical surface beneath "
                    "the HRF scatter — where the neural activity originates."
                )
                ui.switch(
                    "Show MNI head",
                    value=state.library_show_scalp,
                    on_change=_on_scalp_toggle,
                ).props("dense").tooltip(
                    "Translucent fsaverage scalp (outer-skin) surface — "
                    "where forehead/head-mounted fNIRS optodes physically sit."
                )
                ui.switch(
                    "Show info",
                    value=state.library_show_info,
                    on_change=_on_info_toggle,
                ).props("dense").tooltip(
                    "Show the per-HRF metadata popup on hover in the 3D view. "
                    "Turn off to declutter dense clouds — the detail column on "
                    "the right stays, and clicks / shift-hover ROI painting "
                    "still work."
                )

    _viz_body()
    # Note: NiceGUI's ``RefreshableContainer`` template is just a
    # ``<slot>``, which Vue renders as a fragment with no root DOM
    # element. That means classes / styles applied to the container
    # have nowhere to land (Vue prints a "non-prop attribute could not
    # be inherited" warning). The refreshable's children are direct
    # layout children of the splitter slot already, so no wrapper-
    # styling is needed — height propagates through the slot directly.

    # Re-render the viz on both filter and selection change.
    #
    # Selection changes (clicking an HRF, shift-paint adding to the
    # painted set) update both ``state.library_selected_hrf`` AND
    # ``state.cluster_center_*_mm`` (the click handler seeds the
    # cluster centre from the clicked HRF's location). Without the
    # selection_changed subscription, the figure's shape overlay
    # would stay at the old centre until the user happened to toggle
    # something on the Filter sub-tab or the MNI overlay switches --
    # confusing because the detail pane + Cluster sub-tab DO update
    # immediately (both subscribe to selection_changed), so the user
    # sees the readout update while the 3D shape stays put.
    def _refresh_viz(_payload=None) -> None:
        _viz_body.refresh()

    def _refresh_viz_after_event(_payload=None) -> None:
        # Defer the rebuild to the next tick. The HRF click handler
        # (``_on_click``) lives INSIDE ``_viz_body``; refreshing the body
        # synchronously from there tears down the very plotly element that's
        # mid-click, which drops that event's UI update batch — the detail
        # pane (which refreshes later in the same ``publish``) then wouldn't
        # repaint until the user's NEXT interaction (e.g. toggling a switch).
        # Deferring lets the click event finish and flush first, then we
        # rebuild the viz to re-centre the ROI shape overlay on the new anchor.
        #
        # Cancel any still-pending deferred refresh first: clicking through
        # HRFs faster than 0.05 s would otherwise stack timers, and a fired
        # one's ``_viz_body.refresh()`` could delete a sibling timer's slot
        # mid-flight. The timer is parked on the stable ``_deferred_anchor``
        # (not the refreshable body) so it survives the rebuild it triggers.
        prev = _pending_viz_timer["timer"]
        if prev is not None and not prev.is_deleted:
            prev.cancel()
        if _deferred_anchor.is_deleted:
            return
        with _deferred_anchor:
            _pending_viz_timer["timer"] = ui.timer(
                0.05, _refresh_viz, once=True
            )

    state.subscribe("hrtree_filter_changed", _refresh_viz)
    # Selection comes from a plotly click inside this body, so defer (above).
    state.subscribe("hrtree_selection_changed", _refresh_viz_after_event)


def _extract_clicked_hrf_key(event) -> Optional[str]:
    """Pull the clicked HRF's key from a plotly click event payload."""
    try:
        args = getattr(event, "args", None) or {}
        points = args.get("points") or []
        if not points:
            return None
        first = points[0]
        return first.get("customdata")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "hrtree: failed to extract HRF key from click event: %s", exc
        )
        return None


# ---------------------------------------------------------------------------
# Right pane: HRF detail
# ---------------------------------------------------------------------------


def _render_detail_pane(state: AppState) -> None:
    """The selected-HRF detail card.

    Shows context metadata + a matplotlib trace plot for the picked HRF.
    """

    @ui.refreshable
    def _detail_body() -> None:
        hrf = state.library_selected_hrf
        with ui.column().classes("w-full h-full p-4 gap-3 overflow-auto"):
            ui.label("Detail").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            # When an ROI is the current selection, lead with the ROI's
            # aggregate HRF (its averaged trace) instead of a single anchor
            # HRF — that's the thing the user picked.
            if _showing_active_roi(state) and _render_active_roi_detail(state):
                return
            if hrf is None:
                ui.label("Click an HRF in the viz to inspect.").classes(
                    "text-sm opacity-60"
                )
                return

            key = hrf.get("_key", "")
            ui.label(key or "(no key)").classes(
                "text-lg font-mono break-all"
            )

            _kv("oxygenation", "HbO" if hrf.get("oxygenation") else "HbR")
            _kv("sfreq", f"{float(hrf.get('sfreq', 0)):.4g} Hz")
            loc = hrf.get("location") or [0, 0, 0]
            _kv(
                "location",
                f"x={loc[0]:.3f}  y={loc[1]:.3f}  z={loc[2]:.3f}",
            )
            trace = hrf.get("hrf_mean") or []
            _kv("trace length", str(len(trace)))

            context = hrf.get("context") or {}
            if context:
                ui.separator()
                ui.label("Context").classes(
                    "text-xs uppercase opacity-60 tracking-wide"
                )
                # Cap the context block to its own scroll area so a long
                # context (many populated fields) can't push the trace plot
                # down / squish it — it scrolls within this box instead while
                # the Trace section below keeps its space.
                with ui.column().classes(
                    "w-full max-h-40 overflow-auto gap-1 pr-1 shrink-0"
                ):
                    for ctx_key, value in context.items():
                        if value is None:
                            continue
                        _kv(ctx_key, str(value))

            if trace:
                ui.separator()
                ui.label(
                    f"Trace · {'HbO' if hrf.get('oxygenation') else 'HbR'}"
                ).classes("text-xs uppercase opacity-60 tracking-wide")
                png = _render_trace_png(hrf)
                if png is not None:
                    # shrink-0 so the plot keeps its natural size regardless
                    # of how much context sits above it.
                    ui.image(png).classes("max-w-md shrink-0")

            # ROI average plot (only renders when the ROI has at least
            # 2 averageable same-oxygenation HRFs — fewer than that
            # means there's nothing useful to average).
            _render_roi_average(state)

    _detail_body()
    # (See _render_viz_pane note on refreshable wrappers — no styling
    # needed; RefreshableContainer renders as a Vue fragment with no
    # root DOM element.)

    def _refresh_detail(_payload=None) -> None:
        _detail_body.refresh()
    state.subscribe("hrtree_selection_changed", _refresh_detail)
    # The radius slider publishes ``hrtree_filter_changed`` because the
    # viz already listens there; the detail pane needs to refresh too
    # so the ROI-average plot updates when the user widens the radius.
    state.subscribe("hrtree_filter_changed", _refresh_detail)


def _showing_active_roi(state: AppState) -> bool:
    """True when the detail pane should lead with the active ROI's HRF.

    The user "selected an ROI" (vs. clicking a lone HRF) when the current
    selection IS that ROI's own anchor object — selecting a slot re-seeds
    ``library_selected_hrf = active.anchor`` (same identity). An anchorless
    shape ROI selects with ``library_selected_hrf = None`` while still having
    members, so a ``None`` selection with an active ROI also counts. Clicking
    a different individual HRF makes a fresh dict, so identity differs and the
    single-HRF detail shows instead.
    """
    active = state.active_roi
    if active is None:
        return False
    sel = state.library_selected_hrf
    return sel is None or sel is active.anchor


def _roi_average_oxygenations(state: AppState) -> "list":
    """Oxygenations to render ROI averages for — NEVER a mixed pool.

    Averaging HbO and HbR traces together is scientifically wrong (they are
    inverse responses, so the pool cancels into a meaningless trace). So:
    - a determinate oxygenation (the clicked anchor's, or the HbO/HbR filter)
      → that single value;
    - indeterminate ("Both" with no anchor) → BOTH, rendered as separate,
      labelled HbO and HbR averages rather than one pooled trace.
    """
    oxy = _resolve_cluster_oxygenation(state)
    if oxy is not None:
        return [bool(oxy)]
    return [True, False]


def _render_one_roi_average(
    state: AppState, matched, shape, alignment, oxy: bool, *, header: str
) -> bool:
    """Render a single oxygenation-pure ROI average, labelled HbO/HbR.

    Returns True when something rendered (the ROI had ≥2 averageable
    same-oxygenation members), False otherwise.
    """
    roi_keys = compute_roi_keys_by_shape(
        matched, shape, state.library_roi_painted,
        oxygenation_filter=oxy,
        alignment_affine=alignment,
    )
    result = compute_roi_average(matched, roi_keys)
    if result is None:
        return False
    mean, std, n_estimates, n_channels = result
    sfreq = _resolve_roi_sfreq(state.library_selected_hrf, matched, roi_keys)
    tag = "HbO" if oxy else "HbR"
    # ``n_estimates`` is the count of pooled subject-level estimate TRACES
    # across all ROI channels (one subject contributing on K channels adds K
    # traces) -- it is NOT a distinct-subject count. Label it "estimates" so
    # the readout doesn't overstate the sample as N subjects.
    ui.label(
        f"{header} · {tag} ({n_estimates} estimates from {n_channels} channels)"
    ).classes("text-xs uppercase opacity-60 tracking-wide")
    png = _render_roi_average_png(mean, std, sfreq, n_estimates)
    if png is not None:
        ui.image(png).classes("max-w-md shrink-0")
    return True


def _render_active_roi_detail(state: AppState) -> bool:
    """Render the active ROI's aggregate HRF(s) as the primary detail.

    Returns True when it rendered (the ROI has an averageable HRF), False when
    there's nothing to show — the caller then falls back to the single-HRF
    detail. Renders one labelled average PER oxygenation so HbO and HbR are
    never pooled together.
    """
    active = state.active_roi
    if active is None:
        return False
    all_hrfs = gather_library_hrfs(state)
    matched = filter_by_oxygenation(
        apply_filter(all_hrfs, state.library_filter),
        state.library_oxygenation,
    )
    shape = _build_current_shape(state)
    alignment = _alignment_for_shape(state, shape)
    oxygenations = _roi_average_oxygenations(state)

    name = getattr(active, "name", None) or "ROI"
    rendered = False
    # Emit the ROI title/header exactly ONCE, before the first oxygenation
    # pass. Gating on ``rendered`` (the old behaviour) re-emitted the header
    # for the second oxygenation whenever the first one rendered nothing,
    # producing a doubled "ROI HRF" title above a single HbR plot.
    header_shown = False
    for oxy in oxygenations:
        if not header_shown:
            ui.label(name).classes("text-lg font-mono break-all")
            ui.separator()
            ui.label("ROI HRF (averaged trace)").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            header_shown = True
        rendered |= _render_one_roi_average(
            state, matched, shape, alignment, oxy, header="ROI HRF",
        )
    return rendered


def _render_roi_average(state: AppState) -> None:
    """Render the averaged-trace plot(s) for the current ROI.

    Read-only display below the single-HRF detail. Renders one labelled
    average per oxygenation (HbO and HbR are never pooled together — see
    :func:`_roi_average_oxygenations`), staying in sync with the Cluster
    sub-tab's current shape + filters.
    """
    all_hrfs = gather_library_hrfs(state)
    matched = filter_by_oxygenation(
        apply_filter(all_hrfs, state.library_filter),
        state.library_oxygenation,
    )
    shape = _build_current_shape(state)
    alignment = _alignment_for_shape(state, shape)
    sep_done = False
    for oxy in _roi_average_oxygenations(state):
        roi_keys = compute_roi_keys_by_shape(
            matched, shape, state.library_roi_painted,
            oxygenation_filter=oxy, alignment_affine=alignment,
        )
        if compute_roi_average(matched, roi_keys) is None:
            continue
        if not sep_done:
            ui.separator()
            sep_done = True
        _render_one_roi_average(
            state, matched, shape, alignment, oxy, header="ROI average",
        )


def _render_roi_average_png(
    mean, std, sfreq: float, n: int
) -> Optional[str]:
    """Plot ROI-averaged trace with ±1 std shading."""
    try:
        import base64
        import io as _io
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for ROI average: %s", exc)
        return None
    fig = None
    try:
        t = np.arange(len(mean)) / sfreq
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
        ax.plot(t, mean, lw=1.4, color="#f59e0b", label=f"mean (n={n})")
        ax.fill_between(
            t, mean - std, mean + std,
            alpha=0.18, color="#f59e0b",
            label="±1 std",
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (a.u.)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROI average render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def _kv(key: str, value: str) -> None:
    with ui.row().classes("w-full gap-4"):
        ui.label(key).classes("text-xs uppercase opacity-60 w-32")
        ui.label(value).classes("text-sm break-all")


def _resolve_roi_sfreq(
    anchor: Optional[Dict[str, Any]],
    hrfs: Dict[str, Dict[str, Any]],
    roi_keys: Any,
) -> float:
    """Pick the sample rate for an ROI-averaged trace.

    Prefers the click-anchor's ``sfreq`` when there is one (matches the
    v1.2 behaviour). For free-floating ROIs with no anchor, falls back
    to the first ROI member's ``sfreq``. Final fallback is 1.0 Hz so
    the time axis on the plot has *some* scale even when the HRFs are
    missing rate metadata.
    """
    if anchor is not None:
        anchor_sfreq = float(anchor.get("sfreq") or 0.0)
        if anchor_sfreq > 0:
            return anchor_sfreq
    for key in roi_keys:
        hrf = hrfs.get(key)
        if hrf is None:
            continue
        member_sfreq = float(hrf.get("sfreq") or 0.0)
        if member_sfreq > 0:
            return member_sfreq
    return 1.0


# ---------------------------------------------------------------------------
# Data helpers (module-level so tests can call them)
# ---------------------------------------------------------------------------


def gather_library_hrfs(state: AppState) -> Dict[str, Dict[str, Any]]:
    """Combine the HbO + HbR trees into a single name → HRF-dict map.

    Returns empty dict if the trees aren't loaded.

    Two filters applied while merging:

    1. **Global sentinels excluded.** ``montage.estimate_hrf`` and friends
       seed every Montage with ``global_hbo`` / ``global_hbr`` placeholder
       entries at the sentinel location ``[~360, ~360, ~360]`` (out-of-
       MNI-range so they don't collide with real optodes — see
       ``montage._merge_montages`` and ``tree.get_canonical_hrf``). Those
       entries leak into the bundled HRF databases when a researcher
       saves their montage. They have no business in the user-facing
       library browser, and at ``[360, 360, 360]`` they dominate
       plotly's ``aspectmode="data"`` axis range, compressing the real
       optode cluster (~0.07 m) to a single invisible pixel. Skip
       anything whose key starts with ``global_``.
    2. **Re-keyed by oxygenation prefix.** The bundled HbO and HbR
       JSONs share at least one key (``s8_d4_hbr-temp`` appears in
       both — community-contributed entries can be duplicated across
       files). A plain ``dict.update`` would silently drop one copy on
       collision. Prefixing with ``hbo:`` / ``hbr:`` preserves both
       oxygenation flavors even when their optode-pair keys match.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for tree_obj, prefix in (
        (state.library_hbo, "hbo:"),
        (state.library_hbr, "hbr:"),
    ):
        if tree_obj is None:
            continue
        hrfs = tree_obj.gather(tree_obj.root)
        if not hrfs:
            continue
        for key, hrf in hrfs.items():
            if key.startswith("global_"):
                continue
            out[f"{prefix}{key}"] = hrf
    return out


def filter_by_oxygenation(
    hrfs: Dict[str, Dict[str, Any]],
    mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Filter HRFs by oxygenation channel.

    Args:
        hrfs: name → HRF-dict map (post-context-filter).
        mode: ``"both"`` returns unchanged; ``"hbo"`` keeps only HRFs
            with ``oxygenation is True``; ``"hbr"`` keeps only HRFs
            with ``oxygenation is False``. Unknown mode strings
            fall through as ``"both"`` so a typo doesn't blank the
            entire viz.

    Module-level so tests can hit it without spinning up the GUI.
    """
    if mode == "hbo":
        return {k: v for k, v in hrfs.items() if v.get("oxygenation") is True}
    if mode == "hbr":
        return {k: v for k, v in hrfs.items() if v.get("oxygenation") is False}
    return dict(hrfs)


def apply_filter(
    hrfs: Dict[str, Dict[str, Any]],
    filter_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Filter an HRF dict by case-insensitive substring match on context.

    Each key in ``filter_kwargs`` must appear in the HRF's context with
    a value whose string representation contains the filter value (case-
    insensitive). HRFs missing a filtered key are excluded. Empty filter
    returns the input unchanged.
    """
    if not filter_kwargs:
        return dict(hrfs)

    out: Dict[str, Dict[str, Any]] = {}
    for key, hrf in hrfs.items():
        context = hrf.get("context") or {}
        if _hrf_matches_filter(context, filter_kwargs):
            out[key] = hrf
    return out


def _hrf_matches_filter(
    context: Dict[str, Any],
    filter_kwargs: Dict[str, Any],
) -> bool:
    """True if every (key, value) in the filter is reflected in context."""
    for field, needle in filter_kwargs.items():
        if needle is None or needle == "":
            continue
        haystack = context.get(field)
        if haystack is None:
            return False
        if isinstance(haystack, (list, tuple)):
            if not any(_str_match(item, needle) for item in haystack):
                return False
        else:
            if not _str_match(haystack, needle):
                return False
    return True


def _str_match(value: Any, needle: str) -> bool:
    if value is None:
        return False
    return str(needle).lower() in str(value).lower()


# ---------------------------------------------------------------------------
# Plotly figure builder
# ---------------------------------------------------------------------------


def build_plotly_figure(
    hrfs: Dict[str, Dict[str, Any]],
    *,
    show_brain: bool = False,
    show_scalp: bool = False,
    show_info: bool = True,
    roi_keys: Optional[Any] = None,
    roi_shape: Optional[Shape] = None,
    roi_shapes: Optional[List[Shape]] = None,
):
    """Build the 3D scatter figure for the given HRF dict.

    Up to six traces, ordered so the ROI highlight renders on top:

    - **Scalp** (``go.Mesh3d``, only when ``show_scalp=True``):
      fsaverage outer-skin surface -- anatomically where the optodes
      sit. Drawn first (outermost in 3D-painter order).
    - **Brain** (``go.Mesh3d``, only when ``show_brain=True``):
      fsaverage pial cortical surface -- where the neural activity
      originates. Drawn inside the scalp.
    - **ROI shape overlay** (``go.Mesh3d``, only when ``roi_shape``
      is a :class:`~hrfunc.spatial.shapes.Box` or
      :class:`~hrfunc.spatial.shapes.Sphere`): a translucent violet
      cuboid / UV-sphere showing where the Cluster sub-tab's ROI
      selector currently sits. The shape's centre/extent state is
      converted from MNI mm to MNE-meter coordinates so it renders
      in the same coordinate frame as the HRF scatter.
    - **HbO** (``go.Scatter3d``, red): oxygenated HRFs.
    - **HbR** (``go.Scatter3d``, blue): deoxygenated HRFs.
    - **ROI** (``go.Scatter3d``, gold, larger): every HRF whose key
      is in ``roi_keys``. Drawn last so the highlight sits above the
      regular markers. Skipped when ``roi_keys`` is None or empty.

    Mesh overlays have ``hoverinfo="skip"`` and ``showlegend=False``
    so they don't clutter legend / hover UX. Each scatter point's
    ``customdata`` is the HRF key for click + shift-hover handlers.
    """
    import plotly.graph_objects as go

    hbo_x, hbo_y, hbo_z, hbo_keys, hbo_hover = [], [], [], [], []
    hbr_x, hbr_y, hbr_z, hbr_keys, hbr_hover = [], [], [], [], []

    for key, hrf in hrfs.items():
        loc = hrf.get("location")
        # Skip HRFs without a real 3D location rather than fabricating
        # (0,0,0) — clustering location-less nodes at the origin would be
        # visually misleading and the GUI's spatial story (kd-tree) only
        # makes sense for HRFs with measured coordinates.
        if loc is None or len(loc) < 3:
            continue
        is_hbo = bool(hrf.get("oxygenation"))
        hover = _hover_text_for(key, hrf)
        if is_hbo:
            hbo_x.append(loc[0])
            hbo_y.append(loc[1])
            hbo_z.append(loc[2])
            hbo_keys.append(key)
            hbo_hover.append(hover)
        else:
            hbr_x.append(loc[0])
            hbr_y.append(loc[1])
            hbr_z.append(loc[2])
            hbr_keys.append(key)
            hbr_hover.append(hover)

    traces = []

    # Overlay meshes first (scalp outside, brain inside, both more
    # transparent than the HRF markers) so the HRF scatter renders on
    # top. Scalp is drawn FIRST so it's the outermost in 3D-painter
    # order — when both are on, the brain visually nests inside the head.
    if show_scalp:
        mesh = load_mesh("scalp")
        if mesh is not None:
            verts, faces = mesh
            traces.append(
                make_surface_trace(
                    verts, faces,
                    color="#c4b5a0",  # warm skin tone
                    opacity=0.12,
                    name="MNI head",
                    ambient=0.6, diffuse=0.5,
                )
            )
    if show_brain:
        mesh = load_mesh("pial")
        if mesh is not None:
            verts, faces = mesh
            traces.append(
                make_surface_trace(
                    verts, faces,
                    color="#9ca3af",  # cool grey for cortex
                    opacity=0.30,
                    name="MNI brain",
                    ambient=0.5, diffuse=0.6,
                )
            )

    # ROI shape overlay (PR #49 box/sphere). HRF coords are in meters;
    # the spatial-layer shape is in mm, so we down-convert before
    # building the trace -- the resulting Mesh3d is in meters and
    # overlays directly on the HRF scatter.
    #
    # Two parameter forms: ``roi_shape`` (legacy single-shape, kept
    # for back-compat) and ``roi_shapes`` (list, used by the multi-
    # ROI visibility flow). Internally normalised into one list and
    # rendered with one overlay trace per shape.
    shape_list: List[Shape] = []
    if roi_shapes:
        shape_list.extend(roi_shapes)
    elif roi_shape is not None:
        shape_list.append(roi_shape)
    for one_shape in shape_list:
        if isinstance(one_shape, Box):
            box_m = Box(
                center_mm=(c / 1000.0 for c in one_shape.center_mm),
                half_extents_mm=(h / 1000.0 for h in one_shape.half_extents_mm),
            )
            traces.append(make_box_overlay_trace(box_m))
        elif isinstance(one_shape, Sphere):
            sphere_m = Sphere(
                center_mm=(c / 1000.0 for c in one_shape.center_mm),
                radius_mm=one_shape.radius_mm / 1000.0,
            )
            traces.append(make_sphere_overlay_trace(sphere_m))

    # "text" shows the per-HRF metadata popup on hover; "none" hides it but
    # still fires hover/click events, so ROI clicks + shift-hover paint keep
    # working with the popups off (the "Show info" toggle).
    hover_mode = "text" if show_info else "none"
    if hbo_x:
        traces.append(
            go.Scatter3d(
                x=hbo_x, y=hbo_y, z=hbo_z,
                mode="markers",
                # HbO and HbR for the same optode pair share the exact 3D
                # location (one source-detector → two measurements at one
                # spot). Plotly Scatter3d draws traces in order, so without
                # distinct symbols the second trace fully occludes the first.
                # Distinct symbols + sizes keep both visible at the same xyz.
                marker=dict(
                    size=6,
                    color="#fb7185",
                    opacity=0.9,
                    symbol="circle",
                    line=dict(width=1, color="#7f1d1d"),
                ),
                name="HbO",
                customdata=hbo_keys,
                hovertext=hbo_hover,
                hoverinfo=hover_mode,
            )
        )
    if hbr_x:
        traces.append(
            go.Scatter3d(
                x=hbr_x, y=hbr_y, z=hbr_z,
                mode="markers",
                marker=dict(
                    size=4,
                    color="#38bdf8",
                    opacity=0.85,
                    symbol="diamond",
                    line=dict(width=1, color="#0c4a6e"),
                ),
                name="HbR",
                customdata=hbr_keys,
                hovertext=hbr_hover,
                hoverinfo=hover_mode,
            )
        )

    # ROI highlight — drawn LAST so the gold markers visually sit above
    # the regular HbO/HbR scatter (same points still appear in the
    # underlying trace; the ROI layer is an emphasis halo).
    if roi_keys:
        roi_x, roi_y, roi_z, roi_keys_list, roi_hover = [], [], [], [], []
        for key in roi_keys:
            hrf = hrfs.get(key)
            if hrf is None:
                continue
            loc = hrf.get("location")
            if loc is None or len(loc) < 3:
                continue
            roi_x.append(loc[0])
            roi_y.append(loc[1])
            roi_z.append(loc[2])
            roi_keys_list.append(key)
            roi_hover.append(_hover_text_for(key, hrf))
        if roi_x:
            traces.append(
                go.Scatter3d(
                    x=roi_x, y=roi_y, z=roi_z,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="#fbbf24",   # gold
                        opacity=0.9,
                        line=dict(width=1, color="#92400e"),
                    ),
                    name=f"ROI ({len(roi_x)})",
                    customdata=roi_keys_list,
                    hovertext=roi_hover,
                    hoverinfo=hover_mode,
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode="data",
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def compute_roi_keys(
    hrfs: Dict[str, Dict[str, Any]],
    anchor: Optional[Dict[str, Any]],
    radius_m: float,
    painted: Optional[Any] = None,
) -> "set":
    """Return the set of HRF keys that belong to the current ROI.

    Membership rules:
    - If ``anchor`` is set and has a 3-element ``location``, every HRF
      with the SAME ``oxygenation`` as the anchor whose Euclidean
      distance to the anchor is ``<= radius_m`` is included.
    - Every key in ``painted`` is included (filtered to the anchor's
      oxygenation when an anchor is set, so a stray paint on the wrong
      haemoglobin doesn't contaminate the average).

    The anchor's own key is always part of the ROI when it's still in
    ``hrfs`` (otherwise filtering away the anchor's neighbourhood would
    be confusing).

    The radius check is delegated to :class:`hrfunc.spatial.shapes.Sphere`.
    HRF locations are stored in meters internally; the spatial layer
    works in mm per the v1.3 compartmentalization convention, so this
    function converts both the anchor and each candidate location to mm
    before asking the sphere. The result is bit-identical to the
    pre-refactor Euclidean check (``loc * 1000`` is exact for the
    head-scale magnitudes here in float64).

    Module-level so tests can hit it without spinning up the GUI.
    """
    out: set = set()
    if anchor is None and not painted:
        return out

    anchor_loc = None
    anchor_oxy = None
    anchor_key = None
    sphere: Optional[Sphere] = None
    if anchor is not None:
        anchor_loc = anchor.get("location")
        anchor_oxy = anchor.get("oxygenation")
        anchor_key = anchor.get("_key")
        if anchor_key is not None and anchor_key in hrfs:
            out.add(anchor_key)
        if anchor_loc is not None and len(anchor_loc) >= 3 and radius_m > 0:
            sphere = Sphere(
                center_mm=meters_to_mm(anchor_loc[:3]).tolist(),
                radius_mm=float(meters_to_mm(radius_m)),
            )

    if sphere is not None:
        for key, hrf in hrfs.items():
            loc = hrf.get("location")
            if loc is None or len(loc) < 3:
                continue
            if anchor_oxy is not None and hrf.get("oxygenation") != anchor_oxy:
                continue
            loc_mm = meters_to_mm(loc[:3]).tolist()
            if sphere.contains(loc_mm):
                out.add(key)

    if painted:
        for key in painted:
            if key not in hrfs:
                continue
            if anchor_oxy is not None:
                if hrfs[key].get("oxygenation") != anchor_oxy:
                    continue
            out.add(key)

    return out


def compute_roi_keys_by_shape(
    hrfs: Dict[str, Dict[str, Any]],
    shape: Optional[Any],
    painted: Optional[Any] = None,
    *,
    oxygenation_filter: Optional[bool] = None,
    alignment_affine: "Optional[np.ndarray]" = None,
) -> "set":
    """Shape-based ROI membership for free-floating Box / Sphere modes.

    Companion to :func:`compute_roi_keys`. The original anchor-based
    API is preserved for the legacy click-anchor + radius workflow;
    this function takes a fully-constructed :class:`hrfunc.spatial.Shape`
    (typically a :class:`Box` or a free-floating :class:`Sphere`) plus
    an explicit oxygenation filter and returns the matching keys.

    Membership rules:

    - If ``shape`` is not None, every HRF whose location (converted
      meters->mm, then through the optional ``alignment_affine``)
      is inside ``shape`` is included. HRFs without a location are
      skipped.
    - Every key in ``painted`` is included (filtered by
      ``oxygenation_filter`` when set, so a stray paint on the
      wrong haemoglobin doesn't contaminate the average).
    - When ``oxygenation_filter`` is ``True`` / ``False``, only HRFs
      with matching ``oxygenation`` survive the membership check.
      ``None`` (the default) skips the oxygenation filter -- callers
      that have already filtered upstream (e.g. via
      ``library_oxygenation``) should leave this as None.

    ``alignment_affine`` (PR #54): a 4x4 homogeneous transform applied
    to the HRF coordinate before the shape predicate. Used for atlas
    mode to map MNE-head-coord HRFs into the atlas's MNI-mm frame.
    Pass ``None`` to skip the transform (default; sphere / box modes
    don't need it because the shape itself is in MNE-head space).

    Module-level so tests can hit it without spinning up the GUI.
    """
    import numpy as np

    out: "set" = set()
    if shape is None and not painted:
        return out

    if shape is not None:
        for key, hrf in hrfs.items():
            loc = hrf.get("location")
            if loc is None or len(loc) < 3:
                continue
            if (
                oxygenation_filter is not None
                and bool(hrf.get("oxygenation")) != bool(oxygenation_filter)
            ):
                continue
            loc_mm = meters_to_mm(loc[:3]).tolist()
            if alignment_affine is not None:
                homo = np.array(
                    [loc_mm[0], loc_mm[1], loc_mm[2], 1.0],
                    dtype=np.float64,
                )
                aligned = alignment_affine @ homo
                loc_mm = [
                    float(aligned[0]),
                    float(aligned[1]),
                    float(aligned[2]),
                ]
            if shape.contains(loc_mm):
                out.add(key)

    if painted:
        for key in painted:
            if key not in hrfs:
                continue
            if (
                oxygenation_filter is not None
                and bool(hrfs[key].get("oxygenation")) != bool(oxygenation_filter)
            ):
                continue
            out.add(key)

    return out


def compute_roi_average(
    hrfs: Dict[str, Dict[str, Any]],
    roi_keys: Any,
):
    """Average the per-subject ``estimates`` of every HRF in the ROI.

    Returns ``(mean, std, n_subjects, n_channels)`` -- the grand mean
    and std across all subject-level estimates pooled from every HRF
    in the ROI, plus the number of subject traces that contributed
    and the number of source channels they came from. Returns ``None``
    if fewer than 2 subject traces are averageable.

    **PR #54 correctness fix:** previously averaged ``hrf_mean`` (the
    per-channel mean), so a 50-subject channel got the same weight as
    a 5-subject channel in the final grand mean. Pooling ``estimates``
    instead gives every subject equal weight, which is what
    researchers report in publications.

    **HRFs without populated ``estimates``** are excluded from the
    average -- :func:`compute_roi_excluded_count` surfaces the count
    so the GUI can warn the user. The bundled library is loaded with
    ``rich=True`` so estimates survive the JSON load; HRFs missing
    estimates are typically those that came from a study where only
    the channel mean was published.

    Skips traces with empty / mismatched length. The modal length
    across the candidate pool is the canonical length; outliers are
    dropped (e.g. a single channel published at a different duration
    doesn't contaminate the average).

    Module-level so tests can call without a UI.
    """
    import numpy as np
    from collections import Counter

    # First pass: parse every subject-level estimate into a numpy array.
    # Track the source-channel count separately so the UI can show
    # "averaged N subjects across M channels".
    # Keep each parsed estimate paired with its source-channel key so the
    # contributing-channel count can be taken AFTER the modal-length filter
    # below -- a channel whose estimates are all an off-modal length
    # contributes zero traces to the mean and must not inflate the reported
    # "M channels" provenance figure.
    candidates: List[Tuple[str, "np.ndarray"]] = []
    for key in roi_keys:
        hrf = hrfs.get(key)
        if hrf is None:
            continue
        estimates = hrf.get("estimates") or []
        if not estimates:
            # No subject-level estimates -> can't contribute to a
            # subject-weighted grand mean. Skip the channel entirely
            # rather than fall back to hrf_mean (would mix two
            # averaging conventions in the same output).
            continue
        for estimate in estimates:
            try:
                arr = np.asarray(estimate, dtype=float)
            except Exception:  # noqa: BLE001
                continue
            if arr.ndim != 1 or arr.size == 0:
                continue
            candidates.append((key, arr))

    if len(candidates) < 2:
        return None

    # Pick the MODAL length as the canonical one rather than the first
    # iterated one. ``roi_keys`` is typically a set (no order guarantees),
    # and taking the first-seen length means an outlier length could
    # throw out the majority. Modal length is robust to iteration order
    # and matches what a researcher would expect: "average the traces
    # that share the typical duration; skip the oddball".
    lengths = Counter(arr.shape[0] for _, arr in candidates)
    canonical_len = lengths.most_common(1)[0][0]
    kept = [(key, arr) for key, arr in candidates if arr.shape[0] == canonical_len]

    if len(kept) < 2:
        return None

    traces = [arr for _, arr in kept]
    contributing_channels = len({key for key, _ in kept})
    stacked = np.vstack(traces)
    return (
        stacked.mean(axis=0),
        stacked.std(axis=0, ddof=0),
        len(traces),
        contributing_channels,
    )


def compute_roi_excluded_count(
    hrfs: Dict[str, Dict[str, Any]],
    roi_keys: Any,
) -> int:
    """Count the ROI HRFs that have no usable per-subject estimates.

    Used by the GUI to warn researchers when their ROI mixes channels
    with and without published subject-level data -- the average will
    silently drop the un-publishable channels.
    """
    excluded = 0
    for key in roi_keys:
        hrf = hrfs.get(key)
        if hrf is None:
            continue
        estimates = hrf.get("estimates") or []
        if not estimates:
            excluded += 1
    return excluded


def _hover_text_for(key: str, hrf: Dict[str, Any]) -> str:
    """Short multi-line hover-text summary for one HRF."""
    context = hrf.get("context") or {}
    bits = [key]
    for field in ("task", "doi", "study", "demographics"):
        value = context.get(field)
        if value:
            bits.append(f"{field}: {value}")
    return "<br>".join(bits)


# ---------------------------------------------------------------------------
# Trace plot PNG
# ---------------------------------------------------------------------------


def _render_trace_png(hrf: Dict[str, Any]) -> Optional[str]:
    """Render the HRF trace as a base64 PNG line plot."""
    try:
        import base64
        import io as _io
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for hrtree trace: %s", exc)
        return None

    trace = hrf.get("hrf_mean") or []
    if not trace:
        return None
    sfreq = float(hrf.get("sfreq") or 1.0)
    if sfreq <= 0:
        sfreq = 1.0

    fig = None
    try:
        t = np.arange(len(trace)) / sfreq
        std = hrf.get("hrf_std") or []
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
        ax.plot(t, trace, lw=1.2, color="#6366f1")
        if std and len(std) == len(trace):
            lower = np.asarray(trace) - np.asarray(std)
            upper = np.asarray(trace) + np.asarray(std)
            ax.fill_between(t, lower, upper, alpha=0.15, color="#6366f1")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (a.u.)")
        fig.tight_layout()
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("hrtree trace render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
