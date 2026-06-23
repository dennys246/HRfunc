"""AppState — single source of truth for the running GUI.

NiceGUI page handlers, components, and background workers all read and write
the same `AppState` instance. A module-level `state` singleton is created at
import time so any module can ``from hrfunc.gui.state import state`` without
threading a reference through every function signature.

Lifecycle:
- One AppState per process. The `state` singleton is created on first import.
- Tests can instantiate fresh AppState() instances for isolated unit testing
  (the class itself is just a dataclass).
- The singleton holds mutable fields by design — pages bind their UI elements
  to these fields and re-render on changes.

What lives here:
- `manifest`             - last folder scan result (None until a scan completes)
- `selected_scan`        - currently inspected ScanEntry, or None
- `raw_cache`            - hot-path LRU(3) loader of source MNE Raw objects
- `processed_cache`      - LRU(3) of *preprocessed* Raw objects (Sprint 3.2);
                           HRFs / Activity tabs read from here
- `preload_path`         - CLI arg from `hrfunc <path>`; consumed by welcome
                           page on first render
- `busy`                 - True while a background task is running (drives
                           spinner UI); gate for estimation, NOT for scan loads
- `estimation_progress`  - (current, total, channel_name) tuple from the latest
                           progress_callback fire; None when no estimation in flight
- `last_error`           - last error message surfaced to the user, or None
- `subscribers`          - event-bus dispatch table (Sprint 3.2); see
                           ``subscribe`` / ``publish``
- `montage`              - most recently estimated Montage from the HRFs tab
                           (Sprint 3.3); None until estimate_hrf runs at least
                           once. Cleared on dataset reset; switching the
                           selected scan does NOT clear it, so users can
                           switch tabs and come back without losing results
                           — but a new estimation overwrites the field
                           regardless of which scan it came from.
- `activity_raw`         - most recent deconvolved Raw from the Activity tab
                           (Sprint 3.4); the output of ``estimate_activity``
                           which mutates a copy of the preprocessed Raw and
                           returns it with neural-activity values in place
                           of haemoglobin values. None until run at least
                           once; cleared on reset.

Event bus (Sprint 3.2, extended in 3.3):
The bus replaces the Sprint 2.3-era ``_inspect_refresh`` private attribute.
Panels subscribe to named events and are called when other parts of the GUI
publish. The bus is dict-of-lists, deliberately minimal — no priorities, no
async dispatch, no payload schemas. Defined events:

- ``"scan_selected"``  — payload: ``ScanEntry`` (or None for deselection).
  Published when the dataset tree updates ``state.selected_scan``.
- ``"scan_loaded"``    — payload: ``ScanEntry``. Published after a background
  Raw load completes successfully; subscribers can read the Raw from
  ``state.raw_cache``.
- ``"preprocess_done"`` — payload: ``ScanEntry``. Published after a successful
  preprocess run; subscribers can read the processed Raw from
  ``state.processed_cache``.
- ``"hrf_estimated"``   — payload: ``ScanEntry``. Published after a successful
  ``estimate_hrf`` (or canonical HRF generation); subscribers can read the
  resulting Montage from ``state.montage``.
- ``"activity_estimated"`` — payload: ``ScanEntry``. Published after a
  successful ``estimate_activity`` run; subscribers can read the deconvolved
  Raw from ``state.activity_raw``.
- ``"quality_computed"`` — payload: ``ScanEntry`` or ``None``. Published
  after a Quality-panel metrics computation finishes (per-scan: ScanEntry;
  dataset-wide aggregate: None). Subscribers can read
  ``state.quality_metrics`` for the results.
- ``"project_changed"`` — payload: ``Manifest`` or ``None``. Published by
  ``set_manifest`` when the active project swaps (load new, switch, or
  close). Panels with persistent refreshables subscribe to blank or
  rebuild their views before reading the new manifest.
- ``"busy_changed"`` — payload: ``bool`` (the new busy value). Published
  by ``set_busy`` when a background worker starts (True) or completes
  (False). The project picker subscribes to disable Open / Close while
  busy so a switch can't strand a half-finished run on the new project.

Subscribers are sync callables. Async handlers can dispatch via
``nicegui.background_tasks.create`` from inside their callback.

Fields are added (not removed) as later sprints integrate more state. Keeping
the AppState surface stable across sprints means GUI components written in
earlier sprints don't need updates as later panels land.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..io.manifest import Manifest, ScanEntry
from ..io.raw_cache import RawCache

logger = logging.getLogger(__name__)

EventCallback = Callable[..., None]

# Matches the auto-generated ROI slot names ("ROI 1", "ROI 2", ...) that
# ``add_roi`` / the default factory produce. Used by ``clear_active_roi`` to
# renumber ONLY auto-named slots after a delete, leaving descriptive names
# ("Montage: S1_D1" from Add Montage, or user renames) untouched.
_AUTO_ROI_NAME_RE = re.compile(r"^ROI \d+$")


@dataclass
class ROISlot:
    """One ROI in the Cluster sub-tab's multi-ROI list (PR #55).

    Holds everything that distinguishes one ROI from another: its shape
    selection, geometry parameters, painted-key set, and the click-
    anchor that seeded it (if any). Atlas alignment is NOT stored
    here -- alignment is a property of the HRF library's coord frame,
    not of any individual ROI, so it lives on AppState as a global
    per scan/dataset (locked decision 2026-05-14).

    The default-constructed slot reproduces the pre-PR-#55 single-ROI
    starting state: sphere mode, free-floating centred at the MNI
    origin with a 20 mm radius. The first slot in ``cluster_rois``
    is therefore safe to leave at defaults if the user never opens
    the Cluster sub-tab.
    """

    # Display name for the ROI list ("ROI 1", "ROI 2", ...). Auto-
    # assigned by ``AppState.add_roi`` but mutable so future iterations
    # can rename. Not included in the saved montage's per-ROI block
    # unless renamed (see ``workspace_io``).
    name: str = "ROI 1"

    # Shape mode: "sphere" | "box" | "atlas_region". Matches the
    # module-level SHAPE_* constants in hrtree_panel; kept as a string
    # here so this module doesn't have to import the panel.
    shape: str = "sphere"

    # Free-floating shape centre, MNI mm. Three separate fields so each
    # binds cleanly to its own ``ui.number`` input. Defaults to MNI
    # origin (0, 0, 0); seeded by clicking an HRF in the viz pane.
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    center_z_mm: float = 0.0

    # Box half-extents, MNI mm. Default 20 mm on each axis = a 40 mm
    # cube, comparable to a 2 cm radius sphere by volume.
    box_half_x_mm: float = 20.0
    box_half_y_mm: float = 20.0
    box_half_z_mm: float = 20.0

    # Sphere radius, MNI mm. Pre-PR-#55 this lived on AppState as
    # ``library_roi_radius_m`` (meters); the per-ROI move converts to
    # mm to match the rest of the spatial-layer convention (MNI mm
    # everywhere from PR #46). Default 20 mm = the legacy 0.02 m.
    radius_mm: float = 20.0

    # Atlas-region label when ``shape == "atlas_region"``. ``None``
    # means "no region picked yet" and the save button disables.
    atlas_label: Optional[str] = None

    # Shift-hover painted keys (the lasso-like accumulation), filtered
    # to the anchor's oxygenation when an anchor is set. Joins the
    # ROI regardless of the shape geometry. Cleared on every new
    # anchor click so paint from a prior anchor doesn't carry over.
    painted: Set[str] = field(default_factory=set)

    # Click-anchor HRF, if any. Same dict shape as
    # ``state.library_selected_hrf`` (gathered HRF + ``_key``).
    # When present, drives the saved JSON's location + oxygenation
    # fields and the sphere's centre seed.
    anchor: Optional[Dict[str, Any]] = None

    # PR #59 (originally planned as part of the v1.3 montage UX
    # follow-up): per-ROI visibility toggle. A "layer" checkbox in the
    # multi-ROI list -- when False, the ROI:
    #   - is hidden from the viz (no shape overlay, no gold halo for
    #     its member HRFs),
    #   - is excluded from the saved montage.json,
    #   - is excluded from the ROI-status summary in the Cluster
    #     sub-tab.
    # Active state (which slot's controls render in the right panel)
    # is independent of visibility -- the user can edit a hidden ROI's
    # parameters without it appearing on the viz. Defaults to True so
    # newly-added ROIs (manual or via Add Montage) show up immediately.
    visible: bool = True

    # Layout refactor (2026-05-16): per-row selection checkbox for the
    # bulk-edit workflow. When two or more slots are selected, the
    # radius slider + centre inputs apply to the whole selected set so
    # the user can move / resize a group of ROIs together. When the
    # selected set is empty, edits fall back to the active slot
    # (single-slot semantics). Auto-set to True by ``AppState.add_roi``
    # and by the Add Montage flow (which clears prior selection and
    # ticks every new slot). Independent of ``visible`` and ``active``.
    selected: bool = False

    def is_pristine_default(self) -> bool:
        """True when this slot is identical to a fresh ``ROISlot()``.

        Used by the Add-Montage flow to decide whether to drop the
        list's single starting ROI (the one created by
        ``_default_rois()``) before appending the per-channel slots.
        Without this drop, a user clicking Add Montage on a fresh
        project ends up with an orphan "ROI 1" at MNI (0, 0, 0)
        cluttering the list above all the per-channel spheres.

        The ``name`` field is ignored because a freshly-added ROI
        could end up with any auto-name ("ROI 1", "ROI 5"). All
        other fields must match the dataclass defaults.
        """
        default = ROISlot()
        return (
            self.shape == default.shape
            and self.center_x_mm == default.center_x_mm
            and self.center_y_mm == default.center_y_mm
            and self.center_z_mm == default.center_z_mm
            and self.box_half_x_mm == default.box_half_x_mm
            and self.box_half_y_mm == default.box_half_y_mm
            and self.box_half_z_mm == default.box_half_z_mm
            and self.radius_mm == default.radius_mm
            and self.atlas_label == default.atlas_label
            and self.painted == default.painted
            and self.anchor == default.anchor
            and self.visible == default.visible
            and self.selected == default.selected
        )


def _default_rois() -> List[ROISlot]:
    """Factory for AppState.cluster_rois.

    Layout refactor (2026-05-16): the list now defaults to EMPTY.
    Pre-refactor it always seeded one slot so the active-index had
    something to point at and the proxy properties never looked at
    an empty list. The new UI removes the master "ROI active" toggle
    and the auto-spawned default ROI -- adding the first ROI (via
    Add ROI or Add Montage) is the implicit activation. Proxy
    properties handle the empty case by returning sensible defaults
    on read and silently no-oping on write (no slot to mutate).
    """
    return []


@dataclass
class AppState:
    """Mutable, single-process GUI state.

    All fields default to `None` / empty so a freshly-constructed AppState
    represents "no data loaded yet" — the state shown by the welcome page.
    """

    manifest: Optional[Manifest] = None
    selected_scan: Optional[ScanEntry] = None
    raw_cache: RawCache = field(default_factory=RawCache)
    processed_cache: RawCache = field(default_factory=RawCache)
    # LRU(3) of *deconvolved* (neural-activity) Raw objects, keyed per scan.
    # ``activity_raw`` only holds the single most-recent result; this cache
    # keeps each scan's deconvolution so channel-wise 3-stage QC (raw vs
    # hemoglobin vs activity) can read a scan's activity without re-running.
    activity_cache: RawCache = field(default_factory=RawCache)
    preload_path: Optional[Path] = None
    busy: bool = False
    estimation_progress: Optional[Tuple[int, int, str]] = None
    # PR #55a: scans the user has checked in the dataset tree. Stored as a
    # set of resolved paths so equality is filesystem-stable across
    # ScanEntry rebuilds (the manifest rebuilds its entries on every
    # rescan, but the on-disk path is the constant identity). Empty set =
    # "nothing checked"; the action buttons fall back to selected_scan
    # in that case so single-scan workflows keep working. Lives outside
    # the manifest so a project rescan that re-walks the directory
    # doesn't silently wipe the user's selection.
    checked_scan_paths: Set[Path] = field(default_factory=set)
    # PR #55a: bulk-run progress -- (current_index, total, current_scan).
    # Lives alongside ``estimation_progress`` (which tracks within-scan
    # channel progress); the action panels render both lines during a
    # bulk run. ``None`` when no bulk run is in flight. Set by
    # ``workers.run_bulk_in_background`` as it advances.
    bulk_progress: Optional[Tuple[int, int, ScanEntry]] = None
    last_error: Optional[str] = None
    subscribers: Dict[str, List[EventCallback]] = field(default_factory=dict)
    # Montage from the most recent HRF estimation (Sprint 3.3). Typed as Any to
    # avoid pulling hrfunc.hrfunc into the GUI import graph at module load —
    # the GUI must stay importable without MNE for tests that disable it.
    montage: Optional[Any] = None
    # Scan that produced the current ``montage`` (Sprint 3.4). The Activity tab
    # uses this to refuse toeplitz-mode estimation when the user has switched
    # to a different scan since estimate_hrf ran — applying scan A's HRFs to
    # scan B's Raw would silently produce wrong results because the library
    # matches by channel name, not by scan identity.
    montage_source_scan: Optional[ScanEntry] = None
    # Per-scan estimated montages, keyed by ``ScanEntry.path.resolve()``.
    # ``montage`` only holds the single most-recent estimate; this cache keeps
    # each scan's HRFs so toeplitz activity can deconvolve every scan with its
    # OWN HRFs (single + bulk) instead of erroring "HRFs belong to another
    # scan". Only real per-channel Montages are stored (never _CanonicalResult).
    montage_cache: Dict[Path, Any] = field(default_factory=dict)
    # Deconvolved Raw from the most recent estimate_activity call (Sprint 3.4).
    # Typed Any for the same import-graph reason. The Activity panel reads
    # the data + annotations for the lens-style preproc/deconv overlay plot.
    activity_raw: Optional[Any] = None
    # Scan that produced the current ``activity_raw``. Like
    # ``montage_source_scan`` for the montage, this exists so the Activity
    # preview and the Export tab don't overlay/ship scan A's deconvolved Raw
    # against scan B's data after the user switches the selected scan (the
    # single ``activity_raw`` slot is not cleared on scan change). Consumers
    # compare ``activity_source_scan.path`` to the selected scan's path.
    activity_source_scan: Optional[ScanEntry] = None
    # Per-scan quality metrics (Sprint 4.1). Keyed by ``ScanEntry.path.resolve()``;
    # each value is a dict {"raw": metrics_dict, "preprocessed": metrics_dict,
    # "deconvolved": metrics_dict}. Each metrics_dict contains numeric summaries
    # (snr_mean, skew_mean, kurtosis_mean, sci_mean when applicable). Entries
    # appear as the Quality panel computes them — either per-scan when the user
    # views Quality for the current scan, or in bulk during the dataset-wide
    # aggregate run.
    quality_metrics: Dict[Path, Dict[str, Any]] = field(default_factory=dict)
    # Lazy-loaded bundled HRF databases for the /library page (Sprint 4.2-4.4).
    # Tree objects from ``hrfunc.hrtree.tree``. Populated on first /library
    # visit; never cleared (the data is read-only from disk so re-loading
    # would just re-read the same files). Typed Any to keep the GUI import
    # graph free of hrfunc.hrtree at module load.
    library_hbo: Optional[Any] = None
    library_hbr: Optional[Any] = None
    # Current context filter state for the Library page. Keys are context
    # field names ('task', 'doi', 'demographics', ...). Empty dict = no
    # filter applied. The Library page rebuilds the visible HRF list from
    # the filter every render.
    library_filter: Dict[str, Any] = field(default_factory=dict)
    # Currently-selected HRF on the Library page (from a click in the
    # plotly viz or a manual list selection). Stored as the gathered-form
    # dict (the value type produced by ``tree.gather``), or None.
    library_selected_hrf: Optional[Dict[str, Any]] = None
    # Channel name selected for the HRF gallery's detail-pane focus
    # (Sprint 5.1). Sprint 3.3 rendered all channel HRFs overlaid on one
    # plot; Sprint 5 replaces that with a clickable grid + a per-channel
    # full-detail view. None = no channel focused yet (grid renders, no
    # detail view).
    hrf_selected_channel: Optional[str] = None
    # User-uploaded events for HRF estimation (HRFs tab). Many scans reach
    # the GUI without usable MNE annotations, so the event picker would be
    # empty; an uploaded events file supplies onsets instead. ``events_rows``
    # is a list of ``events_io.EventRow`` (typed Any to keep events_io out of
    # the state import graph). When set, it OVERRIDES a scan's embedded
    # annotations for the scans it applies to: the scan recorded in
    # ``events_source_scan`` always, plus every scan when ``events_apply_all``
    # is True (paradigms shared across scans). ``events_format`` is
    # "bids"/"simple" for the source badge; ``events_source_label`` is the
    # filename. All cleared on reset / project switch.
    events_rows: Optional[List[Any]] = None
    # Per-sample 0/1 impulse/design vector (events_io "impulse" format),
    # mutually exclusive with events_rows. Applied by sample index.
    events_impulse: Optional[List[int]] = None
    events_format: Optional[str] = None
    events_source_label: Optional[str] = None
    events_apply_all: bool = False
    events_source_scan: Optional[ScanEntry] = None
    # True when the current events were auto-matched from a collocated file
    # (drives the "auto-matched" badge vs an explicit upload).
    events_is_automatched: bool = False
    # Resolved scan paths the auto-matcher should leave alone: scans the user
    # manually set/cleared, or scans where discovery already ran and found
    # nothing. Prevents re-clobbering a manual choice and avoids re-scanning
    # the folder every render. Cleared on reset / project switch.
    events_no_automatch: Set[Path] = field(default_factory=set)
    # Resolved scan paths whose ``processed_cache`` entry was produced with
    # the DECONVOLUTION preprocessing pipeline (deconvolution=True). HRF and
    # neural-activity estimation require deconvolution-preprocessed data, so
    # the HRFs / Activity tabs gate on membership here; a GLM/hemoglobin
    # preprocess (deconvolution=False) is intentionally NOT added, which
    # blocks estimation until the scan is re-preprocessed in deconvolution
    # mode. Updated wherever a processed Raw is cached; cleared on reset.
    processed_deconvolved: Set[Path] = field(default_factory=set)
    # Toggles for the MNI fsaverage overlay surfaces on the /library
    # plotly viz. The "brain" toggle controls the pial cortical
    # surface (where the neural activity originates); the "scalp"
    # toggle controls the outer-skin head surface (where the optodes
    # physically sit). Both are independent so users can show either,
    # both, or neither. Scalp defaults ON because most fNIRS optodes
    # are forehead/scalp-mounted and the head shape gives the most
    # immediate spatial context; brain defaults ON too because the
    # cortex-relative position is the scientifically meaningful one.
    library_show_brain: bool = True
    library_show_scalp: bool = True
    # Toggle for the per-HRF metadata hover popups on the /library viz. When
    # False the scatter markers use hoverinfo="none" -- the tooltip is hidden
    # but hover/click events still fire, so ROI clicks and shift-hover paint
    # keep working. Defaults ON (the popups are the main way to read an HRF's
    # context); users dealing with dense clouds can switch them off.
    library_show_info: bool = True
    # Oxygenation filter for the /library viz. ``"both"`` (default)
    # shows HbO + HbR; ``"hbo"`` / ``"hbr"`` hide the other channel.
    # Researchers often want to inspect one haemoglobin at a time;
    # the toggle lets them do that without writing a context filter.
    library_oxygenation: str = "both"
    # ROI selection on the /library viz. Two ways to add HRFs to the
    # ROI, used in combination:
    #
    # 1. **Anchor + radius**: click an HRF and every same-oxygenation
    #    HRF inside the sphere of radius ``library_roi_radius_m``
    #    around it gets included. The clicked HRF is the same dict
    #    as ``library_selected_hrf`` (anchor + detail-pane focus
    #    share state).
    # 2. **Shift-hover painting**: hold Shift and hover over HRFs.
    #    Each hovered key is added to ``library_roi_painted`` and
    #    joins the ROI regardless of radius. Lets researchers trace
    #    a non-spherical region by mouse, like a lasso.
    #
    # The full ROI is the union of (in-radius set) and the painted
    # set, filtered to the anchor's oxygenation. The averaged trace
    # in the detail pane is computed from this union.
    #
    # PR #55: per-ROI cluster state moved into ``cluster_rois``. The
    # ``library_roi_radius_m`` / ``library_roi_painted`` /
    # ``cluster_shape`` / ``cluster_center_*_mm`` / ``cluster_box_half_*_mm``
    # / ``cluster_atlas_label`` names are now ``@property`` proxies
    # onto ``cluster_rois[cluster_active_index]`` so existing panel
    # bindings and tests keep working. See ``ROISlot``.
    #
    # Multi-ROI list (PR #55). Defaults to EMPTY since the layout
    # refactor (2026-05-16) -- adding the first ROI (Add ROI / Add
    # Montage) is the implicit activation. CLEAR ROI on the last slot
    # removes it, returning the list to empty. Active index picks
    # which slot the proxies read/write; when the list is empty,
    # proxy reads return slot defaults (sphere, origin, 20 mm) and
    # proxy writes silently no-op.
    cluster_rois: List[ROISlot] = field(default_factory=_default_rois)
    cluster_active_index: int = 0
    # PR #54: HRF-coords-to-MNI alignment for atlas membership. Bundled
    # library HRFs are stored in MNE head coordinates (origin near the
    # auditory meatus); the Harvard-Oxford atlas is in MNI mm (origin
    # at the brain centroid). Without alignment, every HRF maps to
    # voxels outside the atlas volume -> atlas mode silently shows
    # empty ROIs. ``cluster_atlas_alignment_affine`` is a 4x4
    # homogeneous transform applied at the membership-check boundary
    # (HRF coord -> MNI coord). Defaults to None (= no transform) so
    # users with already-MNI HRFs aren't affected. Loadable from a
    # JSON or .npy file via the alignment file picker in atlas mode.
    cluster_atlas_alignment_affine: Optional[Any] = None
    # PR #54: human-friendly atlas alignment offsets (mm). These are a
    # shorthand for users without a full 4x4 affine -- they compose
    # with ``cluster_atlas_alignment_affine`` to produce the full
    # transform applied at lookup. Pure-translation corrections in
    # MNE-head -> MNI mm space.
    cluster_atlas_offset_x_mm: float = 0.0
    cluster_atlas_offset_y_mm: float = 0.0
    cluster_atlas_offset_z_mm: float = 0.0
    # PR #54: persistent "saved to" feedback for the Cluster sub-tab's
    # save action. ``ui.notify`` toasts vanish in seconds; storing the
    # last-saved path lets the sub-tab render an always-visible label
    # below the save button so users can confirm the file went out
    # even after navigating away and back.
    last_saved_roi_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # PR #55: proxy properties onto the active ROI slot.
    # ------------------------------------------------------------------
    # These exist so the Cluster sub-tab UI (which binds widgets to
    # ``state.cluster_shape``, ``state.cluster_center_x_mm``, etc.) and
    # the existing test suite see the per-ROI fields under their pre-
    # PR-#55 names. The actual storage lives in
    # ``cluster_rois[cluster_active_index]``.
    #
    # Layout refactor (2026-05-16): the list can now be empty (Add
    # ROI / Add Montage is the implicit activation). ``active_roi``
    # returns None when the list is empty; proxy reads return slot
    # defaults so panel code that reads ``cluster_shape`` etc. in an
    # empty-state render still sees sensible values, and proxy writes
    # silently no-op so a stray ``state.cluster_shape = ...`` in an
    # empty-state path doesn't ressurect a slot.

    @property
    def active_roi(self) -> Optional[ROISlot]:
        """The currently-active ROI slot, or None when the list is empty.

        If the list is non-empty but the index is out of range, clamps
        to 0 (defensive against external mutation).
        """
        if not self.cluster_rois:
            return None
        if not (0 <= self.cluster_active_index < len(self.cluster_rois)):
            self.cluster_active_index = 0
        return self.cluster_rois[self.cluster_active_index]

    # Cached default slot used as the source of read values when
    # ``active_roi`` is None. Constructed once per AppState so reads
    # don't allocate a fresh slot on every access.
    @property
    def _proxy_default(self) -> ROISlot:
        cached = getattr(self, "__proxy_default_cache", None)
        if cached is None:
            cached = ROISlot()
            object.__setattr__(self, "__proxy_default_cache", cached)
        return cached

    @property
    def cluster_shape(self) -> str:
        slot = self.active_roi
        return slot.shape if slot is not None else self._proxy_default.shape

    @cluster_shape.setter
    def cluster_shape(self, value: str) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.shape = value

    @property
    def cluster_center_x_mm(self) -> float:
        slot = self.active_roi
        return slot.center_x_mm if slot is not None else self._proxy_default.center_x_mm

    @cluster_center_x_mm.setter
    def cluster_center_x_mm(self, value: float) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.center_x_mm = float(value)

    @property
    def cluster_center_y_mm(self) -> float:
        slot = self.active_roi
        return slot.center_y_mm if slot is not None else self._proxy_default.center_y_mm

    @cluster_center_y_mm.setter
    def cluster_center_y_mm(self, value: float) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.center_y_mm = float(value)

    @property
    def cluster_center_z_mm(self) -> float:
        slot = self.active_roi
        return slot.center_z_mm if slot is not None else self._proxy_default.center_z_mm

    @cluster_center_z_mm.setter
    def cluster_center_z_mm(self, value: float) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.center_z_mm = float(value)

    @property
    def cluster_box_half_x_mm(self) -> float:
        slot = self.active_roi
        return slot.box_half_x_mm if slot is not None else self._proxy_default.box_half_x_mm

    @cluster_box_half_x_mm.setter
    def cluster_box_half_x_mm(self, value: float) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.box_half_x_mm = float(value)

    @property
    def cluster_box_half_y_mm(self) -> float:
        slot = self.active_roi
        return slot.box_half_y_mm if slot is not None else self._proxy_default.box_half_y_mm

    @cluster_box_half_y_mm.setter
    def cluster_box_half_y_mm(self, value: float) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.box_half_y_mm = float(value)

    @property
    def cluster_box_half_z_mm(self) -> float:
        slot = self.active_roi
        return slot.box_half_z_mm if slot is not None else self._proxy_default.box_half_z_mm

    @cluster_box_half_z_mm.setter
    def cluster_box_half_z_mm(self, value: float) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.box_half_z_mm = float(value)

    @property
    def cluster_atlas_label(self) -> Optional[str]:
        slot = self.active_roi
        return slot.atlas_label if slot is not None else self._proxy_default.atlas_label

    @cluster_atlas_label.setter
    def cluster_atlas_label(self, value: Optional[str]) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.atlas_label = value

    @property
    def library_roi_radius_m(self) -> float:
        """Sphere radius in meters (legacy unit). The per-ROI storage
        is in mm to match the spatial-layer convention -- the meter
        view is kept for back-compat with pre-PR-#55 panel code and
        tests that compute ``radius_m * 1000`` at the boundary."""
        slot = self.active_roi
        return (slot.radius_mm if slot is not None else self._proxy_default.radius_mm) / 1000.0

    @library_roi_radius_m.setter
    def library_roi_radius_m(self, value: float) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.radius_mm = float(value) * 1000.0

    @property
    def library_roi_painted(self) -> Set[str]:
        """Painted-key set for the active ROI. Returned by reference so
        callers that do ``state.library_roi_painted.add(...)`` or
        ``.clear()`` mutate the active slot's set directly -- matches
        the pre-PR-#55 contract where this attribute *was* a mutable
        set on AppState.

        When the ROI list is empty, returns a transient empty set so
        mutation calls (``.add``, ``.clear``) silently no-op. Reads of
        that empty set return an empty membership, which matches the
        "no ROI yet" semantics.
        """
        slot = self.active_roi
        if slot is not None:
            return slot.painted
        # Transient empty set -- mutations on it are dropped on the
        # next access (it's reconstructed each read), which is the
        # right "no slot to write to" semantic.
        return set()

    @library_roi_painted.setter
    def library_roi_painted(self, value: Set[str]) -> None:
        slot = self.active_roi
        if slot is not None:
            slot.painted = set(value)

    # ------------------------------------------------------------------
    # PR #55: ROI list manipulation helpers.
    # ------------------------------------------------------------------

    def add_roi(self) -> ROISlot:
        """Append a fresh ROI to ``cluster_rois`` and make it active.

        The new ROI inherits no state from the previous active slot --
        it starts at the dataclass defaults. Auto-names it
        "ROI <n>" where n is the new length of the list.

        Layout refactor (2026-05-16): the new slot is also auto-
        selected (``selected=True``) so it joins the bulk-edit set
        immediately. Users adding one ROI at a time still see
        single-slot behaviour (the selected set has one entry, the
        active slot); users iterating Add ROI multiple times build
        up a selected group ready for bulk edits.

        Returns the new slot so callers can stamp additional state
        onto it before publishing the change.
        """
        name = f"ROI {len(self.cluster_rois) + 1}"
        slot = ROISlot(name=name, selected=True)
        self.cluster_rois.append(slot)
        self.cluster_active_index = len(self.cluster_rois) - 1
        return slot

    def set_active_roi(self, index: int) -> None:
        """Switch the active ROI index. No-op if out of range."""
        if 0 <= index < len(self.cluster_rois):
            self.cluster_active_index = index

    def selected_rois(self) -> List[ROISlot]:
        """Return every slot with ``selected=True`` in list order."""
        return [s for s in self.cluster_rois if s.selected]

    def bulk_edit_targets(self) -> List[ROISlot]:
        """Slots an edit (radius slider, centre input) should apply to.

        Returns the selected set when non-empty; falls back to a
        one-element list with the active slot when nothing is
        selected. Returns an empty list when the active is None AND
        nothing is selected (no ROI to edit). The radius/centre
        handlers iterate this list so a single proxy write naturally
        becomes a multi-slot update when the user has multiple slots
        checked.
        """
        selected = self.selected_rois()
        if selected:
            return selected
        active = self.active_roi
        return [active] if active is not None else []

    def set_all_selected(self, value: bool) -> None:
        """Tick (or untick) every slot in ``cluster_rois`` in one shot.

        Used by Add Montage to clear prior selection before ticking
        the new slots. Also useful for a future "Select all / None"
        button in the ROI-list header.
        """
        for slot in self.cluster_rois:
            slot.selected = bool(value)

    def clear_active_roi(self) -> None:
        """Remove the active ROI from the list.

        Layout refactor (2026-05-16): the list is now allowed to be
        empty. Clearing the last slot drops it entirely -- pre-refactor
        we kept one resetting-in-place because the proxy properties
        needed a slot to point at, but the proxies now handle the
        empty case via the cached default slot. Empty list = "no ROI
        active, viz shows raw HRFs, save button disabled".

        With 2+ ROIs: removes the active slot and advances the index
        to the previous slot (or 0 if we removed slot 0). With 1 ROI:
        clears the list; index becomes 0 (which now means "no slot",
        a state the proxies handle).
        """
        if not self.cluster_rois:
            return
        del self.cluster_rois[self.cluster_active_index]
        if self.cluster_rois:
            self.cluster_active_index = max(
                0, self.cluster_active_index - 1
            )
            # Renumber ONLY auto-generated "ROI N" names so the auto-named
            # slots stay "ROI 1 ... ROI N" in order. Descriptive names --
            # "Montage: S1_D1" from Add Montage, or a user rename -- are left
            # untouched. The pre-fix code overwrote EVERY name on any delete,
            # which clobbered montage provenance and corrupted the saved
            # montage.json (build_roi_entry reads slot.name verbatim).
            auto_index = 0
            for slot in self.cluster_rois:
                if _AUTO_ROI_NAME_RE.match(slot.name):
                    auto_index += 1
                    slot.name = f"ROI {auto_index}"
        else:
            self.cluster_active_index = 0

    def subscribe(self, event: str, callback: EventCallback) -> None:
        """Register ``callback`` to be called on ``publish(event, ...)``.

        Multiple subscribers per event are supported and called in
        registration order. Re-subscribing the same callable for the same
        event adds a duplicate registration — callers responsible for
        avoiding duplicate registration if they re-run their setup
        (e.g. ``ui.refreshable`` bodies should subscribe once at module
        scope, not inside the refreshable function).
        """
        self.subscribers.setdefault(event, []).append(callback)

    def unsubscribe(self, event: str, callback: EventCallback) -> bool:
        """Remove ``callback`` from ``event``'s subscriber list.

        Returns True if a registration was removed, False if no matching
        registration was found. Removes only one registration per call
        (matching the first occurrence) so duplicate registrations from
        accidental re-subscription require multiple unsubscribe calls.
        """
        callbacks = self.subscribers.get(event)
        if not callbacks:
            return False
        try:
            callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def publish(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Call every subscriber of ``event`` with the given args.

        Subscriber exceptions are logged and swallowed so one buggy panel
        cannot break event delivery to the others. This matches the GUI's
        broader "errors go to state.last_error, not the user's view" stance.
        """
        for callback in list(self.subscribers.get(event, [])):
            try:
                callback(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Subscriber %r raised on event %r: %s",
                    callback, event, exc,
                )

    def set_busy(self, value: bool) -> None:
        """Toggle the busy flag and notify subscribers.

        ``workers.run_in_background`` calls this with ``True`` before
        dispatching the worker thread and ``False`` after it returns
        (success or failure). The project picker subscribes to
        ``busy_changed`` to disable Open / Close menu items while a
        long task is running — without this, switching projects mid-
        estimate would silently land the result on the new project.
        """
        if self.busy == value:
            return
        self.busy = value
        self.publish("busy_changed", value)

    def set_manifest(self, manifest: Optional[Manifest]) -> None:
        """Swap the active project manifest and notify subscribers.

        Pass ``None`` to clear. The ``project_changed`` event fires AFTER the
        manifest field is updated so subscribers reading ``state.manifest``
        from inside their callback see the new value. Subscribers themselves
        are not cleared — panels stay subscribed across project switches in
        the single-shell GUI; their handlers re-read state and refresh.

        Per-scan state that is meaningless against a different project is
        cleared here, not just in ``reset()``: the picker drives every
        project change (open / switch-to-recent / close) through this
        method, never through ``reset()``. Without the clear, the previous
        project's ``selected_scan`` and cached Raws leak into the Inspect
        panel, and its ``checked_scan_paths`` can silently pre-check (and
        bulk-include) a colliding path in the new project. A ``scan_selected``
        None is published so panels keyed only on that event (Inspect) blank
        themselves. Idempotent: setting the manifest already in place is a
        no-op so a stray re-set can't wipe a live selection.
        """
        if manifest is self.manifest:
            return
        self.manifest = manifest
        self.selected_scan = None
        self.raw_cache.clear()
        self.processed_cache.clear()
        self.checked_scan_paths.clear()
        self.bulk_progress = None
        self.publish("project_changed", manifest)
        self.publish("scan_selected", None)

    def reset(self) -> None:
        """Return to the welcome-screen state.

        Used when the user closes the current project / switches datasets.
        Drops cached Raws (both source and processed) so memory is released.
        The RawCache instances are kept (not reassigned) so any references
        held elsewhere stay valid. Event subscribers and the estimated
        Montage are also cleared — a fresh dataset is a clean slate.
        """
        self.manifest = None
        self.selected_scan = None
        self.raw_cache.clear()
        self.processed_cache.clear()
        self.activity_cache.clear()
        self.montage_cache.clear()
        self.preload_path = None
        self.busy = False
        self.estimation_progress = None
        # PR #55a: bulk-iterate state cleared on project switch -- the
        # checked set is meaningless against a different manifest, and
        # bulk_progress can't survive a busy=False without leaking a
        # stale "still running" display.
        self.checked_scan_paths.clear()
        self.bulk_progress = None
        self.last_error = None
        self.subscribers.clear()
        self.montage = None
        self.montage_source_scan = None
        self.activity_raw = None
        self.events_rows = None
        self.events_impulse = None
        self.events_format = None
        self.events_source_label = None
        self.events_apply_all = False
        self.events_source_scan = None
        self.events_is_automatched = False
        self.events_no_automatch.clear()
        self.processed_deconvolved.clear()
        self.activity_source_scan = None
        self.quality_metrics.clear()
        # Note: library_hbo / library_hbr are deliberately NOT cleared by
        # reset(). They hold immutable bundled data loaded once per process;
        # re-loading on every dataset switch would burn ~100 ms unnecessarily.
        self.library_filter.clear()
        self.library_selected_hrf = None
        # Reset to the default-on state — researchers expect both
        # context overlays when they re-enter the library page.
        self.library_show_brain = True
        self.library_show_scalp = True
        self.library_show_info = True
        self.library_oxygenation = "both"
        # Per-ROI cluster state lives in ``cluster_rois``. Layout
        # refactor (2026-05-16) made the empty list the default; adding
        # a ROI is the implicit activation, so there's no separate
        # ``cluster_roi_active`` field to reset.
        self.cluster_rois = _default_rois()
        self.cluster_active_index = 0
        self.cluster_atlas_alignment_affine = None
        self.cluster_atlas_offset_x_mm = 0.0
        self.cluster_atlas_offset_y_mm = 0.0
        self.cluster_atlas_offset_z_mm = 0.0
        self.last_saved_roi_path = None
        self.hrf_selected_channel = None


# Module-level singleton. Page handlers and components import this directly.
state = AppState()
