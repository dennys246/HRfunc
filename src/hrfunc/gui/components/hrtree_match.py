"""Match a scan's channels to HRtree HRFs for per-channel deconvolution.

The Activity tab's "HRtree HRF" source can deconvolve each channel with its
OWN spatially-matched HRF (instead of one shared kernel) using the ROIs the
user built in the HRtree. This module does the matching and reports coverage
so the UI can show how many channels are covered vs. uncovered.

Both the scan's channel locations (read straight from ``raw.info['chs']``)
and the HRtree HRF locations are in **MNE head coordinates (meters)**, so the
nearest-neighbour match is a plain Euclidean distance in meters — no MNI
alignment needed (that only matters for atlas lookups). Matching is
oxygenation-respecting: HbO channels match only HbO HRFs, HbR only HbR, so
the matched trace is used as-given (no sign flip).

Two strategies (user-selectable):
- ``"individual"`` — each channel takes the single nearest member HRF across
  all visible ROIs.
- ``"roi_mean"`` — each channel takes the mean trace of the nearest visible
  ROI (ROI location = the head-coords centroid of its member HRFs, so this is
  also alignment-free).

Channels with no same-oxygenation candidate within ``radius_mm`` are
"uncovered" and reported; the caller decides whether to skip or canonical-fill
them (``estimate_activity(library_uncovered=...)``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..state import AppState

STRATEGY_INDIVIDUAL = "individual"
STRATEGY_ROI_MEAN = "roi_mean"


@dataclass
class ChannelMatch:
    """One scan channel's match outcome."""

    ch_name: str  # standardized name (matches estimate_activity's channel keys)
    oxygenation: bool  # True = HbO, False = HbR
    matched: bool
    trace: Optional[List[float]] = None  # the kernel for this channel, or None
    source: Optional[str] = None  # which HRF key / ROI name supplied the trace
    distance_mm: Optional[float] = None


@dataclass
class MatchResult:
    """Per-channel matching outcome + headline counts for the UI."""

    matches: List[ChannelMatch] = field(default_factory=list)
    n_candidate_hrfs: int = 0  # HRFs available across visible ROIs
    n_rois: int = 0  # visible ROIs that contributed candidates
    strategy: str = STRATEGY_INDIVIDUAL
    radius_mm: float = 20.0

    @property
    def covered(self) -> List[ChannelMatch]:
        return [m for m in self.matches if m.matched]

    @property
    def uncovered(self) -> List[ChannelMatch]:
        return [m for m in self.matches if not m.matched]

    def library_traces(self) -> Dict[str, List[float]]:
        """The ``{ch_name: trace}`` map for ``estimate_activity(library_traces=...)``."""
        return {
            m.ch_name: m.trace
            for m in self.matches
            if m.matched and m.trace
        }


# Internal candidate: (label, oxygenation, location_m (3,), trace list)
_Candidate = Tuple[str, bool, "np.ndarray", List[float]]


def _channel_geometry(raw) -> List[Tuple[str, bool, "np.ndarray"]]:
    """Read (standardized_name, oxygenation, location_m) for each fNIRS channel.

    Channels whose name can't be parsed for oxygenation, or that carry no
    usable location (all-zero / non-finite ``loc``), are skipped — they can't
    be matched spatially and would otherwise masquerade as covered.
    """
    from ..._utils import _is_oxygenated, standardize_name

    out: List[Tuple[str, bool, np.ndarray]] = []
    for ch in raw.info["chs"]:
        raw_name = ch.get("ch_name", "")
        try:
            std = standardize_name(raw_name)
            oxy = _is_oxygenated(std)
        except Exception:  # noqa: BLE001 — non-fNIRS / unparsable name
            continue
        loc = np.asarray(ch.get("loc", [])[:3], dtype=float)
        if loc.shape != (3,) or not np.all(np.isfinite(loc)) or np.allclose(loc, 0.0):
            # No usable location -> can't match; record as uncovered geometry.
            out.append((std, oxy, None))
            continue
        out.append((std, oxy, loc))
    return out


def _hrf_loc_trace(hrf: Dict[str, Any]) -> Optional[Tuple["np.ndarray", List[float], bool]]:
    """Pull (location_m, trace, oxygenation) from a gathered HRF dict, or None."""
    loc = hrf.get("location")
    trace = hrf.get("hrf_mean")
    if not loc or len(loc) < 3 or trace is None or len(trace) == 0:
        return None
    arr = np.asarray(loc[:3], dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return arr, list(trace), bool(hrf.get("oxygenation"))


def _individual_candidates(
    state: AppState, all_hrfs: Dict[str, Dict[str, Any]]
) -> Tuple[List[_Candidate], int]:
    """Each member HRF across visible ROIs becomes a candidate."""
    from .hrtree_panel import _visible_roi_keys

    union_keys, pairs = _visible_roi_keys(state, all_hrfs)
    candidates: List[_Candidate] = []
    for key in union_keys:
        hrf = all_hrfs.get(key)
        if hrf is None:
            continue
        parsed = _hrf_loc_trace(hrf)
        if parsed is None:
            continue
        loc, trace, oxy = parsed
        candidates.append((key, oxy, loc, trace))
    return candidates, len(pairs)


def _roi_mean_candidates(
    state: AppState, all_hrfs: Dict[str, Dict[str, Any]]
) -> Tuple[List[_Candidate], int]:
    """Each visible ROI becomes one candidate at its member-centroid (head
    coords), carrying the ROI's mean trace."""
    from .hrtree_panel import (
        _alignment_for_shape,
        _visible_shapes,
        compute_roi_average,
        compute_roi_keys_by_shape,
    )

    pairs = _visible_shapes(state)
    candidates: List[_Candidate] = []
    contributing = 0
    for slot, shape in pairs:
        anchor = slot.anchor
        alignment = _alignment_for_shape(state, shape)
        roi_contributed = False
        # Oxygenation-pure: build a SEPARATE mean per haemoglobin so an ROI
        # spanning both never averages HbO and HbR together (they're inverses
        # — pooling them cancels the response). A channel then matches only the
        # same-oxygenation ROI candidate, mirroring the individual strategy's
        # hard same-oxygenation rule. Oxygenations absent from the (already
        # filtered) set yield empty keys and are skipped.
        for oxy in (True, False):
            keys = compute_roi_keys_by_shape(
                all_hrfs, shape, slot.painted,
                oxygenation_filter=oxy,
                alignment_affine=alignment,
            )
            locs = []
            for key in keys:
                hrf = all_hrfs.get(key)
                if hrf is None:
                    continue
                parsed = _hrf_loc_trace(hrf)
                if parsed is not None:
                    locs.append(parsed[0])
            if not locs:
                continue
            centroid = np.mean(np.vstack(locs), axis=0)
            # ROI kernel: prefer the subject-weighted grand mean; fall back to
            # the anchor's own trace ONLY when it matches this oxygenation, so
            # the fallback never crosses haemoglobins.
            avg = compute_roi_average(all_hrfs, keys)
            if avg is not None:
                trace = list(np.asarray(avg[0], dtype=float))
            elif (
                anchor is not None
                and anchor.get("hrf_mean")
                and bool(anchor.get("oxygenation")) == oxy
            ):
                trace = list(anchor["hrf_mean"])
            else:
                continue
            label = f"{slot.name} ({'HbO' if oxy else 'HbR'})"
            candidates.append((label, oxy, centroid, trace))
            roi_contributed = True
        if roi_contributed:
            contributing += 1
    return candidates, contributing


def match_channels_to_hrtree(
    state: AppState,
    raw,
    *,
    strategy: str = STRATEGY_INDIVIDUAL,
    radius_mm: float = 20.0,
) -> MatchResult:
    """Match each scan channel to an HRtree HRF from the user's ROIs.

    :param raw: the (preprocessed) MNE Raw whose channels to match.
    :param strategy: ``"individual"`` (nearest member HRF) or ``"roi_mean"``
        (nearest ROI's mean trace).
    :param radius_mm: max match distance; channels with no same-oxygenation
        candidate within this radius are uncovered.
    """
    from .hrtree_panel import (
        apply_filter,
        filter_by_oxygenation,
        gather_library_hrfs,
    )

    # Use the SAME filtered HRF set the Cluster view operates on, so the
    # context filters (task / doi / demographics …) and the HbO/HbR/Both
    # oxygenation choice scope the deconvolution candidates exactly as they
    # scope what the user sees — Filter and Cluster stay complementary.
    all_hrfs = filter_by_oxygenation(
        apply_filter(gather_library_hrfs(state), state.library_filter),
        state.library_oxygenation,
    )
    if strategy == STRATEGY_ROI_MEAN:
        candidates, n_rois = _roi_mean_candidates(state, all_hrfs)
    else:
        strategy = STRATEGY_INDIVIDUAL
        candidates, n_rois = _individual_candidates(state, all_hrfs)

    radius_m = float(radius_mm) / 1000.0
    geom = _channel_geometry(raw)
    matches: List[ChannelMatch] = []
    for ch_name, oxy, loc in geom:
        if loc is None:
            matches.append(ChannelMatch(ch_name, oxy, matched=False))
            continue
        # Restrict to same-oxygenation candidates so the trace applies as-given.
        pool = [c for c in candidates if c[1] == oxy]
        best = None
        best_dist = None
        for label, _c_oxy, c_loc, trace in pool:
            dist = float(np.linalg.norm(loc - c_loc))
            if best_dist is None or dist < best_dist:
                best_dist, best = dist, (label, trace)
        if best is not None and best_dist is not None and best_dist <= radius_m:
            label, trace = best
            matches.append(ChannelMatch(
                ch_name, oxy, matched=True, trace=trace, source=label,
                distance_mm=best_dist * 1000.0,
            ))
        else:
            matches.append(ChannelMatch(ch_name, oxy, matched=False))

    return MatchResult(
        matches=matches,
        n_candidate_hrfs=len(candidates),
        n_rois=n_rois,
        strategy=strategy,
        radius_mm=float(radius_mm),
    )
