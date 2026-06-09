"""Detect the fNIRS format of a filesystem path.

Three formats are supported:

- ``snirf``: a file ending in ``.snirf``. Always treated as fNIRS by definition
  of the format.
- ``nirx_dir``: a directory containing a NIRx acquisition. Identified by the
  presence of both a ``*_probeInfo.mat`` file and at least one ``*.wl1`` file
  inside the directory (these are the unambiguous NIRx markers).
- ``fif``: a file ending in ``.fif``. MNE ``.fif`` files can contain MEG, EEG,
  or fNIRS data — to confirm a file is fNIRS we read only the info header
  (no data load) and check whether MNE's fnirs channel selector returns any
  picks.

The classifier is path-shape first, content second. SNIRF and NIRx are
identified by filesystem markers alone; FIF requires a cheap MNE info read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FormatHit:
    """A successful classification of a filesystem path.

    Attributes:
        format: ``"snirf"``, ``"nirx_dir"``, or ``"fif"``.
        path: The canonical entry point for the format. For snirf/fif this is
            the file itself; for nirx_dir this is the directory (which is what
            ``mne.io.read_raw_nirx`` expects).
        is_fnirs: ``True`` if the path is confirmed to contain fNIRS data.
            Always ``True`` for snirf and nirx_dir (those formats are
            fNIRS-only by definition). For fif this is ``True`` iff the file
            has at least one fNIRS channel.
        n_channels: Channel count when cheaply available, otherwise ``None``.
            Populated for fif (from the info read); not populated for snirf
            or nirx_dir to keep classification fast.
        sfreq: Sampling frequency in Hz when cheaply available, otherwise
            ``None``. Same population rules as ``n_channels``.
    """

    format: str
    path: Path
    is_fnirs: bool
    n_channels: Optional[int] = None
    sfreq: Optional[float] = None


def classify_path(path: Union[str, Path]) -> Optional[FormatHit]:
    """Classify a filesystem path as an fNIRS dataset entry point.

    Args:
        path: A file or directory path.

    Returns:
        A FormatHit if the path matches a known fNIRS format, otherwise None.
        Returns None for missing paths, unrelated files, and FIF files that do
        not contain fNIRS channels.

    Notes:
        - The returned ``FormatHit.path`` is the path AS PASSED (wrapped in a
          ``Path``); it is not resolved or symlink-followed. Callers that need
          an absolute/canonical path should resolve before calling (the folder
          scanner does — it walks an already-resolved root). Resolving here
          would change ``FormatHit.path``'s shape, which downstream BIDS
          metadata parsing relies on being root-relative.
        - This function does not raise on bad input. A path that does not
          exist, a directory missing NIRx markers, a non-fNIRS FIF, etc., all
          return None. Errors during the FIF info read are logged and treated
          as a non-match — the goal is to never crash a folder scan over a
          single bad file.
    """
    p = Path(path)
    if not p.exists():
        return None

    if p.is_dir():
        return _classify_dir(p)

    suffix = p.suffix.lower()
    if suffix == ".snirf":
        return FormatHit(format="snirf", path=p, is_fnirs=True)
    if suffix == ".fif":
        return _classify_fif(p)

    return None


def _classify_dir(p: Path) -> Optional[FormatHit]:
    """Check whether a directory is a NIRx acquisition folder.

    A NIRx folder must contain both:
    - at least one file matching ``*_probeInfo.mat`` (the probe geometry, the
      most distinctive NIRx marker), and
    - at least one file matching ``*.wl1`` (the wavelength-1 data file, paired
      with ``.wl2`` in every real NIRx recording).

    Either marker alone is insufficient. ``*_probeInfo.mat`` is unique enough
    that it would rarely appear outside a NIRx folder, but pairing with
    ``*.wl1`` guards against shared-probe-geometry directories that don't
    contain actual recordings.
    """
    try:
        has_probe = any(p.glob("*_probeInfo.mat"))
        has_wl1 = any(p.glob("*.wl1"))
    except OSError as exc:
        logger.debug("Skipping %s: %s", p, exc)
        return None

    if has_probe and has_wl1:
        return FormatHit(format="nirx_dir", path=p, is_fnirs=True)
    return None


def _classify_fif(p: Path) -> Optional[FormatHit]:
    """Read the FIF header to check whether it contains fNIRS channels.

    ``mne.io.read_info`` reads only the header — no data is loaded, so this
    is cheap even for large recordings. We then ask MNE's channel-type
    selector whether any fNIRS channels are present.

    Returns None for FIF files that contain only MEG/EEG/other modalities,
    and for files that fail to parse (logged at debug level — a folder scan
    should never crash on one bad file).
    """
    import mne  # imported lazily so classification of snirf/nirx_dir paths
                # does not pay the MNE import cost

    try:
        info = mne.io.read_info(p, verbose="ERROR")
    except Exception as exc:
        logger.debug("Could not read FIF info from %s: %s", p, exc)
        return None

    fnirs_picks = mne.pick_types(info, fnirs=True, exclude=[])
    if len(fnirs_picks) == 0:
        return None

    return FormatHit(
        format="fif",
        path=p,
        is_fnirs=True,
        n_channels=int(len(fnirs_picks)),
        sfreq=float(info["sfreq"]),
    )
