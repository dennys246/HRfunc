"""In-memory LRU cache of loaded MNE Raw objects.

The v1.3.0 GUI lets users navigate between scans via the dataset tree. Each
selection needs the underlying ``mne.io.Raw`` for plotting, channel info, and
event extraction. Re-reading a SNIRF/NIRx/FIF from disk on every click would
be slow (hundreds of ms for SNIRF, several seconds for large NIRx folders).
This cache holds the last ``maxsize`` Raws in memory so common "previous /
current / next" navigation is instant.

Cache semantics:
- **LRU eviction**: when the cache is full and a new scan is loaded, the
  least-recently-used Raw is dropped.
- **Cache hit promotes to MRU**: accessing a cached Raw bumps it to the
  most-recently-used position.
- **Format dispatch**: the loader is chosen from ``classify_path`` so the
  caller can pass either a ``ScanEntry``, a ``Path``, or a string and the
  cache figures out which MNE reader to call.
- **In-memory only**: nothing is persisted; the cache starts empty each
  process. (The on-disk manifest cache in ``scan.py`` is the persistent
  layer; this complements it.)

Thread safety:
- The cache is **not** thread-safe in v1.3.0. The GUI runs estimation in a
  single background worker at a time, so concurrent access does not occur
  in practice. If a future GUI iteration parallelizes I/O, wrap operations
  in a lock at the call site.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Union

from .detect import classify_path
from .manifest import ScanEntry

if TYPE_CHECKING:
    import mne


DEFAULT_MAXSIZE = 3


class RawCache:
    """LRU cache of loaded MNE Raw objects keyed by absolute path.

    Args:
        maxsize: Maximum number of Raws to retain. Must be >= 1.
            Default 3 — covers "previous / current / next" navigation
            while bounding memory to roughly 3 × scan size. Large NIRx
            recordings are ~100MB in memory; smaller SNIRF ones are
            a few MB.

    Raises:
        ValueError: If ``maxsize < 1``.
    """

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self.maxsize = maxsize
        self._cache: "OrderedDict[Path, mne.io.BaseRaw]" = OrderedDict()

    def get(self, scan_or_path: Union[ScanEntry, Path, str]) -> "mne.io.BaseRaw":
        """Load and return a Raw for the given scan.

        On cache hit, the Raw is returned and promoted to most-recently-used.
        On cache miss, the Raw is read from disk via the appropriate MNE
        loader (chosen from the path's classified format), inserted into the
        cache, and the least-recently-used entry is evicted if needed.

        Args:
            scan_or_path: A ScanEntry, a Path, or a string path. The path
                must point to a recognized fNIRS dataset (SNIRF file, NIRx
                directory, or fNIRS-containing FIF file).

        Returns:
            The loaded MNE Raw object. The cache retains a reference, so
            the caller should not assume exclusive ownership — multiple
            cache hits return the same instance.

        Raises:
            ValueError: If the path does not classify as a recognized
                fNIRS dataset.
        """
        path = self._extract_path(scan_or_path)
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        raw = self._load_from_disk(path)
        self._cache[path] = raw
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return raw

    def put(
        self, scan_or_path: Union[ScanEntry, Path, str], raw: "mne.io.BaseRaw"
    ) -> None:
        """Insert (or replace) a Raw the caller already holds, honoring LRU.

        Used by callers that produce a Raw themselves rather than loading it
        through :meth:`get` — e.g. the Preprocess / Quality panels storing a
        *preprocessed* Raw under its scan path. Writing straight into
        ``_cache`` (as those sites previously did) bypasses the ``maxsize``
        eviction that only :meth:`get` performed, so a bulk run could retain
        every result and grow memory without bound. Routing the writes here
        enforces the bound in one place. The entry is promoted to MRU.
        """
        path = self._extract_path(scan_or_path)
        self._cache[path] = raw
        self._cache.move_to_end(path)
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def evict(self, scan_or_path: Union[ScanEntry, Path, str]) -> bool:
        """Remove a specific entry from the cache.

        Used when the underlying file has changed and the cached Raw is
        stale (the GUI's Rescan flow calls this for any open scan).

        Returns:
            True if an entry was removed, False if it was not in the cache.
        """
        path = self._extract_path(scan_or_path)
        if path in self._cache:
            del self._cache[path]
            return True
        return False

    def clear(self) -> None:
        """Drop all cached Raws.

        Called by the GUI when switching to a new dataset or closing the
        current project — releases any large in-memory NIRx recordings.
        """
        self._cache.clear()

    def __contains__(self, scan_or_path: Union[ScanEntry, Path, str]) -> bool:
        return self._extract_path(scan_or_path) in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @staticmethod
    def _extract_path(scan_or_path: Union[ScanEntry, Path, str]) -> Path:
        """Normalize the cache key to an absolute Path.

        Symlinks are resolved so two different paths pointing at the same
        file share a cache entry. Without resolution, opening the same scan
        through different paths would double-fill the cache.
        """
        if isinstance(scan_or_path, ScanEntry):
            return scan_or_path.path.resolve()
        return Path(scan_or_path).resolve()

    @staticmethod
    def _load_from_disk(path: Path) -> "mne.io.BaseRaw":
        """Read a Raw from disk, dispatching by format."""
        import mne

        hit = classify_path(path)
        if hit is None:
            raise ValueError(
                f"Path is not a recognized fNIRS dataset: {path}"
            )

        if hit.format == "snirf":
            return mne.io.read_raw_snirf(path, verbose="ERROR")
        if hit.format == "nirx_dir":
            return mne.io.read_raw_nirx(path, verbose="ERROR")
        if hit.format == "fif":
            return mne.io.read_raw_fif(path, verbose="ERROR")

        # classify_path only ever returns one of the three formats; this is
        # a defensive guard in case a future format is added to detect.py
        # without updating this dispatcher.
        raise ValueError(f"No loader registered for format {hit.format!r}")
