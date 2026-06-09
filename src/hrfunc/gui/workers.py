"""Background-task helpers for long-running GUI operations.

NiceGUI runs page handlers on its event loop; any synchronous work that takes
more than ~100ms blocks the UI. Scanning a large folder or estimating HRFs
take seconds to minutes — they must run off the main thread.

This module provides a thin wrapper around `asyncio.to_thread` plus a
progress-state helper for surfacing `progress_callback` events to the UI.
Sprint 2.1 ships the helper; Sprint 3 (estimate panel) wires it up to actual
estimation calls.

Design constraints:
- **Single background worker at a time.** AppState.busy is a binary flag,
  not a counter. The GUI disables long-task buttons while busy=True so the
  user can't queue overlapping work. This matches the RawCache's not-thread-
  safe contract (see hrfunc.io.raw_cache).
- **Progress is pushed, not polled.** The callback writes (current, total,
  name) into `state.estimation_progress`; UI components bind to that field
  and re-render via NiceGUI's reactivity. No timer needed.
- **Errors surface to `state.last_error`.** The worker catches exceptions
  raised in the threaded function and stores the string message; the GUI
  displays a toast / banner from there.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Tuple

from .state import AppState
from ..io.manifest import ScanEntry

logger = logging.getLogger(__name__)


def make_progress_callback(state: AppState) -> Callable[[int, int, str], None]:
    """Return a `progress_callback` that writes into the given AppState.

    The returned callable matches the signature expected by
    `montage.estimate_hrf` and `montage.estimate_activity`:
    `(current_index, total_channels, channel_name) -> None`.

    Each call writes a `(current, total, name)` tuple into
    `state.estimation_progress`. UI components bound to that field re-render
    automatically via NiceGUI's reactivity.
    """

    def _callback(current: int, total: int, name: str) -> None:
        state.estimation_progress = (current, total, name)

    return _callback


async def run_in_background(
    state: AppState,
    func: Callable[..., Any],
    *args: Any,
    on_done: Optional[Callable[[Any], Awaitable[None]]] = None,
    **kwargs: Any,
) -> Any:
    """Run a blocking function off the main thread, surfacing busy/error state.

    Sets `state.busy = True` before dispatch, clears it (and resets
    `estimation_progress`) when the function returns. Any exception is
    logged and stored in `state.last_error` as a string; the exception is
    NOT re-raised so the GUI stays responsive.

    Args:
        state: AppState whose `busy`, `estimation_progress`, and `last_error`
            fields will be updated.
        func: Synchronous callable to run on a worker thread.
        *args, **kwargs: Forwarded to `func`.
        on_done: Optional async callable invoked with the result after `func`
            completes successfully. Useful for "estimate, then refresh the
            HRF gallery" flows.

    Returns:
        The result of `func`, or `None` if `func` raised.
    """
    if state.busy:
        logger.warning(
            "run_in_background: state.busy is already True; refusing to "
            "start a second worker. The GUI should disable trigger buttons "
            "while busy."
        )
        return None

    state.set_busy(True)
    state.last_error = None
    result: Any = None
    try:
        # Use run_in_executor instead of asyncio.to_thread (3.9+) so the GUI
        # works on Python 3.8 to match the library's requires-python pin.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)
        )
    except Exception as exc:  # noqa: BLE001 — see module docstring
        state.last_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Background worker failed: %s", exc)
        return None
    finally:
        # Clear progress BEFORE flipping busy: set_busy(False) synchronously
        # publishes busy_changed, so a subscriber that re-renders would
        # otherwise observe busy=False while estimation_progress still holds
        # the last channel tuple. Matches the bulk worker's finally ordering.
        state.estimation_progress = None
        state.set_busy(False)

    if on_done is not None:
        try:
            await on_done(result)
        except Exception as exc:  # noqa: BLE001
            state.last_error = f"on_done failed: {type(exc).__name__}: {exc}"
            logger.exception("on_done callback failed: %s", exc)

    return result


BulkResult = Tuple[List[ScanEntry], List[Tuple[ScanEntry, str]]]


async def run_bulk_in_background(
    state: AppState,
    scans: Sequence[ScanEntry],
    build_call: Callable[[ScanEntry], Optional[Tuple[Callable[..., Any], tuple, dict]]],
    *,
    on_each_done: Optional[Callable[[ScanEntry, Any], Awaitable[None]]] = None,
    label: str = "bulk run",
) -> Optional[BulkResult]:
    """Run a synchronous callable against each scan in ``scans`` in order.

    PR #55a -- the "checked N scans, click Run" workflow on the Preprocess
    / HRF / Activity tabs. Acquires the busy gate once for the whole
    batch (matching the single-worker contract on AppState.busy) and
    advances ``state.bulk_progress`` per scan so panels can render a
    "Scan i/N: name" line above the within-scan progress.

    The per-scan call is built by ``build_call(scan) -> (func, args, kwargs)``
    so each panel can layer its own preflight (e.g. "is the raw cached
    for this scan?") without duplicating the dispatch machinery. Returning
    ``None`` from ``build_call`` skips the scan with a "not ready"
    reason -- counted as skipped, not failed.

    Continue-on-error semantics: per-scan exceptions are caught, logged,
    stamped onto ``state.last_error`` (overwritten per failure), and the
    loop continues to the next scan. The returned tuple
    ``(successes, failures)`` lists which scans landed in which bucket so
    the caller can surface a "N succeeded, M failed" toast.

    ``on_each_done`` runs after each successful per-scan call -- caches
    the result, publishes the per-scan event, etc. Exceptions from it
    move the scan from success → failure (so a failed cache write is
    visible to the user, not silently masked).

    The function returns ``None`` if the busy gate is already held
    (matches ``run_in_background`` so callers can detect "already
    running" symmetrically).
    """
    if state.busy:
        logger.warning(
            "run_bulk_in_background: state.busy already True; refusing to "
            "start a bulk run on top of an in-flight task."
        )
        return None
    if not scans:
        return ([], [])

    state.set_busy(True)
    state.last_error = None
    successes: List[ScanEntry] = []
    failures: List[Tuple[ScanEntry, str]] = []
    total = len(scans)
    loop = asyncio.get_event_loop()
    try:
        for index, scan in enumerate(scans):
            state.bulk_progress = (index, total, scan)
            # Per-scan within-channel progress is reset between scans
            # so the previous scan's last channel number doesn't bleed
            # into the next scan's progress line.
            state.estimation_progress = None

            try:
                built = build_call(scan)
            except Exception as exc:  # noqa: BLE001
                msg = f"build_call failed: {type(exc).__name__}: {exc}"
                logger.exception("%s: %s", label, msg)
                state.last_error = f"{scan.path.name}: {msg}"
                failures.append((scan, msg))
                continue

            if built is None:
                failures.append((scan, "skipped (preflight)"))
                continue

            func, args, kwargs = built
            try:
                result = await loop.run_in_executor(
                    None, lambda f=func, a=args, k=kwargs: f(*a, **k)
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                logger.exception("%s on %s: %s", label, scan.path.name, msg)
                state.last_error = f"{scan.path.name}: {msg}"
                failures.append((scan, msg))
                continue

            if on_each_done is not None:
                try:
                    await on_each_done(scan, result)
                except Exception as exc:  # noqa: BLE001
                    msg = f"on_each_done failed: {type(exc).__name__}: {exc}"
                    logger.exception("%s on %s: %s", label, scan.path.name, msg)
                    state.last_error = f"{scan.path.name}: {msg}"
                    failures.append((scan, msg))
                    continue

            successes.append(scan)
    finally:
        state.bulk_progress = None
        state.estimation_progress = None
        state.set_busy(False)

    return (successes, failures)
