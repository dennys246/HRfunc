"""HRfunc GUI entry point.

``hrfunc`` (the CLI command installed by ``pip install hrfunc``)
calls ``main()``, which dispatches three classes of invocation:

1. **Bare launch** (``hrfunc`` or ``hrfunc <path>``) — the dominant path.
   Boots the NiceGUI desktop window. An optional positional ``path``
   preloads a folder / file before the welcome page renders.
2. **Shortcut subcommands** — ``hrfunc install-shortcut`` /
   ``hrfunc uninstall-shortcut``. Adds or removes a system-level
   launcher (Spotlight on macOS, Start menu on Windows, Activities on
   Linux) via ``pyshortcuts``. Researchers who don't live in a terminal
   only need to run this once and can then click HRfunc like any other
   desktop app.
3. **Help / version** — ``hrfunc --help``, ``hrfunc help``, or
   ``hrfunc --version``. Print and exit.

Usage:
    hrfunc                              # launch the GUI
    hrfunc /path/to/study               # launch with that folder preloaded
    hrfunc subject_01.snirf             # launch with a single file preloaded
    hrfunc install-shortcut             # add HRfunc to your system menu
    hrfunc uninstall-shortcut           # remove the HRfunc system menu entry
    hrfunc --version                    # print version and exit
    hrfunc --help                       # show CLI help
    hrfunc help                         # alias for --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Subcommands handled outside argparse — see ``main`` for why.
_SUBCOMMANDS = {"install-shortcut", "uninstall-shortcut", "help"}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns the exit code (0 on success).

    Why the manual subcommand prefilter (vs argparse subparsers): the
    bare-launch form ``hrfunc /path/to/study`` takes a positional path
    argument. argparse subparsers consume the first positional as the
    subcommand name, which would force users to type ``hrfunc launch
    /path/...`` — exactly the friction Denny rejected. The prefilter
    here treats the first argv element as a subcommand only when it
    matches one of the known names; anything else (including paths)
    falls through to the existing bare-launch handling.

    Args:
        argv: Argument list (excluding the program name). When None
            (production), pulls from sys.argv. Tests pass an explicit
            list to exercise without touching sys.argv.

    Returns:
        Exit code. 0 if the invocation completed cleanly. Non-zero on
        argument-parsing failure (argparse exits via SystemExit) or
        subcommand failure.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Subcommand prefilter — match the literal subcommand name before
    # handing argv to argparse.
    if argv:
        head = argv[0]
        if head == "install-shortcut":
            return _run_install_shortcut(argv[1:])
        if head == "uninstall-shortcut":
            return _run_uninstall_shortcut(argv[1:])
        if head == "help":
            # Friendly alias for --help / -h. Forward to argparse so
            # the same usage block renders.
            argv = ["--help"]

    return _launch_gui(argv)


# ---------------------------------------------------------------------------
# Bare-launch GUI path
# ---------------------------------------------------------------------------


def _launch_gui(argv: List[str]) -> int:
    """Parse argv as ``hrfunc [path]`` and boot the NiceGUI window."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # Import NiceGUI lazily so `hrfunc --version` and `hrfunc --help` work
    # even before the heavier GUI imports resolve. The GUI stack is now a
    # core dependency (see pyproject), so this should always succeed on a
    # normal install -- but guard it anyway: if the env is broken/partial,
    # a clear "reinstall" hint beats a raw ModuleNotFoundError traceback for
    # the non-technical researchers who are the GUI's audience.
    try:
        from nicegui import ui
    except ModuleNotFoundError as exc:
        if exc.name not in ("nicegui", "plotly", "pywebview", "webview"):
            raise
        print(
            "HRfunc's desktop GUI couldn't start because a required package "
            f"is missing ({exc.name}). Your install looks incomplete.\n\n"
            "Reinstall HRfunc to pull the GUI dependencies:\n"
            "    pip install --force-reinstall hrfunc\n\n"
            "From a source checkout:\n"
            "    pip install -e .",
            file=sys.stderr,
        )
        return 1

    from .state import state

    if args.path is not None:
        state.preload_path = args.path.resolve()
        logger.info("Preloading path from CLI: %s", state.preload_path)

    _register_pages()

    # Resolve the bundled executable icon to use as the favicon — same
    # PNG that powers the OS-level shortcut so the in-app and out-of-app
    # branding match. Falls back to NiceGUI's default when missing.
    favicon = _resolve_favicon()

    ui.run(
        title="HRfunc",
        native=True,
        window_size=(1400, 900),
        reload=False,
        show=True,
        port=_find_free_port(),
        show_welcome_message=False,
        favicon=favicon,
    )
    return 0


def _resolve_favicon() -> Optional[str]:
    """Return the bundled executable PNG path for ``ui.run(favicon=...)``.

    Returns ``None`` on any resolution failure (broken install, asset
    missing); NiceGUI then uses its default favicon. Keeping this
    fail-safe means a broken asset bundle never blocks GUI startup.
    """
    try:
        from importlib import resources

        ref = resources.files("hrfunc.assets") / "executable_icon.png"
        with resources.as_file(ref) as path:
            return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("favicon unavailable: %s", exc)
        return None


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrfunc",
        description=(
            "HRfunc — fNIRS hemodynamic response function estimation and "
            "neural activity recovery. Bare `hrfunc` launches the desktop "
            "GUI; use `hrfunc install-shortcut` to add HRfunc to your "
            "system menu so you can launch it without the terminal."
        ),
        epilog=(
            "Subcommands:\n"
            "  install-shortcut   Add HRfunc to system menu (Spotlight / "
            "Start menu / Activities)\n"
            "  uninstall-shortcut Remove the HRfunc system menu entry\n"
            "  help               Show this help text (alias for --help)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        type=Path,
        help=(
            "Optional dataset to preload — a folder of scans, a single "
            ".snirf/.fif file, or a NIRx acquisition directory."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hrfunc {_get_version()}",
    )
    return parser


def _register_pages() -> None:
    """Register all `@ui.page` handlers.

    v1.4 single-shell layout: the tabbed shell at ``/`` is the only
    route. Tabs: HRtree, Inspect, Preprocess, Estimate, Activity,
    Quality, Export. The legacy ``welcome.py`` / ``library.py`` /
    ``workspace.py`` modules were deleted across Phases 4-5 — the
    HRtree tab subsumes the old Library route, Inspect was reinstated
    as a tab (Phase 5), and the workspace's three-pane layout is
    superseded by the per-tab dataset-tree + content split.
    """
    from .pages import shell

    shell.register()


def _find_free_port() -> int:
    """Return an unused localhost port for NiceGUI to bind."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_version() -> str:
    """Return the installed hrfunc version, or 'unknown' if not resolvable."""
    try:
        from importlib.metadata import version
        return version("hrfunc")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Shortcut subcommands
# ---------------------------------------------------------------------------


def _run_install_shortcut(argv: List[str]) -> int:
    """Handle ``hrfunc install-shortcut`` — install + report.

    ``argv`` is currently ignored (no flags). Reserved for future use
    (e.g. ``--desktop`` to also write a desktop shortcut).
    """
    if argv:
        print(
            f"hrfunc install-shortcut: unexpected arguments {argv!r}",
            file=sys.stderr,
        )
        return 2

    from ..cli.install_shortcut import install_shortcut, set_prompted

    result = install_shortcut()
    print(result.message)
    if result.ok:
        # Treat manual install as the user's definitive answer to the
        # first-launch prompt so the welcome page doesn't ask again.
        set_prompted()
    return 0 if result.ok else 1


def _run_uninstall_shortcut(argv: List[str]) -> int:
    """Handle ``hrfunc uninstall-shortcut``."""
    if argv:
        print(
            f"hrfunc uninstall-shortcut: unexpected arguments {argv!r}",
            file=sys.stderr,
        )
        return 2

    from ..cli.install_shortcut import uninstall_shortcut

    result = uninstall_shortcut()
    print(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
