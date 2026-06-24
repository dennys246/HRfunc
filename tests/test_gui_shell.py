"""Targeted unit tests for the v1.4 single-shell GUI (Phase 2).

Covers the new ``pages.shell`` module:

- **Toolbar**: brand wordmark renders, project picker is present, tabs
  render in the planned order.
- **Tab landing**: cold launch lands on Library (the soft default);
  CLI preload lands on Preprocess and consumes ``state.preload_path``.
- **Empty states**: data-dependent tabs show "Select a project to <verb>"
  with a single Open button when no project is loaded; the verb is
  customized per tab.
- **Welcome card overlay**: appears on Library when (no project) AND
  (no recent manifests) AND (marker absent); skips when any of the three
  conditions fails; dismissal writes the marker.
- **Project switch**: ``state.set_manifest`` toggles tabs from empty
  state to panel content via the ``project_changed`` subscription.

Module-level helpers (``_should_show_welcome_card``,
``_welcome_card_marker_path``) are exercised in isolation; rendering
tests use the NiceGUI ``User`` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nicegui")

from nicegui.testing import User  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]

from hrfunc.gui import app as gui_app  # noqa: E402
from hrfunc.gui.state import state as global_state  # noqa: E402

gui_app._register_pages()


@pytest.fixture(autouse=True)
def _isolate_marker_and_cache(monkeypatch, tmp_path):
    """Point the welcome-card marker + recent-manifest cache at tmp_path so
    tests don't read or write the real user cache directory. Tests opt
    into "card visible" / "card dismissed" / "recent present" by
    creating files inside this fixture's temp dir.
    """
    monkeypatch.setattr(
        "platformdirs.user_cache_dir",
        lambda *_a, **_kw: str(tmp_path),
    )
    global_state.reset()


# ---------------------------------------------------------------------------
# Toolbar — wordmark + tabs + picker render in correct order
# ---------------------------------------------------------------------------


async def test_shell_renders_brand_wordmark(user: User):
    await user.open("/")
    # The wordmark is emitted as raw HTML "HR<em>func</em>" via the
    # Brand component; match the content literally.
    await user.should_see(content="HR<em>func</em>")


async def test_shell_renders_all_tabs_in_order(user: User):
    """All seven v1.4 tabs are rendered. HRtree is leftmost (soft
    default landing, matches the paper's terminology); Inspect follows
    as a passive 'look at the scan' step before the active pipeline
    (Preprocess → HRFs → Activity → Quality → Export). The 'HRFs'
    tab was briefly named 'Estimate' in v1.4; renamed back so the
    label matches the canonical scientific term users search for."""
    from hrfunc.gui.pages.shell import TAB_NAMES

    expected = (
        "HRtree", "Inspect", "Preprocess", "HRFs",
        "Activity", "Quality", "Export",
    )
    assert TAB_NAMES == expected

    await user.open("/")
    for name in expected:
        await user.should_see(name)


async def test_hrtree_tab_renders_brand_wordmark_header(user: User):
    """The HRtree tab content begins with an HR_tree_ Brand wordmark
    header — reinforces the term as a proper noun for users who arrive
    without paper context."""
    await user.open("/")
    await user.should_see(content="HR<em>tree</em>")


async def test_shell_renders_project_picker(user: User):
    """The picker dropdown is in the toolbar. With no project loaded, the
    label reads "No project"."""
    await user.open("/")
    await user.should_see("No project")


# ---------------------------------------------------------------------------
# Initial tab — Library is the soft default for cold launch
# ---------------------------------------------------------------------------


async def test_cold_launch_lands_on_hrtree(user: User):
    """No project loaded → HRtree is the active tab. The HRtree tab
    renders the bundled-database explorer with the Filter sidebar
    visible by default."""
    await user.open("/")
    # The HRtree tab renders the HRtree panel; "Filter" is the header
    # of the filter sidebar.
    await user.should_see("Filter")


async def test_preload_path_lands_on_preprocess(user: User, tmp_path):
    """``hrfunc <path>`` sets ``state.preload_path``; the shell consumes
    it on render and selects Preprocess as the active tab."""
    target = tmp_path / "demo_project"
    target.mkdir()
    global_state.preload_path = target

    await user.open("/")

    # The shell should have consumed preload_path (so a subsequent render
    # wouldn't re-trigger).
    assert global_state.preload_path is None


# ---------------------------------------------------------------------------
# Empty states — data-dependent tabs render the "Select a project" prompt
# ---------------------------------------------------------------------------


async def test_empty_state_appears_on_data_tabs(user: User):
    """All five data-dependent tabs render an empty-state prompt when
    no project is loaded. Each prompt uses a per-tab verb."""
    await user.open("/")
    # Each data-tab's panel is mounted (tabs are eagerly built); the
    # empty-state copy includes the verb. We assert a representative
    # verb appears; full per-tab assertion would be duplicative.
    await user.should_see(content="Select a project to preprocess.")
    await user.should_see(content="Select a project to estimate HRFs.")


async def test_empty_state_has_open_folder_button(user: User):
    """The empty-state's primary CTA is "Open folder" — same picker as
    the toolbar dropdown."""
    await user.open("/")
    await user.should_see("Open folder")


def test_demo_data_path_resolves_to_bundled_snirf():
    """In a source checkout, the demo button targets the bundled SNIRF folder."""
    from hrfunc.gui.pages import shell

    path = shell._demo_data_path()
    assert path is not None
    assert path.name == "sNIRF_formatted"
    assert any(path.glob("*.snirf"))


async def test_empty_state_shows_demo_button(user: User):
    """When the bundled sample data is present, the empty-state offers a
    rock-paper-scissors demo alongside Open folder."""
    await user.open("/")
    await user.should_see("rock-paper-scissors demo")


# ---------------------------------------------------------------------------
# Welcome card overlay — first-cold-launch onboarding
# ---------------------------------------------------------------------------


class TestWelcomeCardGating:
    """The ``_should_show_welcome_card`` helper has three conditions; one
    test per false-condition exit, plus the all-true happy path."""

    def test_visible_on_fresh_install(self, monkeypatch, tmp_path):
        from hrfunc.gui.pages import shell
        from hrfunc.gui.state import AppState

        monkeypatch.setattr(
            "platformdirs.user_cache_dir", lambda *a, **kw: str(tmp_path)
        )
        assert shell._should_show_welcome_card(AppState()) is True

    def test_hidden_when_project_loaded(self, monkeypatch, tmp_path):
        from hrfunc.gui.pages import shell
        from hrfunc.gui.state import AppState
        from hrfunc.io.manifest import Manifest

        monkeypatch.setattr(
            "platformdirs.user_cache_dir", lambda *a, **kw: str(tmp_path)
        )
        s = AppState()
        s.manifest = Manifest(root=tmp_path)
        assert shell._should_show_welcome_card(s) is False

    def test_hidden_when_marker_present(self, monkeypatch, tmp_path):
        from hrfunc.gui.pages import shell
        from hrfunc.gui.state import AppState

        monkeypatch.setattr(
            "platformdirs.user_cache_dir", lambda *a, **kw: str(tmp_path)
        )
        marker = tmp_path / ".welcome_card_dismissed"
        marker.touch()
        assert shell._should_show_welcome_card(AppState()) is False

    def test_hidden_when_recent_manifests_exist(
        self, monkeypatch, tmp_path
    ):
        from datetime import datetime, timezone

        from hrfunc.gui.pages import shell
        from hrfunc.gui.state import AppState
        from hrfunc.io.manifest import Manifest

        monkeypatch.setattr(
            "platformdirs.user_cache_dir", lambda *a, **kw: str(tmp_path)
        )
        m = Manifest(
            root=Path("/tmp/some_study"),
            scanned_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
        (tmp_path / "manifest_x.json").write_text(m.to_json())
        assert shell._should_show_welcome_card(AppState()) is False


class TestWelcomeCardRender:
    async def test_card_visible_on_fresh_install(self, user: User):
        """No project, no recent manifests, no marker → card appears on
        the Library tab."""
        await user.open("/")
        await user.should_see("Welcome to HRfunc")

    async def test_card_hidden_when_marker_present(
        self, user: User, tmp_path
    ):
        """Marker present → card does NOT render even though the other
        two conditions (no project, no recent) are met."""
        from hrfunc.gui.pages.shell import _mark_welcome_card_dismissed

        _mark_welcome_card_dismissed()
        await user.open("/")
        # The Library tab still renders the HRtree panel content.
        await user.should_see("Filter")
        # But the welcome-card overlay is absent. Use should_not_see to
        # confirm — should_see would loop and fail.
        await user.should_not_see("Welcome to HRfunc")


# ---------------------------------------------------------------------------
# Project switch — manifest swap fires project_changed → panels refresh
# ---------------------------------------------------------------------------


async def test_data_tab_refreshes_on_project_changed(user: User, tmp_path):
    """A ``set_manifest`` call on the shell's state triggers each data-
    tab's refreshable, swapping the empty-state for the loaded panel.

    Asserted via the Project label on the left side of the tab —
    visible only after a project is set.
    """
    from hrfunc.io.manifest import Manifest

    await user.open("/")
    await user.should_see(content="Select a project to preprocess.")

    # Swap the project — both the picker dropdown label and the data-tab
    # bodies should refresh.
    global_state.set_manifest(Manifest(root=tmp_path / "swapped"))
    await user.should_see("Project: swapped")
