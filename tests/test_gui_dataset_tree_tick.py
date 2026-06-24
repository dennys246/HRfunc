"""Interaction tests for the dataset-tree checkbox (tick) pipeline.

The sync tests in ``test_gui_dataset_tree.py`` cover ``build_nodes`` /
``find_scan`` only — the pure data transforms. They never exercise the
*interactive* tick path: ``ui.tree(on_tick=...)`` → ``_on_tick`` →
``state.checked_scan_paths`` → ``checked_changed`` → action-panel bulk
mode. That gap let two separate "ticking is broken" fixes ship without a
regression guard:

- ``bbff389`` ("ticking scans now makes them eligible to estimate") wired
  the ``checked_changed`` publish so the Preprocess / HRF / Activity panels
  recompute bulk mode when the checked set changes.
- ``d17a6c4`` ("checkboxes stay in sync across tabs") subscribed each tree
  to ``checked_changed`` so the per-tab trees re-render their visual ticks
  off the shared set.

These tests drive the REAL pipeline in-process via NiceGUI's ``User``
fixture: ``user.find(ui.tree).trigger("update:ticked", [...])`` dispatches
the exact ``GenericEventArguments`` the Quasar ``q-tree`` sends on a
checkbox click, so ``_on_tick`` runs with a faithful payload (including its
mid-handler ``_tree_body.refresh()``).

Scope note: the ``User`` fixture calls the registered Python handlers
directly — it does NOT run the browser/Quasar client. So these tests lock
the server-side contract (tick → state → bulk mode). A purely client-side
rendering regression would need the Selenium ``Screen`` fixture instead.
"""
from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from hrfunc.gui.components import dataset_tree  # noqa: E402
from hrfunc.gui.state import AppState  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]


def _manifest(tmp_path) -> Manifest:
    """Two BIDS subjects, one scan each — leaf ids are stringified paths."""
    scans = (
        ScanEntry(
            format="snirf",
            path=tmp_path / "sub-01" / "ses-01" / "a.snirf",
            bids_subject="01", bids_session="01", display_name="A",
        ),
        ScanEntry(
            format="snirf",
            path=tmp_path / "sub-02" / "ses-01" / "b.snirf",
            bids_subject="02", bids_session="01", display_name="B",
        ),
    )
    return Manifest(root=tmp_path, scans=scans)


def _tick(user: User, *paths) -> None:
    """Simulate the q-tree emitting its full ticked-id list after a click.

    Quasar sends the COMPLETE set of ticked leaf ids on every change (not a
    delta); the leaf id is ``str(scan.path)``.
    """
    user.find(ui.tree).trigger("update:ticked", [str(p) for p in paths])


def _dispatch_tick(element, client, *paths) -> None:
    """Fire ``update:ticked`` on ONE specific tree element (not every match).

    Used by the cross-tree sync test where two trees coexist and we must
    tick exactly one to prove the other re-syncs off the shared state.
    """
    from nicegui import events

    for listener in element._event_listeners.values():
        if listener.type == "update:ticked":
            events.handle_event(
                listener.handler,
                events.GenericEventArguments(
                    sender=element, client=client,
                    args=[str(p) for p in paths],
                ),
            )


# ---------------------------------------------------------------------------
# Core tick → checked_scan_paths round-trip
# ---------------------------------------------------------------------------


class TestTickUpdatesCheckedSet:
    @pytest.mark.asyncio
    async def test_ticking_a_leaf_adds_resolved_path(self, user: User, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        a = state.manifest.scans[0]

        @ui.page("/_tt_add")
        def _p() -> None:
            dataset_tree.render(state)

        await user.open("/_tt_add")
        assert state.checked_scan_paths == set()

        _tick(user, a.path)
        # Stored as a RESOLVED path so equality is filesystem-stable across
        # manifest re-walks (the on-disk path is the constant identity).
        assert state.checked_scan_paths == {a.path.resolve()}

    @pytest.mark.asyncio
    async def test_ticking_second_leaf_accumulates(self, user: User, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        a, b = state.manifest.scans

        @ui.page("/_tt_acc")
        def _p() -> None:
            dataset_tree.render(state)

        await user.open("/_tt_acc")
        _tick(user, a.path)
        _tick(user, a.path, b.path)
        assert state.checked_scan_paths == {a.path.resolve(), b.path.resolve()}

    @pytest.mark.asyncio
    async def test_unticking_removes_only_that_leaf(self, user: User, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        a, b = state.manifest.scans

        @ui.page("/_tt_untick")
        def _p() -> None:
            dataset_tree.render(state)

        await user.open("/_tt_untick")
        _tick(user, a.path, b.path)
        _tick(user, b.path)  # untick A
        assert state.checked_scan_paths == {b.path.resolve()}

    @pytest.mark.asyncio
    async def test_tick_publishes_checked_changed(self, user: User, tmp_path):
        """Panels recompute bulk mode off this event (bbff389). Without the
        publish, ticking would update the set but leave the action buttons
        stuck in single-scan mode until some other event fired."""
        state = AppState()
        state.manifest = _manifest(tmp_path)
        a = state.manifest.scans[0]

        seen = []
        state.subscribe("checked_changed", lambda payload=None: seen.append(payload))

        @ui.page("/_tt_pub")
        def _p() -> None:
            dataset_tree.render(state)

        await user.open("/_tt_pub")
        _tick(user, a.path)
        assert seen, "ticking a scan did not publish 'checked_changed'"
        assert seen[-1] == {a.path.resolve()}


# ---------------------------------------------------------------------------
# Select-all checkbox
# ---------------------------------------------------------------------------


class TestSelectAll:
    @pytest.mark.asyncio
    async def test_select_all_ticks_every_visible_scan(self, user: User, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)

        @ui.page("/_tt_all")
        def _p() -> None:
            dataset_tree.render(state)

        await user.open("/_tt_all")
        # The "Select all" checkbox is the only top-level ui.checkbox.
        user.find(ui.checkbox).trigger("update:modelValue", True)
        assert state.checked_scan_paths == {
            s.path.resolve() for s in state.manifest.scans
        }

    @pytest.mark.asyncio
    async def test_select_all_off_clears_visible_scans(self, user: User, tmp_path):
        state = AppState()
        state.manifest = _manifest(tmp_path)
        for s in state.manifest.scans:
            state.checked_scan_paths.add(s.path.resolve())

        @ui.page("/_tt_all_off")
        def _p() -> None:
            dataset_tree.render(state)

        await user.open("/_tt_all_off")
        user.find(ui.checkbox).trigger("update:modelValue", False)
        assert state.checked_scan_paths == set()


# ---------------------------------------------------------------------------
# Cross-tab sync (d17a6c4): a second tree on the shared state reflects ticks
# ---------------------------------------------------------------------------


class TestCrossTreeSync:
    @pytest.mark.asyncio
    async def test_second_tree_reflects_shared_checked_set(self, user: User, tmp_path):
        """Two trees, one shared AppState (the real shell mounts one tree per
        data tab). Ticking in the first must leave the second's ticks in sync
        via the ``checked_changed`` refresh — no split-brain between a panel's
        "1 selected" label and an unchecked tree."""
        state = AppState()
        state.manifest = _manifest(tmp_path)
        a = state.manifest.scans[0]

        @ui.page("/_tt_two")
        def _p() -> None:
            with ui.row():
                dataset_tree.render(state)
                dataset_tree.render(state)

        await user.open("/_tt_two")
        trees = list(user.find(ui.tree).elements)
        assert len(trees) == 2, "expected two independent tree instances"

        # Tick on the FIRST tree only. Both trees share one AppState, so the
        # checked set must update regardless of which tree originated the tick.
        _dispatch_tick(trees[0], user.client, a.path)
        assert state.checked_scan_paths == {a.path.resolve()}

        # The originating tree re-applies its own ticks (its _on_tick refreshes
        # its body before the cross-tree publish), so the active-tab tree the
        # user is looking at always reflects the click.
        live_ticks = [
            t._props.get("ticked", [])
            for t in user.find(ui.tree).elements
            if not t.is_deleted
        ]
        assert any(str(a.path) in ticked for ticked in live_ticks), (
            "no live tree reflected the tick in its visual ticked state"
        )


# ---------------------------------------------------------------------------
# Integration: ticking alone makes the action panels eligible (no row click)
# ---------------------------------------------------------------------------
#
# This is the user-facing contract behind the "disjointed selection" report:
# a checkbox tick — with NO row click setting state.selected_scan — must put
# the Preprocess / HRF / Activity panels into bulk mode so the action engages
# the checked file. Locked per-panel so a regression in any one surfaces.


def _render_tab(state: AppState, panel_render):
    with ui.row():
        with ui.column():
            dataset_tree.render(state)
        with ui.column():
            panel_render(state)


class TestTickEngagesBulkModeWithoutRowClick:
    @pytest.mark.asyncio
    async def test_preprocess_panel(self, user: User, tmp_path):
        from hrfunc.gui.components import preprocess_panel

        state = AppState()
        state.manifest = _manifest(tmp_path)
        a = state.manifest.scans[0]

        @ui.page("/_tt_pre")
        def _p() -> None:
            _render_tab(state, preprocess_panel.render)

        await user.open("/_tt_pre")
        await user.should_see("Select a scan from the dataset tree")
        assert state.selected_scan is None  # no row click

        _tick(user, a.path)
        await user.should_see("Bulk run on 1 checked scan")

    @pytest.mark.asyncio
    async def test_hrf_panel(self, user: User, tmp_path):
        from hrfunc.gui.components import hrf_panel

        state = AppState()
        state.manifest = _manifest(tmp_path)
        a = state.manifest.scans[0]

        @ui.page("/_tt_hrf")
        def _p() -> None:
            _render_tab(state, hrf_panel.render)

        await user.open("/_tt_hrf")
        assert state.selected_scan is None

        _tick(user, a.path)
        await user.should_see("Bulk run on 1 checked scan")

    @pytest.mark.asyncio
    async def test_activity_panel(self, user: User, tmp_path):
        from hrfunc.gui.components import activity_panel

        state = AppState()
        state.manifest = _manifest(tmp_path)
        a = state.manifest.scans[0]

        @ui.page("/_tt_act")
        def _p() -> None:
            _render_tab(state, activity_panel.render)

        await user.open("/_tt_act")
        assert state.selected_scan is None

        _tick(user, a.path)
        await user.should_see("Bulk run on 1 checked scan")
