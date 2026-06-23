"""Unit tests for the background-task client-liveness guards.

A long bulk run can outlive its page client (the user navigates away,
reloads, or the native window reconnects). Touching a deleted client via
``ui.notify`` then trips NiceGUI's "Client has been deleted but is still
being used" warning. ``client_alive`` / ``notify_if_alive`` guard against
that. These tests don't need a running server — they exercise the pure
liveness check and the no-op path with stand-in clients.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from hrfunc.gui import workers  # noqa: E402


class _FakeClient:
    def __init__(self, client_id: str):
        self.id = client_id


def test_client_alive_none_is_false():
    assert workers.client_alive(None) is False


def test_client_alive_unknown_id_is_false():
    # A client whose id isn't in nicegui's registry is treated as gone.
    assert workers.client_alive(_FakeClient("not-a-real-client")) is False


def test_notify_if_alive_noop_when_dead(caplog):
    # Must not raise when the client is gone (the bug this guards against).
    workers.notify_if_alive(None, "hello")
    workers.notify_if_alive(_FakeClient("gone"), "hello", type="warning")


def test_client_alive_true_for_registered(monkeypatch):
    # A client present in nicegui's instances registry reads as alive.
    from nicegui import Client

    fake = _FakeClient("live-id")
    monkeypatch.setitem(Client.instances, "live-id", fake)
    try:
        assert workers.client_alive(fake) is True
    finally:
        Client.instances.pop("live-id", None)
