"""Notifier behaviour.

The backoff test covers a failure mode that is easy to ship and unpleasant to
run into: a bot token revoked while the process is live. BotFather's /revoke
invalidates a token instantly, so a long-running bot suddenly gets 401 on every
poll — and a 401 comes back immediately rather than waiting out the long-poll
timeout, so a naive loop spins at full speed against Telegram's API.
"""

from __future__ import annotations

import threading

import httpx
import pytest
import respx

from tqt.notify.base import Level, NotifierGroup, NullNotifier, build_notifier
from tqt.notify.telegram import TelegramNotifier

TOKEN = "8956560738:AAtest-token-value-goes-here-0000000"
API = f"https://api.telegram.org/bot{TOKEN}"


@respx.mock
def test_poll_once_returns_none_on_auth_failure():
    """None means "the poll failed" — distinct from "no commands arrived"."""
    respx.get(f"{API}/getUpdates").mock(
        return_value=httpx.Response(
            401, json={"ok": False, "error_code": 401, "description": "Unauthorized"}
        )
    )
    n = TelegramNotifier(TOKEN, "123456")
    assert n.poll_once({}) is None


@respx.mock
def test_poll_once_returns_zero_when_idle():
    respx.get(f"{API}/getUpdates").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": []})
    )
    n = TelegramNotifier(TOKEN, "123456")
    assert n.poll_once({}) == 0


@respx.mock
def test_command_loop_backs_off_instead_of_spinning_on_a_revoked_token():
    """With a dead token the loop must sleep between attempts, not hot-loop.

    Asserted via the wait intervals requested from the stop event: they have to
    grow, and none may be zero.
    """
    respx.get(f"{API}/getUpdates").mock(
        return_value=httpx.Response(
            401, json={"ok": False, "error_code": 401, "description": "Unauthorized"}
        )
    )
    n = TelegramNotifier(TOKEN, "123456")

    waits: list[float] = []

    class RecordingEvent(threading.Event):
        def wait(self, timeout=None):  # type: ignore[override]
            waits.append(timeout)
            if len(waits) >= 5:
                self.set()
            return self.is_set()

    n.run_command_loop({}, RecordingEvent())

    assert len(waits) >= 5
    assert all(w and w > 0 for w in waits), f"loop spun without sleeping: {waits}"
    assert waits[-1] > waits[0], f"backoff did not grow: {waits}"
    assert max(waits) <= 60


@respx.mock
def test_backoff_resets_after_recovery():
    route = respx.get(f"{API}/getUpdates")
    route.side_effect = [
        httpx.Response(401, json={"ok": False}),
        httpx.Response(401, json={"ok": False}),
        httpx.Response(200, json={"ok": True, "result": []}),
        httpx.Response(401, json={"ok": False}),
    ]
    n = TelegramNotifier(TOKEN, "123456")
    waits: list[float] = []

    class RecordingEvent(threading.Event):
        def wait(self, timeout=None):  # type: ignore[override]
            waits.append(timeout)
            if len(waits) >= 3:
                self.set()
            return self.is_set()

    n.run_command_loop({}, RecordingEvent())
    # Two failures, then a success, then a failure: the last wait must be back
    # down to the first-failure delay rather than continuing to grow.
    assert waits[-1] == waits[0], f"backoff failed to reset: {waits}"


@respx.mock
def test_commands_from_an_unauthorised_chat_are_ignored():
    """TELEGRAM_CHAT_ID is the authorisation boundary for /halt and friends."""
    called: list[str] = []
    respx.get(f"{API}/getUpdates").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "result": [
                    {
                        "update_id": 1,
                        "message": {"chat": {"id": 999999}, "text": "/halt"},
                    }
                ],
            },
        )
    )
    respx.post(f"{API}/sendMessage").mock(return_value=httpx.Response(200, json={"ok": True}))

    n = TelegramNotifier(TOKEN, "123456")
    handled = n.poll_once({"halt": lambda a: called.append("halt") or "halted"})
    assert handled == 0
    assert called == [], "a stranger must not be able to halt trading"


@respx.mock
def test_authorised_command_is_dispatched():
    respx.get(f"{API}/getUpdates").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "result": [
                    {"update_id": 7, "message": {"chat": {"id": 123456}, "text": "/status now"}}
                ],
            },
        )
    )
    respx.post(f"{API}/sendMessage").mock(return_value=httpx.Response(200, json={"ok": True}))

    seen: list[list[str]] = []
    n = TelegramNotifier(TOKEN, "123456")
    handled = n.poll_once({"status": lambda args: seen.append(args) or "ok"})
    assert handled == 1
    assert seen == [["now"]]


def test_non_numeric_chat_id_warns(caplog):
    """Pasting the bot's @username here is a silent killer: sends fail and the
    command allowlist can never match, so /halt looks configured but is dead."""
    with caplog.at_level("WARNING"):
        TelegramNotifier(TOKEN, "toss_quanttrading_bot")
    assert any("neither numeric nor an @channel" in r.message for r in caplog.records)


@pytest.mark.parametrize("chat_id", ["123456789", "-1001234567890", "@somechannel"])
def test_valid_chat_ids_do_not_warn(caplog, chat_id):
    with caplog.at_level("WARNING"):
        TelegramNotifier(TOKEN, chat_id)
    assert not any("neither numeric" in r.message for r in caplog.records)


def test_notifier_group_survives_one_channel_failing():
    """A Telegram outage must never take down trading."""

    class Broken(NullNotifier):
        name = "broken"

        def send(self, text, *, level=Level.INFO):
            raise RuntimeError("channel down")

    group = NotifierGroup([Broken(), NullNotifier()])
    assert group.send("hello") is True  # the healthy channel still delivered


def test_build_notifier_with_nothing_configured_returns_empty_group(settings):
    group = build_notifier(settings)
    assert group.channels == []
    assert group.send("no-op") is True
