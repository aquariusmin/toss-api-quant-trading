"""Telegram: notifications plus two-way remote control.

Telegram is the recommended primary channel because it does the one thing a
trading bot's operator actually needs and Discord webhooks cannot: accept
commands back. Being able to send ``/halt`` from your phone, on a train, is the
difference between watching a problem and stopping it.

Security model: commands are only honoured from the configured chat id.
``TELEGRAM_CHAT_ID`` therefore acts as the authorisation boundary — anyone else
who finds the bot gets ignored, silently. The bot token is a bearer credential
for that bot; treat it like a password.

Implemented with plain ``httpx`` long-polling rather than a framework, to keep
the dependency footprint small enough to be comfortable on a Raspberry Pi.
"""

from __future__ import annotations

import html
import logging
import threading
from collections.abc import Callable

import httpx

from .base import Level, Notifier

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
MAX_MESSAGE = 4000  # Telegram's hard limit is 4096

CommandHandler = Callable[[list[str]], str]


class TelegramNotifier(Notifier):
    name = "telegram"

    def __init__(self, token: str, chat_id: str, *, timeout: float = 10.0) -> None:
        self.token = token
        self.chat_id = str(chat_id)
        self.http = httpx.Client(base_url=f"{API}/bot{token}", timeout=timeout)
        self._offset: int | None = None

        # A very easy mistake is pasting the *bot's* @username here instead of
        # the numeric chat id. Telegram then answers "chat not found" on every
        # send, and — worse — the command allowlist can never match, so /halt
        # silently stops working. Only a public channel legitimately uses a
        # name, and those start with '@'.
        stripped = self.chat_id.lstrip("-")
        if not stripped.isdigit() and not self.chat_id.startswith("@"):
            log.warning(
                "TELEGRAM_CHAT_ID=%r is neither numeric nor an @channel. It is "
                "probably the bot's username. Message the bot once, then run "
                "`tqt telegram-id --write` to fix it.",
                self.chat_id,
            )

    def close(self) -> None:
        self.http.close()

    # ------------------------------------------------------------------
    def send(self, text: str, *, level: Level = Level.INFO) -> bool:
        body = f"{level.emoji} {html.escape(text)}"
        for chunk in _chunks(body, MAX_MESSAGE):
            resp = self.http.post(
                "/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            if resp.status_code != 200:
                log.warning("telegram send failed %s: %s", resp.status_code, resp.text[:200])
                return False
        return True

    def send_pre(self, title: str, block: str, *, level: Level = Level.INFO) -> bool:
        """Send a monospace block — for tables that must stay aligned."""
        text = f"{level.emoji} <b>{html.escape(title)}</b>\n<pre>{html.escape(block)}</pre>"
        resp = self.http.post(
            "/sendMessage",
            json={"chat_id": self.chat_id, "text": text[:MAX_MESSAGE], "parse_mode": "HTML"},
        )
        return resp.status_code == 200

    # ------------------------------------------------------------------
    def poll_once(self, handlers: dict[str, CommandHandler], *, timeout: int = 25) -> int | None:
        """Long-poll for commands.

        Returns the number of commands handled, or ``None`` if the poll itself
        failed — the caller must back off on ``None``. A failing request returns
        *immediately* (a 401 does not wait out the long-poll timeout), so treating
        failure as "0 handled" would spin the loop at full speed against
        Telegram's API.
        """
        params = {"timeout": timeout}
        if self._offset is not None:
            params["offset"] = self._offset
        try:
            resp = self.http.get("/getUpdates", params=params, timeout=timeout + 10)
        except httpx.HTTPError as exc:
            log.debug("telegram poll error: %s", exc)
            return None
        if resp.status_code != 200:
            log.debug("telegram getUpdates %s: %s", resp.status_code, resp.text[:200])
            return None

        handled = 0
        for update in resp.json().get("result", []):
            self._offset = update["update_id"] + 1
            msg = update.get("message") or update.get("edited_message") or {}
            chat = str((msg.get("chat") or {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            # Authorisation: only the configured chat may issue commands.
            if chat != self.chat_id:
                log.warning("ignoring command from unauthorised chat %s: %r", chat, text[:40])
                continue

            parts = text.split()
            cmd = parts[0].lstrip("/").split("@")[0].lower()
            args = parts[1:]

            handler = handlers.get(cmd)
            if handler is None:
                self.send(
                    f"알 수 없는 명령: /{cmd}\n사용 가능: "
                    + ", ".join(f"/{k}" for k in sorted(handlers))
                )
                continue
            try:
                reply = handler(args)
            except Exception as exc:
                log.exception("command /%s failed", cmd)
                reply = f"명령 처리 중 오류: {type(exc).__name__}: {exc}"
            if reply:
                self.send_pre(f"/{cmd}", reply) if "\n" in reply else self.send(reply)
            handled += 1
        return handled

    def run_command_loop(
        self, handlers: dict[str, CommandHandler], stop_event: threading.Event
    ) -> None:
        """Poll until ``stop_event`` is set. Intended for a daemon thread.

        Backs off exponentially while polling keeps failing. Without this, a
        revoked bot token (BotFather's /revoke invalidates it instantly) turns
        this into a hot loop hammering the Telegram API — which matters on a
        Raspberry Pi, and is exactly the state the bot lands in when a token is
        rotated out from under a running process.
        """
        log.info("telegram command loop started (%d commands)", len(handlers))
        failures = 0
        while not stop_event.is_set():
            try:
                result = self.poll_once(handlers)
            except Exception as exc:  # never let the loop die
                log.warning("telegram loop error: %s", exc)
                result = None

            if result is None:
                failures += 1
                delay = min(2 ** min(failures, 6), 60)
                if failures in (1, 5, 20) or failures % 50 == 0:
                    log.warning(
                        "telegram polling has failed %d time(s); retrying in %ds. "
                        "If this persists, the bot token is probably invalid — "
                        "check BotFather and run `tqt notify-test`.",
                        failures,
                        delay,
                    )
                stop_event.wait(delay)
            else:
                if failures:
                    log.info("telegram polling recovered after %d failure(s)", failures)
                failures = 0
        log.info("telegram command loop stopped")

    def start_command_thread(
        self, handlers: dict[str, CommandHandler]
    ) -> tuple[threading.Thread, threading.Event]:
        stop = threading.Event()
        t = threading.Thread(
            target=self.run_command_loop, args=(handlers, stop), daemon=True, name="telegram"
        )
        t.start()
        return t, stop

    # ------------------------------------------------------------------
    def whoami(self) -> dict:
        """Verify the token and report the bot identity — used by `tqt notify test`."""
        resp = self.http.get("/getMe")
        return resp.json()


def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > size:
            out.append(cur)
            cur = line
        else:
            cur += line
    if cur:
        out.append(cur)
    return out
