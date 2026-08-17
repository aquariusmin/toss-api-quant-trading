"""Discord notifications via incoming webhook.

Complements Telegram rather than replacing it: a Discord channel is a much nicer
place to *read back* weeks of trade history and daily reports, since it keeps
searchable scrollback and renders code blocks well. It is one-way, though —
webhooks can post but cannot receive commands — so remote control stays on
Telegram.
"""

from __future__ import annotations

import logging

import httpx

from .base import Level, Notifier

log = logging.getLogger(__name__)

MAX_CONTENT = 1900  # Discord's limit is 2000


class DiscordNotifier(Notifier):
    name = "discord"

    def __init__(self, webhook_url: str, *, timeout: float = 10.0, username: str = "tqt") -> None:
        self.webhook_url = webhook_url
        self.username = username
        self.http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.http.close()

    def send(self, text: str, *, level: Level = Level.INFO) -> bool:
        body = f"{level.emoji} {text}"
        ok = True
        for chunk in _chunks(body, MAX_CONTENT):
            resp = self.http.post(
                self.webhook_url, json={"content": chunk, "username": self.username}
            )
            # Discord returns 204 No Content on success.
            if resp.status_code not in (200, 204):
                log.warning("discord send failed %s: %s", resp.status_code, resp.text[:200])
                ok = False
        return ok

    def send_pre(self, title: str, block: str, *, level: Level = Level.INFO) -> bool:
        content = f"{level.emoji} **{title}**\n```\n{block[:MAX_CONTENT - 100]}\n```"
        resp = self.http.post(
            self.webhook_url, json={"content": content, "username": self.username}
        )
        return resp.status_code in (200, 204)


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
