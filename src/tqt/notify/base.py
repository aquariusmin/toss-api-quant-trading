"""Notification abstraction.

A trading bot you cannot see is a trading bot you should not run. The whole point
of this layer is that silence is never ambiguous: you get a message when it
trades, when it halts, when your IP changes, and a daily summary even on days
nothing happened — so no message at all means the process is dead, which is
itself information.

Notifiers never raise. A Telegram outage must not take down trading, and a failed
send is logged rather than propagated.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum

log = logging.getLogger(__name__)


class Level(str, Enum):
    INFO = "INFO"
    TRADE = "TRADE"
    WARN = "WARN"
    ERROR = "ERROR"

    @property
    def emoji(self) -> str:
        return {
            Level.INFO: "ℹ️",
            Level.TRADE: "💱",
            Level.WARN: "⚠️",
            Level.ERROR: "🚨",
        }[self]


class Notifier(ABC):
    name = "abstract"

    @abstractmethod
    def send(self, text: str, *, level: Level = Level.INFO) -> bool:
        ...

    def send_safe(self, text: str, *, level: Level = Level.INFO) -> bool:
        try:
            return self.send(text, level=level)
        except Exception as exc:
            log.warning("%s notification failed: %s", self.name, exc)
            return False


class NullNotifier(Notifier):
    name = "null"

    def send(self, text: str, *, level: Level = Level.INFO) -> bool:
        log.info("[notify:%s] %s", level.value, text)
        return True


class NotifierGroup(Notifier):
    """Fan out to every configured channel."""

    name = "group"

    def __init__(self, notifiers: list[Notifier]) -> None:
        self.notifiers = [n for n in notifiers if n is not None]

    def send(self, text: str, *, level: Level = Level.INFO) -> bool:
        if not self.notifiers:
            log.info("[notify:%s] %s", level.value, text)
            return True
        return any(n.send_safe(text, level=level) for n in self.notifiers)

    @property
    def channels(self) -> list[str]:
        return [n.name for n in self.notifiers]


def build_notifier(settings) -> NotifierGroup:
    """Assemble notifiers from configuration."""
    channels: list[Notifier] = []

    if settings.telegram_bot_token and settings.telegram_chat_id:
        from .telegram import TelegramNotifier

        channels.append(
            TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        )
    if settings.discord_webhook_url:
        from .discord import DiscordNotifier

        channels.append(DiscordNotifier(settings.discord_webhook_url))

    if not channels:
        log.warning(
            "no notification channel configured — set TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
            "or DISCORD_WEBHOOK_URL so you find out when something breaks"
        )
    return NotifierGroup(channels)
