"""Operational plumbing: market clock, IP watchdog, and startup preflight.

The IP watchdog earns its place on a home server. Toss blocks any call from an IP
that isn't on the allowlist, and a residential connection rotates its address
whenever the router reconnects. Without this check the failure looks like the bot
simply stopping — no orders, no errors visible, until you go looking. With it, you
get a Telegram message naming the new IP to paste into
설정 → Open API → 허용 IP 관리.

The market clock uses Toss's own calendar rather than a hardcoded schedule, so
Korean substitute holidays (today, 2026-08-17, is one) and US DST shifts are
handled by the exchange's own answer instead of my guess about it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .config import Settings
from .data.store import Store

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9), name="KST")
STATE_LAST_IP = "last_public_ip"

IP_SERVICES = (
    "https://api.ipify.org",
    "https://checkip.amazonaws.com",
    "https://ifconfig.me/ip",
)


def public_ip(timeout: float = 6.0) -> str | None:
    """This machine's public IP, trying several services."""
    for url in IP_SERVICES:
        try:
            resp = httpx.get(url, timeout=timeout)
            if resp.status_code == 200:
                ip = resp.text.strip()
                if ip and len(ip) <= 45:
                    return ip
        except httpx.HTTPError:
            continue
    return None


@dataclass
class IPStatus:
    current: str | None
    previous: str | None
    changed: bool

    @property
    def message(self) -> str:
        if self.current is None:
            return "공인 IP를 확인할 수 없습니다 (네트워크 문제일 수 있음)"
        if self.changed:
            return (
                f"⚠️ 공인 IP가 {self.previous} → {self.current} 로 변경되었습니다.\n"
                f"토스증권 WTS → 설정 → Open API → 허용 IP 관리에 "
                f"{self.current} 를 등록하세요. 등록 전까지 모든 주문이 403으로 막힙니다."
            )
        return f"공인 IP {self.current} (변경 없음)"


def check_ip(store: Store) -> IPStatus:
    """Compare the current public IP against the last one we saw."""
    current = public_ip()
    previous = store.get_state(STATE_LAST_IP)
    changed = bool(current and previous and current != previous)
    if current:
        store.set_state(STATE_LAST_IP, current)
        if changed:
            store.log_event(
                "WARN",
                "ip_changed",
                f"public IP changed {previous} -> {current}",
                {"previous": previous, "current": current},
            )
    return IPStatus(current=current, previous=previous, changed=changed)


# ---------------------------------------------------------------------------
# market clock
# ---------------------------------------------------------------------------
def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


@dataclass
class SessionWindow:
    name: str
    start: datetime | None
    end: datetime | None

    def contains(self, now: datetime) -> bool:
        return bool(self.start and self.end and self.start <= now <= self.end)


class MarketClock:
    """Answers "can I trade right now?" from Toss's published calendar.

    Calendars are cached per (country, date) — they don't change intraday, and
    ``MARKET_INFO`` is limited to 3 requests/second.
    """

    def __init__(self, client) -> None:
        self.client = client
        self._cache: dict[tuple[str, str], Any] = {}

    def _calendar(self, country: str, date: str | None = None):
        key = (country, date or datetime.now(KST).strftime("%Y-%m-%d"))
        if key not in self._cache:
            self._cache[key] = (
                self.client.market_calendar_kr(date)
                if country == "KR"
                else self.client.market_calendar_us(date)
            )
        return self._cache[key]

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    def regular_session(self, country: str, date: str | None = None) -> SessionWindow:
        cal = self._calendar(country, date)
        day = cal.today
        if country == "KR":
            sessions = day.integrated
            reg = sessions.regular_market if sessions else None
        else:
            reg = day.regular_market
        return SessionWindow(
            "regular",
            _parse(reg.start_time) if reg else None,
            _parse(reg.end_time) if reg else None,
        )

    def is_open(self, country: str, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(KST)
        try:
            return self.regular_session(country).contains(now)
        except Exception as exc:
            log.warning("calendar lookup failed for %s: %s", country, exc)
            return False

    def is_business_day(self, country: str) -> bool:
        try:
            cal = self._calendar(country)
            return cal.today.is_open_day
        except Exception:
            return False

    def minutes_to_close(self, country: str, *, now: datetime | None = None) -> float | None:
        now = now or datetime.now(KST)
        win = self.regular_session(country)
        if not win.end or not win.contains(now):
            return None
        return (win.end - now).total_seconds() / 60.0

    def next_open(self, country: str) -> datetime | None:
        """Start of the next regular session (today's if it hasn't begun yet)."""
        now = datetime.now(KST)
        try:
            today = self.regular_session(country)
            if today.start and now < today.start:
                return today.start
            cal = self._calendar(country)
            nxt = cal.next_business_day
            if country == "KR":
                reg = nxt.integrated.regular_market if nxt.integrated else None
            else:
                reg = nxt.regular_market
            return _parse(reg.start_time) if reg else None
        except Exception as exc:
            log.warning("next_open lookup failed for %s: %s", country, exc)
            return None

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for country in ("KR", "US"):
            try:
                win = self.regular_session(country)
                out[country] = {
                    "open": self.is_open(country),
                    "business_day": self.is_business_day(country),
                    "session_start": win.start.isoformat() if win.start else None,
                    "session_end": win.end.isoformat() if win.end else None,
                    "next_open": (
                        n.isoformat() if (n := self.next_open(country)) else None
                    ),
                }
            except Exception as exc:
                out[country] = {"error": str(exc)}
        return out


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = False


def preflight(settings: Settings, store: Store, client=None) -> list[Check]:
    """Everything that must be true before trading. Powers ``tqt doctor``."""
    checks: list[Check] = []

    checks.append(
        Check(
            "credentials",
            settings.has_credentials,
            "toss_client_id / toss_client_secret present"
            if settings.has_credentials
            else "missing — copy .env.example to .env and fill them in",
            fatal=True,
        )
    )

    ip = check_ip(store)
    checks.append(
        Check(
            "public IP",
            ip.current is not None and not ip.changed,
            ip.message,
        )
    )

    checks.append(
        Check(
            "database",
            store.path.exists(),
            f"{store.path} ({store.path.stat().st_size // 1024}KB)"
            if store.path.exists()
            else "not created yet",
        )
    )

    n_bars = store.conn.execute("SELECT COUNT(*) n FROM candles").fetchone()["n"]
    checks.append(
        Check(
            "market history",
            n_bars > 1000,
            f"{n_bars:,} candles stored"
            if n_bars
            else "empty — run `tqt data sync` before backtesting",
        )
    )

    if settings.mode.value == "live":
        checks.append(
            Check(
                "live confirmation",
                settings.live_confirm == "I_UNDERSTAND_REAL_MONEY",
                "TQT_LIVE_CONFIRM is set",
                fatal=True,
            )
        )

    notify_on = bool(settings.telegram_bot_token) or bool(settings.discord_webhook_url)
    checks.append(
        Check(
            "notifications",
            notify_on,
            "configured"
            if notify_on
            else "none configured — you will not be told if the bot stops",
        )
    )

    checks.append(
        Check(
            "dashboard token",
            bool(settings.dashboard_token),
            "set" if settings.dashboard_token else "unset — dashboard will refuse to start",
        )
    )

    if client is not None:
        try:
            accounts = client.accounts()
            seq = client.account_seq
            checks.append(
                Check("Toss API", True, f"{len(accounts)} account(s), using accountSeq={seq}")
            )
        except Exception as exc:
            checks.append(Check("Toss API", False, f"{type(exc).__name__}: {exc}", fatal=True))

        try:
            clock = MarketClock(client)
            desc = clock.describe()
            checks.append(
                Check(
                    "market hours",
                    True,
                    "KR "
                    + ("OPEN" if desc.get("KR", {}).get("open") else "closed")
                    + " / US "
                    + ("OPEN" if desc.get("US", {}).get("open") else "closed"),
                )
            )
        except Exception as exc:
            checks.append(Check("market hours", False, str(exc)))

        try:
            rates = {c.market_country: str(c.commission_rate) for c in client.commissions()}
            checks.append(Check("commission rates", True, str(rates)))
        except Exception as exc:
            checks.append(Check("commission rates", False, str(exc)))

    return checks
