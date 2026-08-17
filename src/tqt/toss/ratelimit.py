"""Client-side token-bucket rate limiting, one bucket per Toss rate-limit group.

Toss limits requests per *client x API group*, in requests per second. Getting
429s is not merely wasteful: during the 09:00-09:10 KST open, `ORDER_INFO` drops
from 6/s to 3/s, exactly when a bot is most likely to be hammering it. So we
throttle proactively rather than reacting to 429s.

Two adaptive behaviours:

* ``observe_headers`` narrows a bucket when the server advertises a lower
  ``X-RateLimit-Limit`` than we assumed. The docs warn limits can change with no
  notice, so the hardcoded table below is a starting guess, not gospel.
* ``penalize`` drains a bucket after a 429 so the next call actually waits.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from datetime import time as dtime

KST = timezone(timedelta(hours=9), name="KST")


@dataclass(frozen=True)
class Group:
    """A documented rate-limit group.

    ``peak_tps`` applies inside ``peak_window`` (KST wall-clock), used for the
    market-open throttle on ORDER_INFO.
    """

    name: str
    tps: float
    peak_tps: float | None = None
    peak_window: tuple[dtime, dtime] | None = None

    def limit_at(self, now_kst: datetime) -> float:
        if self.peak_tps is None or self.peak_window is None:
            return self.tps
        start, end = self.peak_window
        return self.peak_tps if start <= now_kst.time() <= end else self.tps


_OPEN_WINDOW = (dtime(9, 0), dtime(9, 10))

# Straight from the published Rate Limits table (requests/second).
GROUPS: dict[str, Group] = {
    "AUTH": Group("AUTH", 5),
    "ACCOUNT": Group("ACCOUNT", 1),
    "ASSET": Group("ASSET", 5),
    "STOCK": Group("STOCK", 5),
    "STOCK_ALL": Group("STOCK_ALL", 1),
    "STOCK_TRADING_TREND": Group("STOCK_TRADING_TREND", 10),
    "MARKET_INFO": Group("MARKET_INFO", 3),
    "MARKET_DATA": Group("MARKET_DATA", 15),
    "MARKET_DATA_CHART": Group("MARKET_DATA_CHART", 20),
    "RANKING": Group("RANKING", 5),
    "MARKET_INDICATOR_PRICE": Group("MARKET_INDICATOR_PRICE", 10),
    "MARKET_INDICATOR": Group("MARKET_INDICATOR", 10),
    "MARKET_INDICATOR_CHART": Group("MARKET_INDICATOR_CHART", 5),
    "ORDER": Group("ORDER", 10, peak_tps=10, peak_window=_OPEN_WINDOW),
    "ORDER_HISTORY": Group("ORDER_HISTORY", 5),
    "ORDER_INFO": Group("ORDER_INFO", 6, peak_tps=3, peak_window=_OPEN_WINDOW),
    "CONDITIONAL_ORDER": Group("CONDITIONAL_ORDER", 5),
    "CONDITIONAL_ORDER_HISTORY": Group("CONDITIONAL_ORDER_HISTORY", 10),
}

# Keep a safety margin: burn at 85% of the documented rate so clock skew and
# in-flight concurrency don't tip us over the edge.
SAFETY_FACTOR = 0.85


@dataclass
class _Bucket:
    group: Group
    tokens: float
    updated: float
    server_limit: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class RateLimiter:
    """Thread-safe token buckets keyed by group name."""

    def __init__(self, *, safety_factor: float = SAFETY_FACTOR, clock=time.monotonic) -> None:
        self._clock = clock
        self._safety = safety_factor
        self._buckets: dict[str, _Bucket] = {}
        self._registry_lock = threading.Lock()
        self.waited_seconds = 0.0  # cumulative, for observability

    def _bucket(self, group_name: str) -> _Bucket:
        with self._registry_lock:
            b = self._buckets.get(group_name)
            if b is None:
                group = GROUPS.get(group_name) or Group(group_name, 1)
                b = _Bucket(group=group, tokens=group.tps, updated=self._clock())
                self._buckets[group_name] = b
            return b

    def _limit(self, b: _Bucket) -> float:
        """The limit in force right now, after peak window + server hints."""
        documented = b.group.limit_at(datetime.now(KST))
        return min(documented, b.server_limit) if b.server_limit else documented

    def _rate(self, b: _Bucket) -> float:
        """Refill rate in tokens/second — the documented limit, minus safety margin."""
        return max(self._limit(b) * self._safety, 0.05)

    def _capacity(self, b: _Bucket) -> float:
        """Maximum tokens the bucket holds (the burst allowance).

        Must never fall below 1.0: a 1 req/s group would otherwise have a
        sub-token ceiling and every single-token acquire would block forever.
        The safety margin belongs on the *refill rate*, not on capacity.
        """
        return max(self._limit(b), 1.0)

    def acquire(self, group_name: str, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available. Returns seconds spent waiting."""
        b = self._bucket(group_name)
        total_wait = 0.0
        while True:
            with b.lock:
                rate = self._rate(b)
                capacity = max(self._capacity(b), tokens)
                now = self._clock()
                elapsed = max(now - b.updated, 0.0)
                b.updated = now
                b.tokens = min(capacity, b.tokens + elapsed * rate)
                if b.tokens >= tokens:
                    b.tokens -= tokens
                    self.waited_seconds += total_wait
                    return total_wait
                deficit = tokens - b.tokens
                sleep_for = max(deficit / rate, 0.005)
            time.sleep(sleep_for)
            total_wait += sleep_for

    def observe_headers(self, group_name: str, headers) -> None:
        """Learn the server's real limit from ``X-RateLimit-Limit``."""
        raw = headers.get("X-RateLimit-Limit") or headers.get("x-ratelimit-limit")
        if not raw:
            return
        try:
            limit = float(raw)
        except (TypeError, ValueError):
            return
        if limit <= 0:
            return
        b = self._bucket(group_name)
        with b.lock:
            b.server_limit = limit
            b.tokens = min(b.tokens, limit * self._safety)

    def penalize(self, group_name: str, retry_after: float | None = None) -> None:
        """Empty the bucket after a 429 so the next acquire() genuinely waits."""
        b = self._bucket(group_name)
        with b.lock:
            b.tokens = 0.0
            if retry_after and retry_after > 0:
                # Push `updated` into the future: refill won't start until then.
                b.updated = self._clock() + retry_after

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Current bucket state, for the dashboard."""
        out: dict[str, dict[str, float]] = {}
        for name, b in list(self._buckets.items()):
            with b.lock:
                out[name] = {
                    "tokens": round(b.tokens, 3),
                    "effective_tps": round(self._rate(b), 3),
                    "documented_tps": b.group.tps,
                    "server_limit": b.server_limit or 0.0,
                }
        return out
