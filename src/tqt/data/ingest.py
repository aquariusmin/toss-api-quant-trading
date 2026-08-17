"""Candle ingestion: pull history from Toss into the local store.

Backtests must never read the live API — they need a stable, repeatable dataset,
and paginating 16 years of daily bars on every run would be both slow and rude to
the rate limiter. So history is downloaded once and topped up incrementally.

Two measured facts about the Toss candle API shape this module:

* A page holds at most 200 bars, ordered **newest-first**, and you walk backwards
  through the ``nextBefore`` cursor.
* Daily history reaches back to at least 2010, but **1-minute history only covers
  about four days**. Intraday backtesting is therefore not viable on this API,
  which is a good reason for the strategies here to be daily-bar based.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..toss.client import TossClient
from ..toss.errors import TossError, TossNotFoundError
from .store import Store

log = logging.getLogger(__name__)

#: ~16 years of trading days. Enough to span 2011, 2015, 2018, 2020 and 2022
#: drawdowns, which is the point: a momentum strategy that has only seen a bull
#: market has not been tested.
DEFAULT_MAX_BARS = 4200


@dataclass
class SyncResult:
    symbol: str
    interval: str
    written: int
    earliest: str | None
    latest: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Ingestor:
    def __init__(self, client: TossClient, store: Store) -> None:
        self.client = client
        self.store = store

    # ------------------------------------------------------------------
    def sync_symbol(
        self,
        symbol: str,
        interval: str = "1d",
        *,
        max_bars: int = DEFAULT_MAX_BARS,
        full: bool = False,
    ) -> SyncResult:
        """Download history for one symbol, resuming from what we already hold.

        ``full=True`` re-walks the entire window — use it after a corporate action,
        since adjusted prices for *past* bars change when a split happens.
        """
        lo, hi, n = self.store.candle_range(symbol, interval)

        # Incremental top-up: stop paginating once we reach a bar we already have.
        # A one-page overlap is intentional, so a partially-formed last bar gets
        # overwritten with its final values.
        stop_at = None if (full or not hi) else hi

        try:
            batch = list(
                self.client.iter_candles(
                    symbol, interval, max_bars=max_bars, stop_at=stop_at, adjusted=True
                )
            )
        except TossNotFoundError:
            return SyncResult(symbol, interval, 0, lo, hi, error="stock-not-found")
        except TossError as exc:
            return SyncResult(symbol, interval, 0, lo, hi, error=str(exc))

        written = self.store.upsert_candles(symbol, interval, batch)
        lo2, hi2, n2 = self.store.candle_range(symbol, interval)
        log.info(
            "%s %s: +%d bars (held %d -> %d, %s .. %s)",
            symbol,
            interval,
            written,
            n,
            n2,
            (lo2 or "")[:10],
            (hi2 or "")[:10],
        )
        return SyncResult(symbol, interval, written, lo2, hi2)

    def sync(
        self,
        symbols: Sequence[str],
        interval: str = "1d",
        *,
        max_bars: int = DEFAULT_MAX_BARS,
        full: bool = False,
        on_progress: Callable[[int, int, SyncResult], None] | None = None,
    ) -> list[SyncResult]:
        """Sync many symbols. One bad symbol never aborts the run."""
        results: list[SyncResult] = []
        total = len(symbols)
        for i, sym in enumerate(symbols, start=1):
            res = self.sync_symbol(sym, interval, max_bars=max_bars, full=full)
            results.append(res)
            if on_progress:
                on_progress(i, total, res)
        failed = [r for r in results if not r.ok]
        if failed:
            log.warning(
                "%d/%d symbols failed: %s",
                len(failed),
                total,
                ", ".join(f"{r.symbol}({r.error})" for r in failed[:8]),
            )
        return results

    # ------------------------------------------------------------------
    def sync_stock_master(self, symbols: Sequence[str]) -> int:
        """Cache symbol metadata (name, market, currency, tradability)."""
        written = 0
        chunk = 100
        for i in range(0, len(symbols), chunk):
            batch = self.client.stocks(symbols[i : i + chunk])
            written += self.store.upsert_stocks(batch)
        return written

    def coverage(self, symbols: Sequence[str], interval: str = "1d") -> list[dict]:
        """What we hold locally, for `tqt data status`."""
        out = []
        for sym in symbols:
            lo, hi, n = self.store.candle_range(sym, interval)
            row = self.store.get_stock(sym)
            out.append(
                {
                    "symbol": sym,
                    "name": (row["name"] if row else None) or "",
                    "bars": n,
                    "from": (lo or "")[:10],
                    "to": (hi or "")[:10],
                }
            )
        return out
