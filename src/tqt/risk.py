"""Risk limits — the layer that decides a strategy's wish is not going to happen.

Written on the assumption that the strategy, the data feed and my own code will
each eventually be wrong. Every check here exists because of a specific way
automated trading destroys accounts:

``halted``           A manual or automatic stop. Checked first, always.
``daily loss``       Caps how much a single bad day (or a bug placing 400 orders)
                     can cost before everything stops until tomorrow.
``order budget``     A runaway loop is the classic way to turn a small edge into a
                     large commission bill. Hard cap per day.
``position weight``  Concentration limit, so one wrong signal can't become the
                     whole account.
``gross exposure``   Never above 100% — this bot does not borrow (미수 없음).
``stock warnings``   Refuses to *buy* anything in 정리매매, 투자경고 or 단기과열.
                     Toss exposes this; not checking it is how a bot ends up
                     accumulating a stock on its way to delisting.

The kill switch is persisted in SQLite, not held in memory, so it survives a
restart. A bot that forgets it was halted when the Pi reboots is not halted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

from .config import Settings
from .data.store import Store
from .execution.broker import AccountSnapshot, OrderIntent
from .universe import country_of

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9), name="KST")

STATE_HALTED = "halted"
STATE_HALT_REASON = "halt_reason"
STATE_HALTED_DATE = "halted_for_date"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOW = Verdict(True)


def today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


class RiskManager:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        broker_name: str,
        client=None,
        defensive_symbols: frozenset[str] | None = None,
        max_defensive_weight: Decimal = Decimal("0.60"),
    ) -> None:
        self.settings = settings
        self.store = store
        self.broker_name = broker_name
        self.client = client
        # Bond / cash-proxy ETFs that strategies retreat into. The concentration
        # limit exists to stop one *bet* becoming the account; parking 25% in a
        # 3-year government bond ETF is not that bet. Without this exemption the
        # limit silently caps how defensive the portfolio can get: a Faber sleeve
        # with 4 of 8 assets out of trend wants 25% in 국고채, gets refused, and
        # leaves the money in cash instead — quietly changing the strategy.
        self.defensive_symbols = defensive_symbols or frozenset()
        self.max_defensive_weight = max_defensive_weight
        self._warning_cache: dict[str, list[str]] = {}

    def weight_limit_for(self, symbol: str) -> Decimal:
        if symbol in self.defensive_symbols:
            return max(self.max_defensive_weight, self.settings.max_position_weight)
        return self.settings.max_position_weight

    # ------------------------------------------------------------------
    # kill switch
    # ------------------------------------------------------------------
    @property
    def is_halted(self) -> bool:
        if bool(self.store.get_state(STATE_HALTED, False)):
            return True
        # A daily-loss halt expires at the next KST date rollover.
        halted_for = self.store.get_state(STATE_HALTED_DATE)
        return bool(halted_for) and halted_for == today_kst()

    @property
    def halt_reason(self) -> str:
        return str(self.store.get_state(STATE_HALT_REASON, "") or "")

    def halt(self, reason: str, *, for_today_only: bool = False) -> None:
        if for_today_only:
            self.store.set_state(STATE_HALTED_DATE, today_kst())
        else:
            self.store.set_state(STATE_HALTED, True)
        self.store.set_state(STATE_HALT_REASON, reason)
        self.store.log_event("WARN", "halt", reason, {"for_today_only": for_today_only})
        log.warning("TRADING HALTED: %s", reason)

    def resume(self) -> None:
        self.store.set_state(STATE_HALTED, False)
        self.store.set_state(STATE_HALTED_DATE, "")
        self.store.set_state(STATE_HALT_REASON, "")
        self.store.log_event("INFO", "resume", "trading resumed by operator")
        log.info("trading resumed")

    # ------------------------------------------------------------------
    # portfolio-level gates, evaluated once per cycle
    # ------------------------------------------------------------------
    def check_cycle(self, snapshot: AccountSnapshot) -> Verdict:
        if self.is_halted:
            return Verdict(False, f"halted: {self.halt_reason or 'no reason recorded'}")

        loss = self.daily_loss(snapshot)
        if loss is not None and loss <= -self.settings.max_daily_loss_pct:
            reason = (
                f"daily loss {loss:.2%} breached limit "
                f"-{self.settings.max_daily_loss_pct:.2%}"
            )
            self.halt(reason, for_today_only=True)
            return Verdict(False, reason)

        used = self.orders_today()
        if used >= self.settings.max_orders_per_day:
            reason = f"order budget exhausted ({used}/{self.settings.max_orders_per_day})"
            self.halt(reason, for_today_only=True)
            return Verdict(False, reason)

        return ALLOW

    def daily_loss(self, snapshot: AccountSnapshot) -> Decimal | None:
        """Today's return vs the first equity mark of the day, or None if unknown."""
        opening = self.store.first_equity_of_day(self.broker_name, today_kst())
        if opening is None or opening <= 0:
            return None
        return (snapshot.equity_krw - opening) / opening

    def orders_today(self) -> int:
        row = self.store.conn.execute(
            "SELECT COUNT(*) n FROM orders WHERE broker=? AND substr(submitted_at,1,10)=?",
            (self.broker_name, datetime.now(UTC).strftime("%Y-%m-%d")),
        ).fetchone()
        return int(row["n"] or 0)

    def remaining_order_budget(self) -> int:
        return max(self.settings.max_orders_per_day - self.orders_today(), 0)

    # ------------------------------------------------------------------
    # per-order gates
    # ------------------------------------------------------------------
    def check_order(
        self, intent: OrderIntent, snapshot: AccountSnapshot, *, price: Decimal | None = None
    ) -> Verdict:
        equity = snapshot.equity_krw
        if equity <= 0:
            return Verdict(False, "account equity is zero")

        if self.remaining_order_budget() <= 0:
            return Verdict(False, "daily order budget exhausted")

        side = intent.side.upper()
        px = price or Decimal(0)
        qty = intent.quantity or Decimal(0)
        fx = snapshot.usd_krw if country_of(intent.symbol) == "US" else Decimal(1)
        notional_krw = px * qty * fx
        if intent.order_amount is not None:
            notional_krw = intent.order_amount * fx

        if side == "BUY":
            # Concentration: existing exposure plus what we're about to add.
            current = Decimal(0)
            pos = snapshot.positions.get(intent.symbol)
            if pos is not None:
                mark = snapshot.prices.get(intent.symbol, pos.avg_price)
                current = mark * pos.quantity * fx
            projected_weight = (current + notional_krw) / equity
            limit = self.weight_limit_for(intent.symbol)
            if projected_weight > limit:
                return Verdict(
                    False,
                    f"{intent.symbol}: weight would reach {projected_weight:.1%} "
                    f"> limit {limit:.1%}",
                )

            gross = (snapshot.position_value_krw() + notional_krw) / equity
            if gross > self.settings.max_gross_exposure:
                return Verdict(
                    False,
                    f"gross exposure would reach {gross:.1%} "
                    f"> limit {self.settings.max_gross_exposure:.1%}",
                )

            if notional_krw > snapshot.cash_krw:
                return Verdict(
                    False,
                    f"{intent.symbol}: needs {notional_krw:,.0f} KRW, "
                    f"cash is {snapshot.cash_krw:,.0f} (no margin)",
                )

            blocked = self.buy_warnings(intent.symbol)
            if blocked:
                return Verdict(False, f"{intent.symbol}: 매수 유의종목 {', '.join(blocked)}")

        elif side == "SELL":
            pos = snapshot.positions.get(intent.symbol)
            if pos is None or pos.quantity <= 0:
                return Verdict(False, f"{intent.symbol}: nothing held to sell")
            if qty > pos.quantity:
                return Verdict(
                    False, f"{intent.symbol}: sell {qty} exceeds holding {pos.quantity}"
                )

        return ALLOW

    def buy_warnings(self, symbol: str) -> list[str]:
        """Disqualifying Toss warnings for a buy, cached per cycle."""
        if self.client is None:
            return []
        if symbol in self._warning_cache:
            return self._warning_cache[symbol]
        try:
            reasons = self.client.buy_block_reasons(symbol)
        except Exception as exc:
            log.warning("warning check failed for %s: %s", symbol, exc)
            reasons = []
        self._warning_cache[symbol] = reasons
        return reasons

    def clear_cache(self) -> None:
        self._warning_cache.clear()

    # ------------------------------------------------------------------
    def status(self, snapshot: AccountSnapshot | None = None) -> dict:
        out = {
            "halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "orders_today": self.orders_today(),
            "order_budget": self.settings.max_orders_per_day,
            "max_position_weight": float(self.settings.max_position_weight),
            "max_daily_loss_pct": float(self.settings.max_daily_loss_pct),
        }
        if snapshot is not None:
            loss = self.daily_loss(snapshot)
            out["daily_pnl_pct"] = float(loss) if loss is not None else None
            out["gross_exposure"] = (
                float(snapshot.position_value_krw() / snapshot.equity_krw)
                if snapshot.equity_krw > 0
                else 0.0
            )
        return out
