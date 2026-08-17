"""Paper broker: simulated fills against **real, live** Toss market data.

This is stage 2 of the plan, and its value depends entirely on being honest. So:

* Fills use the live order book — a BUY crosses the real best ask, a SELL hits the
  real best bid. That captures the actual spread you would pay, which for a thin
  KR ETF is a genuine cost, not a rounding error.
* Commission, transaction tax and FX spread are charged at the same rates the
  live account pays (read from the Toss commissions API).
* Whole shares for KR, fractional allowed for US, matching what Toss accepts.
* The ledger lives in the same SQLite tables as live trading, tagged
  ``broker='paper'``, so the dashboard, reports and metrics code are shared.

What it still cannot capture: queue position on a limit order, partial fills
across a day, and the market impact of your own size. For a 1,000,000 KRW account
in liquid ETFs those are negligible; if you scale to where they aren't, revisit
this.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Any

from ..backtest.costs import CostModel
from ..data.store import Store, d, s, utcnow
from ..toss.client import TossClient
from ..universe import country_of
from .broker import AccountSnapshot, Broker, OrderIntent, OrderResult, Position

log = logging.getLogger(__name__)

CASH_KEY = "KRW"


class PaperBroker(Broker):
    name = "paper"

    def __init__(
        self,
        client: TossClient,
        store: Store,
        *,
        cost_model: CostModel | None = None,
        starting_cash_krw: Decimal = Decimal(1_000_000),
        etf_symbols: frozenset[str] | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.cost_model = cost_model or CostModel()
        self.etf_symbols = etf_symbols or frozenset()
        self._ensure_cash(starting_cash_krw)

    # ------------------------------------------------------------------
    def _ensure_cash(self, starting: Decimal) -> None:
        row = self.store.conn.execute(
            "SELECT amount FROM cash WHERE broker=? AND currency=?", (self.name, CASH_KEY)
        ).fetchone()
        if row is None:
            self.store.conn.execute(
                "INSERT INTO cash(broker,currency,amount) VALUES(?,?,?)",
                (self.name, CASH_KEY, s(starting)),
            )
            self.store.log_event(
                "INFO", "paper_init", f"paper account funded with {starting:,.0f} KRW"
            )

    @property
    def cash(self) -> Decimal:
        row = self.store.conn.execute(
            "SELECT amount FROM cash WHERE broker=? AND currency=?", (self.name, CASH_KEY)
        ).fetchone()
        return d(row["amount"]) if row else Decimal(0)

    def _set_cash(self, amount: Decimal) -> None:
        self.store.conn.execute(
            "UPDATE cash SET amount=? WHERE broker=? AND currency=?",
            (s(amount), self.name, CASH_KEY),
        )

    def reset(self, starting_cash_krw: Decimal) -> None:
        """Wipe the paper account. Never touches live rows."""
        with self.store.transaction():
            for table in ("orders", "fills", "positions", "equity"):
                self.store.conn.execute(f"DELETE FROM {table} WHERE broker=?", (self.name,))  # noqa: S608
            self.store.conn.execute("DELETE FROM cash WHERE broker=?", (self.name,))
        self._ensure_cash(starting_cash_krw)
        log.info("paper account reset to %s KRW", starting_cash_krw)

    # ------------------------------------------------------------------
    def positions(self) -> dict[str, Position]:
        rows = self.store.conn.execute(
            "SELECT * FROM positions WHERE broker=? AND CAST(quantity AS REAL) > 0", (self.name,)
        ).fetchall()
        return {
            r["symbol"]: Position(
                symbol=r["symbol"],
                quantity=d(r["quantity"]),
                avg_price=d(r["avg_price"]),
                currency=r["currency"],
            )
            for r in rows
        }

    def _write_position(self, pos: Position, strategy: str = "") -> None:
        if pos.quantity <= 0:
            self.store.conn.execute(
                "DELETE FROM positions WHERE broker=? AND symbol=?", (self.name, pos.symbol)
            )
            return
        self.store.conn.execute(
            """INSERT INTO positions(broker,symbol,quantity,avg_price,currency,strategy,
                                     opened_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(broker,symbol) DO UPDATE SET
                 quantity=excluded.quantity, avg_price=excluded.avg_price,
                 currency=excluded.currency, updated_at=excluded.updated_at""",
            (
                self.name,
                pos.symbol,
                s(pos.quantity),
                s(pos.avg_price),
                pos.currency,
                strategy,
                utcnow(),
                utcnow(),
            ),
        )

    # ------------------------------------------------------------------
    def price_of(self, symbol: str) -> Decimal | None:
        try:
            return self.client.price(symbol).last_price
        except Exception as exc:
            log.warning("price lookup failed for %s: %s", symbol, exc)
            return None

    def _execution_price(self, symbol: str, side: str) -> tuple[Decimal | None, str]:
        """Cross the real spread; fall back to last trade if the book is empty.

        Returns ``(price, source)``. The source is recorded so a later review can
        tell which fills were spread-accurate and which were approximated.
        """
        try:
            book = self.client.orderbook(symbol)
            px = book.best_ask if side == "BUY" else book.best_bid
            if px:
                return px, "book"
        except Exception as exc:
            log.debug("orderbook unavailable for %s: %s", symbol, exc)
        px = self.price_of(symbol)
        return px, "last" if px else "none"

    def snapshot(self) -> AccountSnapshot:
        positions = self.positions()
        prices: dict[str, Decimal] = {}
        if positions:
            try:
                for p in self.client.prices(list(positions)):
                    prices[p.symbol] = p.last_price
            except Exception as exc:
                log.warning("mark-to-market price fetch failed: %s", exc)
        try:
            fx = self.client.usd_krw()
        except Exception:
            fx = d(self.store.get_state("last_usd_krw", "1400"))
        else:
            self.store.set_state("last_usd_krw", str(fx))

        return AccountSnapshot(
            cash_krw=self.cash, positions=positions, usd_krw=fx, prices=prices
        )

    def open_orders(self) -> list[Any]:
        return []  # paper fills are immediate, so nothing is ever pending

    # ------------------------------------------------------------------
    def submit(self, intent: OrderIntent) -> OrderResult:
        symbol = intent.symbol
        side = intent.side.upper()
        country = country_of(symbol)
        currency = "KRW" if country == "KR" else "USD"

        px, source = self._execution_price(symbol, side)
        if px is None or px <= 0:
            return OrderResult(
                ok=False, symbol=symbol, side=side, error="no market price available"
            )

        fx = self.snapshot().usd_krw if currency == "USD" else Decimal(1)

        # Resolve quantity, honouring KRX's whole-share rule.
        qty = intent.quantity
        if qty is None and intent.order_amount is not None:
            qty = intent.order_amount / px
        if qty is None:
            return OrderResult(ok=False, symbol=symbol, side=side, error="no quantity")
        if country == "KR":
            qty = Decimal(int(qty))
        if qty <= 0:
            return OrderResult(
                ok=False, symbol=symbol, side=side, error="quantity rounds to zero"
            )

        positions = self.positions()
        held = positions.get(symbol)

        if side == "SELL":
            available = held.quantity if held else Decimal(0)
            qty = min(qty, available)
            if qty <= 0:
                return OrderResult(
                    ok=False, symbol=symbol, side=side, error="no position to sell"
                )

        cost = self.cost_model.cost_of(
            side=side,
            country=country,
            price=px,
            quantity=qty,
            is_etf=symbol in self.etf_symbols,
            include_slippage=False,  # crossing the real spread already is the slippage
        )
        gross_local = px * qty
        gross_krw = gross_local * fx
        cost_krw = cost.total * fx

        cash = self.cash
        if side == "BUY":
            needed = gross_krw + cost_krw
            if needed > cash:
                return OrderResult(
                    ok=False,
                    symbol=symbol,
                    side=side,
                    error=f"insufficient paper cash: need {needed:,.0f} have {cash:,.0f}",
                )
            new_cash = cash - needed
            prev_qty = held.quantity if held else Decimal(0)
            prev_avg = held.avg_price if held else Decimal(0)
            new_qty = prev_qty + qty
            new_avg = ((prev_avg * prev_qty) + (px * qty)) / new_qty
        else:
            new_cash = cash + gross_krw - cost_krw
            new_qty = held.quantity - qty
            new_avg = held.avg_price  # realised P&L is derived from fills, not avg cost

        order_id = f"paper-{uuid.uuid4().hex[:16]}"
        with self.store.transaction():
            self._set_cash(new_cash)
            self._write_position(
                Position(symbol, new_qty, new_avg, currency), strategy=intent.strategy
            )
            self.store.conn.execute(
                """INSERT INTO orders(order_id,broker,client_order_id,strategy,symbol,side,
                       order_type,quantity,price,currency,status,filled_quantity,
                       avg_fill_price,commission,tax,submitted_at,updated_at,reason)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    order_id,
                    self.name,
                    order_id,
                    intent.strategy,
                    symbol,
                    side,
                    intent.order_type,
                    s(qty),
                    s(px),
                    currency,
                    "FILLED",
                    s(qty),
                    s(px),
                    s(cost.commission),
                    s(cost.tax),
                    utcnow(),
                    utcnow(),
                    f"{intent.reason} [px={source}]",
                ),
            )
            self.store.conn.execute(
                """INSERT INTO fills(broker,order_id,symbol,side,quantity,price,commission,tax,
                                     currency,filled_at,strategy)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    self.name,
                    order_id,
                    symbol,
                    side,
                    s(qty),
                    s(px),
                    s(cost.commission),
                    s(cost.tax + cost.regulatory + cost.fx),
                    currency,
                    utcnow(),
                    intent.strategy,
                ),
            )

        log.info(
            "PAPER %s %s x%s @ %s (%s) cost=%s",
            side,
            symbol,
            qty,
            px,
            source,
            round(cost.total, 2),
        )
        return OrderResult(
            ok=True,
            symbol=symbol,
            side=side,
            order_id=order_id,
            filled_quantity=qty,
            fill_price=px,
            commission=cost.commission,
            tax=cost.tax + cost.regulatory + cost.fx,
            currency=currency,
            status="FILLED",
            reason=intent.reason,
        )
