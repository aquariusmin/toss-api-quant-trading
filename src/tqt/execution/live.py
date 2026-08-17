"""Live broker: real orders, real money.

Everything here is deliberately conservative:

* **Idempotent submission.** Every order carries a ``clientOrderId``, so a network
  timeout on the POST — where the order may already be live at the exchange — can
  be retried without risking a duplicate position.
* **Limit prices are tick-rounded** before submission, biased in the aggressive
  direction, because KRX rejects off-tick prices outright and a bot whose every
  order bounces looks identical to a bot that has decided not to trade.
* **Fills are confirmed, not assumed.** ``submit`` polls the order briefly and
  reports what actually filled. A market order that got 0 shares because the stock
  was halted must not be recorded as a position.
* **No leverage, ever.** Sizing comes from cash buying power, which excludes 미수.

The account state here is read from Toss, not from our own ledger. The broker is
the source of truth for what we own — our tables are an audit log, and where they
disagree, Toss wins.
"""

from __future__ import annotations

import logging
import time
import uuid
from decimal import Decimal

from ..data.store import Store, s, utcnow
from ..tick import round_to_tick
from ..toss.client import TossClient
from ..toss.errors import SKIP_SYMBOL_CODES, TossError, TossOrderRejectedError
from ..toss.models import Order, Side
from ..universe import country_of
from .broker import AccountSnapshot, Broker, OrderIntent, OrderResult, Position

log = logging.getLogger(__name__)

FILL_POLL_ATTEMPTS = 5
FILL_POLL_DELAY = 1.0


class LiveBroker(Broker):
    name = "live"

    def __init__(
        self,
        client: TossClient,
        store: Store,
        *,
        etf_symbols: frozenset[str] | None = None,
        confirm_high_value: bool = False,
    ) -> None:
        self.client = client
        self.store = store
        self.etf_symbols = etf_symbols or frozenset()
        self.confirm_high_value = confirm_high_value

    # ------------------------------------------------------------------
    def snapshot(self) -> AccountSnapshot:
        holdings = self.client.holdings()
        fx = self.client.usd_krw()

        positions: dict[str, Position] = {}
        prices: dict[str, Decimal] = {}
        for item in holdings.items:
            if item.quantity <= 0:
                continue
            positions[item.symbol] = Position(
                symbol=item.symbol,
                quantity=item.quantity,
                avg_price=item.average_purchase_price,
                currency=item.currency,
            )
            prices[item.symbol] = item.last_price

        cash_krw = self.client.buying_power("KRW").cash_buying_power
        try:
            usd_cash = self.client.buying_power("USD").cash_buying_power
            cash_krw += usd_cash * fx
        except TossError as exc:
            log.debug("USD buying power unavailable: %s", exc)

        return AccountSnapshot(
            cash_krw=cash_krw, positions=positions, usd_krw=fx, prices=prices
        )

    def open_orders(self) -> list[Order]:
        return self.client.open_orders()

    def cancel_all(self) -> int:
        cancelled = 0
        for order in self.client.open_orders():
            try:
                self.client.cancel_order(order.order_id)
                cancelled += 1
            except TossError as exc:
                log.warning("cancel failed for %s: %s", order.order_id, exc)
        if cancelled:
            self.store.log_event("WARN", "cancel_all", f"cancelled {cancelled} open orders")
        return cancelled

    def price_of(self, symbol: str) -> Decimal | None:
        try:
            return self.client.price(symbol).last_price
        except TossError:
            return None

    def sellable(self, symbol: str) -> Decimal:
        try:
            return self.client.sellable_quantity(symbol)
        except TossError as exc:
            log.warning("sellable_quantity failed for %s: %s", symbol, exc)
            return Decimal(0)

    # ------------------------------------------------------------------
    def submit(self, intent: OrderIntent) -> OrderResult:
        symbol = intent.symbol
        side = intent.side.upper()
        country = country_of(symbol)
        currency = "KRW" if country == "KR" else "USD"
        client_order_id = f"tqt-{uuid.uuid4().hex[:24]}"

        qty = intent.quantity
        price = intent.price

        # KRX trades whole shares only; a fractional request would be rejected.
        if qty is not None and country == "KR":
            qty = Decimal(int(qty))
            if qty <= 0:
                return OrderResult(
                    ok=False, symbol=symbol, side=side, error="quantity rounds to zero"
                )

        # Never try to sell more than the broker says is sellable (T+2 settlement
        # means freshly bought shares may not be sellable yet).
        if side == "SELL" and qty is not None:
            avail = self.sellable(symbol)
            if avail <= 0:
                return OrderResult(
                    ok=False, symbol=symbol, side=side, error="sellable quantity is 0"
                )
            if qty > avail:
                log.info("%s: trimming sell %s -> %s (sellable)", symbol, qty, avail)
                qty = avail if country != "KR" else Decimal(int(avail))

        if intent.order_type == "LIMIT" and price is not None:
            price = round_to_tick(
                price, country=country, is_etf=symbol in self.etf_symbols, side=side
            )

        try:
            ack = self.client.create_order(
                symbol,
                Side(side),
                intent.order_type,
                quantity=qty,
                price=price,
                order_amount=intent.order_amount,
                client_order_id=client_order_id,
                confirm_high_value=self.confirm_high_value,
            )
        except TossOrderRejectedError as exc:
            severity = "INFO" if exc.code in SKIP_SYMBOL_CODES else "ERROR"
            self.store.log_event(
                severity,
                "order_rejected",
                f"{side} {symbol}: {exc.code} {exc.message}",
                {"symbol": symbol, "side": side, "code": exc.code},
            )
            return OrderResult(
                ok=False, symbol=symbol, side=side, error=f"{exc.code}: {exc.message}"
            )
        except TossError as exc:
            self.store.log_event("ERROR", "order_failed", f"{side} {symbol}: {exc}")
            return OrderResult(ok=False, symbol=symbol, side=side, error=str(exc))

        self._record_submission(ack.order_id, client_order_id, intent, qty, price, currency)
        final = self._await_fill(ack.order_id)

        if final is None:
            return OrderResult(
                ok=True,
                symbol=symbol,
                side=side,
                order_id=ack.order_id,
                status="PENDING",
                currency=currency,
                reason=intent.reason,
            )

        ex = final.execution
        result = OrderResult(
            ok=final.status not in {"REJECTED", "CANCEL_REJECTED", "REPLACE_REJECTED"},
            symbol=symbol,
            side=side,
            order_id=final.order_id,
            filled_quantity=ex.filled_quantity if ex else Decimal(0),
            fill_price=(ex.average_filled_price if ex else None),
            commission=(ex.commission or Decimal(0)) if ex else Decimal(0),
            tax=(ex.tax or Decimal(0)) if ex else Decimal(0),
            currency=final.currency,
            status=final.status,
            reason=intent.reason,
        )
        self._record_final(final, intent)
        return result

    # ------------------------------------------------------------------
    def _await_fill(self, order_id: str) -> Order | None:
        """Poll briefly for a terminal state.

        Market orders in an open session normally fill within a second. If it is
        still pending we return None rather than blocking the run — the next cycle
        reconciles it, and ``sync_orders`` records the eventual outcome.
        """
        for _ in range(FILL_POLL_ATTEMPTS):
            try:
                order = self.client.get_order(order_id)
            except TossError as exc:
                log.warning("order poll failed for %s: %s", order_id, exc)
                return None
            if order.is_terminal:
                return order
            time.sleep(FILL_POLL_DELAY)
        return None

    def _record_submission(
        self,
        order_id: str,
        client_order_id: str,
        intent: OrderIntent,
        qty: Decimal | None,
        price: Decimal | None,
        currency: str,
    ) -> None:
        self.store.conn.execute(
            """INSERT OR REPLACE INTO orders(order_id,broker,client_order_id,strategy,symbol,
                   side,order_type,quantity,price,currency,status,submitted_at,updated_at,reason)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                order_id,
                self.name,
                client_order_id,
                intent.strategy,
                intent.symbol,
                intent.side,
                intent.order_type,
                s(qty) if qty is not None else s(intent.order_amount),
                s(price) if price is not None else None,
                currency,
                "SUBMITTED",
                utcnow(),
                utcnow(),
                intent.reason,
            ),
        )

    def _record_final(self, order: Order, intent: OrderIntent) -> None:
        ex = order.execution
        with self.store.transaction():
            self.store.conn.execute(
                """UPDATE orders SET status=?, filled_quantity=?, avg_fill_price=?,
                       commission=?, tax=?, updated_at=? WHERE broker=? AND order_id=?""",
                (
                    order.status,
                    s(ex.filled_quantity if ex else Decimal(0)),
                    s(ex.average_filled_price) if ex and ex.average_filled_price else None,
                    s(ex.commission) if ex and ex.commission else None,
                    s(ex.tax) if ex and ex.tax else None,
                    utcnow(),
                    self.name,
                    order.order_id,
                ),
            )
            if ex and ex.filled_quantity > 0 and ex.average_filled_price:
                self.store.conn.execute(
                    """INSERT INTO fills(broker,order_id,symbol,side,quantity,price,commission,
                                         tax,currency,filled_at,strategy)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        self.name,
                        order.order_id,
                        order.symbol,
                        order.side,
                        s(ex.filled_quantity),
                        s(ex.average_filled_price),
                        s(ex.commission or Decimal(0)),
                        s(ex.tax or Decimal(0)),
                        order.currency,
                        ex.filled_at or utcnow(),
                        intent.strategy,
                    ),
                )

    # ------------------------------------------------------------------
    def sync_orders(self, days: int = 7) -> int:
        """Reconcile our ledger with Toss's order history.

        Run at startup: it catches orders that filled while the bot was down, which
        is exactly the state that makes a bot double-buy on restart.
        """
        updated = 0
        for order in self.client.iter_orders("CLOSED", limit=100, max_pages=5):
            ex = order.execution
            cur = self.store.conn.execute(
                "SELECT status FROM orders WHERE broker=? AND order_id=?",
                (self.name, order.order_id),
            ).fetchone()
            if cur and cur["status"] == order.status:
                continue
            self.store.conn.execute(
                """INSERT INTO orders(order_id,broker,symbol,side,order_type,quantity,price,
                       currency,status,filled_quantity,avg_fill_price,submitted_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(broker,order_id) DO UPDATE SET
                       status=excluded.status, filled_quantity=excluded.filled_quantity,
                       avg_fill_price=excluded.avg_fill_price, updated_at=excluded.updated_at""",
                (
                    order.order_id,
                    self.name,
                    order.symbol,
                    order.side,
                    order.order_type,
                    s(order.quantity),
                    s(order.price) if order.price else None,
                    order.currency,
                    order.status,
                    s(ex.filled_quantity if ex else Decimal(0)),
                    s(ex.average_filled_price) if ex and ex.average_filled_price else None,
                    order.ordered_at,
                    utcnow(),
                ),
            )
            updated += 1
        if updated:
            log.info("reconciled %d orders from Toss history", updated)
        return updated
