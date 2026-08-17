"""Synchronous Toss Securities Open API client.

Why synchronous: at this strategy's frequency (daily/weekly rebalance, 1-minute
polling at the fastest) concurrency buys nothing, while sync code is far easier
to reason about when real money is at stake. FastAPI endpoints declared with
plain ``def`` run in a threadpool, so the dashboard can call this directly.

Safety properties worth knowing about:

* **Order creation is idempotent by default.** ``create_order`` generates a
  ``clientOrderId`` unless you pass one. Toss treats it as an idempotency key, so
  a read timeout on a POST — where the order may well have reached the exchange —
  can be retried safely instead of risking a duplicate position. This is the
  single most important correctness property in the whole client.
* **Retries are error-aware.** 401 re-issues the token; 429 honours
  ``Retry-After``; 5xx backs off exponentially with jitter; 422 (order rejected)
  is never retried because the request itself must change.
* **Rate limits are respected before the fact**, per documented API group.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Any

import httpx

from ..config import Settings, get_settings
from .auth import TokenManager
from .errors import (
    TossAuthError,
    TossConflictError,
    TossError,
    TossNotFoundError,
    TossRateLimitError,
    TossTransportError,
    from_response,
)
from .models import (
    BLOCKING_WARNINGS,
    Account,
    BuyingPower,
    Candle,
    CandlePage,
    Commission,
    ConditionalOrder,
    ConditionalOrderAck,
    ConditionalOrdersPage,
    ConditionalType,
    Currency,
    ExchangeRate,
    Holdings,
    KrMarketCalendar,
    Order,
    OrderAck,
    OrderBook,
    OrdersPage,
    OrderType,
    Price,
    PriceLimits,
    RankingEntry,
    Side,
    Stock,
    StockBrief,
    StockWarning,
    TimeInForce,
    Trade,
    UsMarketCalendar,
)

log = logging.getLogger(__name__)

MAX_CANDLES_PER_CALL = 200
MAX_SYMBOLS_PER_PRICE_CALL = 200
DEFAULT_MAX_RETRIES = 4


def _dec(v: Any) -> str:
    """Render a Decimal for the wire without scientific notation."""
    if isinstance(v, Decimal):
        return format(v.normalize(), "f")
    return str(v)


class TossClient:
    """Thin, typed, rate-limited wrapper over the Toss Open API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http: httpx.Client | None = None,
        limiter=None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.require_credentials()

        # Imported lazily so tests can inject a limiter with a fake clock.
        if limiter is None:
            from .ratelimit import RateLimiter

            limiter = RateLimiter()
        self.limiter = limiter
        self.max_retries = max_retries

        self._owns_http = http is None
        self.http = http or httpx.Client(
            base_url=self.settings.toss_base_url,
            timeout=httpx.Timeout(self.settings.toss_timeout_seconds, connect=5.0),
            headers={"User-Agent": "tqt/0.1 (+github.com/aquariusmin/toss-api-quant-trading)"},
        )
        self.tokens = TokenManager(
            self.http,
            self.settings.toss_client_id,
            self.settings.toss_client_secret,
            limiter=self.limiter,
        )
        self._account_seq: int | None = self.settings.toss_account_seq
        self._accounts_cache: list[Account] | None = None
        self.request_count = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_http:
            self.http.close()

    def __enter__(self) -> TossClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # core request path
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        group: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        account: bool = False,
        idempotent: bool = True,
    ) -> Any:
        """Issue one API call, returning the unwrapped ``result`` payload.

        ``idempotent=False`` means "this call may have taken effect even if we
        never saw the response", so transport-level failures are surfaced rather
        than retried.
        """
        clean_params = None
        if params:
            clean_params = {k: v for k, v in params.items() if v is not None}

        attempt = 0
        while True:
            attempt += 1
            self.limiter.acquire(group)

            headers = {"Authorization": f"Bearer {self.tokens.token()}"}
            if account:
                headers["X-Tossinvest-Account"] = str(self.account_seq)

            try:
                resp = self.http.request(
                    method, path, params=clean_params, json=json_body, headers=headers
                )
            except httpx.HTTPError as exc:
                err: TossError = TossTransportError(f"{method} {path} failed: {exc}")
                if not idempotent or attempt > self.max_retries:
                    raise err from exc
                self._sleep_backoff(attempt, reason=str(exc))
                continue

            self.request_count += 1
            self.limiter.observe_headers(group, resp.headers)

            if resp.status_code < 400:
                if resp.status_code == 204 or not resp.content:
                    return None
                payload = resp.json()
                if isinstance(payload, dict) and "result" in payload:
                    return payload["result"]
                return payload

            # --- error path -------------------------------------------------
            try:
                body = resp.json()
            except ValueError:
                body = {"error": {"message": resp.text[:400]}}

            retry_after = _parse_float(resp.headers.get("Retry-After"))
            exc_obj = from_response(
                resp.status_code,
                body,
                request_id=resp.headers.get("X-Request-Id"),
                retry_after=retry_after,
            )

            if isinstance(exc_obj, TossRateLimitError):
                self.limiter.penalize(group, exc_obj.retry_after)

            # An in-flight duplicate of our own idempotent order: wait it out and
            # re-ask, which returns the original order instead of creating a new one.
            transient_conflict = (
                isinstance(exc_obj, TossConflictError) and exc_obj.code == "request-in-progress"
            )

            if attempt > self.max_retries or not (
                exc_obj.retryable or transient_conflict
            ):
                raise exc_obj

            if isinstance(exc_obj, TossAuthError):
                self.tokens.invalidate()

            self._sleep_backoff(attempt, reason=str(exc_obj), retry_after=exc_obj_retry(exc_obj))

    def _sleep_backoff(
        self, attempt: int, *, reason: str, retry_after: float | None = None
    ) -> None:
        """Exponential backoff (1s, 2s, 4s, ...) with jitter, per the docs' advice."""
        delay = retry_after if retry_after else min(2.0 ** (attempt - 1), 30.0)
        delay += random.uniform(0, 0.25 * delay)
        log.warning("retrying in %.2fs (attempt %d): %s", delay, attempt, reason)
        time.sleep(delay)

    # ------------------------------------------------------------------
    # account resolution
    # ------------------------------------------------------------------
    @property
    def account_seq(self) -> int:
        """The numeric ``accountSeq`` for the X-Tossinvest-Account header.

        Users know their account *number*; the API wants an opaque sequence. If
        ``TOSS_ACCOUNT_SEQ`` isn't configured we resolve it once by matching the
        configured account number against the account list.
        """
        if self._account_seq is None:
            self._account_seq = self.resolve_account_seq()
        return self._account_seq

    def resolve_account_seq(self) -> int:
        accounts = self.accounts()
        if not accounts:
            raise TossError(
                "No brokerage (종합매매) account found on this Toss login. "
                "Open one in the Toss app before trading."
            )

        wanted = self.settings.toss_bank_account
        if wanted:
            for acc in accounts:
                if acc.digits.endswith(wanted) or wanted.endswith(acc.digits):
                    return acc.account_seq
            log.warning(
                "configured account %s matched none of %s; falling back to the first account",
                wanted[-4:],
                [a.account_no for a in accounts],
            )
        if len(accounts) > 1:
            log.warning(
                "multiple accounts found; using accountSeq=%s. "
                "Set TOSS_ACCOUNT_SEQ to pin one explicitly.",
                accounts[0].account_seq,
            )
        return accounts[0].account_seq

    def accounts(self, *, refresh: bool = False) -> list[Account]:
        """The account list, cached for the client's lifetime.

        ``ACCOUNT`` is the tightest rate-limit group at 1 request/second, and the
        list does not change while the process runs — so callers that need it
        twice (resolving accountSeq, then displaying it) must not pay for it
        twice or they trip a 429 on startup.
        """
        if self._accounts_cache is None or refresh:
            raw = self._request("GET", "/api/v1/accounts", group="ACCOUNT")
            self._accounts_cache = [Account.model_validate(a) for a in raw or []]
        return self._accounts_cache

    # ------------------------------------------------------------------
    # market data
    # ------------------------------------------------------------------
    def prices(self, symbols: Sequence[str]) -> list[Price]:
        """Current price for up to 200 symbols per call (auto-chunked beyond that)."""
        out: list[Price] = []
        syms = list(symbols)
        for i in range(0, len(syms), MAX_SYMBOLS_PER_PRICE_CALL):
            chunk = syms[i : i + MAX_SYMBOLS_PER_PRICE_CALL]
            raw = self._request(
                "GET", "/api/v1/prices", group="MARKET_DATA", params={"symbols": ",".join(chunk)}
            )
            out.extend(Price.model_validate(p) for p in raw or [])
        return out

    def price(self, symbol: str) -> Price:
        got = self.prices([symbol])
        if not got:
            raise TossNotFoundError(f"no price returned for {symbol}", code="stock-not-found")
        return got[0]

    def orderbook(self, symbol: str) -> OrderBook:
        raw = self._request(
            "GET", "/api/v1/orderbook", group="MARKET_DATA", params={"symbol": symbol}
        )
        return OrderBook.model_validate(raw)

    def trades(self, symbol: str, count: int | None = None) -> list[Trade]:
        raw = self._request(
            "GET",
            "/api/v1/trades",
            group="MARKET_DATA",
            params={"symbol": symbol, "count": count},
        )
        return [Trade.model_validate(t) for t in raw or []]

    def price_limits(self, symbol: str) -> PriceLimits:
        raw = self._request(
            "GET", "/api/v1/price-limits", group="MARKET_DATA", params={"symbol": symbol}
        )
        return PriceLimits.model_validate(raw)

    def candles(
        self,
        symbol: str,
        interval: str = "1d",
        *,
        count: int = MAX_CANDLES_PER_CALL,
        before: str | None = None,
        adjusted: bool = True,
    ) -> CandlePage:
        """One page of OHLCV. ``interval`` is ``1m`` or ``1d`` only; max 200 bars."""
        if interval not in {"1m", "1d"}:
            raise ValueError(f"interval must be '1m' or '1d', got {interval!r}")
        raw = self._request(
            "GET",
            "/api/v1/candles",
            group="MARKET_DATA_CHART",
            params={
                "symbol": symbol,
                "interval": interval,
                "count": min(count, MAX_CANDLES_PER_CALL),
                "before": before,
                "adjusted": str(adjusted).lower(),
            },
        )
        return CandlePage.model_validate(raw)

    def iter_candles(
        self,
        symbol: str,
        interval: str = "1d",
        *,
        max_bars: int = 2000,
        stop_at: str | None = None,
        adjusted: bool = True,
    ) -> Iterator[Candle]:
        """Walk history backwards through the ``nextBefore`` cursor.

        Yields oldest-first *within each page*, newest page first. Callers that
        need a fully sorted series should sort by timestamp (the data store does).

        ``stop_at``: stop once a bar at or before this ISO timestamp is seen —
        used for incremental top-ups so we don't re-download years of history.
        """
        fetched = 0
        before: str | None = None
        seen_cursors: set[str] = set()

        while fetched < max_bars:
            page = self.candles(
                symbol,
                interval,
                count=min(MAX_CANDLES_PER_CALL, max_bars - fetched),
                before=before,
                adjusted=adjusted,
            )
            if not page.candles:
                return

            for c in sorted(page.candles, key=lambda x: x.timestamp):
                yield c
                fetched += 1

            if stop_at and min(c.timestamp for c in page.candles) <= stop_at:
                return
            if not page.next_before or page.next_before in seen_cursors:
                return
            seen_cursors.add(page.next_before)
            before = page.next_before

    # ------------------------------------------------------------------
    # stock info
    # ------------------------------------------------------------------
    def stocks(self, symbols: Sequence[str]) -> list[Stock]:
        raw = self._request(
            "GET", "/api/v1/stocks", group="STOCK", params={"symbols": ",".join(symbols)}
        )
        return [Stock.model_validate(s) for s in raw or []]

    def stock(self, symbol: str) -> Stock:
        got = self.stocks([symbol])
        if not got:
            raise TossNotFoundError(f"unknown symbol {symbol}", code="stock-not-found")
        return got[0]

    def stocks_all(
        self,
        market: str,
        *,
        status: str | None = "ACTIVE",
        security_type: str | None = None,
        common_share: bool | None = None,
    ) -> list[StockBrief]:
        """Full symbol master for one market. Rate-limited to 1/s — cache it."""
        raw = self._request(
            "GET",
            "/api/v1/stocks/all",
            group="STOCK_ALL",
            params={
                "market": market,
                "status": status,
                "securityType": security_type,
                "commonShare": None if common_share is None else str(common_share).lower(),
            },
        )
        return [StockBrief.model_validate(s) for s in raw or []]

    def stock_warnings(self, symbol: str) -> list[StockWarning]:
        raw = self._request(
            "GET", f"/api/v1/stocks/{symbol}/warnings", group="STOCK", params=None
        )
        return [StockWarning.model_validate(w) for w in raw or []]

    def buy_block_reasons(self, symbol: str) -> list[str]:
        """Pre-trade gate: returns disqualifying warning types, empty if clear.

        Cheap insurance against the classic retail disaster of a bot happily
        accumulating a stock that is in 정리매매 (delisting liquidation).
        """
        try:
            warnings = self.stock_warnings(symbol)
        except TossNotFoundError:
            return ["stock-not-found"]
        return [w.warning_type for w in warnings if w.warning_type in BLOCKING_WARNINGS]

    def _trend(self, symbol: str, kind: str, count: int | None, until: str | None) -> dict:
        return self._request(
            "GET",
            f"/api/v1/stocks/{symbol}/{kind}",
            group="STOCK_TRADING_TREND",
            params={"count": count, "until": until},
        )

    def investor_trading(self, symbol: str, count: int = 30, until: str | None = None) -> dict:
        return self._trend(symbol, "investor-trading", count, until)

    def program_trades(self, symbol: str, count: int = 30, until: str | None = None) -> dict:
        return self._trend(symbol, "program-trades", count, until)

    def short_selling(self, symbol: str, count: int = 30, until: str | None = None) -> dict:
        return self._trend(symbol, "short-selling", count, until)

    def credit_trades(self, symbol: str, count: int = 30, until: str | None = None) -> dict:
        return self._trend(symbol, "credit-trades", count, until)

    def securities_lending(self, symbol: str, count: int = 30, until: str | None = None) -> dict:
        return self._trend(symbol, "securities-lending", count, until)

    # ------------------------------------------------------------------
    # market info
    # ------------------------------------------------------------------
    def exchange_rate(
        self, base: str = "USD", quote: str = "KRW", date_time: str | None = None
    ) -> ExchangeRate:
        raw = self._request(
            "GET",
            "/api/v1/exchange-rate",
            group="MARKET_INFO",
            params={"baseCurrency": base, "quoteCurrency": quote, "dateTime": date_time},
        )
        return ExchangeRate.model_validate(raw)

    def usd_krw(self) -> Decimal:
        """KRW per 1 USD — needed to value a mixed KR/US portfolio in one currency."""
        return self.exchange_rate("USD", "KRW").rate

    def market_calendar_kr(self, date: str | None = None) -> KrMarketCalendar:
        raw = self._request(
            "GET", "/api/v1/market-calendar/KR", group="MARKET_INFO", params={"date": date}
        )
        return KrMarketCalendar.model_validate(raw)

    def market_calendar_us(self, date: str | None = None) -> UsMarketCalendar:
        raw = self._request(
            "GET", "/api/v1/market-calendar/US", group="MARKET_INFO", params={"date": date}
        )
        return UsMarketCalendar.model_validate(raw)

    def rankings(
        self,
        ranking_type: str,
        market_country: str,
        duration: str = "1d",
        *,
        exclude_investment_caution: bool = True,
        count: int | None = 30,
    ) -> list[RankingEntry]:
        raw = self._request(
            "GET",
            "/api/v1/rankings",
            group="RANKING",
            params={
                "type": ranking_type,
                "marketCountry": market_country,
                "duration": duration,
                "excludeInvestmentCaution": str(exclude_investment_caution).lower(),
                "count": count,
            },
        )
        return [RankingEntry.model_validate(r) for r in (raw or {}).get("rankings", [])]

    def market_indicator_prices(self, symbols: Sequence[str]) -> list[dict]:
        return self._request(
            "GET",
            "/api/v1/market-indicators/prices",
            group="MARKET_INDICATOR_PRICE",
            params={"symbols": ",".join(symbols)},
        )

    def market_indicator_candles(
        self, symbol: str, interval: str = "1d", *, count: int = 200, before: str | None = None
    ) -> CandlePage:
        raw = self._request(
            "GET",
            f"/api/v1/market-indicators/{symbol}/candles",
            group="MARKET_INDICATOR_CHART",
            params={"interval": interval, "count": count, "before": before},
        )
        return CandlePage.model_validate(raw)

    # ------------------------------------------------------------------
    # assets
    # ------------------------------------------------------------------
    def holdings(self, symbol: str | None = None) -> Holdings:
        raw = self._request(
            "GET", "/api/v1/holdings", group="ASSET", params={"symbol": symbol}, account=True
        )
        return Holdings.model_validate(raw)

    def buying_power(self, currency: str | Currency = Currency.KRW) -> BuyingPower:
        cur = currency.value if isinstance(currency, Currency) else currency
        raw = self._request(
            "GET",
            "/api/v1/buying-power",
            group="ORDER_INFO",
            params={"currency": cur},
            account=True,
        )
        return BuyingPower.model_validate(raw)

    def sellable_quantity(self, symbol: str) -> Decimal:
        raw = self._request(
            "GET",
            "/api/v1/sellable-quantity",
            group="ORDER_INFO",
            params={"symbol": symbol},
            account=True,
        )
        from .models import _to_decimal

        return _to_decimal(raw["sellableQuantity"])

    def commissions(self) -> list[Commission]:
        """This account's real commission rates — used to calibrate backtest costs."""
        raw = self._request("GET", "/api/v1/commissions", group="ORDER_INFO", account=True)
        return [Commission.model_validate(c) for c in raw or []]

    # ------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------
    def create_order(
        self,
        symbol: str,
        side: Side | str,
        order_type: OrderType | str = OrderType.LIMIT,
        *,
        quantity: Decimal | int | str | None = None,
        price: Decimal | int | str | None = None,
        order_amount: Decimal | str | None = None,
        time_in_force: TimeInForce | str | None = None,
        client_order_id: str | None = None,
        confirm_high_value: bool = False,
    ) -> OrderAck:
        """Place an order.

        Exactly one of ``quantity`` or ``order_amount`` must be given;
        ``order_amount`` (dollar-denominated) is US market-order only.

        ``client_order_id`` defaults to a fresh UUID so that retrying a timed-out
        request cannot create a second order. Do not disable this.
        """
        if (quantity is None) == (order_amount is None):
            raise ValueError("pass exactly one of quantity= or order_amount=")

        side_v = side.value if isinstance(side, Side) else str(side)
        type_v = order_type.value if isinstance(order_type, OrderType) else str(order_type)

        if type_v == "LIMIT" and price is None:
            raise ValueError("LIMIT orders require price=")
        if type_v == "MARKET" and price is not None:
            raise ValueError("MARKET orders must not carry a price")
        if order_amount is not None and type_v != "MARKET":
            raise ValueError("order_amount= is only valid for MARKET orders (US)")

        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side_v,
            "orderType": type_v,
            "clientOrderId": client_order_id or f"tqt-{uuid.uuid4().hex[:24]}",
        }
        if quantity is not None:
            body["quantity"] = _dec(quantity)
            if price is not None:
                body["price"] = _dec(price)
            if time_in_force is not None:
                body["timeInForce"] = (
                    time_in_force.value
                    if isinstance(time_in_force, TimeInForce)
                    else str(time_in_force)
                )
        else:
            body["orderAmount"] = _dec(order_amount)
        if confirm_high_value:
            body["confirmHighValueOrder"] = True

        log.info("submitting order %s", body)
        raw = self._request(
            "POST", "/api/v1/orders", group="ORDER", json_body=body, account=True, idempotent=True
        )
        return OrderAck.model_validate(raw)

    def modify_order(
        self,
        order_id: str,
        *,
        order_type: OrderType | str = OrderType.LIMIT,
        quantity: Decimal | int | None = None,
        price: Decimal | int | None = None,
        confirm_high_value: bool = False,
    ) -> str:
        type_v = order_type.value if isinstance(order_type, OrderType) else str(order_type)
        body: dict[str, Any] = {"orderType": type_v}
        if quantity is not None:
            body["quantity"] = _dec(quantity)
        if price is not None:
            body["price"] = _dec(price)
        if confirm_high_value:
            body["confirmHighValueOrder"] = True
        raw = self._request(
            "POST",
            f"/api/v1/orders/{order_id}/modify",
            group="ORDER",
            json_body=body,
            account=True,
            idempotent=False,
        )
        return (raw or {}).get("orderId", order_id)

    def cancel_order(self, order_id: str) -> str:
        raw = self._request(
            "POST",
            f"/api/v1/orders/{order_id}/cancel",
            group="ORDER",
            json_body={},
            account=True,
            idempotent=False,
        )
        return (raw or {}).get("orderId", order_id)

    def get_order(self, order_id: str) -> Order:
        raw = self._request(
            "GET", f"/api/v1/orders/{order_id}", group="ORDER_HISTORY", account=True
        )
        return Order.model_validate(raw)

    def list_orders(
        self,
        status: str = "OPEN",
        *,
        symbol: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> OrdersPage:
        raw = self._request(
            "GET",
            "/api/v1/orders",
            group="ORDER_HISTORY",
            params={
                "status": status,
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "cursor": cursor,
                "limit": limit,
            },
            account=True,
        )
        return OrdersPage.model_validate(raw)

    def iter_orders(
        self,
        status: str = "CLOSED",
        *,
        symbol: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 100,
        max_pages: int = 50,
    ) -> Iterator[Order]:
        """Page through order history. ``status=OPEN`` returns everything at once."""
        cursor: str | None = None
        for _ in range(max_pages):
            page = self.list_orders(
                status,
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                cursor=cursor,
                limit=limit,
            )
            yield from page.orders
            if not page.has_next or not page.next_cursor:
                return
            cursor = page.next_cursor

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return self.list_orders("OPEN", symbol=symbol).orders

    # ------------------------------------------------------------------
    # conditional orders (server-side stop / take-profit)
    # ------------------------------------------------------------------
    def create_conditional_order(
        self,
        symbol: str,
        *,
        conditional_type: ConditionalType | str,
        quantity: Decimal | int,
        order_type: OrderType | str,
        expire_date: str,
        first_side: Side | str,
        first_trigger_price: Decimal | int,
        first_order_price: Decimal | int | None = None,
        second_side: Side | str | None = None,
        second_trigger_price: Decimal | int | None = None,
        second_order_price: Decimal | int | None = None,
        client_order_id: str | None = None,
        confirm_high_value: bool = False,
    ) -> ConditionalOrderAck:
        """Register a server-side conditional order.

        Worth preferring over bot-side stops: an OCO stop-loss / take-profit pair
        sits on Toss's servers, so it still protects the position when the Pi
        loses power or the process crashes. Note OCO/OTO are limited to one per
        symbol.
        """

        def leg(side: Any, trigger: Any, order_price: Any) -> dict[str, Any]:
            d: dict[str, Any] = {
                "orderSide": side.value if isinstance(side, Side) else str(side),
                "triggerPrice": _dec(trigger),
            }
            if order_price is not None:
                d["orderPrice"] = _dec(order_price)
            return d

        body: dict[str, Any] = {
            "symbol": symbol,
            "type": (
                conditional_type.value
                if isinstance(conditional_type, ConditionalType)
                else str(conditional_type)
            ),
            "quantity": _dec(quantity),
            "orderType": order_type.value if isinstance(order_type, OrderType) else str(order_type),
            "expireDate": expire_date,
            "clientOrderId": client_order_id or f"tqt-c-{uuid.uuid4().hex[:22]}",
            "first": leg(first_side, first_trigger_price, first_order_price),
        }
        if second_side is not None and second_trigger_price is not None:
            body["second"] = leg(second_side, second_trigger_price, second_order_price)
        if confirm_high_value:
            body["confirmHighValueOrder"] = True

        raw = self._request(
            "POST",
            "/api/v1/conditional-orders",
            group="CONDITIONAL_ORDER",
            json_body=body,
            account=True,
        )
        return ConditionalOrderAck.model_validate(raw)

    def list_conditional_orders(
        self,
        status: str = "OPEN",
        *,
        symbol: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ConditionalOrdersPage:
        raw = self._request(
            "GET",
            "/api/v1/conditional-orders",
            group="CONDITIONAL_ORDER_HISTORY",
            params={"status": status, "symbol": symbol, "cursor": cursor, "limit": limit},
            account=True,
        )
        return ConditionalOrdersPage.model_validate(raw)

    def get_conditional_order(self, conditional_order_id: str) -> ConditionalOrder:
        raw = self._request(
            "GET",
            f"/api/v1/conditional-orders/{conditional_order_id}",
            group="CONDITIONAL_ORDER_HISTORY",
            account=True,
        )
        return ConditionalOrder.model_validate(raw)

    def cancel_conditional_order(self, conditional_order_id: str) -> None:
        self._request(
            "DELETE",
            f"/api/v1/conditional-orders/{conditional_order_id}",
            group="CONDITIONAL_ORDER",
            account=True,
            idempotent=False,
        )


def exc_obj_retry(exc: TossError) -> float | None:
    return getattr(exc, "retry_after", None)


def _parse_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
