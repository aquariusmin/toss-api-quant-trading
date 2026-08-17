"""Pydantic models for Toss API payloads.

Two rules drive everything here:

1. **Every monetary/quantity field is ``Decimal``.** Toss returns them as JSON
   strings precisely so clients don't lose precision; parsing them as ``float``
   would reintroduce the error the API design avoided. ``Money`` does the
   str -> Decimal coercion.

2. **Inbound enums stay ``str``.** The docs explicitly say "클라이언트는 unknown
   enum 값을 허용하도록 구현해야 합니다" — a new order status or market code must
   not crash a running bot. Enums are used only for values *we* send, where a
   closed set is what we want. Constants for the known inbound values live at the
   bottom of this module.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _to_decimal(v: Any) -> Any:
    if v is None or isinstance(v, Decimal):
        return v
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if s == "":
            return None
        try:
            return Decimal(s)
        except InvalidOperation as exc:  # pragma: no cover - defensive
            raise ValueError(f"not a decimal: {v!r}") from exc
    if isinstance(v, (int, float)):
        return Decimal(str(v))
    return v


Money = Annotated[Decimal, BeforeValidator(_to_decimal)]
OptMoney = Annotated[Decimal | None, BeforeValidator(_to_decimal)]


class TossModel(BaseModel):
    """Base: tolerate new fields Toss adds without a client release."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Outbound enums (values we send — a closed set is desirable here)
# ---------------------------------------------------------------------------
class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class TimeInForce(str, Enum):
    DAY = "DAY"
    CLS = "CLS"  # LIMIT + CLS == LOC (limit-on-close)


class Currency(str, Enum):
    KRW = "KRW"
    USD = "USD"


class MarketCountry(str, Enum):
    KR = "KR"
    US = "US"


class ConditionalType(str, Enum):
    SINGLE = "SINGLE"
    OCO = "OCO"  # one-cancels-the-other
    OTO = "OTO"  # one-triggers-the-other


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
class Price(TossModel):
    symbol: str
    timestamp: str | None = None
    last_price: Money = Field(alias="lastPrice")
    currency: str


class BookLevel(TossModel):
    price: Money
    volume: Money


class OrderBook(TossModel):
    timestamp: str | None = None
    currency: str
    asks: list[BookLevel] = Field(default_factory=list)
    bids: list[BookLevel] = Field(default_factory=list)

    @property
    def best_ask(self) -> Decimal | None:
        """Lowest sell offer — what a market BUY would pay."""
        return self.asks[0].price if self.asks else None

    @property
    def best_bid(self) -> Decimal | None:
        """Highest bid — what a market SELL would receive."""
        return self.bids[0].price if self.bids else None

    @property
    def mid(self) -> Decimal | None:
        if self.best_ask is None or self.best_bid is None:
            return None
        return (self.best_ask + self.best_bid) / Decimal(2)

    @property
    def spread_bps(self) -> Decimal | None:
        """Half-spread cost estimate in basis points, for slippage calibration."""
        m = self.mid
        if not m or m == 0 or self.best_ask is None or self.best_bid is None:
            return None
        return (self.best_ask - self.best_bid) / m * Decimal(10000)


class Trade(TossModel):
    price: Money
    volume: Money
    timestamp: str
    currency: str


class PriceLimits(TossModel):
    timestamp: str
    upper_limit_price: OptMoney = Field(default=None, alias="upperLimitPrice")
    lower_limit_price: OptMoney = Field(default=None, alias="lowerLimitPrice")
    currency: str


class Candle(TossModel):
    timestamp: str
    open: Money = Field(alias="openPrice")
    high: Money = Field(alias="highPrice")
    low: Money = Field(alias="lowPrice")
    close: Money = Field(alias="closePrice")
    volume: Money
    currency: str | None = None


class CandlePage(TossModel):
    candles: list[Candle] = Field(default_factory=list)
    next_before: str | None = Field(default=None, alias="nextBefore")


# ---------------------------------------------------------------------------
# Stock info
# ---------------------------------------------------------------------------
class KrMarketDetail(TossModel):
    liquidation_trading: bool = Field(default=False, alias="liquidationTrading")
    nxt_supported: bool = Field(default=False, alias="nxtSupported")
    krx_trading_suspended: bool = Field(default=False, alias="krxTradingSuspended")
    nxt_trading_suspended: bool | None = Field(default=None, alias="nxtTradingSuspended")


class Stock(TossModel):
    symbol: str
    name: str
    english_name: str | None = Field(default=None, alias="englishName")
    isin_code: str | None = Field(default=None, alias="isinCode")
    market: str
    security_type: str = Field(alias="securityType")
    is_common_share: bool = Field(default=True, alias="isCommonShare")
    status: str
    currency: str
    list_date: str | None = Field(default=None, alias="listDate")
    delist_date: str | None = Field(default=None, alias="delistDate")
    shares_outstanding: OptMoney = Field(default=None, alias="sharesOutstanding")
    leverage_factor: OptMoney = Field(default=None, alias="leverageFactor")
    korean_market_detail: KrMarketDetail | None = Field(default=None, alias="koreanMarketDetail")

    @property
    def market_country(self) -> str:
        return "KR" if self.market in {"KOSPI", "KOSDAQ", "KR_ETC"} else "US"

    @property
    def is_tradable(self) -> bool:
        """Cheap pre-trade gate on static metadata."""
        if self.status != "ACTIVE":
            return False
        d = self.korean_market_detail
        return not (d and (d.liquidation_trading or d.krx_trading_suspended))

    @property
    def is_leveraged(self) -> bool:
        """Leveraged/inverse ETFs decay; strategies usually want to exclude them."""
        lf = self.leverage_factor
        return lf is not None and lf != Decimal(1)


class StockBrief(TossModel):
    symbol: str
    name: str
    security_type: str = Field(alias="securityType")
    is_common_share: bool = Field(default=True, alias="isCommonShare")
    isin_code: str | None = Field(default=None, alias="isinCode")


class StockWarning(TossModel):
    warning_type: str = Field(alias="warningType")
    exchange: str | None = None
    start_date: str | None = Field(default=None, alias="startDate")
    end_date: str | None = Field(default=None, alias="endDate")


#: Warnings that should block a *new buy* outright. VI (변동성 완화장치) is
#: transient and not disqualifying on its own, so it is not in this set.
BLOCKING_WARNINGS = frozenset(
    {
        "LIQUIDATION_TRADING",  # 정리매매 — delisting in progress
        "INVESTMENT_WARNING",  # 투자경고
        "INVESTMENT_RISK",  # 투자위험
        "OVERHEATED",  # 단기과열
        "STOCK_WARRANTS",  # 신주인수권
    }
)


# ---------------------------------------------------------------------------
# Market info
# ---------------------------------------------------------------------------
class ExchangeRate(TossModel):
    base_currency: str = Field(alias="baseCurrency")
    quote_currency: str = Field(alias="quoteCurrency")
    rate: Money
    mid_rate: Money = Field(alias="midRate")
    basis_point: Money = Field(alias="basisPoint")
    rate_change_type: str = Field(alias="rateChangeType")
    valid_from: str | None = Field(default=None, alias="validFrom")
    valid_until: str | None = Field(default=None, alias="validUntil")


class Session(TossModel):
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    single_price_auction_start_time: str | None = Field(
        default=None, alias="singlePriceAuctionStartTime"
    )


class KrSessions(TossModel):
    pre_market: Session | None = Field(default=None, alias="preMarket")
    regular_market: Session | None = Field(default=None, alias="regularMarket")
    after_market: Session | None = Field(default=None, alias="afterMarket")


class KrMarketDay(TossModel):
    date: str
    integrated: KrSessions | None = None

    @property
    def is_open_day(self) -> bool:
        """A holiday comes back with no session block at all."""
        return self.integrated is not None and self.integrated.regular_market is not None


class UsMarketDay(TossModel):
    date: str
    day_market: Session | None = Field(default=None, alias="dayMarket")
    pre_market: Session | None = Field(default=None, alias="preMarket")
    regular_market: Session | None = Field(default=None, alias="regularMarket")
    after_market: Session | None = Field(default=None, alias="afterMarket")

    @property
    def is_open_day(self) -> bool:
        return self.regular_market is not None


class KrMarketCalendar(TossModel):
    today: KrMarketDay
    previous_business_day: KrMarketDay = Field(alias="previousBusinessDay")
    next_business_day: KrMarketDay = Field(alias="nextBusinessDay")


class UsMarketCalendar(TossModel):
    today: UsMarketDay
    previous_business_day: UsMarketDay = Field(alias="previousBusinessDay")
    next_business_day: UsMarketDay = Field(alias="nextBusinessDay")


class RankingEntry(TossModel):
    rank: int
    symbol: str
    currency: str
    trading_volume: OptMoney = Field(default=None, alias="tradingVolume")
    trading_amount: OptMoney = Field(default=None, alias="tradingAmount")
    price: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Account & assets
# ---------------------------------------------------------------------------
class Account(TossModel):
    account_no: str = Field(alias="accountNo")
    account_seq: int = Field(alias="accountSeq")
    account_type: str = Field(alias="accountType")

    @property
    def digits(self) -> str:
        return "".join(c for c in self.account_no if c.isdigit())


class CurrencyAmount(TossModel):
    krw: Money
    usd: OptMoney = None


class OverviewMarketValue(TossModel):
    amount: CurrencyAmount
    amount_after_cost: CurrencyAmount = Field(alias="amountAfterCost")


class OverviewProfitLoss(TossModel):
    amount: CurrencyAmount
    amount_after_cost: CurrencyAmount | None = Field(default=None, alias="amountAfterCost")
    rate: Money
    rate_after_cost: OptMoney = Field(default=None, alias="rateAfterCost")


class ItemMarketValue(TossModel):
    purchase_amount: Money = Field(alias="purchaseAmount")
    amount: Money
    amount_after_cost: Money = Field(alias="amountAfterCost")


class ItemProfitLoss(TossModel):
    amount: Money
    amount_after_cost: OptMoney = Field(default=None, alias="amountAfterCost")
    rate: Money
    rate_after_cost: OptMoney = Field(default=None, alias="rateAfterCost")


class ItemDailyProfitLoss(TossModel):
    amount: Money
    rate: Money


class ItemCost(TossModel):
    commission: Money
    tax: OptMoney = None


class HoldingItem(TossModel):
    symbol: str
    name: str
    market_country: str = Field(alias="marketCountry")
    currency: str
    quantity: Money
    last_price: Money = Field(alias="lastPrice")
    average_purchase_price: Money = Field(alias="averagePurchasePrice")
    market_value: ItemMarketValue = Field(alias="marketValue")
    profit_loss: ItemProfitLoss = Field(alias="profitLoss")
    daily_profit_loss: ItemDailyProfitLoss | None = Field(default=None, alias="dailyProfitLoss")
    cost: ItemCost | None = None


class Holdings(TossModel):
    total_purchase_amount: CurrencyAmount = Field(alias="totalPurchaseAmount")
    market_value: OverviewMarketValue = Field(alias="marketValue")
    profit_loss: OverviewProfitLoss = Field(alias="profitLoss")
    daily_profit_loss: OverviewProfitLoss | None = Field(default=None, alias="dailyProfitLoss")
    items: list[HoldingItem] = Field(default_factory=list)

    def by_symbol(self) -> dict[str, HoldingItem]:
        return {i.symbol: i for i in self.items}


class BuyingPower(TossModel):
    currency: str
    cash_buying_power: Money = Field(alias="cashBuyingPower")


class Commission(TossModel):
    market_country: str = Field(alias="marketCountry")
    commission_rate: Money = Field(alias="commissionRate")
    start_date: str | None = Field(default=None, alias="startDate")
    end_date: str | None = Field(default=None, alias="endDate")


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
class Execution(TossModel):
    filled_quantity: Money = Field(alias="filledQuantity")
    average_filled_price: OptMoney = Field(default=None, alias="averageFilledPrice")
    filled_amount: OptMoney = Field(default=None, alias="filledAmount")
    commission: OptMoney = None
    tax: OptMoney = None
    filled_at: str | None = Field(default=None, alias="filledAt")
    settlement_date: str | None = Field(default=None, alias="settlementDate")


class Order(TossModel):
    order_id: str = Field(alias="orderId")
    symbol: str
    side: str
    order_type: str = Field(alias="orderType")
    time_in_force: str = Field(default="DAY", alias="timeInForce")
    status: str
    price: OptMoney = None
    quantity: Money
    order_amount: OptMoney = Field(default=None, alias="orderAmount")
    currency: str
    ordered_at: str = Field(alias="orderedAt")
    canceled_at: str | None = Field(default=None, alias="canceledAt")
    execution: Execution | None = None

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_ORDER_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_ORDER_STATUSES

    @property
    def filled_quantity(self) -> Decimal:
        return self.execution.filled_quantity if self.execution else Decimal(0)

    @property
    def remaining_quantity(self) -> Decimal:
        return max(self.quantity - self.filled_quantity, Decimal(0))


OPEN_ORDER_STATUSES = frozenset({"PENDING", "PARTIAL_FILLED", "PENDING_CANCEL", "PENDING_REPLACE"})
TERMINAL_ORDER_STATUSES = frozenset(
    {"FILLED", "CANCELED", "REJECTED", "REPLACED", "CANCEL_REJECTED", "REPLACE_REJECTED"}
)


class OrdersPage(TossModel):
    orders: list[Order] = Field(default_factory=list)
    next_cursor: str | None = Field(default=None, alias="nextCursor")
    has_next: bool = Field(default=False, alias="hasNext")


class OrderAck(TossModel):
    order_id: str = Field(alias="orderId")
    client_order_id: str | None = Field(default=None, alias="clientOrderId")


class ConditionalLeg(TossModel):
    type: str | None = None
    status: str | None = None
    trigger_price: OptMoney = Field(default=None, alias="triggerPrice")
    target_profit_rate: OptMoney = Field(default=None, alias="targetProfitRate")
    order_price: OptMoney = Field(default=None, alias="orderPrice")
    triggered_order_id: str | None = Field(default=None, alias="triggeredOrderId")
    order_side: str | None = Field(default=None, alias="orderSide")


class ConditionalOrder(TossModel):
    conditional_order_id: str = Field(alias="conditionalOrderId")
    type: str
    status: str
    symbol: str
    market: str | None = None
    quantity: Money
    order_type: str = Field(alias="orderType")
    expire_date: str | None = Field(default=None, alias="expireDate")
    first: ConditionalLeg
    second: ConditionalLeg | None = None
    created_at: str | None = Field(default=None, alias="createdAt")


class ConditionalOrdersPage(TossModel):
    conditional_orders: list[ConditionalOrder] = Field(
        default_factory=list, alias="conditionalOrders"
    )
    next_cursor: str | None = Field(default=None, alias="nextCursor")
    has_next: bool = Field(default=False, alias="hasNext")


class ConditionalOrderAck(TossModel):
    conditional_order_id: str = Field(alias="conditionalOrderId")
    client_order_id: str | None = Field(default=None, alias="clientOrderId")
