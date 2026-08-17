"""Broker abstraction shared by paper and live trading.

The point of this interface is that ``runner.py`` cannot tell which broker it is
driving. Paper and live take the identical code path — same strategy, same risk
checks, same sizing, same order construction — so the paper stage genuinely
rehearses live behaviour instead of testing a parallel implementation that might
differ in exactly the place it matters.

Accounting convention: a single **KRW cash pool**. Buying a USD-denominated asset
converts KRW at the live rate plus the FX spread; selling converts back. This
mirrors how a Korean brokerage account with 통합증거금 actually behaves, and it
means "equity" is one number in one currency rather than two that need mental
arithmetic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ..universe import country_of


@dataclass
class Position:
    symbol: str
    quantity: Decimal
    avg_price: Decimal  # in the instrument's own currency
    currency: str

    @property
    def country(self) -> str:
        return country_of(self.symbol)


@dataclass
class OrderIntent:
    """What the strategy wants, before risk checks and broker translation."""

    symbol: str
    side: str  # BUY | SELL
    quantity: Decimal | None = None
    order_type: str = "MARKET"
    price: Decimal | None = None
    #: USD notional for dollar-based US market buys (Toss ``orderAmount``).
    order_amount: Decimal | None = None
    strategy: str = ""
    reason: str = ""

    @property
    def country(self) -> str:
        return country_of(self.symbol)

    def __post_init__(self) -> None:
        if self.quantity is None and self.order_amount is None:
            raise ValueError(f"{self.symbol}: intent needs quantity or order_amount")
        if self.order_type == "LIMIT" and self.price is None:
            raise ValueError(f"{self.symbol}: LIMIT intent needs a price")


@dataclass
class OrderResult:
    ok: bool
    symbol: str
    side: str
    order_id: str | None = None
    filled_quantity: Decimal = Decimal(0)
    fill_price: Decimal | None = None
    commission: Decimal = Decimal(0)
    tax: Decimal = Decimal(0)
    currency: str = "KRW"
    status: str = "UNKNOWN"
    error: str | None = None
    reason: str = ""

    @property
    def notional(self) -> Decimal:
        if self.fill_price is None:
            return Decimal(0)
        return self.fill_price * self.filled_quantity


@dataclass
class AccountSnapshot:
    cash_krw: Decimal
    positions: dict[str, Position]
    usd_krw: Decimal
    prices: dict[str, Decimal] = field(default_factory=dict)

    def position_value_krw(self) -> Decimal:
        total = Decimal(0)
        for sym, pos in self.positions.items():
            px = self.prices.get(sym)
            if px is None:
                px = pos.avg_price
            value = px * pos.quantity
            if pos.currency == "USD":
                value *= self.usd_krw
            total += value
        return total

    @property
    def equity_krw(self) -> Decimal:
        return self.cash_krw + self.position_value_krw()

    def weights(self) -> dict[str, Decimal]:
        eq = self.equity_krw
        if eq <= 0:
            return {}
        out: dict[str, Decimal] = {}
        for sym, pos in self.positions.items():
            px = self.prices.get(sym, pos.avg_price)
            value = px * pos.quantity
            if pos.currency == "USD":
                value *= self.usd_krw
            out[sym] = value / eq
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "equity_krw": float(self.equity_krw),
            "cash_krw": float(self.cash_krw),
            "positions_krw": float(self.position_value_krw()),
            "n_positions": len(self.positions),
            "usd_krw": float(self.usd_krw),
        }


class Broker(ABC):
    name: str = "abstract"

    @abstractmethod
    def snapshot(self) -> AccountSnapshot:
        """Current cash, positions, and marks."""

    @abstractmethod
    def submit(self, intent: OrderIntent) -> OrderResult:
        """Place one order."""

    @abstractmethod
    def open_orders(self) -> list[Any]:
        ...

    def cancel_all(self) -> int:
        return 0

    def price_of(self, symbol: str) -> Decimal | None:
        raise NotImplementedError
