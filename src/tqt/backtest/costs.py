"""Transaction cost model for KR and US trading through Toss.

Costs are the difference between a backtest that means something and one that
doesn't. A monthly-rebalanced momentum strategy turns over its portfolio roughly
10x a year; at a round-trip cost of 50bp that is 5% of annual return handed over
before the strategy has done anything. Understating costs is the single most
common way a promising backtest turns into a losing live account.

Default rates are **calibrated from this account's live API values** rather than
guessed — ``CostModel.from_api()`` reads ``GET /api/v1/commissions``, which
returned KR 0.015% and US 0.10% for this account.

What is modelled:

===============  ========================================================
commission       Per-side, from the Toss commissions API.
KR sell tax      증권거래세 + 농특세, 0.15% on stocks. **ETFs are exempt.**
US sell fees     SEC Section 31 fee + FINRA TAF (small, sells only).
FX spread        Charged when KRW must be converted to trade a US symbol.
slippage         Half-spread plus impact, in basis points.
===============  ========================================================

What is deliberately *not* in here: income tax on gains. That is an annual,
portfolio-level, jurisdiction-specific calculation, not a per-trade one, so it
lives in ``metrics.after_tax_summary`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

BPS = Decimal("0.0001")

# --- documented Korean statutory rates (2026) ------------------------------
# 증권거래세 0% + 농어촌특별세 0.15% for KOSPI; KOSDAQ 0.15% 거래세.
# Both land on 0.15% of *proceeds*, charged on sells only.
KR_STOCK_SELL_TAX = Decimal("0.0015")
# ETFs and ETNs are exempt from 증권거래세 — a large edge for ETF strategies.
KR_ETF_SELL_TAX = Decimal("0")

# --- US regulatory fees (sells only; rates are revised periodically) -------
US_SEC_FEE_RATE = Decimal("0.0000278")  # SEC §31, per USD of proceeds
US_TAF_PER_SHARE = Decimal("0.000166")  # FINRA TAF
US_TAF_CAP = Decimal("8.30")


@dataclass(frozen=True)
class TradeCost:
    """Cost breakdown for a single fill, in the instrument's own currency."""

    commission: Decimal = Decimal(0)
    tax: Decimal = Decimal(0)
    regulatory: Decimal = Decimal(0)
    slippage: Decimal = Decimal(0)
    fx: Decimal = Decimal(0)

    @property
    def total(self) -> Decimal:
        return self.commission + self.tax + self.regulatory + self.slippage + self.fx

    @property
    def explicit(self) -> Decimal:
        """Costs that appear on a statement (excludes modelled slippage)."""
        return self.commission + self.tax + self.regulatory + self.fx

    def __add__(self, other: TradeCost) -> TradeCost:
        return TradeCost(
            commission=self.commission + other.commission,
            tax=self.tax + other.tax,
            regulatory=self.regulatory + other.regulatory,
            slippage=self.slippage + other.slippage,
            fx=self.fx + other.fx,
        )


@dataclass(frozen=True)
class CostModel:
    kr_commission_rate: Decimal = Decimal("0.00015")
    us_commission_rate: Decimal = Decimal("0.001")

    kr_stock_sell_tax: Decimal = KR_STOCK_SELL_TAX
    kr_etf_sell_tax: Decimal = KR_ETF_SELL_TAX

    us_sec_fee_rate: Decimal = US_SEC_FEE_RATE
    us_taf_per_share: Decimal = US_TAF_PER_SHARE

    #: One-way FX spread applied when KRW is converted to buy a USD asset.
    #: Toss quotes a real-time rate with a spread; measure yours and set it here.
    fx_spread_rate: Decimal = Decimal("0.001")
    #: Set False if you hold a USD cash balance and never convert per-trade.
    charge_fx_on_us_trades: bool = True

    #: Slippage in basis points of notional, per side.
    slippage_bps_kr: Decimal = Decimal("10")
    slippage_bps_us: Decimal = Decimal("5")

    def __post_init__(self) -> None:
        for name in (
            "kr_commission_rate",
            "us_commission_rate",
            "fx_spread_rate",
            "slippage_bps_kr",
            "slippage_bps_us",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    # ------------------------------------------------------------------
    @classmethod
    def from_api(cls, client, **overrides) -> CostModel:
        """Calibrate commission rates from the account's real Toss values."""
        rates: dict[str, Decimal] = {}
        for c in client.commissions():
            if c.market_country == "KR":
                rates["kr_commission_rate"] = c.commission_rate
            elif c.market_country == "US":
                rates["us_commission_rate"] = c.commission_rate
        return cls(**{**rates, **overrides})

    def with_(self, **kw) -> CostModel:
        return replace(self, **kw)

    # ------------------------------------------------------------------
    def commission_rate(self, country: str) -> Decimal:
        return self.kr_commission_rate if country == "KR" else self.us_commission_rate

    def slippage_rate(self, country: str) -> Decimal:
        bps = self.slippage_bps_kr if country == "KR" else self.slippage_bps_us
        return bps * BPS

    def sell_tax_rate(self, country: str, *, is_etf: bool) -> Decimal:
        if country != "KR":
            return Decimal(0)
        return self.kr_etf_sell_tax if is_etf else self.kr_stock_sell_tax

    # ------------------------------------------------------------------
    def cost_of(
        self,
        *,
        side: str,
        country: str,
        price: Decimal,
        quantity: Decimal,
        is_etf: bool = True,
        include_slippage: bool = True,
    ) -> TradeCost:
        """Cost of one fill, denominated in the instrument's own currency.

        ``price`` should be the *intended* price; slippage is returned separately
        so the caller can either shift the fill price or book it as a cost, but
        must not do both.
        """
        side = side.upper()
        notional = abs(price * quantity)
        if notional == 0:
            return TradeCost()

        commission = notional * self.commission_rate(country)
        slippage = notional * self.slippage_rate(country) if include_slippage else Decimal(0)

        tax = Decimal(0)
        regulatory = Decimal(0)
        if side == "SELL":
            tax = notional * self.sell_tax_rate(country, is_etf=is_etf)
            if country == "US":
                sec = notional * self.us_sec_fee_rate
                taf = min(abs(quantity) * self.us_taf_per_share, US_TAF_CAP)
                regulatory = sec + taf

        fx = Decimal(0)
        if country == "US" and self.charge_fx_on_us_trades:
            fx = notional * self.fx_spread_rate

        return TradeCost(
            commission=commission,
            tax=tax,
            regulatory=regulatory,
            slippage=slippage,
            fx=fx,
        )

    def round_trip_bps(self, country: str, *, is_etf: bool = True) -> Decimal:
        """Total buy+sell cost in basis points — the number to sanity-check turnover against.

        For this account: KR ETF ~23bp, KR stock ~53bp, US ETF ~32bp per round trip.
        A strategy trading 12x a year at 30bp round trip gives up ~3.6%/yr.
        """
        price, qty = Decimal(10_000), Decimal(100)
        buy = self.cost_of(
            side="BUY", country=country, price=price, quantity=qty, is_etf=is_etf
        )
        sell = self.cost_of(
            side="SELL", country=country, price=price, quantity=qty, is_etf=is_etf
        )
        return (buy.total + sell.total) / (price * qty) / BPS

    def describe(self) -> dict[str, str]:
        return {
            "KR commission": f"{self.kr_commission_rate * 100:.4f}%",
            "US commission": f"{self.us_commission_rate * 100:.4f}%",
            "KR stock sell tax": f"{self.kr_stock_sell_tax * 100:.3f}%",
            "KR ETF sell tax": f"{self.kr_etf_sell_tax * 100:.3f}% (면제)",
            "FX spread": f"{self.fx_spread_rate * 100:.3f}% one-way",
            "slippage KR / US": f"{self.slippage_bps_kr}bp / {self.slippage_bps_us}bp",
            "round trip KR ETF": f"{self.round_trip_bps('KR', is_etf=True):.1f}bp",
            "round trip KR stock": f"{self.round_trip_bps('KR', is_etf=False):.1f}bp",
            "round trip US ETF": f"{self.round_trip_bps('US', is_etf=True):.1f}bp",
        }
