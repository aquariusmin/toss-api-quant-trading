"""Tradable universes, grouped into strategy "sleeves".

Every symbol here was verified against the live Toss API: it resolves, it is
ACTIVE, and it returns daily candles. Leveraged and inverse products are
deliberately excluded — daily-rebalanced leverage decays under volatility, so a
momentum rule applied to it measures the decay as much as the trend.

The KR sleeve is worth understanding, because it is unusual and useful: Korean
exchanges list ETFs tracking the S&P 500, Nasdaq 100, EuroStoxx and gold. Trading
those gives global exposure **in KRW**, with no currency conversion, no FX
spread, and no US capital-gains filing. The tax treatment differs from domestic
equity ETFs though — see docs/TAXES.md.

References for the sleeve designs:
  * Gary Antonacci, *Dual Momentum Investing* (2014) — relative + absolute momentum.
  * Wouter Keller & Jan Willem Keuning, *Vigilant Asset Allocation* (2017) —
    offensive/defensive switching driven by a breadth signal.
  * Mebane Faber, *A Quantitative Approach to Tactical Asset Allocation* (2007) —
    the 10-month moving-average trend filter.
  * Andreas Clenow, *Stocks on the Move* (2015) — volatility-scaled position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str
    country: str  # "KR" | "US"
    currency: str  # "KRW" | "USD"
    role: str  # equity | bond | gold | commodity | cash
    #: ETFs are exempt from Korean 증권거래세; individual stocks pay 0.15% on
    #: every sell. Getting this wrong understates a KR stock strategy's costs by
    #: 30bp per round trip, so it is stated per asset rather than inferred.
    is_etf: bool = True


@dataclass(frozen=True)
class Sleeve:
    """A self-contained strategy universe.

    ``risk`` assets are candidates to hold when momentum is positive; ``safe``
    assets are where the strategy hides when it isn't. Keeping them separate is
    what makes absolute-momentum ("is this thing even going up?") expressible.
    """

    key: str
    title: str
    description: str
    risk: list[Asset]
    safe: list[Asset]
    country: str = "KR"
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def all_assets(self) -> list[Asset]:
        return [*self.risk, *self.safe]

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.all_assets]

    @property
    def risk_symbols(self) -> list[str]:
        return [a.symbol for a in self.risk]

    @property
    def safe_symbols(self) -> list[str]:
        return [a.symbol for a in self.safe]

    def asset(self, symbol: str) -> Asset | None:
        return next((a for a in self.all_assets if a.symbol == symbol), None)


# ---------------------------------------------------------------------------
# KR-listed ETFs — global exposure, settled in KRW
# ---------------------------------------------------------------------------
KR_GLOBAL = Sleeve(
    key="kr-global-etf",
    title="국내상장 글로벌 ETF 모멘텀",
    description=(
        "KRW로 거래되는 국내 상장 ETF만 사용해 미국·유럽·중국·금까지 분산합니다. "
        "환전이 없으므로 환 스프레드가 들지 않고, 미국 양도소득세 신고 대상도 아닙니다."
    ),
    country="KR",
    risk=[
        Asset("069500", "KODEX 200", "KR", "KRW", "equity"),
        Asset("229200", "KODEX 코스닥150", "KR", "KRW", "equity"),
        Asset("360750", "TIGER 미국S&P500", "KR", "KRW", "equity"),
        Asset("133690", "TIGER 미국나스닥100", "KR", "KRW", "equity"),
        Asset("195930", "TIGER 유로스탁스50", "KR", "KRW", "equity"),
        Asset("192090", "TIGER 차이나CSI300", "KR", "KRW", "equity"),
        Asset("132030", "KODEX 골드선물(H)", "KR", "KRW", "gold"),
        Asset("305080", "TIGER 미국채10년선물", "KR", "KRW", "bond"),
    ],
    safe=[
        Asset("114260", "KODEX 국고채3년", "KR", "KRW", "bond"),
        Asset("273130", "KODEX 종합채권액티브", "KR", "KRW", "bond"),
    ],
    notes=(
        "국내주식형 ETF(069500·229200)의 매매차익은 비과세지만, "
        "해외지수·금·채권 ETF의 매매차익은 배당소득세 15.4% 대상입니다."
    ),
)

# ---------------------------------------------------------------------------
# US-listed ETFs — the canonical VAA / dual-momentum asset set
# ---------------------------------------------------------------------------
US_GLOBAL = Sleeve(
    key="us-global-etf",
    title="미국상장 ETF 자산배분 (VAA 계열)",
    description=(
        "Keller의 VAA 논문에서 쓰인 공격자산 4종 + 방어자산 3종 구성. "
        "유동성이 매우 크고 스프레드가 좁아 슬리피지가 작습니다."
    ),
    country="US",
    risk=[
        Asset("SPY", "S&P 500", "US", "USD", "equity"),
        Asset("VEA", "Developed ex-US", "US", "USD", "equity"),
        Asset("VWO", "Emerging Markets", "US", "USD", "equity"),
        Asset("AGG", "US Aggregate Bond", "US", "USD", "bond"),
    ],
    safe=[
        Asset("SHY", "1-3Y Treasury", "US", "USD", "cash"),
        Asset("IEF", "7-10Y Treasury", "US", "USD", "bond"),
        Asset("LQD", "Investment Grade Corp", "US", "USD", "bond"),
    ],
    notes="미국 주식 양도차익은 연 250만원 공제 후 22% 양도소득세 신고 대상입니다.",
)

US_SECTOR = Sleeve(
    key="us-sector",
    title="미국 섹터 로테이션",
    description="S&P 섹터 ETF 중 상대모멘텀 상위만 보유, 추세가 꺾이면 단기채로 회피.",
    country="US",
    risk=[
        Asset("XLK", "Technology", "US", "USD", "equity"),
        Asset("XLV", "Health Care", "US", "USD", "equity"),
        Asset("XLF", "Financials", "US", "USD", "equity"),
        Asset("XLE", "Energy", "US", "USD", "equity"),
        Asset("XLU", "Utilities", "US", "USD", "equity"),
        Asset("QQQ", "Nasdaq 100", "US", "USD", "equity"),
        Asset("IWM", "Russell 2000", "US", "USD", "equity"),
        Asset("VNQ", "US REITs", "US", "USD", "equity"),
        Asset("GLD", "Gold", "US", "USD", "gold"),
        Asset("TLT", "20Y+ Treasury", "US", "USD", "bond"),
    ],
    safe=[
        Asset("BIL", "1-3M T-Bill", "US", "USD", "cash"),
        Asset("SHY", "1-3Y Treasury", "US", "USD", "cash"),
    ],
)

KR_LARGE_CAP = Sleeve(
    key="kr-large-cap",
    title="국내 대형주 스윙",
    description=(
        "코스피 대형주 추세추종 슬리브. 개별주는 ETF보다 변동성이 크고 "
        "매도 시 증권거래세 0.15%가 붙으므로 회전율을 낮게 유지해야 합니다."
    ),
    country="KR",
    risk=[
        Asset("005930", "삼성전자", "KR", "KRW", "equity", is_etf=False),
        Asset("000660", "SK하이닉스", "KR", "KRW", "equity", is_etf=False),
        Asset("035420", "NAVER", "KR", "KRW", "equity", is_etf=False),
        Asset("005380", "현대차", "KR", "KRW", "equity", is_etf=False),
        Asset("051910", "LG화학", "KR", "KRW", "equity", is_etf=False),
    ],
    safe=[
        Asset("114260", "KODEX 국고채3년", "KR", "KRW", "bond"),
    ],
)

SLEEVES: dict[str, Sleeve] = {
    s.key: s for s in (KR_GLOBAL, US_GLOBAL, US_SECTOR, KR_LARGE_CAP)
}


def get_sleeve(key: str) -> Sleeve:
    try:
        return SLEEVES[key]
    except KeyError:
        raise KeyError(f"unknown sleeve {key!r}; available: {', '.join(SLEEVES)}") from None


def all_symbols() -> list[str]:
    """Every symbol referenced by any sleeve — what `tqt data sync` downloads."""
    seen: dict[str, None] = {}
    for sleeve in SLEEVES.values():
        for sym in sleeve.symbols:
            seen.setdefault(sym, None)
    return list(seen)


def country_of(symbol: str) -> str:
    """KR symbols are 6 digits; US symbols are alphabetic tickers."""
    return "KR" if symbol.isdigit() and len(symbol) == 6 else "US"
