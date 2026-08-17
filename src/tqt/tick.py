"""Korean exchange tick sizes (호가가격단위).

A KRX limit order whose price isn't a multiple of the tick size for its price
band is rejected with ``400 invalid-tick-size``. This is a classic way for a
freshly written bot to have every single order bounce, so rounding happens here
rather than being sprinkled through the execution code.

Bands below follow the KRX schedule effective from the January 2023 revision.
ETFs/ETNs use their own, flatter schedule. US symbols are decimal-priced and need
no tick rounding above $1, so ``round_to_tick`` passes them through.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_UP, Decimal

# (upper bound exclusive, tick). The final entry is the open-ended top band.
KRX_STOCK_TICKS: tuple[tuple[Decimal | None, Decimal], ...] = (
    (Decimal(2_000), Decimal(1)),
    (Decimal(5_000), Decimal(5)),
    (Decimal(20_000), Decimal(10)),
    (Decimal(50_000), Decimal(50)),
    (Decimal(200_000), Decimal(100)),
    (Decimal(500_000), Decimal(500)),
    (None, Decimal(1_000)),
)

# ETF / ETN: 5 KRW flat, except below 2,000 KRW where it is 1 KRW.
KRX_ETF_TICKS: tuple[tuple[Decimal | None, Decimal], ...] = (
    (Decimal(2_000), Decimal(1)),
    (None, Decimal(5)),
)


def tick_size(price: Decimal, *, is_etf: bool = False) -> Decimal:
    """Tick size applicable at ``price`` on KRX."""
    table = KRX_ETF_TICKS if is_etf else KRX_STOCK_TICKS
    for upper, tick in table:
        if upper is None or price < upper:
            return tick
    return table[-1][1]  # pragma: no cover - unreachable, table ends open


def round_to_tick(
    price: Decimal,
    *,
    country: str = "KR",
    is_etf: bool = False,
    side: str | None = None,
) -> Decimal:
    """Snap ``price`` onto a valid tick.

    ``side`` biases the rounding so the order stays *at least* as aggressive as
    intended: a BUY rounds up (willing to pay the next tick), a SELL rounds down.
    Rounding a buy limit downward can silently make it unfillable, which looks
    like a mysteriously idle bot rather than an error.
    """
    if country != "KR":
        # US equities quote in cents; anything finer is rejected above $1.
        return price.quantize(Decimal("0.01"))

    t = tick_size(price, is_etf=is_etf)
    if side == "BUY":
        rounded = (price / t).quantize(Decimal(1), rounding=ROUND_UP) * t
    elif side == "SELL":
        rounded = (price / t).quantize(Decimal(1), rounding=ROUND_DOWN) * t
    else:
        rounded = (price / t).quantize(Decimal(1)) * t

    # Crossing into a higher band can land on an invalid tick for that band;
    # re-snap once with the band the rounded price actually falls in.
    t2 = tick_size(rounded, is_etf=is_etf)
    if t2 != t:
        rounding = ROUND_UP if side == "BUY" else ROUND_DOWN
        rounded = (rounded / t2).quantize(Decimal(1), rounding=rounding) * t2
    return rounded


def is_valid_tick(price: Decimal, *, country: str = "KR", is_etf: bool = False) -> bool:
    if country != "KR":
        return price == price.quantize(Decimal("0.01"))
    return price % tick_size(price, is_etf=is_etf) == 0
