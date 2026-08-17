"""KRX tick sizes. An off-tick limit price is rejected outright, so this is the
difference between a bot that trades and one whose every order bounces."""

from decimal import Decimal

import pytest

from tqt.tick import is_valid_tick, round_to_tick, tick_size


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (1_500, 1),
        (1_999, 1),
        (2_000, 5),
        (4_999, 5),
        (5_000, 10),
        (19_999, 10),
        (20_000, 50),
        (49_999, 50),
        (50_000, 100),
        (199_999, 100),
        (200_000, 500),
        (499_999, 500),
        (500_000, 1_000),
        (1_200_000, 1_000),
    ],
)
def test_stock_tick_bands(price, expected):
    assert tick_size(Decimal(price)) == Decimal(expected)


def test_etf_ticks_are_flat_five_won():
    # ETFs use their own flatter schedule regardless of price band.
    assert tick_size(Decimal(50_000), is_etf=True) == Decimal(5)
    assert tick_size(Decimal(300_000), is_etf=True) == Decimal(5)
    assert tick_size(Decimal(1_500), is_etf=True) == Decimal(1)


def test_buy_rounds_up_and_sell_rounds_down():
    """Rounding must never make an order less aggressive than intended.

    A buy limit rounded *down* below the tick can sit unfilled forever, which
    presents as an idle bot rather than an error.
    """
    px = Decimal("74_321".replace("_", ""))
    buy = round_to_tick(px, country="KR", side="BUY")
    sell = round_to_tick(px, country="KR", side="SELL")
    assert buy == Decimal(74_400)  # up to the next 100 KRW tick
    assert sell == Decimal(74_300)
    assert buy > px > sell


def test_rounded_prices_are_always_valid_ticks():
    for raw in (1_234, 2_001, 19_998, 49_991, 199_950, 499_999, 762_345):
        for side in ("BUY", "SELL"):
            out = round_to_tick(Decimal(raw), country="KR", side=side)
            assert is_valid_tick(out, country="KR"), (raw, side, out)


def test_band_crossing_resnaps():
    """Rounding up across a band boundary must re-snap to the new band's tick.

    49,980 -> BUY rounds up to 50,000, which is the first price in the 100-KRW
    band; a naive single-pass round could land on 49,990+50 = an invalid tick.
    """
    out = round_to_tick(Decimal(49_980), country="KR", side="BUY")
    assert is_valid_tick(out, country="KR")
    assert out >= Decimal(49_980)


def test_us_prices_quantize_to_cents():
    assert round_to_tick(Decimal("307.0837"), country="US") == Decimal("307.08")
    assert is_valid_tick(Decimal("307.08"), country="US")
    assert not is_valid_tick(Decimal("307.0837"), country="US")
