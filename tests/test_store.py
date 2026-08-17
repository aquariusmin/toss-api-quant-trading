"""Store round-trips. Money is stored as TEXT so a position's cost basis stays
exact; if it ever silently became REAL, these tests fail."""

from decimal import Decimal

from tqt.data.store import Store, d, s
from tqt.toss.models import Candle


def _candle(date: str, close: str = "105") -> Candle:
    return Candle.model_validate(
        {
            "timestamp": f"{date}T00:00:00.000+09:00",
            "openPrice": "100",
            "highPrice": "110",
            "lowPrice": "90",
            "closePrice": close,
            "volume": "1000",
            "currency": "KRW",
        }
    )


def test_decimal_survives_a_round_trip_exactly(store: Store):
    """0.1 + 0.2 must never creep into a P&L figure."""
    tricky = Decimal("74321.123456789012345")
    store.set_state("x", s(tricky))
    assert d(store.get_state("x")) == tricky

    store.record_equity("paper", tricky, Decimal("1.005"), Decimal("0.0001"))
    row = store.equity_curve("paper")[0]
    assert d(row["equity_krw"]) == tricky


def test_candles_upsert_is_idempotent(store: Store):
    batch = [_candle("2026-01-01"), _candle("2026-01-02")]
    store.upsert_candles("005930", "1d", batch)
    store.upsert_candles("005930", "1d", batch)
    lo, hi, n = store.candle_range("005930", "1d")
    assert n == 2
    assert lo.startswith("2026-01-01")
    assert hi.startswith("2026-01-02")


def test_candle_upsert_overwrites_a_partial_bar(store: Store):
    """The last bar of a live session is incomplete; a later sync must correct it."""
    store.upsert_candles("005930", "1d", [_candle("2026-01-02", close="100")])
    store.upsert_candles("005930", "1d", [_candle("2026-01-02", close="123")])
    rows = store.load_candles("005930", "1d")
    assert len(rows) == 1
    assert d(rows[0]["close"]) == Decimal(123)


def test_load_frame_pivots_and_leaves_gaps_as_nan(store: Store):
    """A symbol that didn't trade must be NaN, never a fabricated price."""
    store.upsert_candles("AAA", "1d", [_candle("2026-01-01"), _candle("2026-01-02")])
    store.upsert_candles("BBB", "1d", [_candle("2026-01-02")])
    df = store.load_frame(["AAA", "BBB"], "1d")
    assert list(df.columns) == ["AAA", "BBB"]
    assert len(df) == 2
    assert df["BBB"].isna().sum() == 1


def test_load_frame_rejects_an_unknown_field(store: Store):
    import pytest

    store.upsert_candles("AAA", "1d", [_candle("2026-01-01")])
    with pytest.raises(ValueError, match="invalid field"):
        store.load_frame(["AAA"], "1d", field="close; DROP TABLE candles")


def test_state_round_trips_json_and_plain_strings(store: Store):
    store.set_state("dict", {"a": 1})
    store.set_state("plain", "hello")
    assert store.get_state("dict") == {"a": 1}
    assert store.get_state("plain") == "hello"
    assert store.get_state("missing", "fallback") == "fallback"


def test_first_equity_of_day_is_the_earliest_mark(store: Store):
    """This is the daily-loss kill switch's baseline, so ordering matters."""
    for hh, eq in (("09", 1_000_000), ("12", 990_000), ("15", 980_000)):
        store.record_equity(
            "paper", Decimal(eq), Decimal(eq), Decimal(0), ts=f"2026-08-18T{hh}:00:00+00:00"
        )
    assert store.first_equity_of_day("paper", "2026-08-18") == Decimal(1_000_000)
    assert store.first_equity_of_day("paper", "2026-08-19") is None


def test_events_come_back_newest_first(store: Store):
    for i in range(3):
        store.log_event("INFO", "test", f"msg{i}")
    rows = store.recent_events(10)
    assert rows[0]["message"] == "msg2"


def test_brokers_are_isolated(store: Store):
    """A paper run must never contaminate live statistics."""
    store.record_equity("paper", Decimal(1), Decimal(1), Decimal(0))
    store.record_equity("live", Decimal(2), Decimal(2), Decimal(0))
    assert len(store.equity_curve("paper")) == 1
    assert len(store.equity_curve("live")) == 1
    assert d(store.equity_curve("live")[0]["equity_krw"]) == Decimal(2)


def test_transaction_rolls_back_on_error(store: Store):
    try:
        with store.transaction():
            store.conn.execute(
                "INSERT INTO state(key,value,updated_at) VALUES('k','v','t')"
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert store.get_state("k") is None


def test_signals_record_and_read_back(store: Store):
    store.record_signals("2026-08-18T15:00:00", "portfolio", {"069500": Decimal("0.25")}, "why")
    rows = store.latest_signals()
    assert rows[0]["symbol"] == "069500"
    assert d(rows[0]["target_weight"]) == Decimal("0.25")
