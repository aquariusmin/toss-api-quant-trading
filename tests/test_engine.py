"""Backtest engine invariants.

The first two tests are the ones that matter most. Lookahead bias and implicit
leverage are the two failure modes that make a backtest *look* excellent while
being unachievable, and neither announces itself — you only find out with real
money. So they are asserted directly rather than trusted.
"""

from decimal import Decimal

import pandas as pd

from tqt.backtest.costs import CostModel
from tqt.backtest.engine import BacktestConfig, Backtester
from tqt.strategy.base import Strategy, month_end_dates


class AllInA(Strategy):
    """Always wants 100% of the first risk asset. Records what it was shown."""

    key = "all-in-a"
    warmup_days = 0

    def __init__(self, sleeve, **kw):
        super().__init__(sleeve, **kw)
        self.seen_max_ts: list[pd.Timestamp] = []
        self.asof_seen: list[pd.Timestamp] = []

    def target_weights(self, closes, asof):
        self.seen_max_ts.append(closes.index.max())
        self.asof_seen.append(asof)
        return {"111111": Decimal(1)}


def _config(**kw) -> BacktestConfig:
    base = {
        "initial_cash": Decimal(1_000_000),
        "country": "KR",
        "etf_symbols": frozenset({"111111", "222222", "333333"}),
        "cost_model": CostModel(),
    }
    base.update(kw)
    return BacktestConfig(**base)


def test_strategy_never_sees_data_past_the_decision_date(sleeve, prices):
    """The engine slices history before calling, so peeking is impossible."""
    opens, closes = prices
    strat = AllInA(sleeve)
    Backtester(opens, closes, _config()).run(strat)

    assert strat.asof_seen, "strategy was never called"
    for shown, asof in zip(strat.seen_max_ts, strat.asof_seen, strict=True):
        assert shown <= asof, f"strategy saw {shown} while deciding on {asof}"


def test_fills_happen_at_the_next_bar_open_not_the_decision_close(sleeve, prices):
    """Fills must use the *next* bar's open.

    The fixture puts opens 5% below closes precisely so that filling at the
    decision bar's close would produce an unmistakably different price.
    """
    opens, closes = prices
    cfg = _config()
    strat = AllInA(sleeve)
    result = Backtester(opens, closes, cfg).run(strat)

    assert result.fills, "no fills produced"
    first = result.fills[0]

    decision_dates = sorted(pd.Timestamp(d) for d in month_end_dates(closes.index))
    decision = next(d for d in decision_dates if str(d.date()) < first.date)
    next_bar = closes.index[closes.index.get_loc(decision) + 1]

    assert first.date == str(next_bar.date())

    slip = cfg.cost_model.slippage_rate("KR")
    expected = Decimal(str(opens.loc[next_bar, "111111"])) * (Decimal(1) + slip)
    assert abs(first.price - expected) < Decimal("0.01")

    # And explicitly NOT the decision bar's close — the 5% open/close gap in the
    # fixture makes these two hypotheses unmistakably different.
    decision_close = Decimal(str(closes.loc[decision, "111111"]))
    assert abs(first.price - decision_close) > decision_close * Decimal("0.03")


def test_cash_never_goes_negative(sleeve, prices):
    """No implicit leverage: a backtest that borrows reports returns you can't get."""
    opens, closes = prices
    result = Backtester(opens, closes, _config()).run(AllInA(sleeve))
    assert (result.equity > 0).all()

    # Reconstruct cash from fills and confirm it stays solvent throughout.
    cash = Decimal(1_000_000)
    for f in result.fills:
        if f.side == "BUY":
            cash -= f.notional + f.cost.total
        else:
            cash += f.notional - f.cost.total
        assert cash >= Decimal("-0.01"), f"cash went negative at {f.date}: {cash}"


def test_kr_positions_are_whole_shares(sleeve, prices):
    opens, closes = prices
    result = Backtester(opens, closes, _config(country="KR")).run(AllInA(sleeve))
    for f in result.fills:
        assert f.quantity == f.quantity.to_integral_value(), f"fractional KR fill: {f}"


def test_us_allows_fractional_quantities(sleeve, prices):
    opens, closes = prices
    result = Backtester(opens, closes, _config(country="US", initial_cash=Decimal(10_000))).run(
        AllInA(sleeve)
    )
    assert result.fills
    assert any(f.quantity != f.quantity.to_integral_value() for f in result.fills)


def test_dust_threshold_is_currency_aware():
    """5,000 is a sane floor in KRW and absurd in USD. Getting this wrong once
    made every US order silently vanish as 'dust'."""
    assert _config(country="KR").dust_threshold == Decimal(5_000)
    assert _config(country="US").dust_threshold == Decimal(5)
    assert _config(country="US", min_order_value=Decimal(50)).dust_threshold == Decimal(50)


def test_costs_are_accumulated(sleeve, prices):
    opens, closes = prices
    result = Backtester(opens, closes, _config()).run(AllInA(sleeve))
    assert result.total_cost.total > 0
    assert result.total_cost.commission > 0
    # KR ETFs are tax-exempt, so the tax bucket must stay empty here.
    assert result.total_cost.tax == 0


def test_free_costs_beat_real_costs(sleeve, prices):
    """Sanity check that the cost model actually bites."""
    opens, closes = prices
    free = CostModel(
        kr_commission_rate=Decimal(0),
        slippage_bps_kr=Decimal(0),
        kr_stock_sell_tax=Decimal(0),
    )
    expensive = CostModel(kr_commission_rate=Decimal("0.01"), slippage_bps_kr=Decimal(100))
    a = Backtester(opens, closes, _config(cost_model=free)).run(AllInA(sleeve))
    b = Backtester(opens, closes, _config(cost_model=expensive)).run(AllInA(sleeve))
    assert a.equity.iloc[-1] > b.equity.iloc[-1]


def test_empty_data_raises_clearly():
    import pytest

    with pytest.raises(ValueError, match="no price data"):
        Backtester(pd.DataFrame(), pd.DataFrame(), _config())
