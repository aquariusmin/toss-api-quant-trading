"""Metrics against known answers, and strategy logic against its source papers."""

import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from tqt.backtest.metrics import (
    after_tax_summary,
    cagr,
    calmar,
    deflated_sharpe,
    drawdown_details,
    max_drawdown,
    monthly_returns,
    sharpe,
    summarize,
)
from tqt.portfolio import Allocation, PortfolioPlan
from tqt.strategy.base import momentum_13612w, month_end_dates
from tqt.strategy.momentum import VAA, BuyAndHold, DualMomentum, FaberTiming


# ---------------------------------------------------------------------------
# metrics — known answers
# ---------------------------------------------------------------------------
def _series(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_cagr_of_exact_doubling_over_one_year():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01"])
    eq = pd.Series([100.0, 200.0], index=idx)
    assert cagr(eq) == pytest.approx(1.0, abs=0.01)


def test_monotonic_growth_has_no_drawdown():
    eq = _series([100 * 1.001**i for i in range(300)])
    assert max_drawdown(eq) == pytest.approx(0.0, abs=1e-12)
    assert calmar(eq) == 0.0  # undefined with zero MDD; reported as 0, not inf


def test_max_drawdown_is_measured_peak_to_trough():
    eq = _series([100, 120, 60, 80, 130])
    assert max_drawdown(eq) == pytest.approx(-0.5)  # 120 -> 60
    det = drawdown_details(eq)
    assert det["max_drawdown"] == pytest.approx(-0.5)
    assert det["recovery_date"] is not None  # it did recover past 120


def test_drawdown_without_recovery_reports_none():
    eq = _series([100, 120, 60, 70])
    assert drawdown_details(eq)["recovery_date"] is None


def test_sharpe_of_a_flat_curve_is_zero_not_nan():
    eq = _series([100.0] * 100)
    assert sharpe(eq) == 0.0
    assert math.isfinite(sharpe(eq))


def test_risk_free_rate_reduces_sharpe():
    rng = np.random.default_rng(42)
    eq = _series(100 * np.cumprod(1 + rng.normal(0.0006, 0.01, 500)))
    assert sharpe(eq, rf=0.0) > sharpe(eq, rf=0.05)


def test_deflated_sharpe_penalises_multiple_testing():
    """Try 50 parameter sets and the winner's Sharpe is biased upward."""
    assert deflated_sharpe(1.5, n_trials=1, n_obs=1000) == 1.5
    assert deflated_sharpe(1.5, n_trials=50, n_obs=1000) < 1.5


def test_summarize_handles_an_empty_curve():
    assert "error" in summarize(pd.Series(dtype=float))


def test_monthly_returns_length():
    eq = _series([100 * 1.0005**i for i in range(260)])
    assert 10 <= len(monthly_returns(eq)) <= 13


def test_domestic_etf_gains_are_untaxed_while_us_gains_are_not():
    """Same gross return, very different net — the point of this function.

    Uses a 100M KRW account so annual gains clear the 2.5M KRW 양도소득 deduction;
    see the next test for the small-account case.
    """
    eq = _series([100_000_000 * 1.0005**i for i in range(600)])
    kr = after_tax_summary(eq, asset_class="kr_domestic_etf")
    us = after_tax_summary(eq, asset_class="us_stock")
    foreign = after_tax_summary(eq, asset_class="kr_foreign_etf")

    assert kr["estimated_tax"] == 0
    assert kr["net_cagr"] == pytest.approx(kr["gross_cagr"])
    assert us["estimated_tax"] > 0
    assert us["net_cagr"] < us["gross_cagr"]
    # 15.4% 배당소득세 on foreign-index ETFs vs 22% 양도소득세 on US stock.
    assert 0 < foreign["estimated_tax"] < us["estimated_tax"]


def test_small_account_us_gains_fall_under_the_annual_deduction():
    """A genuinely useful consequence for a ~1M KRW seed: the 2,500,000 KRW annual
    deduction wipes out US capital-gains tax entirely, so tax is not a reason to
    avoid US-listed assets at this size. Costs and FX spread still are."""
    eq = _series([1_000_000 * 1.0005**i for i in range(600)])
    us = after_tax_summary(eq, asset_class="us_stock")
    assert us["gross_gain"] > 0
    assert us["estimated_tax"] == 0
    assert us["net_cagr"] == pytest.approx(us["gross_cagr"])


def test_after_tax_rejects_unknown_asset_class():
    eq = _series([100.0, 101.0])
    with pytest.raises(ValueError, match="asset_class"):
        after_tax_summary(eq, asset_class="crypto")


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------
def test_month_end_dates_pick_the_last_session_of_each_month():
    idx = pd.bdate_range("2026-01-01", "2026-03-31")
    ends = {pd.Timestamp(x) for x in month_end_dates(idx)}
    assert pd.Timestamp("2026-01-30") in ends  # last business day of Jan 2026
    assert len(ends) == 3


def test_13612w_is_positive_for_an_uptrend_and_negative_for_a_downtrend():
    up = pd.Series(np.linspace(100, 200, 300))
    down = pd.Series(np.linspace(200, 100, 300))
    assert momentum_13612w(up) > 0
    assert momentum_13612w(down) < 0
    assert momentum_13612w(pd.Series(np.linspace(100, 110, 50))) is None  # too short


def test_weights_never_exceed_one(sleeve, prices):
    _, closes = prices
    for strat in (
        BuyAndHold(sleeve),
        FaberTiming(sleeve),
        DualMomentum(sleeve, top_n=2),
        VAA(sleeve),
    ):
        w = strat.target_weights(closes, closes.index[-1])
        assert sum(w.values(), Decimal(0)) <= Decimal("1.0001"), strat.key
        assert all(v >= 0 for v in w.values()), strat.key


def test_absolute_momentum_avoids_a_falling_asset(sleeve, prices):
    """Relative momentum alone would hold the best of a set of falling assets.
    The absolute filter is what stops that."""
    _, closes = prices
    # Force everything downward so no asset clears the floor.
    falling = closes.copy()
    falling["111111"] = np.linspace(20_000, 10_000, len(falling))
    falling["222222"] = np.linspace(20_000, 12_000, len(falling))

    strat = DualMomentum(sleeve, top_n=2, min_momentum=0.0)
    w = strat.target_weights(falling, falling.index[-1])
    assert "111111" not in w and "222222" not in w
    assert "333333" in w  # parked in the safe asset instead


def test_faber_holds_only_assets_above_their_moving_average(sleeve, prices):
    _, closes = prices
    strat = FaberTiming(sleeve, sma_days=100)
    w = strat.target_weights(closes, closes.index[-1])
    # Fixture: 111111 trends up, 222222 trends down.
    assert w.get("111111", Decimal(0)) > 0
    assert w.get("222222", Decimal(0)) == 0


def test_vaa_breadth_threshold_scales_with_universe_size(sleeve):
    """Keller's own pairs anchor this: B=1 over 4 assets, B=4 over 12. Hardcoding
    B=1 for a large universe makes the strategy permanently defensive."""
    v = VAA(sleeve)
    assert v._breadth_threshold(4) == 1
    assert v._breadth_threshold(8) == 3
    assert v._breadth_threshold(12) == 4
    # An explicit override still wins.
    assert VAA(sleeve, breadth_b=1)._breadth_threshold(12) == 1


def test_strategies_survive_all_nan_history(sleeve, prices):
    _, closes = prices
    empty = closes.copy()
    empty[:] = np.nan
    for strat in (BuyAndHold(sleeve), FaberTiming(sleeve), DualMomentum(sleeve), VAA(sleeve)):
        w = strat.target_weights(empty, empty.index[-1])
        assert sum(w.values(), Decimal(0)) <= Decimal("1.0001")


# ---------------------------------------------------------------------------
# portfolio plan
# ---------------------------------------------------------------------------
def test_plan_rejects_weights_that_would_need_leverage():
    with pytest.raises(ValueError, match="leverage"):
        PortfolioPlan(
            allocations=[
                Allocation("kr-global-etf", "faber", Decimal("0.7")),
                Allocation("us-global-etf", "buy-and-hold", Decimal("0.7")),
            ]
        )


def test_plan_rejects_negative_weights():
    with pytest.raises(ValueError, match="short selling"):
        PortfolioPlan(allocations=[Allocation("kr-global-etf", "faber", Decimal("-0.1"))])


def test_plan_reports_the_uninvested_remainder_as_cash():
    plan = PortfolioPlan(
        allocations=[Allocation("kr-global-etf", "faber", Decimal("0.60"))]
    )
    assert plan.cash_weight == Decimal("0.40")
    assert "cash" in plan.describe()


def test_default_plan_is_valid_and_fully_allocated():
    from tqt.portfolio import DEFAULT_PLAN

    assert DEFAULT_PLAN.cash_weight == Decimal(0)
    assert len(DEFAULT_PLAN.symbols) > 5


def test_shipped_plan_file_parses():
    """config/portfolio.toml is user-facing; a typo there must fail loudly here."""
    from tqt.portfolio import DEFAULT_PLAN_PATH, load_plan

    if DEFAULT_PLAN_PATH.exists():
        plan = load_plan(DEFAULT_PLAN_PATH)
        assert plan.allocations
        assert plan.cash_weight >= 0
