"""Out-of-sample validation: walk-forward testing and honest parameter sweeps.

A single full-sample backtest tells you almost nothing about whether a strategy
will work next year — it tells you what would have worked over a period you can
already see. These two tools are the cheapest available defence:

* ``walk_forward`` splits history into consecutive train/test folds and stitches
  together **only the test segments**. The resulting curve is what the strategy
  would have delivered while genuinely blind to the future. If OOS CAGR is a
  fraction of in-sample CAGR, the strategy is fitted, not found.

* ``parameter_sweep`` runs a grid and reports the *deflated* Sharpe alongside the
  raw one. Try 30 lookbacks and the best will look good by luck alone; the
  deflated figure prices that in (Bailey & López de Prado, 2014).

The rule of thumb worth keeping: prefer the parameter value sitting in a broad
*plateau* of decent results over the lone spike. A spike is usually noise, and it
will not be there next year.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from .engine import BacktestConfig, Backtester
from .metrics import deflated_sharpe, summarize

log = logging.getLogger(__name__)

StrategyFactory = Callable[..., Any]


def walk_forward(
    factory: StrategyFactory,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    config: BacktestConfig,
    *,
    train_years: float = 5.0,
    test_years: float = 2.0,
    strategy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Anchored-start walk-forward.

    Each fold trains on everything up to a cut-off and is evaluated on the
    following ``test_years``. Note that these strategies have no fitted
    parameters — the "training" window exists to supply warmup history, so the
    test segments are true OOS by construction. When you *do* start tuning
    parameters, refit inside ``factory`` per fold and this becomes a real
    walk-forward optimisation.
    """
    kwargs = strategy_kwargs or {}
    if closes.empty:
        return {"error": "no data"}

    index = closes.index
    start, end = index[0], index[-1]
    train_td = pd.Timedelta(days=int(train_years * 365.25))
    test_td = pd.Timedelta(days=int(test_years * 365.25))

    folds: list[dict[str, Any]] = []
    oos_returns: list[pd.Series] = []

    cut = start + train_td
    while cut < end:
        test_end = min(cut + test_td, end)
        # Feed the full history up to test_end so indicators are warm, but only
        # score the slice after the cut-off.
        o = opens.loc[:test_end]
        c = closes.loc[:test_end]
        if len(c) < 60:
            break

        strat = factory(**kwargs)
        result = Backtester(o, c, config).run(strat)
        eq = result.equity
        test_eq = eq.loc[eq.index > cut]
        if len(test_eq) > 5:
            seg = test_eq.pct_change().dropna()
            oos_returns.append(seg)
            folds.append(
                {
                    "test_start": str(test_eq.index[0].date()),
                    "test_end": str(test_eq.index[-1].date()),
                    **{
                        k: v
                        for k, v in summarize(test_eq).items()
                        if k in {"cagr", "sharpe", "max_drawdown", "total_return"}
                    },
                    "n_trades": len(
                        [f for f in result.fills if f.date > str(cut.date())]
                    ),
                }
            )
        cut = test_end

    if not oos_returns:
        return {"error": "history too short for the requested train/test split"}

    stitched = pd.concat(oos_returns).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    oos_equity = float(config.initial_cash) * (1 + stitched).cumprod()

    in_sample = Backtester(opens, closes, config).run(factory(**kwargs)).equity

    oos = summarize(oos_equity)
    ins = summarize(in_sample)
    return {
        "folds": folds,
        "n_folds": len(folds),
        "oos": oos,
        "in_sample": ins,
        "oos_equity": oos_equity,
        # >1 means OOS held up; well under 1 means the full-sample result was
        # flattered by hindsight.
        "decay_ratio_cagr": (oos.get("cagr", 0) / ins["cagr"]) if ins.get("cagr") else 0.0,
        "decay_ratio_sharpe": (
            (oos.get("sharpe", 0) / ins["sharpe"]) if ins.get("sharpe") else 0.0
        ),
    }


def parameter_sweep(
    factory: StrategyFactory,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    config: BacktestConfig,
    *,
    param: str,
    values: Sequence[Any],
    base_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Grid one parameter and report raw + deflated Sharpe.

    Read the returned table for a *plateau*, not a maximum.
    """
    base = dict(base_kwargs or {})
    rows: list[dict[str, Any]] = []
    n_trials = len(values)

    for v in values:
        strat = factory(**{**base, param: v})
        try:
            result = Backtester(opens, closes, config).run(strat)
        except Exception as exc:  # a bad parameter shouldn't kill the sweep
            log.warning("sweep %s=%s failed: %s", param, v, exc)
            continue
        m = result.metrics()
        rows.append(
            {
                param: v,
                "CAGR%": round(m.get("cagr", 0) * 100, 2),
                "Sharpe": round(m.get("sharpe", 0), 3),
                "DeflSharpe": round(
                    deflated_sharpe(m.get("sharpe", 0), n_trials, len(result.equity)), 3
                ),
                "MDD%": round(m.get("max_drawdown", 0) * 100, 2),
                "Calmar": round(m.get("calmar", 0), 3),
                "trades": m.get("n_trades", 0),
                "cost%": round(m.get("cost_pct_of_initial", 0), 2),
            }
        )
    return pd.DataFrame(rows)
