"""Performance metrics.

Reported deliberately as a set rather than a single number. CAGR alone hides the
drawdown you must actually live through; Sharpe alone hides skew; and both hide
whether the result rests on three lucky months. The ones that decide whether a
strategy is worth real money here are **max drawdown**, **Calmar**, and the
**out-of-sample** figures from ``walk_forward``.

On overfitting: with enough parameter tries, an in-sample Sharpe of 1.5 is easy
and meaningless (Bailey & López de Prado, *The Deflated Sharpe Ratio*, 2014).
``deflated_sharpe`` adjusts for how many variants you tested — if you tried 20
lookbacks and picked the best, pass ``n_trials=20`` and watch the number fall.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(drawdown_series(equity).min())


def drawdown_details(equity: pd.Series) -> dict[str, Any]:
    """Depth, dates, and the longest underwater stretch in calendar days."""
    if len(equity) < 2:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None, "longest_days": 0}
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()

    underwater = dd < -1e-12
    longest = 0
    run_start = None
    for ts, is_under in underwater.items():
        if is_under and run_start is None:
            run_start = ts
        elif not is_under and run_start is not None:
            longest = max(longest, (ts - run_start).days)
            run_start = None
    if run_start is not None:
        longest = max(longest, (underwater.index[-1] - run_start).days)

    recovered = equity.loc[trough:] >= equity.loc[peak]
    recovery_date = recovered[recovered].index[0] if recovered.any() else None

    return {
        "max_drawdown": float(dd.min()),
        "peak_date": str(peak.date()),
        "trough_date": str(trough.date()),
        "recovery_date": str(recovery_date.date()) if recovery_date is not None else None,
        "longest_days": int(longest),
    }


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start_v, end_v = float(equity.iloc[0]), float(equity.iloc[-1])
    if start_v <= 0 or end_v <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (end_v / start_v) ** (1 / years) - 1


def annual_vol(equity: pd.Series) -> float:
    r = daily_returns(equity)
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 2 else 0.0


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    r = daily_returns(equity)
    if len(r) < 3:
        return 0.0
    excess = r - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(excess.mean() / sd * math.sqrt(TRADING_DAYS))


def sortino(equity: pd.Series, rf: float = 0.0) -> float:
    """Like Sharpe but penalises only downside deviation."""
    r = daily_returns(equity)
    if len(r) < 3:
        return 0.0
    excess = r - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if len(downside) < 2 or dd == 0 or not np.isfinite(dd):
        return 0.0
    return float(excess.mean() / dd * math.sqrt(TRADING_DAYS))


def calmar(equity: pd.Series) -> float:
    """CAGR per unit of max drawdown — the "can I sleep at night" ratio."""
    mdd = abs(max_drawdown(equity))
    return float(cagr(equity) / mdd) if mdd > 1e-9 else 0.0


def ulcer_index(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(math.sqrt((dd**2).mean())) if len(dd) else 0.0


def monthly_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    return equity.resample("ME").last().pct_change().dropna()


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_obs: int) -> float:
    """Haircut a Sharpe ratio for multiple-testing bias.

    Subtracts the Sharpe you'd expect from the *best* of ``n_trials`` random
    strategies. If you grid-searched 50 parameter sets, the winner's Sharpe is
    biased upward and this shows by how much.
    """
    if n_trials < 1 or n_obs < 3:
        return observed_sharpe
    if n_trials == 1:
        return observed_sharpe
    euler = 0.5772156649
    # Expected maximum of n_trials standard normals.
    e_max = (1 - euler) * _norm_ppf(1 - 1.0 / n_trials) + euler * _norm_ppf(
        1 - 1.0 / (n_trials * math.e)
    )
    se = math.sqrt((1 + 0.5 * observed_sharpe**2) / max(n_obs - 1, 1))
    return observed_sharpe - e_max * se * math.sqrt(TRADING_DAYS) / math.sqrt(TRADING_DAYS)


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    if not 0 < p < 1:
        return 0.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
          3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            (((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            (((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (
        ((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def summarize(
    equity: pd.Series, *, rf: float = 0.0, n_trials: int = 1, benchmark: pd.Series | None = None
) -> dict[str, Any]:
    """The full metric set for one equity curve."""
    if equity.empty:
        return {"error": "empty equity curve"}

    mr = monthly_returns(equity)
    dd = drawdown_details(equity)
    sr = sharpe(equity, rf)

    out: dict[str, Any] = {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "days": int((equity.index[-1] - equity.index[0]).days),
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": cagr(equity),
        "annual_vol": annual_vol(equity),
        "sharpe": sr,
        "sortino": sortino(equity, rf),
        "max_drawdown": dd["max_drawdown"],
        "mdd_peak": dd["peak_date"],
        "mdd_trough": dd["trough_date"],
        "mdd_recovery": dd["recovery_date"],
        "mdd_longest_days": dd["longest_days"],
        "calmar": calmar(equity),
        "ulcer_index": ulcer_index(equity),
        "monthly_win_rate": float((mr > 0).mean()) if len(mr) else 0.0,
        "best_month": float(mr.max()) if len(mr) else 0.0,
        "worst_month": float(mr.min()) if len(mr) else 0.0,
        "n_months": int(len(mr)),
    }
    if n_trials > 1:
        out["deflated_sharpe"] = deflated_sharpe(sr, n_trials, len(equity))
        out["n_trials"] = n_trials

    if benchmark is not None and not benchmark.empty:
        aligned = pd.concat([equity, benchmark], axis=1, join="inner").dropna()
        if len(aligned) > 3:
            sr_b = aligned.iloc[:, 0].pct_change().dropna()
            bm_b = aligned.iloc[:, 1].pct_change().dropna()
            cov = np.cov(sr_b, bm_b)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else 0.0
            out["benchmark_cagr"] = cagr(aligned.iloc[:, 1])
            out["benchmark_mdd"] = max_drawdown(aligned.iloc[:, 1])
            out["excess_cagr"] = out["cagr"] - out["benchmark_cagr"]
            out["beta_to_benchmark"] = beta
    return out


# ---------------------------------------------------------------------------
# Korean tax estimate
# ---------------------------------------------------------------------------
def after_tax_summary(
    equity: pd.Series,
    *,
    asset_class: str = "kr_domestic_etf",
    annual_deduction_krw: float = 2_500_000,
) -> dict[str, Any]:
    """Rough after-tax annual outcome for a Korean resident investor.

    Not tax advice, and deliberately conservative. The point is to make visible
    that the same gross return nets very differently by wrapper:

    * ``kr_domestic_etf`` — 국내주식형 ETF: price gains untaxed.
    * ``kr_foreign_etf``  — 해외지수/금/채권 ETF: gains taxed as 배당소득 15.4%,
      and they count toward the 2,000만원 금융소득종합과세 threshold.
    * ``us_stock``        — 해외주식 양도소득: 22% on gains above a 2.5M KRW
      annual deduction, self-reported each May.
    """
    rates = {"kr_domestic_etf": 0.0, "kr_foreign_etf": 0.154, "us_stock": 0.22}
    if asset_class not in rates:
        raise ValueError(f"asset_class must be one of {list(rates)}")
    rate = rates[asset_class]
    deduction = annual_deduction_krw if asset_class == "us_stock" else 0.0

    yearly = equity.resample("YE").last()
    if len(yearly) < 2:
        yearly = pd.concat([equity.iloc[[0]], equity.iloc[[-1]]])

    gross_total = 0.0
    tax_total = 0.0
    prev = float(equity.iloc[0])
    for v in yearly:
        gain = float(v) - prev
        gross_total += gain
        if gain > 0:
            taxable = max(gain - deduction, 0.0)
            tax_total += taxable * rate
        prev = float(v)

    net = gross_total - tax_total
    start_v = float(equity.iloc[0])
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    net_cagr = ((start_v + net) / start_v) ** (1 / years) - 1 if start_v > 0 else 0.0

    return {
        "asset_class": asset_class,
        "tax_rate": rate,
        "gross_gain": gross_total,
        "estimated_tax": tax_total,
        "net_gain": net,
        "gross_cagr": cagr(equity),
        "net_cagr": net_cagr,
        "tax_drag_pct_points": (cagr(equity) - net_cagr) * 100,
    }


def compare(results: dict[str, pd.Series], *, rf: float = 0.0) -> pd.DataFrame:
    """Side-by-side metric table for several equity curves."""
    rows = {}
    for name, eq in results.items():
        m = summarize(eq, rf=rf)
        rows[name] = {
            "CAGR": m.get("cagr", 0) * 100,
            "Vol": m.get("annual_vol", 0) * 100,
            "Sharpe": m.get("sharpe", 0),
            "Sortino": m.get("sortino", 0),
            "MDD": m.get("max_drawdown", 0) * 100,
            "Calmar": m.get("calmar", 0),
            "WinMo%": m.get("monthly_win_rate", 0) * 100,
            "WorstMo": m.get("worst_month", 0) * 100,
        }
    return pd.DataFrame(rows).T.round(2)
