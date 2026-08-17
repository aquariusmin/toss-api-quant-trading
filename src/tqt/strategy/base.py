"""Strategy interface.

A strategy's only job is to answer one question: *given history up to today, what
fraction of the portfolio should sit in each symbol tomorrow?* It returns target
weights, never orders. Order sizing, cash management, tick rounding and risk
limits are someone else's problem — which is what lets the identical strategy
object drive a backtest, a paper run, and live trading with no code changes. If
paper and live disagreed about strategy logic, the paper stage would be
worthless.

Weights are ``Decimal`` in [0, 1] and should sum to <= 1. Anything unallocated is
held as cash.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ..universe import Sleeve


class Strategy(ABC):
    key: str = "base"
    title: str = ""
    #: Bars of history needed before the first decision is meaningful.
    warmup_days: int = 0

    def __init__(self, sleeve: Sleeve, **params: Any) -> None:
        self.sleeve = sleeve
        self.params: dict[str, Any] = params
        self.last_reason: str = ""

    # ------------------------------------------------------------------
    @abstractmethod
    def target_weights(self, closes: pd.DataFrame, asof: Any) -> dict[str, Decimal]:
        """Target weights given closes **up to and including** ``asof``.

        The engine slices the frame before calling, so indexing past ``asof`` is
        impossible by construction.
        """

    def rebalance_dates(self, index) -> set:
        """Month-end by default.

        Monthly is the standard cadence in the tactical-allocation literature and
        keeps turnover — and therefore cost — low. Weekly roughly quadruples cost
        for typically marginal signal improvement.
        """
        return month_end_dates(index)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        bits = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.key}({bits})"

    def _safe_weights(self, closes: pd.DataFrame, asof, weight: Decimal) -> dict[str, Decimal]:
        """Park ``weight`` in the best-performing available safe asset.

        "Best" rather than "first" matters: in 2022 both stocks and long bonds
        fell, and a strategy hardwired to hide in IEF lost money doing it while
        SHY was flat.
        """
        if weight <= 0:
            return {}
        candidates = [s for s in self.sleeve.safe_symbols if s in closes.columns]
        if not candidates:
            return {}
        # Ranked on a 126-day lookback, not 63: a short lookback makes the
        # "safest safe asset" flip between SHY/IEF/LQD most months, and each flip
        # is a real round-trip cost paid for no signal. Smoothing the selection
        # cut measured turnover materially.
        lookback = 126
        scored: list[tuple[float, str]] = []
        for sym in candidates:
            series = closes[sym].dropna()
            if len(series) <= lookback:
                continue
            scored.append((float(series.iloc[-1] / series.iloc[-1 - lookback] - 1), sym))
        best = max(scored)[1] if scored else candidates[0]
        return {best: weight}


def month_end_dates(index) -> set:
    """Last available trading day of each month in ``index``.

    Returns ``pd.Timestamp`` values, not ``numpy.datetime64``. The engine tests
    membership with the Timestamps it iterates from the index; those two types
    happen to hash equal today, but depending on that across pandas/numpy
    versions is a silent-breakage risk — a rebalance set that stops matching
    would make the strategy simply never trade.
    """
    import pandas as pd

    if len(index) == 0:
        return set()
    s = pd.Series(index, index=index)
    return {pd.Timestamp(v) for v in s.groupby([index.year, index.month]).last().values}


def week_end_dates(index) -> set:
    """Last available trading day of each ISO week in ``index``."""
    import pandas as pd

    if len(index) == 0:
        return set()
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return {
        pd.Timestamp(v)
        for v in s.groupby([iso.year.values, iso.week.values]).last().values
    }


def total_return(series, lookback: int) -> float | None:
    """Simple total return over ``lookback`` bars, or None if history is short."""
    clean = series.dropna()
    if len(clean) <= lookback:
        return None
    past = clean.iloc[-1 - lookback]
    if past <= 0:
        return None
    return float(clean.iloc[-1] / past - 1)


def momentum_13612w(series) -> float | None:
    """Keller's 13612W score: a weighted blend of 1/3/6/12-month returns.

    Weighting the shortest lookback most heavily makes the signal react faster
    than plain 12-month momentum without the whipsaw of using 1-month alone.
    """
    r1 = total_return(series, 21)
    r3 = total_return(series, 63)
    r6 = total_return(series, 126)
    r12 = total_return(series, 252)
    if None in (r1, r3, r6, r12):
        return None
    return (12 * r1 + 4 * r3 + 2 * r6 + 1 * r12) / 19.0
