"""Tactical asset-allocation strategies from the published literature.

These are implemented as described in their source papers, not "improved". A rule
that has been public for a decade and still works out of sample is worth far more
than one tuned to look good on 16 years of Korean ETF data — the latter is just a
record of what already happened.

Included:

``buy-and-hold``  Equal-weight benchmark. Every other strategy must beat this
                  *after costs* or it is not earning its complexity.
``dual-momentum`` Antonacci (2014). Relative momentum picks the winners;
                  absolute momentum decides whether to be invested at all.
``vaa``           Keller & Keuning (2017). Breadth momentum: if any offensive
                  asset is falling, the whole sleeve goes defensive.
``faber``         Faber (2007). Hold each asset only while it is above its
                  10-month moving average.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .base import Strategy, momentum_13612w, month_end_dates, total_return


class BuyAndHold(Strategy):
    """Equal-weight the risk assets, rebalance yearly. The benchmark to beat."""

    key = "buy-and-hold"
    title = "동일비중 매수보유 (벤치마크)"
    warmup_days = 1

    def __init__(self, sleeve, *, rebalance: str = "Y", **kw) -> None:
        super().__init__(sleeve, rebalance=rebalance, **kw)

    def rebalance_dates(self, index) -> set:
        if self.params.get("rebalance") == "Y" and len(index):
            import pandas as pd

            s = pd.Series(index, index=index)
            return set(s.groupby(index.year).last().values)
        return month_end_dates(index)

    def target_weights(self, closes, asof) -> dict[str, Decimal]:
        available = [s for s in self.sleeve.risk_symbols if s in closes.columns]
        available = [s for s in available if closes[s].dropna().shape[0] > 0]
        if not available:
            return {}
        w = Decimal(1) / Decimal(len(available))
        self.last_reason = f"equal weight {len(available)} assets"
        return dict.fromkeys(available, w)


class DualMomentum(Strategy):
    """Antonacci's dual momentum.

    Two filters stacked:

    * **Relative** momentum — rank the risk assets by total return over
      ``lookback`` bars, take the top ``top_n``.
    * **Absolute** momentum — an asset is only held if its own return also clears
      ``min_momentum``. This is the part that matters: relative momentum alone
      will happily hold the best of a set of falling assets, which is how
      momentum strategies lose 40% in 2008.

    Anything filtered out goes to the best safe asset.
    """

    key = "dual-momentum"
    title = "듀얼 모멘텀 (상대 + 절대)"

    def __init__(
        self,
        sleeve,
        *,
        lookback: int = 252,
        top_n: int = 2,
        min_momentum: float = 0.0,
        **kw: Any,
    ) -> None:
        super().__init__(
            sleeve, lookback=lookback, top_n=top_n, min_momentum=min_momentum, **kw
        )
        self.warmup_days = lookback + 5

    def target_weights(self, closes, asof) -> dict[str, Decimal]:
        lookback = int(self.params["lookback"])
        top_n = int(self.params["top_n"])
        floor = float(self.params["min_momentum"])

        scores: list[tuple[float, str]] = []
        for sym in self.sleeve.risk_symbols:
            if sym not in closes.columns:
                continue
            r = total_return(closes[sym], lookback)
            if r is not None:
                scores.append((r, sym))

        if not scores:
            self.last_reason = "insufficient history"
            return {}

        scores.sort(reverse=True)
        picked = [(r, s) for r, s in scores[:top_n] if r > floor]

        weights: dict[str, Decimal] = {}
        if picked:
            each = Decimal(1) / Decimal(top_n)
            for _, sym in picked:
                weights[sym] = each

        invested = sum(weights.values(), Decimal(0))
        weights.update(self._safe_weights(closes, asof, Decimal(1) - invested))

        held = ", ".join(f"{s}:{r:+.1%}" for r, s in picked) or "none (all below floor)"
        self.last_reason = f"{lookback}d momentum -> {held}"
        return weights


class VAA(Strategy):
    """Vigilant Asset Allocation (Keller & Keuning, 2017).

    Uses the 13612W momentum blend and *breadth* protection: count how many of
    the offensive assets have negative momentum, and if that count reaches
    ``breadth_b``, move the entire sleeve into the single best defensive asset.
    With the default ``breadth_b=1`` this is the aggressive VAA-G4 rule — one
    falling asset anywhere in the offensive set is enough to de-risk everything.

    That sounds drastic, and it is: VAA trades more often than dual momentum and
    spends long stretches in bonds. What it buys is a much shallower drawdown,
    which for a small account is worth more than a slightly higher CAGR.
    """

    key = "vaa"
    title = "VAA (breadth momentum)"
    warmup_days = 260

    def __init__(self, sleeve, *, top_n: int = 1, breadth_b: int | None = None, **kw: Any) -> None:
        super().__init__(sleeve, top_n=top_n, breadth_b=breadth_b, **kw)

    def _breadth_threshold(self, n_assets: int) -> int:
        """How many falling assets trigger a full retreat to defensive.

        Keller's published pairs anchor this: VAA-G4 uses B=1 over 4 offensive
        assets, VAA-G12 uses B=4 over 12. So the threshold scales with universe
        size, roughly n/3 — it is *not* a constant.

        Hardcoding B=1 for a larger universe is a real trap, and measurably so:
        with 8 risk assets, "any one of eight is down" is true nearly every month,
        so the strategy sits in bonds permanently and its CAGR collapses. Scaling
        the threshold is what keeps the rule meaning "breadth has broken down"
        rather than "something, somewhere, fell".
        """
        explicit = self.params.get("breadth_b")
        if explicit:
            return int(explicit)
        return max(1, round(n_assets / 3))

    def target_weights(self, closes, asof) -> dict[str, Decimal]:
        top_n = int(self.params["top_n"])

        scores: list[tuple[float, str]] = []
        for sym in self.sleeve.risk_symbols:
            if sym not in closes.columns:
                continue
            m = momentum_13612w(closes[sym])
            if m is not None:
                scores.append((m, sym))

        if not scores:
            self.last_reason = "insufficient history"
            return {}

        breadth_b = self._breadth_threshold(len(scores))
        n_negative = sum(1 for m, _ in scores if m <= 0)
        if n_negative >= breadth_b:
            self.last_reason = (
                f"breadth: {n_negative}/{len(scores)} negative (B={breadth_b}) -> defensive"
            )
            return self._safe_weights(closes, asof, Decimal(1))

        scores.sort(reverse=True)
        picked = scores[:top_n]
        each = Decimal(1) / Decimal(len(picked))
        self.last_reason = "offensive: " + ", ".join(f"{s}:{m:+.2%}" for m, s in picked)
        return {sym: each for _, sym in picked}


class FaberTiming(Strategy):
    """Faber (2007): hold an asset only while it trades above its 10-month SMA.

    The simplest trend filter that has held up out of sample. It does not try to
    pick winners — it only decides, per asset, invested or not. Weight is split
    equally across the risk assets currently in an uptrend, and whatever is left
    over goes to the safe asset.
    """

    key = "faber"
    title = "Faber 이동평균 타이밍"

    def __init__(self, sleeve, *, sma_days: int = 200, **kw: Any) -> None:
        super().__init__(sleeve, sma_days=sma_days, **kw)
        self.warmup_days = sma_days + 5

    def target_weights(self, closes, asof) -> dict[str, Decimal]:
        sma_days = int(self.params["sma_days"])
        candidates = [s for s in self.sleeve.risk_symbols if s in closes.columns]
        if not candidates:
            return {}

        in_trend: list[str] = []
        for sym in candidates:
            series = closes[sym].dropna()
            if len(series) < sma_days:
                continue
            sma = series.iloc[-sma_days:].mean()
            if float(series.iloc[-1]) > float(sma):
                in_trend.append(sym)

        n_slots = len(candidates)
        weights: dict[str, Decimal] = {}
        if in_trend:
            each = Decimal(1) / Decimal(n_slots)
            for sym in in_trend:
                weights[sym] = each

        invested = sum(weights.values(), Decimal(0))
        weights.update(self._safe_weights(closes, asof, Decimal(1) - invested))
        self.last_reason = f"{len(in_trend)}/{n_slots} above {sma_days}d SMA"
        return weights


STRATEGIES: dict[str, type[Strategy]] = {
    BuyAndHold.key: BuyAndHold,
    DualMomentum.key: DualMomentum,
    VAA.key: VAA,
    FaberTiming.key: FaberTiming,
}


def build_strategy(key: str, sleeve, **params) -> Strategy:
    try:
        cls = STRATEGIES[key]
    except KeyError:
        raise KeyError(
            f"unknown strategy {key!r}; available: {', '.join(STRATEGIES)}"
        ) from None
    return cls(sleeve, **params)
