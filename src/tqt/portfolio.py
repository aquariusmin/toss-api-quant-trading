"""Multi-sleeve portfolio: combine several strategies into one target book.

Each *allocation* is (sleeve, strategy, weight). Running two or three
uncorrelated rules side by side is the cheapest diversification available — when
one signal is in a drawdown the others usually aren't, and the blended equity
curve is smoother than any single component. It also means one strategy quietly
breaking does not take the whole account with it.

Weights are fractions of total capital and must sum to <= 1. Any remainder is
held as cash, which is a legitimate allocation, not an accident.

The default plan is shaped by a measured constraint rather than taste: Toss
charges **0.10% on US trades vs 0.015% on KR trades**, and US trades also incur an
FX spread. That is roughly 50bp per US round trip against 23bp for a KR ETF. A
signal turning over 6x a year therefore bleeds ~3%/yr in the US sleeve and ~1.4%
in the KR sleeve. So high-turnover rules are pointed at KR-listed ETFs — which
happen to track the S&P 500, Nasdaq 100 and EuroStoxx anyway — and the US sleeve
is kept to a low-turnover buy-and-hold.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .strategy.base import Strategy
from .strategy.momentum import build_strategy
from .universe import Sleeve, get_sleeve

log = logging.getLogger(__name__)

DEFAULT_PLAN_PATH = REPO_ROOT / "config" / "portfolio.toml"


@dataclass
class Allocation:
    sleeve_key: str
    strategy_key: str
    weight: Decimal
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.sleeve_key}:{self.strategy_key}"

    def sleeve(self) -> Sleeve:
        return get_sleeve(self.sleeve_key)

    def strategy(self) -> Strategy:
        return build_strategy(self.strategy_key, self.sleeve(), **self.params)


@dataclass
class PortfolioPlan:
    allocations: list[Allocation]
    rebalance: str = "monthly"

    def __post_init__(self) -> None:
        if not self.allocations:
            raise ValueError("portfolio plan has no allocations")
        total = sum((a.weight for a in self.allocations), Decimal(0))
        if total > Decimal("1.0001"):
            raise ValueError(
                f"allocation weights sum to {total}, which would require leverage. "
                "This bot never borrows; reduce the weights so they sum to <= 1."
            )
        for a in self.allocations:
            if a.weight < 0:
                raise ValueError(f"{a.name}: negative weight — short selling is not supported")

    @property
    def cash_weight(self) -> Decimal:
        return Decimal(1) - sum((a.weight for a in self.allocations), Decimal(0))

    @property
    def symbols(self) -> list[str]:
        seen: dict[str, None] = {}
        for a in self.allocations:
            for sym in a.sleeve().symbols:
                seen.setdefault(sym, None)
        return list(seen)

    # ------------------------------------------------------------------
    def target_weights(self, store, asof=None) -> tuple[dict[str, Decimal], dict[str, str]]:
        """Blended target weights across every sleeve.

        Returns ``(weights, reasons)`` where reasons maps allocation name to the
        strategy's own explanation — that text is what gets sent to Telegram, so a
        human can see *why* the bot wants to trade before it does.
        """
        combined: dict[str, Decimal] = {}
        reasons: dict[str, str] = {}

        for alloc in self.allocations:
            sleeve = alloc.sleeve()
            strat = alloc.strategy()
            closes = store.load_frame(sleeve.symbols, "1d")
            if closes.empty:
                log.warning("%s: no local data; run `tqt data sync`", alloc.name)
                reasons[alloc.name] = "no data"
                continue
            if asof is not None:
                closes = closes.loc[:asof]
            if closes.empty:
                reasons[alloc.name] = "no data before asof"
                continue

            stamp = closes.index[-1]
            try:
                sleeve_targets = strat.target_weights(closes, stamp)
            except Exception:
                log.exception("%s: strategy failed", alloc.name)
                reasons[alloc.name] = "strategy error"
                continue

            for sym, w in sleeve_targets.items():
                combined[sym] = combined.get(sym, Decimal(0)) + Decimal(str(w)) * alloc.weight
            reasons[alloc.name] = strat.last_reason or "no reason given"

        # Guard against rounding pushing us over 100% invested.
        total = sum(combined.values(), Decimal(0))
        if total > Decimal(1):
            scale = Decimal(1) / total
            combined = {k: v * scale for k, v in combined.items()}
            log.info("scaled targets by %.4f to keep gross exposure <= 100%%", scale)

        return combined, reasons

    def describe(self) -> str:
        lines = [f"portfolio ({len(self.allocations)} sleeves, rebalance={self.rebalance}):"]
        for a in self.allocations:
            extra = f" {a.params}" if a.params else ""
            lines.append(f"  {a.weight:>6.1%}  {a.sleeve_key:16} {a.strategy_key}{extra}")
        if self.cash_weight > 0:
            lines.append(f"  {self.cash_weight:>6.1%}  cash")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#: Cost-aware default: high-turnover signals on cheap KR ETFs, and the expensive
#: US sleeve limited to a rule that barely trades (0.12x turnover/yr).
DEFAULT_PLAN = PortfolioPlan(
    allocations=[
        Allocation("kr-global-etf", "faber", Decimal("0.50"), {"sma_days": 200}),
        Allocation("kr-global-etf", "dual-momentum", Decimal("0.25"), {"top_n": 2}),
        Allocation("us-global-etf", "buy-and-hold", Decimal("0.25"), {}),
    ]
)


def load_plan(path: str | Path | None = None) -> PortfolioPlan:
    """Load a plan from TOML, falling back to ``DEFAULT_PLAN``."""
    p = Path(path) if path else DEFAULT_PLAN_PATH
    if not p.exists():
        log.info("no plan at %s; using built-in default", p)
        return DEFAULT_PLAN

    with p.open("rb") as fh:
        raw = tomllib.load(fh)

    allocations = []
    for entry in raw.get("sleeve", []):
        allocations.append(
            Allocation(
                sleeve_key=entry["key"],
                strategy_key=entry["strategy"],
                weight=Decimal(str(entry["weight"])),
                params=dict(entry.get("params", {})),
            )
        )
    if not allocations:
        log.warning("%s defined no [[sleeve]] entries; using default plan", p)
        return DEFAULT_PLAN
    return PortfolioPlan(allocations=allocations, rebalance=raw.get("rebalance", "monthly"))
