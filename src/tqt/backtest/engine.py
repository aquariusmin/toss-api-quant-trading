"""Event-driven daily backtester.

Three deliberate design decisions, each guarding against a way backtests lie:

1. **Signals on close, fills at the next open.** A strategy is handed history
   *sliced up to and including* the decision bar, and the resulting orders fill at
   the following bar's open. It is structurally impossible for a strategy here to
   trade on a price it could not have known — the single most common bug in
   hand-rolled backtests (Ernest Chan, *Quantitative Trading*, ch. 3).

2. **Whole shares and real cash.** Positions are integers for KR (KRX has no
   fractional trading) and cash is ``Decimal``. A backtest that buys 3.7183 shares
   of a 250,000-won stock quietly overstates returns on a small account, which is
   exactly this account's situation.

3. **Costs applied as price shift + cash deduction.** Slippage moves the fill
   price against us; commission, tax and FX come out of cash. Never both for the
   same effect.

Single-currency by design: a sleeve of KR assets runs in KRW, a sleeve of US
assets in USD. Blending FX moves into a strategy's return series measures the
currency, not the strategy. Portfolio-level FX exposure is an explicit allocation
decision made in ``tqt.portfolio``, not something smuggled into a Sharpe ratio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from .costs import CostModel, TradeCost

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

log = logging.getLogger(__name__)

ZERO = Decimal(0)


def _dec(v: Any) -> Decimal:
    return v if isinstance(v, Decimal) else Decimal(str(v))


@dataclass
class Fill:
    date: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    cost: TradeCost
    reason: str = ""

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity


@dataclass
class BacktestConfig:
    initial_cash: Decimal = Decimal(1_000_000)
    country: str = "KR"
    #: Which assets are ETFs — drives the KR sell-tax exemption.
    etf_symbols: frozenset[str] = frozenset()
    allow_fractional: bool | None = None  # default: True for US, False for KR
    #: Skip rebalancing a position whose weight is already within this band of
    #: target. Turnover is a direct cost; churning 0.5% deltas is pure leakage.
    rebalance_band: Decimal = Decimal("0.01")
    #: Orders below this *notional, in the sleeve's own currency* are skipped as
    #: dust. Must be currency-aware: 5,000 is a sane floor in KRW and an absurd
    #: one in USD, where it would silently reject every order on a small account.
    min_order_value: Decimal | None = None
    cost_model: CostModel = field(default_factory=CostModel)

    @property
    def dust_threshold(self) -> Decimal:
        if self.min_order_value is not None:
            return self.min_order_value
        return Decimal(5_000) if self.country == "KR" else Decimal(5)

    def fractional(self) -> bool:
        if self.allow_fractional is not None:
            return self.allow_fractional
        # Toss supports dollar-amount US market buys and fractional US market
        # sells, so fractional US sizing is realistic. KRX is whole-share only.
        return self.country == "US"

    def charges_fx(self, cost_model: CostModel) -> bool:
        """Whether a buy in this sleeve also incurs an FX conversion cost."""
        return self.country == "US" and cost_model.charge_fx_on_us_trades


@dataclass
class BacktestResult:
    equity: pd.Series
    fills: list[Fill]
    weights: pd.DataFrame
    config: BacktestConfig
    strategy_key: str
    strategy_params: dict[str, Any]
    total_cost: TradeCost
    turnover: pd.Series

    @property
    def start(self) -> str:
        return str(self.equity.index[0].date()) if len(self.equity) else ""

    @property
    def end(self) -> str:
        return str(self.equity.index[-1].date()) if len(self.equity) else ""

    def metrics(self, **kw) -> dict[str, Any]:
        from .metrics import summarize

        m = summarize(self.equity, **kw)
        gross = float(self.config.initial_cash)
        m["n_trades"] = len(self.fills)
        m["total_cost"] = float(self.total_cost.total)
        m["cost_pct_of_initial"] = (
            float(self.total_cost.total) / gross * 100 if gross else 0.0
        )
        m["avg_annual_turnover"] = (
            float(self.turnover.sum()) / max(len(self.equity) / 252.0, 1e-9)
            if len(self.turnover)
            else 0.0
        )
        return m


class Backtester:
    """Runs one strategy over one sleeve, in one currency."""

    def __init__(
        self,
        opens: pd.DataFrame,
        closes: pd.DataFrame,
        config: BacktestConfig | None = None,
    ) -> None:
        if opens.empty or closes.empty:
            raise ValueError("no price data: run `tqt data sync` first")
        self.config = config or BacktestConfig()
        # Align both frames on a common index/columns so a missing open can never
        # be silently paired with a present close.
        common_cols = sorted(set(opens.columns) & set(closes.columns))
        common_idx = opens.index.intersection(closes.index)
        self.opens = opens.loc[common_idx, common_cols].sort_index()
        self.closes = closes.loc[common_idx, common_cols].sort_index()

    # ------------------------------------------------------------------
    def run(self, strategy) -> BacktestResult:
        cfg = self.config
        fractional = cfg.fractional()

        cash = _dec(cfg.initial_cash)
        positions: dict[str, Decimal] = {}
        fills: list[Fill] = []
        total_cost = TradeCost()

        index = self.closes.index
        warmup = int(getattr(strategy, "warmup_days", 0) or 0)
        rebal_dates = strategy.rebalance_dates(index)

        equity_rows: list[tuple[Any, float]] = []
        weight_rows: dict[Any, dict[str, float]] = {}
        turnover_rows: list[tuple[Any, float]] = []

        pending: dict[str, Decimal] | None = None
        pending_reason = ""

        for i, ts in enumerate(index):
            date_s = str(ts.date())

            # --- 1. execute yesterday's decision at today's open -----------
            traded_notional = ZERO
            if pending is not None:
                equity_at_open = self._equity(cash, positions, self.opens.loc[ts])
                new_cash, new_positions, batch, batch_cost, traded_notional = self._rebalance(
                    cash,
                    positions,
                    targets=pending,
                    prices=self.opens.loc[ts],
                    equity=equity_at_open,
                    date_s=date_s,
                    fractional=fractional,
                    reason=pending_reason,
                )
                cash, positions = new_cash, new_positions
                fills.extend(batch)
                total_cost = total_cost + batch_cost
                pending = None
                pending_reason = ""

            # --- 2. mark to market on today's close ------------------------
            closes_today = self.closes.loc[ts]
            equity = self._equity(cash, positions, closes_today)
            equity_rows.append((ts, float(equity)))
            turnover_rows.append(
                (ts, float(traded_notional / equity) if equity > 0 else 0.0)
            )
            if positions:
                weight_rows[ts] = {
                    sym: float(_dec(closes_today.get(sym, 0) or 0) * q / equity)
                    for sym, q in positions.items()
                    if equity > 0
                }

            # --- 3. decide, using only data up to and including today ------
            # There must be a next bar to fill against, else the decision is
            # unexecutable and including it would be lookahead by another name.
            if i + 1 >= len(index) or i < warmup or ts not in rebal_dates:
                continue

            history = self.closes.iloc[: i + 1]
            try:
                targets = strategy.target_weights(history, ts)
            except Exception:  # a broken strategy must not corrupt the ledger
                log.exception("strategy %s failed on %s", strategy.key, date_s)
                continue
            if targets:
                pending = {k: _dec(v) for k, v in targets.items()}
                pending_reason = getattr(strategy, "last_reason", "") or ""

        import pandas as pd

        equity_series = pd.Series(
            [v for _, v in equity_rows], index=[t for t, _ in equity_rows], name="equity"
        )
        turnover_series = pd.Series(
            [v for _, v in turnover_rows], index=[t for t, _ in turnover_rows], name="turnover"
        )
        weights_df = pd.DataFrame.from_dict(weight_rows, orient="index").fillna(0.0)

        return BacktestResult(
            equity=equity_series,
            fills=fills,
            weights=weights_df,
            config=cfg,
            strategy_key=strategy.key,
            strategy_params=dict(getattr(strategy, "params", {}) or {}),
            total_cost=total_cost,
            turnover=turnover_series,
        )

    # ------------------------------------------------------------------
    def _equity(self, cash: Decimal, positions: dict[str, Decimal], prices) -> Decimal:
        total = cash
        for sym, qty in positions.items():
            px = prices.get(sym)
            if px is None or px != px:  # NaN check
                continue
            total += _dec(px) * qty
        return total

    def _rebalance(
        self,
        cash: Decimal,
        positions: dict[str, Decimal],
        *,
        targets: dict[str, Decimal],
        prices,
        equity: Decimal,
        date_s: str,
        fractional: bool,
        reason: str,
    ) -> tuple[dict, dict, list[Fill], TradeCost, Decimal]:
        cfg = self.config
        cm = cfg.cost_model
        fills: list[Fill] = []
        batch_cost = TradeCost()
        traded = ZERO
        positions = dict(positions)

        def tradable(sym: str) -> bool:
            px = prices.get(sym)
            return px is not None and px == px and px > 0

        # Desired share counts at current prices.
        desired: dict[str, Decimal] = {}
        for sym, weight in targets.items():
            if not tradable(sym) or weight <= 0:
                desired[sym] = ZERO
                continue
            px = _dec(prices[sym])
            raw = equity * _dec(weight) / px
            desired[sym] = raw if fractional else Decimal(int(raw))

        # Anything held but not targeted must be liquidated.
        for sym in positions:
            desired.setdefault(sym, ZERO)

        # Sells first: they release the cash the buys need.
        ordered = sorted(desired.items(), key=lambda kv: kv[1] - positions.get(kv[0], ZERO))

        for sym, want in ordered:
            have = positions.get(sym, ZERO)
            delta = want - have
            if delta == 0 or not tradable(sym):
                continue

            ref_px = _dec(prices[sym])
            side = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)

            # Ignore dust: rebalancing a sliver costs more than the tracking error.
            notional_est = ref_px * qty
            if notional_est < cfg.dust_threshold:
                continue
            if equity > 0 and (notional_est / equity) < cfg.rebalance_band and have > 0:
                continue

            slip = cm.slippage_rate(cfg.country)
            fill_px = ref_px * (Decimal(1) + slip) if side == "BUY" else ref_px * (
                Decimal(1) - slip
            )

            is_etf = sym in cfg.etf_symbols
            if side == "BUY":
                cost = cm.cost_of(
                    side=side,
                    country=cfg.country,
                    price=fill_px,
                    quantity=qty,
                    is_etf=is_etf,
                    include_slippage=False,
                )
                needed = fill_px * qty + cost.total
                if needed > cash:
                    # Scale down to what cash allows rather than going negative:
                    # a backtest that borrows implicitly reports leveraged returns.
                    affordable_px = fill_px * (Decimal(1) + cm.commission_rate(cfg.country))
                    if cfg.charges_fx(cm):
                        affordable_px *= Decimal(1) + cm.fx_spread_rate
                    max_qty = cash / affordable_px if affordable_px > 0 else ZERO
                    qty = max_qty if fractional else Decimal(int(max_qty))
                    if qty <= 0 or fill_px * qty < cfg.dust_threshold:
                        continue
                    cost = cm.cost_of(
                        side=side,
                        country=cfg.country,
                        price=fill_px,
                        quantity=qty,
                        is_etf=is_etf,
                        include_slippage=False,
                    )
                cash -= fill_px * qty + cost.total
                positions[sym] = have + qty
            else:
                qty = min(qty, have)
                if qty <= 0:
                    continue
                cost = cm.cost_of(
                    side=side,
                    country=cfg.country,
                    price=fill_px,
                    quantity=qty,
                    is_etf=is_etf,
                    include_slippage=False,
                )
                cash += fill_px * qty - cost.total
                positions[sym] = have - qty

            traded += fill_px * qty
            batch_cost = batch_cost + cost
            fills.append(
                Fill(
                    date=date_s,
                    symbol=sym,
                    side=side,
                    quantity=qty,
                    price=fill_px,
                    cost=cost,
                    reason=reason,
                )
            )

        positions = {k: v for k, v in positions.items() if v > 0}
        return cash, positions, fills, batch_cost, traded


def make_config(sleeve, cost_model: CostModel | None = None, **kw) -> BacktestConfig:
    """Build a config wired to a sleeve: right country, right ETF tax treatment."""
    etfs = frozenset(a.symbol for a in sleeve.all_assets if a.is_etf)
    return BacktestConfig(
        country=sleeve.country,
        etf_symbols=etfs,
        cost_model=cost_model or CostModel(),
        **kw,
    )
