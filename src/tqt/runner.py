"""The trading loop.

One ``cycle()`` is: look at the account, check risk, ask the portfolio what it
wants to hold, and place the difference. The same method drives paper and live —
only the injected broker differs.

Scheduling notes:

* **Rebalance day is derived from Toss's calendar, not a date arithmetic guess.**
  Today is the last business day of the month exactly when the exchange says the
  *next* business day falls in a different month. This handles Korean substitute
  holidays and year-end closures without a hardcoded holiday table.
* **KR and US trade in separate windows** because their sessions barely overlap
  (KR 09:00-15:30 KST, US 22:30-05:00 KST). Each cycle only touches symbols whose
  market is actually open — placing an order into a closed market just earns a
  ``422 order-hours-closed``.
* **We trade near the close, not the open.** The 09:00-09:10 KST window has
  tighter rate limits and the widest spreads of the day; a monthly rebalance has
  no reason to be there.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from .backtest.costs import CostModel
from .config import Mode, Settings, get_settings
from .data.store import Store, d
from .execution.broker import AccountSnapshot, Broker, OrderIntent, OrderResult
from .notify.base import Level, NotifierGroup, build_notifier
from .ops import MarketClock, check_ip
from .portfolio import PortfolioPlan, load_plan
from .risk import RiskManager
from .toss.client import TossClient
from .universe import SLEEVES, country_of

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9), name="KST")


@dataclass
class CycleReport:
    started_at: str
    markets: list[str]
    rebalanced: bool = False
    skipped_reason: str = ""
    equity_krw: Decimal = Decimal(0)
    targets: dict[str, Decimal] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    results: list[OrderResult] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def filled(self) -> list[OrderResult]:
        return [r for r in self.results if r.ok and r.filled_quantity > 0]

    def summary(self) -> str:
        lines = [f"[{self.started_at}] markets={','.join(self.markets) or '-'}"]
        lines.append(f"equity: {self.equity_krw:,.0f} KRW")
        if self.skipped_reason:
            lines.append(f"skipped: {self.skipped_reason}")
        if not self.rebalanced and not self.skipped_reason:
            lines.append("no rebalance scheduled today")
        for name, reason in self.reasons.items():
            lines.append(f"  {name}: {reason}")
        for r in self.results:
            if r.ok and r.filled_quantity > 0:
                lines.append(
                    f"  ✅ {r.side} {r.symbol} x{r.filled_quantity} @ {r.fill_price} "
                    f"({r.currency})"
                )
            elif r.ok:
                lines.append(f"  ⏳ {r.side} {r.symbol} {r.status}")
            else:
                lines.append(f"  ❌ {r.side} {r.symbol}: {r.error}")
        for b in self.blocked:
            lines.append(f"  🚫 {b}")
        return "\n".join(lines)


class Runner:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        client: TossClient,
        plan: PortfolioPlan,
        broker: Broker,
        risk: RiskManager,
        notifier: NotifierGroup,
        clock: MarketClock,
    ) -> None:
        self.settings = settings
        self.store = store
        self.client = client
        self.plan = plan
        self.broker = broker
        self.risk = risk
        self.notifier = notifier
        self.clock = clock
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, settings: Settings | None = None, *, plan_path: str | None = None) -> Runner:
        settings = settings or get_settings()
        settings.require_credentials()

        store = Store(settings.db_path)
        client = TossClient(settings)
        plan = load_plan(plan_path)
        clock = MarketClock(client)
        notifier = build_notifier(settings)

        cost_model = CostModel.from_api(client)
        etfs = frozenset(
            a.symbol for sl in SLEEVES.values() for a in sl.all_assets if a.is_etf
        )

        broker: Broker
        if settings.mode is Mode.LIVE:
            from .execution.live import LiveBroker

            broker = LiveBroker(client, store, etf_symbols=etfs)
            broker.sync_orders()
        else:
            from .execution.paper import PaperBroker

            broker = PaperBroker(
                client,
                store,
                cost_model=cost_model,
                starting_cash_krw=settings.paper_starting_cash_krw,
                etf_symbols=etfs,
            )

        defensive = frozenset(
            a.symbol
            for sl in SLEEVES.values()
            for a in sl.safe
            if a.role in {"bond", "cash"}
        )
        risk = RiskManager(
            settings, store, broker.name, client=client, defensive_symbols=defensive
        )
        return cls(settings, store, client, plan, broker, risk, notifier, clock)

    # ------------------------------------------------------------------
    # rebalance scheduling
    # ------------------------------------------------------------------
    def is_rebalance_day(self, country: str) -> bool:
        """True on the last business day of the period, per the exchange calendar."""
        cadence = (self.plan.rebalance or "monthly").lower()
        try:
            cal = self.clock._calendar(country)
        except Exception as exc:
            log.warning("calendar unavailable for %s: %s", country, exc)
            return False

        today = cal.today
        if not today.is_open_day:
            return False

        try:
            today_d = datetime.fromisoformat(today.date).date()
            next_d = datetime.fromisoformat(cal.next_business_day.date).date()
        except ValueError:
            return False

        if cadence == "weekly":
            # Crossing into a new ISO week means today is the week's last session.
            return today_d.isocalendar()[1] != next_d.isocalendar()[1]
        if cadence == "daily":
            return True
        return today_d.month != next_d.month

    def open_markets(self) -> list[str]:
        return [c for c in ("KR", "US") if self.clock.is_open(c)]

    # ------------------------------------------------------------------
    def cycle(self, markets: list[str] | None = None, *, force: bool = False) -> CycleReport:
        """One evaluation. ``force=True`` rebalances regardless of the calendar."""
        with self._lock:
            return self._cycle(markets, force=force)

    def _cycle(self, markets: list[str] | None, *, force: bool) -> CycleReport:
        self.clock.clear_cache()
        self.risk.clear_cache()
        now = datetime.now(KST).isoformat(timespec="seconds")
        markets = markets if markets is not None else self.open_markets()
        report = CycleReport(started_at=now, markets=list(markets))

        snapshot = self.broker.snapshot()
        report.equity_krw = snapshot.equity_krw
        self.store.record_equity(
            self.broker.name,
            snapshot.equity_krw,
            snapshot.cash_krw,
            snapshot.position_value_krw(),
            snapshot.usd_krw,
        )

        verdict = self.risk.check_cycle(snapshot)
        if not verdict:
            report.skipped_reason = verdict.reason
            self.notifier.send(f"거래 중단 상태: {verdict.reason}", level=Level.WARN)
            return report

        if not markets:
            report.skipped_reason = "no market open"
            return report

        should_rebalance = force or any(self.is_rebalance_day(c) for c in markets)
        if not should_rebalance:
            return report

        targets, reasons = self.plan.target_weights(self.store)
        report.targets = targets
        report.reasons = reasons
        if not targets:
            report.skipped_reason = "strategies produced no targets"
            return report
        self.store.record_signals(now, "portfolio", targets, "; ".join(reasons.values())[:400])

        report.rebalanced = True
        intents = self._build_intents(targets, snapshot, markets)

        for intent, price in intents:
            check = self.risk.check_order(intent, snapshot, price=price)
            if not check:
                report.blocked.append(check.reason)
                log.info("blocked: %s", check.reason)
                continue
            result = self.broker.submit(intent)
            report.results.append(result)
            if result.ok and result.filled_quantity > 0:
                # Re-read state so subsequent orders see the new cash/positions.
                snapshot = self.broker.snapshot()

        if report.results or report.blocked:
            level = Level.TRADE if report.filled else Level.INFO
            self.notifier.send(report.summary(), level=level)
        self.store.log_event(
            "INFO",
            "cycle",
            f"rebalance markets={markets} filled={len(report.filled)}",
            {"equity": float(report.equity_krw)},
        )
        return report

    # ------------------------------------------------------------------
    def _build_intents(
        self, targets: dict[str, Decimal], snapshot: AccountSnapshot, markets: list[str]
    ) -> list[tuple[OrderIntent, Decimal]]:
        """Translate target weights into orders, sells first."""
        equity = snapshot.equity_krw
        if equity <= 0:
            return []

        current = snapshot.weights()
        band = Decimal("0.01")  # ignore drift under 1% of equity
        symbols = set(targets) | set(snapshot.positions)

        planned: list[tuple[Decimal, OrderIntent, Decimal]] = []
        for symbol in sorted(symbols):
            country = country_of(symbol)
            if country not in markets:
                continue

            want = targets.get(symbol, Decimal(0))
            have = current.get(symbol, Decimal(0))
            drift = want - have
            if abs(drift) < band:
                continue

            price = self.broker.price_of(symbol)
            if price is None or price <= 0:
                log.warning("%s: no price, skipping", symbol)
                continue

            fx = snapshot.usd_krw if country == "US" else Decimal(1)
            notional_krw = abs(drift) * equity
            qty = notional_krw / (price * fx)
            if country == "KR":
                qty = Decimal(int(qty))
            if qty <= 0:
                continue

            side = "BUY" if drift > 0 else "SELL"
            if side == "SELL":
                pos = snapshot.positions.get(symbol)
                if pos is None:
                    continue
                qty = min(qty, pos.quantity)
                if country == "KR":
                    qty = Decimal(int(qty))
                if qty <= 0:
                    continue

            intent = OrderIntent(
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type="MARKET",
                strategy="portfolio",
                reason=f"target {want:.1%} vs held {have:.1%}",
            )
            # Sort key: sells (negative drift) first so their cash is available.
            planned.append((drift, intent, price))

        planned.sort(key=lambda t: t[0])
        return [(i, p) for _, i, p in planned]

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def daily_report(self) -> str:
        snapshot = self.broker.snapshot()
        curve = self.store.equity_curve(self.broker.name, limit=400)
        lines = [
            f"📊 일일 리포트 ({datetime.now(KST):%Y-%m-%d %H:%M} KST)",
            f"모드: {self.settings.mode.value} / broker: {self.broker.name}",
            f"평가금액: {snapshot.equity_krw:,.0f} KRW",
            f"현금: {snapshot.cash_krw:,.0f} KRW  주식: {snapshot.position_value_krw():,.0f} KRW",
            f"USD/KRW: {snapshot.usd_krw:,.2f}",
        ]

        loss = self.risk.daily_loss(snapshot)
        if loss is not None:
            lines.append(f"당일 손익: {loss:+.2%}")
        if len(curve) > 1:
            first = d(curve[0]["equity_krw"])
            if first > 0:
                lines.append(f"누적 손익: {(snapshot.equity_krw - first) / first:+.2%}")

        if snapshot.positions:
            lines.append("\n보유 종목:")
            for sym, pos in snapshot.positions.items():
                mark = snapshot.prices.get(sym, pos.avg_price)
                pnl = (mark / pos.avg_price - 1) if pos.avg_price > 0 else Decimal(0)
                lines.append(
                    f"  {sym:8} x{pos.quantity:>10} @ {pos.avg_price:>12,.2f} "
                    f"→ {mark:>12,.2f} ({pnl:+.2%})"
                )
        else:
            lines.append("\n보유 종목 없음 (전액 현금)")

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        fills = self.store.conn.execute(
            "SELECT * FROM fills WHERE broker=? AND substr(filled_at,1,10)=? ORDER BY filled_at",
            (self.broker.name, today),
        ).fetchall()
        if fills:
            lines.append(f"\n오늘 체결 {len(fills)}건:")
            for f in fills:
                lines.append(f"  {f['side']} {f['symbol']} x{f['quantity']} @ {f['price']}")

        rs = self.risk.status(snapshot)
        lines.append(
            f"\n리스크: 중단={'예' if rs['halted'] else '아니오'} "
            f"주문 {rs['orders_today']}/{rs['order_budget']}"
        )
        if rs["halted"]:
            lines.append(f"중단 사유: {rs['halt_reason']}")
        return "\n".join(lines)

    def send_daily_report(self) -> None:
        self.notifier.send(self.daily_report(), level=Level.INFO)

    def check_ip_and_alert(self) -> None:
        status = check_ip(self.store)
        if status.changed or status.current is None:
            self.notifier.send(status.message, level=Level.ERROR)

    def mark_equity(self) -> None:
        """Periodic equity mark so the dashboard curve and the daily-loss
        baseline both have data even on days with no trades."""
        try:
            snap = self.broker.snapshot()
            self.store.record_equity(
                self.broker.name,
                snap.equity_krw,
                snap.cash_krw,
                snap.position_value_krw(),
                snap.usd_krw,
            )
        except Exception as exc:
            log.warning("equity mark failed: %s", exc)

    # ------------------------------------------------------------------
    # Telegram command surface
    # ------------------------------------------------------------------
    def command_handlers(self) -> dict[str, Any]:
        def h_status(_args: list[str]) -> str:
            snap = self.broker.snapshot()
            rs = self.risk.status(snap)
            clock = self.clock.describe()
            return (
                f"mode={self.settings.mode.value} broker={self.broker.name}\n"
                f"equity={snap.equity_krw:,.0f} KRW cash={snap.cash_krw:,.0f}\n"
                f"positions={len(snap.positions)} halted={rs['halted']}\n"
                f"orders today={rs['orders_today']}/{rs['order_budget']}\n"
                f"KR {'OPEN' if clock.get('KR', {}).get('open') else 'closed'} / "
                f"US {'OPEN' if clock.get('US', {}).get('open') else 'closed'}"
            )

        def h_positions(_args: list[str]) -> str:
            snap = self.broker.snapshot()
            if not snap.positions:
                return "보유 종목 없음"
            rows = [f"{'symbol':8} {'qty':>10} {'avg':>12} {'now':>12} {'pnl':>8}"]
            for sym, pos in snap.positions.items():
                mark = snap.prices.get(sym, pos.avg_price)
                pnl = (mark / pos.avg_price - 1) * 100 if pos.avg_price > 0 else 0
                rows.append(
                    f"{sym:8} {pos.quantity:>10} {pos.avg_price:>12,.2f} "
                    f"{mark:>12,.2f} {pnl:>7.2f}%"
                )
            return "\n".join(rows)

        def h_halt(args: list[str]) -> str:
            reason = " ".join(args) or "operator requested via Telegram"
            self.risk.halt(reason)
            return f"🛑 거래를 중단했습니다: {reason}\n재개는 /resume"

        def h_resume(_args: list[str]) -> str:
            self.risk.resume()
            return "▶️ 거래를 재개했습니다."

        def h_report(_args: list[str]) -> str:
            return self.daily_report()

        def h_orders(_args: list[str]) -> str:
            rows = self.store.conn.execute(
                "SELECT * FROM orders WHERE broker=? ORDER BY submitted_at DESC LIMIT 10",
                (self.broker.name,),
            ).fetchall()
            if not rows:
                return "주문 기록 없음"
            return "\n".join(
                f"{r['submitted_at'][:16]} {r['side']:4} {r['symbol']:8} "
                f"x{r['quantity']:>8} {r['status']}"
                for r in rows
            )

        def h_signals(_args: list[str]) -> str:
            rows = self.store.latest_signals(limit=20)
            if not rows:
                return "신호 기록 없음"
            return "\n".join(
                f"{r['ts'][:16]} {r['symbol']:8} {float(r['target_weight']):.1%}" for r in rows
            )

        def h_ip(_args: list[str]) -> str:
            return check_ip(self.store).message

        def h_rebalance(_args: list[str]) -> str:
            markets = self.open_markets()
            if not markets:
                return "장이 열린 시장이 없습니다. 리밸런싱을 건너뜁니다."
            rep = self.cycle(markets, force=True)
            return rep.summary()

        def h_plan(_args: list[str]) -> str:
            return self.plan.describe()

        handlers = {
            "status": h_status,
            "positions": h_positions,
            "pnl": h_report,
            "report": h_report,
            "orders": h_orders,
            "signals": h_signals,
            "halt": h_halt,
            "stop": h_halt,
            "resume": h_resume,
            "ip": h_ip,
            "rebalance": h_rebalance,
            "plan": h_plan,
        }

        def h_help(_args: list[str]) -> str:
            return (
                "사용 가능한 명령:\n"
                "/status    현재 상태 요약\n"
                "/positions 보유 종목\n"
                "/report    일일 리포트 (=/pnl)\n"
                "/orders    최근 주문 10건\n"
                "/signals   최근 목표 비중\n"
                "/plan      포트폴리오 배분\n"
                "/rebalance 지금 강제 리밸런싱\n"
                "/halt      즉시 거래 중단\n"
                "/resume    거래 재개\n"
                "/ip        공인 IP 확인"
            )

        handlers["help"] = h_help
        handlers["start"] = h_help
        return handlers

    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        """Start the scheduler and block. This is what ``tqt run`` calls."""
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        sched = BackgroundScheduler(timezone=KST)

        # KR: 15:00 KST, comfortably inside the regular session and before the
        # 15:20 closing auction.
        sched.add_job(
            lambda: self.cycle(["KR"]),
            CronTrigger(day_of_week="mon-fri", hour=15, minute=0),
            id="kr_cycle",
            misfire_grace_time=1800,
        )
        # US: 04:30 KST — 30 minutes before the 05:00 KST close.
        sched.add_job(
            lambda: self.cycle(["US"]),
            CronTrigger(day_of_week="tue-sat", hour=4, minute=30),
            id="us_cycle",
            misfire_grace_time=1800,
        )
        sched.add_job(
            self.send_daily_report,
            CronTrigger(hour=8, minute=0),
            id="daily_report",
            misfire_grace_time=3600,
        )
        sched.add_job(
            self.check_ip_and_alert, CronTrigger(minute="*/30"), id="ip_watch"
        )
        sched.add_job(self.mark_equity, CronTrigger(minute="*/30"), id="equity_mark")

        stop_event: threading.Event | None = None
        for n in self.notifier.notifiers:
            if n.name == "telegram":
                _, stop_event = n.start_command_thread(self.command_handlers())
                break

        sched.start()
        banner = (
            f"🤖 tqt 시작 — mode={self.settings.mode.value}, broker={self.broker.name}\n"
            f"{self.plan.describe()}\n"
            f"알림 채널: {', '.join(self.notifier.channels) or '없음'}\n"
            "명령어는 /help"
        )
        log.info(banner)
        self.notifier.send(banner, level=Level.INFO)
        self.check_ip_and_alert()
        self.mark_equity()

        try:
            while True:
                threading.Event().wait(60)
        except (KeyboardInterrupt, SystemExit):
            log.info("shutting down")
        finally:
            if stop_event is not None:
                stop_event.set()
            sched.shutdown(wait=False)
            self.notifier.send("🛑 tqt 종료됨", level=Level.WARN)
