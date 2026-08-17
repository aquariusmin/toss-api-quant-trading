"""``tqt`` command line.

The commands map onto the three stages of the plan, in order:

  1. ``tqt data sync`` then ``tqt backtest ...``   — does the idea work at all?
  2. ``tqt run`` with ``TQT_MODE=paper``           — does it work on live prices?
  3. ``tqt run`` with ``TQT_MODE=live``            — real money, small.

``tqt doctor`` is the one to run whenever something looks wrong.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .config import Mode, get_settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Toss Securities quant trading bot: backtest -> paper -> live.",
)
data_app = typer.Typer(no_args_is_help=True, help="Market data ingestion.")
bt_app = typer.Typer(no_args_is_help=True, help="Backtesting and validation.")
paper_app = typer.Typer(no_args_is_help=True, help="Paper account management.")
app.add_typer(data_app, name="data")
app.add_typer(bt_app, name="backtest")
app.add_typer(paper_app, name="paper")

console = Console()


def _setup_logging(level: str | None = None) -> None:
    s = get_settings()
    logging.basicConfig(
        level=(level or s.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def _client():
    from .toss.client import TossClient

    return TossClient(get_settings())


def _store():
    from .data.store import Store

    return Store(get_settings().db_path)


# ---------------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Check everything needed before trading, and say what to fix."""
    _setup_logging("WARNING")
    from .ops import preflight

    settings = get_settings()
    store = _store()
    client = None
    if settings.has_credentials:
        try:
            client = _client()
        except Exception as exc:
            console.print(f"[red]client init failed:[/] {exc}")

    checks = preflight(settings, store, client)
    table = Table(title="tqt doctor", show_lines=False)
    table.add_column("", width=3)
    table.add_column("check", style="bold")
    table.add_column("detail", overflow="fold")
    for c in checks:
        mark = "[green]OK[/]" if c.ok else ("[red]!![/]" if c.fatal else "[yellow]--[/]")
        table.add_row(mark, c.name, c.detail)
    console.print(table)

    fatal = [c for c in checks if c.fatal and not c.ok]
    if fatal:
        console.print(f"[red]{len(fatal)} fatal problem(s) — trading will not work.[/]")
        raise typer.Exit(1)
    console.print("[green]ready.[/]")


@app.command()
def accounts() -> None:
    """List Toss accounts and show which accountSeq will be used."""
    _setup_logging("WARNING")
    client = _client()
    table = Table(title="Toss accounts")
    table.add_column("accountNo")
    table.add_column("accountSeq")
    table.add_column("type")
    for a in client.accounts():
        table.add_row(a.account_no, str(a.account_seq), a.account_type)
    console.print(table)
    console.print(f"resolved accountSeq = [bold]{client.account_seq}[/]")
    console.print("Pin it with TOSS_ACCOUNT_SEQ in .env to skip this lookup.")


@app.command()
def costs() -> None:
    """Show the transaction cost model, calibrated from your account."""
    _setup_logging("WARNING")
    from .backtest.costs import CostModel

    try:
        cm = CostModel.from_api(_client())
        src = "live Toss commissions API"
    except Exception as exc:
        cm = CostModel()
        src = f"defaults (API unavailable: {exc})"

    table = Table(title=f"cost model — {src}")
    table.add_column("item")
    table.add_column("value", justify="right")
    for k, v in cm.describe().items():
        table.add_row(k, v)
    console.print(table)


# ---------------------------------------------------------------------------
@data_app.command("sync")
def data_sync(
    interval: str = typer.Option("1d", help="1d or 1m (1m only covers ~4 days)"),
    max_bars: int = typer.Option(4200, help="max bars per symbol"),
    full: bool = typer.Option(False, "--full", help="re-download all history"),
    symbols: str = typer.Option("", help="comma-separated; default = whole universe"),
) -> None:
    """Download candle history into the local store."""
    _setup_logging()
    from .data.ingest import Ingestor
    from .universe import all_symbols

    client, store = _client(), _store()
    ing = Ingestor(client, store)
    syms = [s.strip() for s in symbols.split(",") if s.strip()] or all_symbols()

    console.print(f"syncing {len(syms)} symbols ({interval}) ...")
    with console.status("downloading") as status:

        def progress(i: int, total: int, res: Any) -> None:
            status.update(f"[{i}/{total}] {res.symbol} +{res.written}")

        results = ing.sync(syms, interval, max_bars=max_bars, full=full, on_progress=progress)

    ing.sync_stock_master(syms)
    failed = [r for r in results if not r.ok]
    total = sum(r.written for r in results)
    console.print(f"[green]{total:,} bars written[/] across {len(results) - len(failed)} symbols")
    for r in failed:
        console.print(f"  [red]{r.symbol}[/]: {r.error}")


@data_app.command("status")
def data_status(interval: str = typer.Option("1d")) -> None:
    """Show what history is stored locally."""
    _setup_logging("WARNING")
    from .universe import all_symbols

    store = _store()
    table = Table(title=f"local data ({interval})")
    table.add_column("symbol")
    table.add_column("name")
    table.add_column("bars", justify="right")
    table.add_column("from")
    table.add_column("to")
    for sym in all_symbols():
        lo, hi, n = store.candle_range(sym, interval)
        row = store.get_stock(sym)
        table.add_row(
            sym,
            (row["name"] if row else "") or "",
            f"{n:,}",
            (lo or "")[:10],
            (hi or "")[:10],
        )
    console.print(table)


# ---------------------------------------------------------------------------
def _load_frames(sleeve, start: str, end: str | None):
    store = _store()
    opens = store.load_frame(sleeve.symbols, "1d", start=start, end=end, field="open")
    closes = store.load_frame(sleeve.symbols, "1d", start=start, end=end, field="close")
    if closes.empty:
        console.print("[red]no local data — run `tqt data sync` first[/]")
        raise typer.Exit(1)
    return opens, closes


def _cost_model():
    from .backtest.costs import CostModel

    try:
        return CostModel.from_api(_client())
    except Exception:
        return CostModel()


@bt_app.command("run")
def backtest_run(
    sleeve: str = typer.Option("kr-global-etf", help="universe key"),
    strategy: str = typer.Option("faber"),
    start: str = typer.Option("2011-01-01"),
    end: str = typer.Option(None),
    cash: float = typer.Option(1_000_000.0),
    compare_all: bool = typer.Option(False, "--all", help="run every strategy"),
) -> None:
    """Backtest one sleeve. ``--all`` compares every strategy on it."""
    _setup_logging("WARNING")
    from .backtest.engine import Backtester, make_config
    from .backtest.metrics import compare
    from .strategy.momentum import STRATEGIES, build_strategy
    from .universe import get_sleeve

    sl = get_sleeve(sleeve)
    opens, closes = _load_frames(sl, start, end)
    cfg = make_config(sl, _cost_model(), initial_cash=Decimal(str(cash)))

    keys = list(STRATEGIES) if compare_all else [strategy]
    curves: dict[str, Any] = {}
    for key in keys:
        strat = build_strategy(key, sl)
        result = Backtester(opens, closes, cfg).run(strat)
        curves[key] = result.equity
        m = result.metrics()
        console.print(
            f"[bold]{key}[/]: trades={m['n_trades']} "
            f"cost={m['total_cost']:,.0f} ({m['cost_pct_of_initial']:.1f}% of initial) "
            f"turnover/yr={m['avg_annual_turnover']:.2f}"
        )
        console.print(f"  현재 신호: {strat.last_reason}")
        _store().save_backtest(
            f"{sleeve}:{key}:{start}",
            key,
            {"sleeve": sleeve, **strat.params},
            m.get("start", start),
            m.get("end", ""),
            m,
            [],
        )

    console.print(
        f"\n[bold]{sl.title}[/] — "
        f"{closes.index[0].date()} .. {closes.index[-1].date()}"
    )
    console.print(compare(curves).to_string())


@bt_app.command("plan")
def backtest_plan(
    start: str = typer.Option("2011-01-01"),
    cash: float = typer.Option(1_000_000.0),
) -> None:
    """Backtest every sleeve/strategy in config/portfolio.toml."""
    _setup_logging("WARNING")
    from .backtest.engine import Backtester, make_config
    from .backtest.metrics import compare
    from .portfolio import load_plan

    plan = load_plan()
    console.print(plan.describe())
    cm = _cost_model()
    curves: dict[str, Any] = {}

    for alloc in plan.allocations:
        sl = alloc.sleeve()
        opens, closes = _load_frames(sl, start, None)
        cfg = make_config(sl, cm, initial_cash=Decimal(str(cash)) * alloc.weight)
        result = Backtester(opens, closes, cfg).run(alloc.strategy())
        curves[alloc.name] = result.equity

    console.print(compare(curves).to_string())
    console.print(
        "\n[dim]각 슬리브를 배정 비중만큼의 자본으로 개별 실행한 결과입니다. "
        "통화가 다르므로 수익률(%) 기준으로만 비교하세요.[/]"
    )


@bt_app.command("walkforward")
def backtest_walkforward(
    sleeve: str = typer.Option("kr-global-etf"),
    strategy: str = typer.Option("faber"),
    start: str = typer.Option("2011-01-01"),
    cash: float = typer.Option(1_000_000.0),
    train: float = typer.Option(5.0, help="train window in years"),
    test: float = typer.Option(2.0, help="test window in years"),
) -> None:
    """Out-of-sample walk-forward test — the number that actually matters."""
    _setup_logging("WARNING")
    from .backtest.engine import make_config
    from .backtest.validate import walk_forward
    from .strategy.momentum import build_strategy
    from .universe import get_sleeve

    sl = get_sleeve(sleeve)
    opens, closes = _load_frames(sl, start, None)
    cfg = make_config(sl, _cost_model(), initial_cash=Decimal(str(cash)))

    res = walk_forward(
        lambda **kw: build_strategy(strategy, sl, **kw),
        opens,
        closes,
        cfg,
        train_years=train,
        test_years=test,
    )
    if "error" in res:
        console.print(f"[red]{res['error']}[/]")
        raise typer.Exit(1)

    table = Table(title=f"walk-forward folds — {sleeve}/{strategy}")
    for c in ("test_start", "test_end", "cagr", "sharpe", "max_drawdown", "n_trades"):
        table.add_column(c)
    for f in res["folds"]:
        table.add_row(
            f["test_start"],
            f["test_end"],
            f"{f.get('cagr', 0):.2%}",
            f"{f.get('sharpe', 0):.2f}",
            f"{f.get('max_drawdown', 0):.2%}",
            str(f.get("n_trades", 0)),
        )
    console.print(table)

    ins, oos = res["in_sample"], res["oos"]
    console.print(
        f"in-sample : CAGR {ins.get('cagr', 0):>7.2%}  Sharpe {ins.get('sharpe', 0):.2f}  "
        f"MDD {ins.get('max_drawdown', 0):.2%}"
    )
    console.print(
        f"out-of-sample: CAGR {oos.get('cagr', 0):>7.2%}  Sharpe {oos.get('sharpe', 0):.2f}  "
        f"MDD {oos.get('max_drawdown', 0):.2%}"
    )
    console.print(
        f"decay: CAGR x{res['decay_ratio_cagr']:.2f}, Sharpe x{res['decay_ratio_sharpe']:.2f}  "
        "[dim](1.0에 가까울수록 과최적화가 아님)[/]"
    )


@bt_app.command("sweep")
def backtest_sweep(
    sleeve: str = typer.Option("kr-global-etf"),
    strategy: str = typer.Option("faber"),
    param: str = typer.Option("sma_days"),
    values: str = typer.Option("100,150,200,250,300"),
    start: str = typer.Option("2011-01-01"),
    cash: float = typer.Option(1_000_000.0),
) -> None:
    """Grid one parameter. Look for a plateau, not the single best value."""
    _setup_logging("WARNING")
    from .backtest.engine import make_config
    from .backtest.validate import parameter_sweep
    from .strategy.momentum import build_strategy
    from .universe import get_sleeve

    sl = get_sleeve(sleeve)
    opens, closes = _load_frames(sl, start, None)
    cfg = make_config(sl, _cost_model(), initial_cash=Decimal(str(cash)))
    vals: list[Any] = []
    for v in values.split(","):
        v = v.strip()
        vals.append(int(v) if v.isdigit() else float(v))

    df = parameter_sweep(
        lambda **kw: build_strategy(strategy, sl, **kw),
        opens,
        closes,
        cfg,
        param=param,
        values=vals,
    )
    console.print(df.to_string(index=False))
    console.print(
        "\n[dim]DeflSharpe는 여러 값을 시험한 편향을 보정한 값입니다. "
        "가장 높은 한 점보다, 넓게 좋은 구간의 중앙값을 고르세요.[/]"
    )


# ---------------------------------------------------------------------------
@paper_app.command("reset")
def paper_reset(
    cash: float = typer.Option(None, help="starting KRW; default from .env"),
    yes: bool = typer.Option(False, "--yes", help="skip confirmation"),
) -> None:
    """Wipe the paper account and start over. Never touches live records."""
    _setup_logging("WARNING")
    from .execution.paper import PaperBroker

    settings = get_settings()
    amount = Decimal(str(cash)) if cash is not None else settings.paper_starting_cash_krw
    if not yes and not typer.confirm(f"paper 기록을 모두 지우고 {amount:,.0f} KRW로 초기화?"):
        raise typer.Exit(0)

    broker = PaperBroker(_client(), _store(), starting_cash_krw=amount)
    broker.reset(amount)
    console.print(f"[green]paper account reset to {amount:,.0f} KRW[/]")


@app.command()
def cycle(
    force: bool = typer.Option(False, "--force", help="rebalance regardless of schedule"),
    market: str = typer.Option("", help="KR, US, or blank for whichever is open"),
) -> None:
    """Run one trading cycle now, then exit. The way to test before `tqt run`."""
    _setup_logging()
    from .runner import Runner

    runner = Runner.build()
    markets = [m.strip().upper() for m in market.split(",") if m.strip()] or None
    report = runner.cycle(markets, force=force)
    console.print(report.summary())


@app.command()
def run(
    dashboard: bool = typer.Option(False, "--dashboard", help="also serve the web dashboard"),
) -> None:
    """Start the bot: scheduler, notifications, and optionally the dashboard."""
    _setup_logging()
    from .runner import Runner

    settings = get_settings()
    if settings.mode is Mode.LIVE:
        console.print("[bold red]LIVE MODE — real orders with real money.[/]")

    runner = Runner.build()

    if dashboard:
        import threading

        from .web.app import create_app

        def _serve() -> None:
            import uvicorn

            uvicorn.run(
                create_app(runner, settings),
                host=settings.dashboard_host,
                port=settings.dashboard_port,
                log_level="warning",
            )

        threading.Thread(target=_serve, daemon=True, name="dashboard").start()
        console.print(
            f"dashboard: http://localhost:{settings.dashboard_port}/"
            f"?token={settings.dashboard_token[:6]}…"
        )

    runner.run_forever()


@app.command()
def dashboard() -> None:
    """Serve only the dashboard (read-only; no scheduler)."""
    _setup_logging()
    from .runner import Runner
    from .web.app import serve

    settings = get_settings()
    try:
        runner = Runner.build()
    except Exception as exc:
        console.print(f"[yellow]running without a live runner: {exc}[/]")
        runner = None
    console.print(f"http://localhost:{settings.dashboard_port}/?token=<TQT_DASHBOARD_TOKEN>")
    serve(runner, settings)


@app.command()
def halt(reason: str = typer.Argument("cli")) -> None:
    """Stop all trading (persists across restarts)."""
    _setup_logging("WARNING")
    from .risk import RiskManager

    settings = get_settings()
    store = _store()
    broker = "live" if settings.mode is Mode.LIVE else "paper"
    RiskManager(settings, store, broker).halt(f"cli: {reason}")
    console.print("[red]halted.[/] resume with `tqt resume`")


@app.command()
def resume() -> None:
    """Clear the halt flag."""
    _setup_logging("WARNING")
    from .risk import RiskManager

    settings = get_settings()
    broker = "live" if settings.mode is Mode.LIVE else "paper"
    RiskManager(settings, _store(), broker).resume()
    console.print("[green]resumed.[/]")


@app.command("telegram-id")
def telegram_id(
    write: bool = typer.Option(False, "--write", help="save the id into .env"),
) -> None:
    """Find your numeric Telegram chat id (message the bot first).

    Telegram needs a numeric chat id, not the bot's @username — a very common
    mix-up that makes every send fail and silently disables /halt.
    """
    _setup_logging("ERROR")
    import re

    import httpx

    from .config import REPO_ROOT

    settings = get_settings()
    if not settings.telegram_bot_token:
        console.print("[red]TELEGRAM_BOT_TOKEN is not set.[/]")
        raise typer.Exit(1)

    http = httpx.Client(
        base_url=f"https://api.telegram.org/bot{settings.telegram_bot_token}", timeout=15
    )
    me = http.get("/getMe").json()
    if not me.get("ok"):
        console.print(f"[red]bad bot token:[/] {me.get('description')}")
        raise typer.Exit(1)
    console.print(f"bot: [bold]@{me['result'].get('username')}[/]")

    updates = http.get("/getUpdates", params={"limit": 50}).json()
    chats: dict[int, str] = {}
    for u in updates.get("result", []):
        msg = u.get("message") or u.get("edited_message") or u.get("my_chat_member") or {}
        chat = msg.get("chat") or {}
        if chat.get("id") is not None:
            chats[chat["id"]] = (
                chat.get("username") or chat.get("title") or chat.get("first_name") or "?"
            )

    if not chats:
        console.print(
            "[yellow]No chats yet.[/] Open Telegram, find "
            f"[bold]@{me['result'].get('username')}[/], press Start or send any "
            "message, then run this again."
        )
        raise typer.Exit(1)

    for cid, name in chats.items():
        console.print(f"  chat_id = [bold]{cid}[/]  ({name})")

    chosen = next(iter(chats))
    if not write:
        console.print(f"\nAdd to .env:  TELEGRAM_CHAT_ID={chosen}")
        console.print("Or re-run with --write to do it automatically.")
        return

    env = REPO_ROOT / ".env"
    raw = env.read_text(encoding="utf-8")
    line = f"TELEGRAM_CHAT_ID={chosen}"
    if re.search(r"(?m)^TELEGRAM_CHAT_ID=.*$", raw):
        raw = re.sub(r"(?m)^TELEGRAM_CHAT_ID=.*$", line, raw)
    else:
        raw = raw.rstrip("\n") + f"\n{line}\n"
    env.write_text(raw, encoding="utf-8")
    console.print(f"[green]wrote {line} to .env[/]")

    http.post(
        "/sendMessage",
        json={"chat_id": chosen, "text": "✅ tqt: 채팅 ID가 설정되었습니다. /help 를 보내보세요."},
    )


@app.command("notify-test")
def notify_test() -> None:
    """Send a test message to every configured channel."""
    _setup_logging("WARNING")
    from .notify.base import Level, build_notifier

    settings = get_settings()
    n = build_notifier(settings)
    if not n.channels:
        console.print("[yellow]no channels configured[/]")
        raise typer.Exit(1)
    ok = n.send("tqt 알림 테스트입니다. 이 메시지가 보이면 설정이 정상입니다.", level=Level.INFO)
    console.print(f"channels={n.channels} sent={ok}")


@app.command()
def status() -> None:
    """Quick account + risk summary."""
    _setup_logging("WARNING")
    from .runner import Runner

    runner = Runner.build()
    console.print(runner.daily_report())


if __name__ == "__main__":
    app()
