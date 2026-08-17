"""FastAPI dashboard — the "app" you check from your phone.

Design constraints that shaped this:

* **Read-mostly, with two dangerous buttons.** Halt and resume are the only state
  changes, and they are POSTs guarded by the same token. Order entry deliberately
  is not exposed: a web form is the wrong place to fat-finger a trade, and
  Telegram already gives remote control with an audit trail.
* **Token-gated, not password-gated.** A single bearer token in
  ``TQT_DASHBOARD_TOKEN`` compared with ``secrets.compare_digest``. The app refuses
  to start without one, because a dashboard exposing account balances and a kill
  switch must never be open by accident.
* **Endpoints are sync ``def``**, so FastAPI runs them in a threadpool and the
  blocking SQLite/httpx calls don't stall the event loop.
"""

from __future__ import annotations

import logging
import secrets
from decimal import Decimal
from typing import Any

from fastapi import Cookie, Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..config import Settings, get_settings
from ..data.store import Store, d
from ..runner import Runner
from .charts import equity_chart, weights_chart

log = logging.getLogger(__name__)

COOKIE = "tqt_token"


def create_app(runner: Runner | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if not settings.dashboard_token:
        raise RuntimeError(
            "TQT_DASHBOARD_TOKEN is not set. The dashboard shows account balances "
            "and exposes a kill switch, so it refuses to run unauthenticated. "
            "Generate one with: openssl rand -hex 24"
        )

    app = FastAPI(title="tqt dashboard", docs_url=None, redoc_url=None)
    app.state.runner = runner
    app.state.settings = settings
    app.state.store = runner.store if runner else Store(settings.db_path)

    templates = _templates()

    # ------------------------------------------------------------------
    def auth(
        token: str | None = Query(default=None),
        cookie: str | None = Cookie(default=None, alias=COOKIE),
    ) -> str:
        supplied = token or cookie or ""
        if not secrets.compare_digest(supplied, settings.dashboard_token):
            raise HTTPException(status_code=401, detail="invalid or missing token")
        return supplied

    # ------------------------------------------------------------------
    def gather(store: Store, runner: Runner | None) -> dict[str, Any]:
        """Everything the dashboard renders, in one place."""
        broker_name = runner.broker.name if runner else settings.mode.value
        curve_rows = store.equity_curve(broker_name, limit=1500)
        curve = [{"ts": r["ts"], "equity": float(d(r["equity_krw"]))} for r in curve_rows]

        snapshot = None
        risk_status: dict[str, Any] = {}
        targets: dict[str, Decimal] = {}
        clock: dict[str, Any] = {}
        if runner is not None:
            try:
                snapshot = runner.broker.snapshot()
                risk_status = runner.risk.status(snapshot)
            except Exception as exc:
                log.warning("snapshot failed: %s", exc)
            try:
                targets, _ = runner.plan.target_weights(store)
            except Exception as exc:
                log.warning("target computation failed: %s", exc)
            try:
                clock = runner.clock.describe()
            except Exception as exc:
                log.warning("clock failed: %s", exc)

        actual = snapshot.weights() if snapshot else {}
        symbols = sorted(set(targets) | set(actual))
        weight_rows = [
            {
                "symbol": sym,
                "name": _name_of(store, sym),
                "target": float(targets.get(sym, 0)),
                "actual": float(actual.get(sym, 0)),
            }
            for sym in symbols
        ]

        positions = []
        if snapshot:
            for sym, pos in snapshot.positions.items():
                mark = snapshot.prices.get(sym, pos.avg_price)
                pnl = (mark / pos.avg_price - 1) if pos.avg_price > 0 else Decimal(0)
                positions.append(
                    {
                        "symbol": sym,
                        "name": _name_of(store, sym),
                        "quantity": float(pos.quantity),
                        "avg_price": float(pos.avg_price),
                        "last_price": float(mark),
                        "pnl_pct": float(pnl) * 100,
                        "currency": pos.currency,
                        "weight": float(actual.get(sym, 0)) * 100,
                    }
                )
            positions.sort(key=lambda p: -p["weight"])

        orders = [
            dict(r)
            for r in store.conn.execute(
                "SELECT submitted_at,side,symbol,quantity,price,status,reason "
                "FROM orders WHERE broker=? ORDER BY submitted_at DESC LIMIT 25",
                (broker_name,),
            ).fetchall()
        ]
        events = [dict(r) for r in store.recent_events(limit=30)]

        equity = float(snapshot.equity_krw) if snapshot else (
            curve[-1]["equity"] if curve else 0.0
        )
        first = curve[0]["equity"] if curve else 0.0
        total_return = (equity / first - 1) * 100 if first else 0.0

        return {
            "mode": settings.mode.value,
            "broker": broker_name,
            "equity": equity,
            "cash": float(snapshot.cash_krw) if snapshot else 0.0,
            "positions_value": float(snapshot.position_value_krw()) if snapshot else 0.0,
            "usd_krw": float(snapshot.usd_krw) if snapshot else 0.0,
            "total_return": total_return,
            "daily_pnl": risk_status.get("daily_pnl_pct"),
            "gross_exposure": risk_status.get("gross_exposure", 0.0),
            "risk": risk_status,
            "clock": clock,
            "curve": curve,
            "weight_rows": weight_rows,
            "positions": positions,
            "orders": orders,
            "events": events,
            "plan": runner.plan.describe() if runner else "",
            "n_bars": store.conn.execute("SELECT COUNT(*) n FROM candles").fetchone()["n"],
        }

    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, token: str = Depends(auth)) -> Any:
        store: Store = app.state.store
        data = gather(store, app.state.runner)
        html = templates.render(
            data=data,
            equity_svg=equity_chart(data["curve"]),
            weights_svg=weights_chart(data["weight_rows"]),
        )
        resp = HTMLResponse(html)
        # Keep the token out of subsequent URLs (and out of browser history).
        resp.set_cookie(
            COOKIE, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30
        )
        return resp

    @app.get("/api/state")
    def api_state(token: str = Depends(auth)) -> JSONResponse:
        data = gather(app.state.store, app.state.runner)
        data.pop("curve", None)  # keep the JSON small for polling
        return JSONResponse(data)

    @app.get("/api/curve")
    def api_curve(token: str = Depends(auth)) -> JSONResponse:
        runner = app.state.runner
        broker = runner.broker.name if runner else app.state.settings.mode.value
        rows = app.state.store.equity_curve(broker, limit=2000)
        return JSONResponse(
            [{"ts": r["ts"], "equity": float(d(r["equity_krw"]))} for r in rows]
        )

    @app.post("/halt")
    def halt(reason: str = Form(default="dashboard"), token: str = Depends(auth)) -> Any:
        runner = app.state.runner
        if runner is None:
            raise HTTPException(503, "no runner attached; start with `tqt run --dashboard`")
        runner.risk.halt(f"dashboard: {reason}")
        runner.notifier.send_safe(f"🛑 대시보드에서 거래 중단: {reason}")
        return RedirectResponse("/", status_code=303)

    @app.post("/resume")
    def resume(token: str = Depends(auth)) -> Any:
        runner = app.state.runner
        if runner is None:
            raise HTTPException(503, "no runner attached")
        runner.risk.resume()
        runner.notifier.send_safe("▶️ 대시보드에서 거래 재개")
        return RedirectResponse("/", status_code=303)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Unauthenticated liveness probe — deliberately leaks nothing."""
        return {"ok": True}

    return app


def _name_of(store: Store, symbol: str) -> str:
    row = store.get_stock(symbol)
    return (row["name"] if row and row["name"] else symbol) or symbol


def _templates():
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader, select_autoescape

    root = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(root)),
        autoescape=select_autoescape(["html"]),
    )
    return env.get_template("dashboard.html")


def serve(runner: Runner | None = None, settings: Settings | None = None) -> None:
    import uvicorn

    settings = settings or get_settings()
    app = create_app(runner, settings)
    uvicorn.run(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level=settings.log_level.lower(),
    )
