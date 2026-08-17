"""SQLite persistence: market history, the trade ledger, and bot state.

SQLite is the right database here — a single file, no server, fine on a
Raspberry Pi, and transactional enough that a power cut mid-write cannot corrupt
the ledger (WAL + synchronous=FULL on the ledger path).

Money is stored as **TEXT**, not REAL. A position's average cost must round-trip
exactly; ``0.1 + 0.2`` must not creep into a P&L figure. Indicator math converts
to float on the way into pandas, which is fine — a moving average is a
statistic, not an accounting record.

Timestamps are stored as the ISO-8601 strings Toss returns (with offset), which
sort lexicographically in the correct order for a fixed offset.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..toss.models import Candle, Stock

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ------------------------------------------------------------------ market data
CREATE TABLE IF NOT EXISTS candles (
    symbol     TEXT NOT NULL,
    interval   TEXT NOT NULL,
    ts         TEXT NOT NULL,          -- ISO-8601 with offset, as returned
    open       TEXT NOT NULL,
    high       TEXT NOT NULL,
    low        TEXT NOT NULL,
    close      TEXT NOT NULL,
    volume     TEXT NOT NULL,
    currency   TEXT,
    PRIMARY KEY (symbol, interval, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS stocks (
    symbol         TEXT PRIMARY KEY,
    name           TEXT,
    market         TEXT,
    market_country TEXT,
    security_type  TEXT,
    currency       TEXT,
    status         TEXT,
    leverage_factor TEXT,
    updated_at     TEXT,
    raw            TEXT
);

-- ------------------------------------------------------------------ ledger
-- One row per order we submit, in paper or live. `broker` distinguishes them so
-- a paper run and a live run can share one database without contaminating stats.
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT NOT NULL,
    broker          TEXT NOT NULL,      -- 'paper' | 'live'
    client_order_id TEXT,
    strategy        TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    quantity        TEXT NOT NULL,
    price           TEXT,
    currency        TEXT,
    status          TEXT NOT NULL,
    filled_quantity TEXT DEFAULT '0',
    avg_fill_price  TEXT,
    commission      TEXT,
    tax             TEXT,
    submitted_at    TEXT NOT NULL,
    updated_at      TEXT,
    reason          TEXT,
    PRIMARY KEY (broker, order_id)
);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(broker, symbol, submitted_at);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(broker, status);

CREATE TABLE IF NOT EXISTS fills (
    fill_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    broker     TEXT NOT NULL,
    order_id   TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    side       TEXT NOT NULL,
    quantity   TEXT NOT NULL,
    price      TEXT NOT NULL,
    commission TEXT NOT NULL DEFAULT '0',
    tax        TEXT NOT NULL DEFAULT '0',
    currency   TEXT NOT NULL,
    filled_at  TEXT NOT NULL,
    strategy   TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_time ON fills(broker, filled_at);

CREATE TABLE IF NOT EXISTS positions (
    broker       TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    quantity     TEXT NOT NULL,
    avg_price    TEXT NOT NULL,
    currency     TEXT NOT NULL,
    strategy     TEXT,
    opened_at    TEXT,
    updated_at   TEXT,
    PRIMARY KEY (broker, symbol)
);

CREATE TABLE IF NOT EXISTS cash (
    broker   TEXT NOT NULL,
    currency TEXT NOT NULL,
    amount   TEXT NOT NULL,
    PRIMARY KEY (broker, currency)
);

CREATE TABLE IF NOT EXISTS equity (
    broker           TEXT NOT NULL,
    ts               TEXT NOT NULL,
    equity_krw       TEXT NOT NULL,
    cash_krw         TEXT NOT NULL,
    positions_krw    TEXT NOT NULL,
    usd_krw          TEXT,
    PRIMARY KEY (broker, ts)
);

CREATE TABLE IF NOT EXISTS signals (
    ts            TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    target_weight TEXT NOT NULL,
    score         TEXT,
    reason        TEXT,
    PRIMARY KEY (ts, strategy, symbol)
);

-- ------------------------------------------------------------------ bot state
CREATE TABLE IF NOT EXISTS state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    level    TEXT NOT NULL,
    kind     TEXT NOT NULL,
    message  TEXT NOT NULL,
    data     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);

CREATE TABLE IF NOT EXISTS backtests (
    run_id     TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    strategy   TEXT NOT NULL,
    params     TEXT,
    start_date TEXT,
    end_date   TEXT,
    metrics    TEXT,
    equity     TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def d(v: Any) -> Decimal:
    """Coerce a stored TEXT value back to Decimal."""
    if isinstance(v, Decimal):
        return v
    if v is None or v == "":
        return Decimal(0)
    return Decimal(str(v))


def s(v: Any) -> str:
    """Render a Decimal for storage without scientific notation."""
    if v is None:
        return ""
    if isinstance(v, Decimal):
        return format(v.normalize(), "f")
    return str(v)


class Store:
    """Thread-confined SQLite wrapper.

    Each thread gets its own connection (``check_same_thread`` would otherwise
    bite when the dashboard's threadpool and the trading loop touch the store
    concurrently).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ------------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=30000")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    def transaction(self):
        """Context manager for an explicit transaction (``with store.transaction():``)."""
        return _Tx(self.conn)

    # ------------------------------------------------------------------
    # candles
    # ------------------------------------------------------------------
    def upsert_candles(self, symbol: str, interval: str, candles: Iterable[Candle]) -> int:
        rows = [
            (
                symbol,
                interval,
                c.timestamp,
                s(c.open),
                s(c.high),
                s(c.low),
                s(c.close),
                s(c.volume),
                c.currency,
            )
            for c in candles
        ]
        if not rows:
            return 0
        with self.transaction():
            self.conn.executemany(
                """INSERT INTO candles(symbol, interval, ts, open, high, low, close, volume, currency)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol, interval, ts) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume,
                     currency=COALESCE(excluded.currency, candles.currency)""",
                rows,
            )
        return len(rows)

    def candle_range(self, symbol: str, interval: str) -> tuple[str | None, str | None, int]:
        """(earliest_ts, latest_ts, count) held locally for this series."""
        r = self.conn.execute(
            "SELECT MIN(ts) lo, MAX(ts) hi, COUNT(*) n FROM candles WHERE symbol=? AND interval=?",
            (symbol, interval),
        ).fetchone()
        return (r["lo"], r["hi"], r["n"] or 0)

    def symbols_with_data(self, interval: str = "1d") -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM candles WHERE interval=? ORDER BY symbol", (interval,)
        ).fetchall()
        return [r["symbol"] for r in rows]

    def load_candles(
        self,
        symbol: str,
        interval: str = "1d",
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> list[sqlite3.Row]:
        q = "SELECT * FROM candles WHERE symbol=? AND interval=?"
        args: list[Any] = [symbol, interval]
        if start:
            q += " AND ts >= ?"
            args.append(start)
        if end:
            q += " AND ts <= ?"
            args.append(end)
        q += " ORDER BY ts ASC"
        return self.conn.execute(q, args).fetchall()

    def load_frame(
        self,
        symbols: Sequence[str],
        interval: str = "1d",
        *,
        start: str | None = None,
        end: str | None = None,
        field: str = "close",
    ):
        """Wide DataFrame: index = date, one column per symbol.

        Uses only *dates* (not full timestamps) as the index so KR and US series
        align on a common calendar. Missing values stay NaN — strategies must
        handle a symbol that did not trade that day rather than see a fabricated
        price.
        """
        import pandas as pd

        if not symbols:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in symbols)
        q = (
            f"SELECT symbol, substr(ts,1,10) AS date, {field} AS v "  # noqa: S608 - field is validated
            f"FROM candles WHERE interval=? AND symbol IN ({placeholders})"
        )
        if field not in {"open", "high", "low", "close", "volume"}:
            raise ValueError(f"invalid field {field!r}")
        args: list[Any] = [interval, *symbols]
        if start:
            q += " AND ts >= ?"
            args.append(start)
        if end:
            q += " AND ts <= ?"
            args.append(end)
        rows = self.conn.execute(q, args).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["v"] = df["v"].astype(float)
        wide = df.pivot_table(index="date", columns="symbol", values="v", aggfunc="last")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()

    # ------------------------------------------------------------------
    # stock master
    # ------------------------------------------------------------------
    def upsert_stocks(self, stocks: Iterable[Stock]) -> int:
        rows = [
            (
                st.symbol,
                st.name,
                st.market,
                st.market_country,
                st.security_type,
                st.currency,
                st.status,
                s(st.leverage_factor) if st.leverage_factor is not None else None,
                utcnow(),
                json.dumps(st.model_dump(mode="json"), ensure_ascii=False),
            )
            for st in stocks
        ]
        if not rows:
            return 0
        with self.transaction():
            self.conn.executemany(
                """INSERT INTO stocks(symbol,name,market,market_country,security_type,currency,
                                      status,leverage_factor,updated_at,raw)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol) DO UPDATE SET
                     name=excluded.name, market=excluded.market,
                     market_country=excluded.market_country,
                     security_type=excluded.security_type, currency=excluded.currency,
                     status=excluded.status, leverage_factor=excluded.leverage_factor,
                     updated_at=excluded.updated_at, raw=excluded.raw""",
                rows,
            )
        return len(rows)

    def get_stock(self, symbol: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM stocks WHERE symbol=?", (symbol,)).fetchone()

    # ------------------------------------------------------------------
    # key/value state  (kill switch, daily counters, last known IP)
    # ------------------------------------------------------------------
    def set_state(self, key: str, value: Any) -> None:
        payload = value if isinstance(value, str) else json.dumps(value, default=str)
        self.conn.execute(
            "INSERT INTO state(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, payload, utcnow()),
        )

    def get_state(self, key: str, default: Any = None) -> Any:
        """Read a state value.

        Only JSON containers and literals are decoded. A bare numeric string is
        returned **as a string**, deliberately: ``json.loads("74321.1234567890123")``
        parses to a float and silently truncates the value. State holds things like
        the last USD/KRW rate, which the caller converts with ``d()`` to a Decimal —
        round-tripping it through float first would defeat the entire
        store-money-as-TEXT design.
        """
        r = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        if r is None:
            return default
        raw = r["value"]
        if not isinstance(raw, str):
            return raw
        stripped = raw.strip()
        if stripped[:1] in ("{", "[", '"') or stripped in ("true", "false", "null"):
            try:
                return json.loads(stripped)
            except ValueError:
                return raw
        return raw

    # ------------------------------------------------------------------
    # events (audit trail shown on the dashboard)
    # ------------------------------------------------------------------
    def log_event(
        self, level: str, kind: str, message: str, data: dict[str, Any] | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO events(ts,level,kind,message,data) VALUES(?,?,?,?,?)",
            (utcnow(), level, kind, message, json.dumps(data, default=str) if data else None),
        )

    def recent_events(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM events ORDER BY event_id DESC LIMIT ?", (limit,)
        ).fetchall()

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------
    def record_signals(self, ts: str, strategy: str, targets: dict[str, Decimal], reason: str = ""):
        rows = [(ts, strategy, sym, s(w), None, reason) for sym, w in targets.items()]
        if rows:
            with self.transaction():
                self.conn.executemany(
                    "INSERT OR REPLACE INTO signals(ts,strategy,symbol,target_weight,score,reason)"
                    " VALUES(?,?,?,?,?,?)",
                    rows,
                )

    def latest_signals(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM signals ORDER BY ts DESC, strategy, symbol LIMIT ?", (limit,)
        ).fetchall()

    # ------------------------------------------------------------------
    # equity curve
    # ------------------------------------------------------------------
    def record_equity(
        self,
        broker: str,
        equity_krw: Decimal,
        cash_krw: Decimal,
        positions_krw: Decimal,
        usd_krw: Decimal | None = None,
        ts: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO equity(broker,ts,equity_krw,cash_krw,positions_krw,usd_krw)"
            " VALUES(?,?,?,?,?,?)",
            (broker, ts or utcnow(), s(equity_krw), s(cash_krw), s(positions_krw), s(usd_krw)),
        )

    def equity_curve(self, broker: str, limit: int = 5000) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            "SELECT * FROM equity WHERE broker=? ORDER BY ts DESC LIMIT ?", (broker, limit)
        ).fetchall()
        return list(reversed(rows))

    def first_equity_of_day(self, broker: str, date_prefix: str) -> Decimal | None:
        """Opening equity for the day — the baseline for the daily-loss kill switch."""
        r = self.conn.execute(
            "SELECT equity_krw FROM equity WHERE broker=? AND ts LIKE ? ORDER BY ts ASC LIMIT 1",
            (broker, f"{date_prefix}%"),
        ).fetchone()
        return d(r["equity_krw"]) if r else None

    # ------------------------------------------------------------------
    # backtest results
    # ------------------------------------------------------------------
    def save_backtest(
        self,
        run_id: str,
        strategy: str,
        params: dict[str, Any],
        start_date: str,
        end_date: str,
        metrics: dict[str, Any],
        equity: list[dict[str, Any]],
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO backtests"
            "(run_id,created_at,strategy,params,start_date,end_date,metrics,equity)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                run_id,
                utcnow(),
                strategy,
                json.dumps(params, default=str),
                start_date,
                end_date,
                json.dumps(metrics, default=str),
                json.dumps(equity, default=str),
            ),
        )

    def list_backtests(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT run_id, created_at, strategy, start_date, end_date, metrics"
            " FROM backtests ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def get_backtest(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM backtests WHERE run_id=?", (run_id,)).fetchone()


class _Tx:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN")
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")
