# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Package management — uv

**Use `uv` as the package manager.**

- Add runtime deps with `uv add <pkg>`.
- Add dev/test deps with `uv add --group test <pkg>`.
- **Do NOT edit the `pyproject.toml` dependency lists by hand**, and **do NOT use
  `pip install`.** `uv add` keeps `uv.lock` in sync; hand-editing does not, and a
  drifted lockfile means the Docker build (`uv sync --locked`) fails or, worse,
  the Pi runs different versions than were tested.
- Run Python and any Python-installed binaries via `uv run`:
  ```bash
  uv run pytest
  uv run ruff check src/ tests/
  uv run tqt doctor
  uv run python -c "..."
  ```
  This guarantees the project venv is used without activating it.
- After pulling changes that touch `pyproject.toml` or `uv.lock`, run `uv sync`
  before running anything.
- **Do NOT reintroduce `requirements.txt`.** The lockfile is `uv.lock`, and it is
  committed.
- Inside the Docker image the venv is on `PATH` (`/app/.venv/bin`), so `python ...`
  and `tqt ...` work directly — **no `uv run` prefix** in `docker-compose.yml`
  commands or in `Dockerfile` `CMD`/`ENTRYPOINT`.

The `test` group is listed in `[tool.uv] default-groups`, so a bare `uv sync`
installs it and the checkout is immediately testable. The image builds with
`--no-default-groups` to keep pytest/ruff out of the runtime.

## Verifying a change

```bash
uv run pytest                          # 113 tests, should be fast (~1s)
uv run ruff check src/ tests/
uv run tqt doctor                      # hits the live Toss API
```

## Safety rules — this bot trades real money

These are not style preferences. Violating them can lose money.

1. **Never use `float` for money, prices, quantities, or costs.** Everything is
   `Decimal`, stored as TEXT in SQLite. Toss returns numbers as JSON strings
   precisely so precision survives; parsing them as float throws that away.
   `float` is fine for indicator math (a moving average is a statistic, not a
   ledger entry).

2. **Never let a strategy see data past its decision bar.** `Backtester` slices
   history before calling `target_weights`, and fills happen at the *next* bar's
   open. If you change the engine, keep
   `tests/test_engine.py::test_strategy_never_sees_data_past_the_decision_date`
   and `::test_fills_happen_at_the_next_bar_open_not_the_decision_close` passing —
   they are the guard against the most damaging class of backtest bug.

3. **Order creation must stay idempotent.** Every order carries a
   `clientOrderId`. This is what makes retrying a timed-out POST safe instead of
   opening a duplicate position. Do not remove it or make it optional.

4. **Never bypass `RiskManager`.** All orders go through `check_cycle` and
   `check_order`. The kill switch lives in SQLite so it survives a restart —
   don't cache it in memory.

5. **Live mode requires two explicit settings** (`TQT_MODE=live` *and*
   `TQT_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY`). Don't add a shortcut around this,
   and never default anything to live.

6. **Don't commit secrets.** `.env` is gitignored and the repo is public. Before
   committing, confirm with `git check-ignore -v .env`. Test fixtures pass
   `_env_file=None` so the suite never reads the real `.env` — keep it that way.

## Costs are not an implementation detail

Toss charges **0.015% on KR trades and 0.10% on US trades** (read live via
`CostModel.from_api()`, not hardcoded). With the FX spread that is ~50bp per US
round trip against ~23bp for a KR ETF. Any new strategy should be checked against
`turnover x round-trip cost` before its returns are taken seriously — a signal
turning over 6x/year bleeds ~3%/yr in the US sleeve.

KR ETFs are exempt from 증권거래세; individual KR stocks pay 0.15% on every sell.
That is why `Asset.is_etf` is set explicitly per asset rather than inferred.

## Architecture notes

- **Strategies return target weights only.** Order construction, cash management
  and risk checks are separate layers. This is what lets the same strategy object
  drive backtest, paper and live. If paper and live ran different logic, the
  paper stage would prove nothing — don't add a code path that only one uses.
- **The client is synchronous.** At this frequency concurrency buys nothing and
  sync code is easier to reason about with money at stake. FastAPI endpoints are
  plain `def` so they run in a threadpool.
- **Inbound API enums stay `str`.** Toss documents that clients must tolerate
  unknown enum values; a new order status must not crash a running bot. Enums are
  only used for values *we* send.
- **Rate limits are respected before the fact**, per documented API group, in
  `toss/ratelimit.py`. `ACCOUNT` at 1 req/s is the tightest.

## API constraints worth remembering

Measured, not assumed. They shaped the design:

- No sandbox → paper trading is our own engine.
- REST only, no WebSocket → polling.
- Candle intervals are `1d` and `1m` only; **1m history covers only ~4 days**, so
  intraday backtesting is not viable. Daily history reaches 2010.
- Candle pages are max 200 bars, **newest-first**, walked via `nextBefore`.
- The IP allowlist is mandatory; an unregistered IP gets `403 edge-blocked`.
- Only 종합매매 (BROKERAGE) accounts are exposed — no pension accounts.

## Style

- Match the surrounding code: type hints, `from __future__ import annotations`,
  module docstrings that explain *why* rather than restating the code.
- Comments earn their place by explaining a non-obvious decision or a trap. Don't
  narrate what the line already says.
- Korean is used for user-facing strings (CLI output, notifications, dashboard,
  README); English for code, comments and commit messages.
