"""Risk limits. Every one of these exists because of a specific way an
automated account gets destroyed, so each is asserted explicitly."""

from decimal import Decimal

from tqt.data.store import Store
from tqt.execution.broker import AccountSnapshot, OrderIntent, Position
from tqt.risk import RiskManager


def _snapshot(cash=Decimal(1_000_000), positions=None, prices=None):
    return AccountSnapshot(
        cash_krw=cash,
        positions=positions or {},
        usd_krw=Decimal(1400),
        prices=prices or {},
    )


def _rm(settings, **kw) -> RiskManager:
    return RiskManager(settings, Store(settings.db_path), "paper", **kw)


def test_kill_switch_persists_across_instances(settings):
    """A bot that forgets it was halted when the Pi reboots is not halted."""
    a = _rm(settings)
    a.halt("manual test")
    assert a.is_halted

    b = _rm(settings)  # simulates a fresh process
    assert b.is_halted
    assert "manual test" in b.halt_reason

    b.resume()
    assert not _rm(settings).is_halted


def test_daily_loss_breach_halts_for_the_day(settings):
    rm = _rm(settings)
    store = rm.store
    from tqt.risk import today_kst

    store.record_equity(
        "paper", Decimal(1_000_000), Decimal(1_000_000), Decimal(0),
        ts=f"{today_kst()}T00:00:01+09:00",
    )
    # -4% against a -3% limit.
    snap = _snapshot(cash=Decimal(960_000))
    verdict = rm.check_cycle(snap)
    assert not verdict
    assert "daily loss" in verdict.reason
    assert rm.is_halted


def test_daily_loss_within_limit_allows_trading(settings):
    rm = _rm(settings)
    from tqt.risk import today_kst

    rm.store.record_equity(
        "paper", Decimal(1_000_000), Decimal(1_000_000), Decimal(0),
        ts=f"{today_kst()}T00:00:01+09:00",
    )
    assert rm.check_cycle(_snapshot(cash=Decimal(985_000)))  # -1.5%


def test_position_weight_limit_blocks_concentration(settings):
    rm = _rm(settings)
    snap = _snapshot()
    # 30% of a 1,000,000 account against a 20% limit.
    intent = OrderIntent(symbol="111111", side="BUY", quantity=Decimal(30))
    v = rm.check_order(intent, snap, price=Decimal(10_000))
    assert not v
    assert "weight would reach" in v.reason


def test_existing_holding_counts_toward_the_weight_limit(settings):
    rm = _rm(settings)
    snap = _snapshot(
        cash=Decimal(850_000),
        positions={"111111": Position("111111", Decimal(15), Decimal(10_000), "KRW")},
        prices={"111111": Decimal(10_000)},
    )
    # Already 15% held; adding another 10% would breach 20%.
    intent = OrderIntent(symbol="111111", side="BUY", quantity=Decimal(10))
    assert not rm.check_order(intent, snap, price=Decimal(10_000))


def test_defensive_assets_get_a_higher_weight_limit(settings):
    """Parking 25% in a government-bond ETF is not the concentration risk the
    limit guards against. Without this exemption the cap silently changes the
    strategy by forcing the money to sit in cash instead."""
    plain = _rm(settings)
    lenient = _rm(settings, defensive_symbols=frozenset({"333333"}))

    intent = OrderIntent(symbol="333333", side="BUY", quantity=Decimal(25))
    snap = _snapshot()
    assert not plain.check_order(intent, snap, price=Decimal(10_000))
    assert lenient.check_order(intent, snap, price=Decimal(10_000))


def test_gross_exposure_cap_enforces_a_cash_buffer(settings):
    """At the default 100% cap the cash check already prevents leverage, so this
    limit's real job is honouring a deliberately lower ceiling — "never be more
    than 50% invested"."""
    settings.max_gross_exposure = Decimal("0.50")
    settings.max_position_weight = Decimal("1.00")  # isolate the exposure check
    rm = _rm(settings)

    snap = _snapshot(
        cash=Decimal(600_000),
        positions={"222222": Position("222222", Decimal(40), Decimal(10_000), "KRW")},
        prices={"222222": Decimal(10_000)},
    )
    # Equity 1,000,000; already 40% invested. Another 20% would reach 60% > 50%.
    v = rm.check_order(
        OrderIntent(symbol="111111", side="BUY", quantity=Decimal(20)),
        snap,
        price=Decimal(10_000),
    )
    assert not v
    assert "exposure" in v.reason

    # A smaller buy that lands at 45% is fine.
    assert rm.check_order(
        OrderIntent(symbol="111111", side="BUY", quantity=Decimal(5)),
        snap,
        price=Decimal(10_000),
    )


def test_buy_beyond_cash_is_refused(settings):
    rm = _rm(settings)
    snap = _snapshot(cash=Decimal(50_000))
    intent = OrderIntent(symbol="111111", side="BUY", quantity=Decimal(10))
    v = rm.check_order(intent, snap, price=Decimal(10_000))
    assert not v


def test_cannot_sell_what_is_not_held(settings):
    rm = _rm(settings)
    intent = OrderIntent(symbol="111111", side="SELL", quantity=Decimal(5))
    v = rm.check_order(intent, _snapshot())
    assert not v
    assert "nothing held" in v.reason


def test_cannot_sell_more_than_held(settings):
    rm = _rm(settings)
    snap = _snapshot(
        positions={"111111": Position("111111", Decimal(3), Decimal(10_000), "KRW")},
        prices={"111111": Decimal(10_000)},
    )
    assert not rm.check_order(
        OrderIntent(symbol="111111", side="SELL", quantity=Decimal(5)), snap
    )
    assert rm.check_order(
        OrderIntent(symbol="111111", side="SELL", quantity=Decimal(3)), snap
    )


def test_order_budget_caps_a_runaway_loop(settings):
    """A bug that places orders in a loop should hit a wall, not a commission bill."""
    rm = _rm(settings)
    from tqt.data.store import utcnow

    for i in range(settings.max_orders_per_day):
        rm.store.conn.execute(
            "INSERT INTO orders(order_id,broker,symbol,side,order_type,quantity,status,"
            "submitted_at) VALUES(?,?,?,?,?,?,?,?)",
            (f"o{i}", "paper", "111111", "BUY", "MARKET", "1", "FILLED", utcnow()),
        )
    assert rm.orders_today() == settings.max_orders_per_day
    assert rm.remaining_order_budget() == 0
    assert not rm.check_cycle(_snapshot())


def test_blocking_stock_warnings_prevent_a_buy(settings):
    """Refuses to accumulate a stock in 정리매매 (delisting liquidation)."""

    class FakeClient:
        def buy_block_reasons(self, symbol):
            return ["LIQUIDATION_TRADING"] if symbol == "111111" else []

    rm = _rm(settings, client=FakeClient())
    snap = _snapshot()
    v = rm.check_order(
        OrderIntent(symbol="111111", side="BUY", quantity=Decimal(1)), snap,
        price=Decimal(10_000),
    )
    assert not v
    assert "LIQUIDATION_TRADING" in v.reason

    assert rm.check_order(
        OrderIntent(symbol="222222", side="BUY", quantity=Decimal(1)), snap,
        price=Decimal(10_000),
    )


def test_zero_equity_is_refused(settings):
    rm = _rm(settings)
    v = rm.check_order(
        OrderIntent(symbol="111111", side="BUY", quantity=Decimal(1)),
        _snapshot(cash=Decimal(0)),
        price=Decimal(10_000),
    )
    assert not v
