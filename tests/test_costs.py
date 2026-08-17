"""Cost model. Understating costs is the main way a good-looking backtest turns
into a losing account, so these assert real rates rather than "close enough"."""

from decimal import Decimal

from tqt.backtest.costs import CostModel


def test_kr_etf_is_exempt_from_transaction_tax():
    """The ETF exemption is worth ~30bp per round trip — it must not be assumed
    away, and it must not be applied to individual stocks."""
    cm = CostModel()
    etf = cm.cost_of(
        side="SELL", country="KR", price=Decimal(10_000), quantity=Decimal(10), is_etf=True
    )
    stock = cm.cost_of(
        side="SELL", country="KR", price=Decimal(10_000), quantity=Decimal(10), is_etf=False
    )
    assert etf.tax == 0
    assert stock.tax == Decimal("0.0015") * Decimal(100_000)  # 150 KRW on 100k
    assert stock.total > etf.total


def test_no_transaction_tax_on_buys():
    cm = CostModel()
    buy = cm.cost_of(
        side="BUY", country="KR", price=Decimal(10_000), quantity=Decimal(10), is_etf=False
    )
    assert buy.tax == 0


def test_us_sell_adds_regulatory_fees_but_buy_does_not():
    cm = CostModel()
    buy = cm.cost_of(side="BUY", country="US", price=Decimal(100), quantity=Decimal(10))
    sell = cm.cost_of(side="SELL", country="US", price=Decimal(100), quantity=Decimal(10))
    assert buy.regulatory == 0
    assert sell.regulatory > 0


def test_fx_spread_only_applies_to_us_trades():
    cm = CostModel()
    kr = cm.cost_of(side="BUY", country="KR", price=Decimal(10_000), quantity=Decimal(1))
    us = cm.cost_of(side="BUY", country="US", price=Decimal(100), quantity=Decimal(1))
    assert kr.fx == 0
    assert us.fx > 0

    no_fx = CostModel(charge_fx_on_us_trades=False)
    assert no_fx.cost_of(
        side="BUY", country="US", price=Decimal(100), quantity=Decimal(1)
    ).fx == 0


def test_us_round_trip_is_materially_more_expensive_than_kr_etf():
    """This asymmetry is the reason the default portfolio points high-turnover
    strategies at KR-listed ETFs. If it ever stops holding, revisit the plan."""
    cm = CostModel()
    kr = cm.round_trip_bps("KR", is_etf=True)
    us = cm.round_trip_bps("US", is_etf=True)
    assert us > kr * Decimal(2)


def test_zero_quantity_costs_nothing():
    cm = CostModel()
    assert cm.cost_of(
        side="BUY", country="KR", price=Decimal(10_000), quantity=Decimal(0)
    ).total == 0


def test_slippage_can_be_excluded_to_avoid_double_counting():
    """The engine shifts the fill price instead of booking slippage as a cash
    cost. Charging both would double-count."""
    cm = CostModel()
    with_s = cm.cost_of(
        side="BUY", country="KR", price=Decimal(10_000), quantity=Decimal(10)
    )
    without = cm.cost_of(
        side="BUY",
        country="KR",
        price=Decimal(10_000),
        quantity=Decimal(10),
        include_slippage=False,
    )
    assert with_s.slippage > 0
    assert without.slippage == 0
    assert without.commission == with_s.commission


def test_from_api_uses_live_rates():
    class FakeCommission:
        def __init__(self, mc, rate):
            self.market_country = mc
            self.commission_rate = Decimal(rate)

    class FakeClient:
        def commissions(self):
            return [FakeCommission("KR", "0.00015"), FakeCommission("US", "0.001")]

    cm = CostModel.from_api(FakeClient())
    assert cm.kr_commission_rate == Decimal("0.00015")
    assert cm.us_commission_rate == Decimal("0.001")


def test_costs_are_additive():
    cm = CostModel()
    a = cm.cost_of(side="BUY", country="KR", price=Decimal(1000), quantity=Decimal(1))
    b = cm.cost_of(side="SELL", country="KR", price=Decimal(1000), quantity=Decimal(1))
    assert (a + b).total == a.total + b.total
