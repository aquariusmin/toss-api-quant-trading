"""Shared fixtures. Nothing here touches the network or the real database."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from tqt.data.store import Store
from tqt.universe import Asset, Sleeve


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "test.db")


@pytest.fixture
def sleeve() -> Sleeve:
    return Sleeve(
        key="test",
        title="test sleeve",
        description="",
        country="KR",
        risk=[
            Asset("111111", "risk A", "KR", "KRW", "equity"),
            Asset("222222", "risk B", "KR", "KRW", "equity"),
        ],
        safe=[Asset("333333", "safe", "KR", "KRW", "bond")],
    )


@pytest.fixture
def prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic OHLC: A trends up, B trends down, safe is flat.

    Business-day index so month-end rebalance dates exist.
    """
    idx = pd.bdate_range("2020-01-01", periods=400)
    n = len(idx)
    up = 10_000 * (1 + np.arange(n) * 0.001)
    down = 10_000 * (1 - np.arange(n) * 0.0005)
    flat = np.full(n, 10_000.0)

    closes = pd.DataFrame(
        {"111111": up, "222222": down, "333333": flat}, index=idx
    )
    # Opens sit a deliberately large 5% below closes. The gap has to be bigger
    # than anything the daily trend plus slippage could produce, otherwise
    # "filled at next open" and "filled at the decision close" evaluate to nearly
    # the same number and a lookahead test cannot tell them apart.
    opens = closes * 0.95
    return opens, closes


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Hermetic settings.

    ``_env_file=None`` stops pydantic-settings reading the developer's real
    ``.env``. Without it the suite silently inherits live credentials and
    notification config — which makes results machine-dependent and leaks real
    values into test output.
    """
    from tqt.config import Settings

    for var in (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "DISCORD_WEBHOOK_URL",
        "TQT_DASHBOARD_TOKEN",
        "TQT_MODE",
        "TOSS_ACCOUNT_SEQ",
    ):
        monkeypatch.delenv(var, raising=False)

    return Settings(
        _env_file=None,
        toss_client_id="test-id",
        toss_client_secret="test-secret",
        toss_bank_account="1234567890",
        db_path=tmp_path / "risk.db",
        max_position_weight=Decimal("0.20"),
        max_daily_loss_pct=Decimal("0.03"),
        max_orders_per_day=10,
        max_gross_exposure=Decimal("1.00"),
    )
