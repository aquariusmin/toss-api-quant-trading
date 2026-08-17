"""Central configuration, loaded from `.env` / process environment.

Design notes
------------
* Money never touches ``float``. Risk knobs that are ratios stay ``Decimal`` too,
  so position sizing is exact and reproducible.
* ``live`` mode is deliberately awkward to enable: it needs both ``TQT_MODE=live``
  and ``TQT_LIVE_CONFIRM`` set to an exact sentinel string. A typo in a config
  file should never be able to start sending real orders.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

LIVE_CONFIRM_SENTINEL = "I_UNDERSTAND_REAL_MONEY"


class Mode(str, Enum):
    """Which execution path the runner takes."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Toss credentials --------------------------------------------------
    toss_client_id: str = ""
    toss_client_secret: str = ""

    # The user's existing .env spells this `toss_bank_acount`; accept both that
    # and the correct spelling so nobody has to edit a working file.
    toss_bank_account: str = Field(
        default="",
        validation_alias=AliasChoices(
            "toss_bank_account", "toss_bank_acount", "TOSS_BANK_ACCOUNT", "TOSS_BANK_ACOUNT"
        ),
    )
    # The API wants a numeric `accountSeq`, not the account number. Left unset,
    # `tqt accounts` resolves it from the account number and prints it.
    toss_account_seq: int | None = None
    toss_base_url: str = "https://openapi.tossinvest.com"
    toss_timeout_seconds: float = 10.0

    # --- Mode --------------------------------------------------------------
    mode: Mode = Field(default=Mode.PAPER, validation_alias=AliasChoices("mode", "TQT_MODE"))
    live_confirm: str = Field(
        default="", validation_alias=AliasChoices("live_confirm", "TQT_LIVE_CONFIRM")
    )

    # --- Risk limits -------------------------------------------------------
    max_position_weight: Decimal = Field(
        default=Decimal("0.20"),
        validation_alias=AliasChoices("max_position_weight", "TQT_MAX_POSITION_WEIGHT"),
    )
    max_daily_loss_pct: Decimal = Field(
        default=Decimal("0.03"),
        validation_alias=AliasChoices("max_daily_loss_pct", "TQT_MAX_DAILY_LOSS_PCT"),
    )
    max_orders_per_day: int = Field(
        default=40, validation_alias=AliasChoices("max_orders_per_day", "TQT_MAX_ORDERS_PER_DAY")
    )
    max_gross_exposure: Decimal = Field(
        default=Decimal("1.00"),
        validation_alias=AliasChoices("max_gross_exposure", "TQT_MAX_GROSS_EXPOSURE"),
    )
    paper_starting_cash_krw: Decimal = Field(
        default=Decimal("1000000"),
        validation_alias=AliasChoices(
            "paper_starting_cash_krw", "TQT_PAPER_STARTING_CASH_KRW"
        ),
    )

    # --- Notifications -----------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""

    # --- Dashboard ---------------------------------------------------------
    dashboard_host: str = Field(
        default="0.0.0.0", validation_alias=AliasChoices("dashboard_host", "TQT_DASHBOARD_HOST")
    )
    dashboard_port: int = Field(
        default=8000, validation_alias=AliasChoices("dashboard_port", "TQT_DASHBOARD_PORT")
    )
    dashboard_token: str = Field(
        default="", validation_alias=AliasChoices("dashboard_token", "TQT_DASHBOARD_TOKEN")
    )

    # --- Storage -----------------------------------------------------------
    db_path: Path = Field(
        default=Path("data/tqt.db"), validation_alias=AliasChoices("db_path", "TQT_DB_PATH")
    )
    log_level: str = Field(
        default="INFO", validation_alias=AliasChoices("log_level", "TQT_LOG_LEVEL")
    )

    # ----------------------------------------------------------------------
    @field_validator("db_path")
    @classmethod
    def _absolutize(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v)

    @field_validator("toss_bank_account")
    @classmethod
    def _normalize_account(cls, v: str) -> str:
        """Strip dashes/spaces so `123-45-678901` compares equal to the API form."""
        return re.sub(r"[^0-9]", "", v or "")

    @model_validator(mode="after")
    def _guard_live_mode(self) -> Settings:
        if self.mode is Mode.LIVE and self.live_confirm != LIVE_CONFIRM_SENTINEL:
            raise ValueError(
                "TQT_MODE=live requires TQT_LIVE_CONFIRM="
                f"{LIVE_CONFIRM_SENTINEL}. Refusing to start in live mode."
            )
        return self

    # --- convenience -------------------------------------------------------
    @property
    def has_credentials(self) -> bool:
        return bool(self.toss_client_id and self.toss_client_secret)

    @property
    def is_live(self) -> bool:
        return self.mode is Mode.LIVE

    def require_credentials(self) -> None:
        if not self.has_credentials:
            raise RuntimeError(
                "toss_client_id / toss_client_secret are not set. "
                "Copy .env.example to .env and fill them in."
            )


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
