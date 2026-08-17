"""Typed exceptions mapped from the Toss error envelope.

Every Toss error body looks like::

    {"error": {"requestId": "...", "code": "invalid-tick-size",
               "message": "...", "data": {...}}}

We map (status, code) onto specific exception classes so callers can branch on
*meaning* instead of string-matching messages. The distinction that matters most
operationally is: which errors are worth retrying, which mean "fix your config",
and which mean "this order will never work".
"""

from __future__ import annotations

from typing import Any


class TossError(Exception):
    """Base class for every Toss API failure."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        data: Any = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.data = data
        self.request_id = request_id

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        bits = [f"HTTP {self.status}" if self.status else "", self.code or "", self.message]
        out = " ".join(b for b in bits if b)
        if self.request_id:
            out += f" (requestId={self.request_id})"
        return out

    @property
    def retryable(self) -> bool:
        """Whether a bare retry could plausibly succeed."""
        return False


class TossTransportError(TossError):
    """Network-level failure: DNS, TLS, connect timeout, read timeout."""

    @property
    def retryable(self) -> bool:
        return True


class TossAuthError(TossError):
    """401. Token missing, malformed, or expired -> re-issue and retry once."""

    @property
    def retryable(self) -> bool:
        # The client re-issues the token before retrying, so this is recoverable.
        return self.code in {"expired-token", "invalid-token"}


class TossIPBlockedError(TossError):
    """403 edge-blocked: this machine's public IP is not on the Toss allowlist.

    The single most common cause of a bot silently dying on a home server, since
    residential IPs rotate. Register the current IP under
    Toss WTS -> 설정 -> Open API -> 허용 IP 관리.
    """


class TossForbiddenError(TossError):
    """403 forbidden: credentials lack the permission for this call."""


class TossNotFoundError(TossError):
    """404: unknown symbol, order, account, or API path."""


class TossConflictError(TossError):
    """409: already filled/canceled/modified, or a duplicate in-flight request."""


class TossRateLimitError(TossError):
    """429. Honour ``retry_after`` before trying again."""

    def __init__(self, *args: Any, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return True


class TossOrderRejectedError(TossError):
    """422: the order was understood but refused (funds, hours, limits, tick size).

    Never retried blindly — the request itself must change.
    """


class TossServerError(TossError):
    """5xx: transient Toss-side failure or scheduled maintenance."""

    @property
    def retryable(self) -> bool:
        return self.code != "maintenance"


# Codes that mean "the market/account will not accept this order right now".
# Useful for the runner to decide between "skip this symbol" and "halt everything".
HALT_WORTHY_CODES = frozenset(
    {
        "account-restricted",
        "prerequisite-required",
        "order-limit-exceeded",
        "maintenance",
    }
)

SKIP_SYMBOL_CODES = frozenset(
    {
        "stock-restricted",
        "price-out-of-range",
        "invalid-tick-size",
        "order-type-not-allowed",
        "market-not-supported-for-stock",
        "opposite-pending-order-exists",
        "stock-not-found",
        "amount-order-outside-regular-hours",
        "fractional-quantity-outside-regular-hours",
    }
)


_STATUS_MAP: dict[int, type[TossError]] = {
    401: TossAuthError,
    404: TossNotFoundError,
    409: TossConflictError,
    422: TossOrderRejectedError,
    429: TossRateLimitError,
}


def from_response(
    status: int, body: Any, *, request_id: str | None = None, retry_after: float | None = None
) -> TossError:
    """Build the right exception from an HTTP status and parsed JSON body."""
    err = {}
    if isinstance(body, dict):
        maybe = body.get("error")
        if isinstance(maybe, dict):
            err = maybe
    code = err.get("code")
    message = err.get("message") or f"Toss API returned HTTP {status}"
    data = err.get("data")
    rid = err.get("requestId") or request_id

    kwargs: dict[str, Any] = {"status": status, "code": code, "data": data, "request_id": rid}

    if status == 403:
        cls: type[TossError] = TossIPBlockedError if code == "edge-blocked" else TossForbiddenError
        return cls(message, **kwargs)
    if status == 429:
        return TossRateLimitError(message, retry_after=retry_after, **kwargs)
    if status >= 500:
        return TossServerError(message, **kwargs)
    if status == 400 and code == "account-header-required":
        return TossForbiddenError(message, **kwargs)

    return _STATUS_MAP.get(status, TossError)(message, **kwargs)
