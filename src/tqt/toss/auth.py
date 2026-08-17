"""OAuth2 client-credentials token management.

The token lives ~1 hour. We refresh it early (default 5 minutes before expiry)
so a long-running bot never presents an expired token mid-order. The token is
held **in memory only** — writing it to disk would put a bearer credential for a
real brokerage account on the filesystem for no real benefit, since re-issuing
costs one cheap call (AUTH group allows 5/s).
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

from .errors import TossTransportError, from_response

log = logging.getLogger(__name__)

TOKEN_PATH = "/oauth2/token"
DEFAULT_REFRESH_MARGIN = 300.0  # seconds before expiry to proactively refresh


class TokenManager:
    """Thread-safe cache for the bearer token."""

    def __init__(
        self,
        http: httpx.Client,
        client_id: str,
        client_secret: str,
        *,
        limiter=None,
        refresh_margin: float = DEFAULT_REFRESH_MARGIN,
        clock=time.monotonic,
    ) -> None:
        self._http = http
        self._client_id = client_id
        self._client_secret = client_secret
        self._limiter = limiter
        self._margin = refresh_margin
        self._clock = clock

        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: float = 0.0
        self.issued_count = 0

    # ------------------------------------------------------------------
    @property
    def expires_in(self) -> float:
        """Seconds until the cached token expires (0 if none/expired)."""
        return max(self._expires_at - self._clock(), 0.0)

    def invalidate(self) -> None:
        """Drop the cached token; the next ``token()`` re-issues.

        Called when the server says ``expired-token``/``invalid-token``, which can
        happen even before our own expiry estimate (e.g. clock skew, or Toss
        revoking on credential rotation).
        """
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    def token(self) -> str:
        """Return a valid bearer token, issuing one if needed."""
        with self._lock:
            if self._token and self._clock() < self._expires_at - self._margin:
                return self._token
            return self._issue_locked()

    # ------------------------------------------------------------------
    def _issue_locked(self) -> str:
        if self._limiter is not None:
            self._limiter.acquire("AUTH")

        try:
            resp = self._http.post(
                TOKEN_PATH,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise TossTransportError(f"token request failed: {exc}") from exc

        if resp.status_code >= 400:
            body: object
            try:
                body = resp.json()
            except ValueError:
                body = {"error": {"message": resp.text[:400]}}
            raise from_response(
                resp.status_code, body, request_id=resp.headers.get("X-Request-Id")
            )

        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise TossTransportError("token response contained no access_token")

        expires_in = float(payload.get("expires_in") or 3600)
        self._token = token
        self._expires_at = self._clock() + expires_in
        self.issued_count += 1
        log.info("issued Toss access token (expires_in=%.0fs)", expires_in)
        return token
