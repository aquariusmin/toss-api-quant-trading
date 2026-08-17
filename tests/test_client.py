"""Toss client behaviour, against a mocked HTTP layer.

The order-idempotency and retry tests are the important ones: they cover the case
where the network fails *after* an order may already have reached the exchange.
Getting that wrong means duplicate positions with real money.
"""

from decimal import Decimal

import httpx
import pytest
import respx

from tqt.config import Settings
from tqt.toss.client import TossClient
from tqt.toss.errors import (
    TossIPBlockedError,
    TossOrderRejectedError,
    TossRateLimitError,
    from_response,
)
from tqt.toss.ratelimit import RateLimiter

BASE = "https://openapi.tossinvest.com"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        toss_client_id="cid",
        toss_client_secret="csec",
        toss_account_seq=1,
        db_path=tmp_path / "c.db",
    )


class FastLimiter(RateLimiter):
    """No real sleeping — these tests are about protocol, not timing."""

    def acquire(self, group_name: str, tokens: float = 1.0) -> float:
        return 0.0


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    http = httpx.Client(base_url=BASE)
    c = TossClient(settings, http=http, limiter=FastLimiter(), max_retries=3)
    yield c
    c.close()


def _token_route():
    return respx.post(f"{BASE}/oauth2/token").mock(
        return_value=httpx.Response(
            200, json={"access_token": "tok", "token_type": "Bearer", "expires_in": 86400}
        )
    )


# ---------------------------------------------------------------------------
@respx.mock
def test_token_is_issued_once_and_reused(client):
    tok = _token_route()
    respx.get(f"{BASE}/api/v1/prices").mock(
        return_value=httpx.Response(
            200,
            json={"result": [{"symbol": "005930", "lastPrice": "74000", "currency": "KRW"}]},
        )
    )
    client.prices(["005930"])
    client.prices(["005930"])
    assert tok.call_count == 1, "token should be cached, not re-issued per call"


@respx.mock
def test_prices_parse_as_decimal_not_float(client):
    _token_route()
    respx.get(f"{BASE}/api/v1/prices").mock(
        return_value=httpx.Response(
            200,
            json={
                "result": [
                    {"symbol": "AAPL", "lastPrice": "307.08", "currency": "USD"}
                ]
            },
        )
    )
    px = client.price("AAPL").last_price
    assert isinstance(px, Decimal)
    assert px == Decimal("307.08")  # exact, not 307.08000000000001


@respx.mock
def test_429_is_retried_and_then_succeeds(client):
    _token_route()
    route = respx.get(f"{BASE}/api/v1/prices")
    route.side_effect = [
        httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={"error": {"code": "rate-limit-exceeded", "message": "too fast"}},
        ),
        httpx.Response(
            200,
            json={"result": [{"symbol": "005930", "lastPrice": "1", "currency": "KRW"}]},
        ),
    ]
    got = client.prices(["005930"])
    assert len(got) == 1
    assert route.call_count == 2


@respx.mock
def test_expired_token_triggers_reissue_then_retry(client):
    tok = _token_route()
    route = respx.get(f"{BASE}/api/v1/prices")
    route.side_effect = [
        httpx.Response(401, json={"error": {"code": "expired-token", "message": "expired"}}),
        httpx.Response(
            200,
            json={"result": [{"symbol": "005930", "lastPrice": "1", "currency": "KRW"}]},
        ),
    ]
    client.prices(["005930"])
    assert tok.call_count == 2, "a 401 must invalidate the cached token"


@respx.mock
def test_403_edge_blocked_maps_to_ip_error(client):
    """The single most common home-server failure: the public IP rotated and is
    no longer on the Toss allowlist."""
    _token_route()
    respx.get(f"{BASE}/api/v1/prices").mock(
        return_value=httpx.Response(
            403, json={"error": {"code": "edge-blocked", "message": "blocked"}}
        )
    )
    with pytest.raises(TossIPBlockedError):
        client.prices(["005930"])


@respx.mock
def test_422_order_rejection_is_not_retried(client):
    """A rejected order must change before being resent — blind retries would
    just hammer the API with the same invalid request."""
    _token_route()
    route = respx.post(f"{BASE}/api/v1/orders").mock(
        return_value=httpx.Response(
            422,
            json={
                "error": {
                    "code": "insufficient-buying-power",
                    "message": "not enough cash",
                }
            },
        )
    )
    with pytest.raises(TossOrderRejectedError) as exc:
        client.create_order("005930", "BUY", "MARKET", quantity=1)
    assert exc.value.code == "insufficient-buying-power"
    assert route.call_count == 1


@respx.mock
def test_order_carries_an_idempotency_key_by_default(client):
    """Without clientOrderId, a retried POST could open a second position."""
    _token_route()
    captured: dict = {}

    def handler(request):
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"result": {"orderId": "o1", "clientOrderId": "x"}})

    respx.post(f"{BASE}/api/v1/orders").mock(side_effect=handler)
    client.create_order("005930", "BUY", "MARKET", quantity=1)
    assert captured.get("clientOrderId"), "orders must be idempotent by default"
    assert captured["clientOrderId"].startswith("tqt-")
    assert captured["quantity"] == "1"
    assert "price" not in captured  # MARKET orders must not carry a price


@respx.mock
def test_request_in_progress_conflict_is_retried(client):
    """409 request-in-progress means our own idempotent order is mid-flight;
    asking again returns the original rather than creating a duplicate."""
    _token_route()
    route = respx.post(f"{BASE}/api/v1/orders")
    route.side_effect = [
        httpx.Response(
            409, json={"error": {"code": "request-in-progress", "message": "wait"}}
        ),
        httpx.Response(200, json={"result": {"orderId": "o1", "clientOrderId": "k"}}),
    ]
    ack = client.create_order("005930", "BUY", "MARKET", quantity=1, client_order_id="k")
    assert ack.order_id == "o1"
    assert route.call_count == 2


@respx.mock
def test_transport_error_on_order_is_not_silently_retried_into_a_duplicate(client):
    """Retries are allowed *because* the order is idempotent; assert the key is
    reused across attempts rather than regenerated."""
    _token_route()
    keys = []

    def handler(request):
        import json

        keys.append(json.loads(request.content)["clientOrderId"])
        if len(keys) == 1:
            raise httpx.ConnectTimeout("boom")
        return httpx.Response(200, json={"result": {"orderId": "o9", "clientOrderId": keys[0]}})

    respx.post(f"{BASE}/api/v1/orders").mock(side_effect=handler)
    ack = client.create_order("005930", "BUY", "MARKET", quantity=1)
    assert ack.order_id == "o9"
    assert len(set(keys)) == 1, "the idempotency key must be identical across retries"


# ---------------------------------------------------------------------------
def test_market_order_with_price_is_rejected_locally(client):
    with pytest.raises(ValueError, match="must not carry a price"):
        client.create_order("005930", "BUY", "MARKET", quantity=1, price=100)


def test_limit_order_without_price_is_rejected_locally(client):
    with pytest.raises(ValueError, match="require price"):
        client.create_order("005930", "BUY", "LIMIT", quantity=1)


def test_quantity_and_amount_are_mutually_exclusive(client):
    with pytest.raises(ValueError, match="exactly one"):
        client.create_order("AAPL", "BUY", "MARKET", quantity=1, order_amount=100)
    with pytest.raises(ValueError, match="exactly one"):
        client.create_order("AAPL", "BUY", "MARKET")


def test_invalid_candle_interval_is_rejected(client):
    with pytest.raises(ValueError, match="1m"):
        client.candles("005930", "5m")


def test_error_mapping_by_status_and_code():
    assert isinstance(
        from_response(403, {"error": {"code": "edge-blocked"}}), TossIPBlockedError
    )
    assert isinstance(
        from_response(429, {"error": {"code": "rate-limit-exceeded"}}), TossRateLimitError
    )
    assert isinstance(
        from_response(422, {"error": {"code": "stock-restricted"}}), TossOrderRejectedError
    )
    assert from_response(429, {}).retryable
    assert not from_response(422, {}).retryable


@respx.mock
def test_candle_pagination_walks_the_cursor(client):
    """History is paginated newest-first via nextBefore; the iterator must follow
    it and stop instead of looping."""
    _token_route()

    def handler(request):
        before = request.url.params.get("before")
        if before is None:
            return httpx.Response(
                200,
                json={
                    "result": {
                        "candles": [
                            _candle("2026-01-02"),
                            _candle("2026-01-01"),
                        ],
                        "nextBefore": "2025-12-31",
                    }
                },
            )
        return httpx.Response(
            200,
            json={"result": {"candles": [_candle("2025-12-30")], "nextBefore": None}},
        )

    respx.get(f"{BASE}/api/v1/candles").mock(side_effect=handler)
    bars = list(client.iter_candles("005930", "1d", max_bars=100))
    assert len(bars) == 3
    assert {b.timestamp[:10] for b in bars} == {"2026-01-02", "2026-01-01", "2025-12-30"}


def _candle(date: str) -> dict:
    return {
        "timestamp": f"{date}T00:00:00.000+09:00",
        "openPrice": "100",
        "highPrice": "110",
        "lowPrice": "90",
        "closePrice": "105",
        "volume": "1000",
        "currency": "KRW",
    }


@respx.mock
def test_account_list_is_cached(client):
    """ACCOUNT is limited to 1 request/second, so a double lookup on startup
    trips a 429."""
    _token_route()
    route = respx.get(f"{BASE}/api/v1/accounts").mock(
        return_value=httpx.Response(
            200,
            json={
                "result": [
                    {"accountNo": "130-11-111111", "accountSeq": 1, "accountType": "BROKERAGE"}
                ]
            },
        )
    )
    client.accounts()
    client.accounts()
    assert route.call_count == 1
    client.accounts(refresh=True)
    assert route.call_count == 2
