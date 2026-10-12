from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraudguard.serve.app import SECURITY_HEADERS, Settings, create_app
from fraudguard.serve.security import KeyStore, RateLimiter, hash_key

KEY = "test-key-123"
HEADERS = {"X-API-Key": KEY}
TX = {
    "amount": 899.0,
    "hour": 3,
    "merchant_category": "electronics",
    "distance_from_home_km": 420.0,
    "card_present": False,
    "is_foreign": True,
    "txn_count_24h": 8,
    "avg_amount_30d": 55.0,
    "account_age_days": 120,
    "new_device": True,
}
NORMAL_TX = TX | {
    "amount": 42.5,
    "hour": 13,
    "merchant_category": "grocery",
    "distance_from_home_km": 3.0,
    "card_present": True,
    "is_foreign": False,
    "txn_count_24h": 2,
    "account_age_days": 2400,
    "new_device": False,
}


def make_client(model_dir, rate: int = 1000, keys: str | None = None) -> TestClient:
    settings = Settings(
        model_dir=model_dir,
        api_key_hashes=hash_key(KEY) if keys is None else keys,
        rate_limit_per_minute=rate,
        enable_docs=False,
    )
    return TestClient(create_app(settings=settings))


@pytest.fixture()
def client(model_dir):
    with make_client(model_dir) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_predict_requires_api_key(client):
    assert client.post("/v1/predict", json=TX).status_code == 401
    assert client.post("/v1/predict", json=TX, headers={"X-API-Key": "wrong"}).status_code == 401


def test_predict_scores_suspicious_above_normal(client):
    risky = client.post("/v1/predict", json=TX, headers=HEADERS)
    normal = client.post("/v1/predict", json=NORMAL_TX, headers=HEADERS)
    assert risky.status_code == normal.status_code == 200
    body = risky.json()
    assert set(body) == {"fraud_score", "is_fraud", "request_id", "threshold", "model_version"}
    assert body["fraud_score"] > normal.json()["fraud_score"]
    assert body["is_fraud"] is True


@pytest.mark.parametrize(
    "patch",
    [
        {"amount": -5},
        {"hour": 24},
        {"merchant_category": "casino"},
        {"card_present": "yes"},
        {"txn_count_24h": "7"},
        {"is_admin": True},
    ],
)
def test_invalid_input_is_rejected(client, patch):
    assert client.post("/v1/predict", json=TX | patch, headers=HEADERS).status_code == 422


def test_nan_and_infinity_are_rejected(client):
    raw = '{"amount": NaN}'
    r = client.post(
        "/v1/predict", content=raw, headers=HEADERS | {"content-type": "application/json"}
    )
    assert r.status_code == 422


def test_oversized_body_is_rejected(client):
    r = client.post(
        "/v1/predict",
        content=b"x" * (70 * 1024),
        headers=HEADERS | {"content-type": "application/json"},
    )
    assert r.status_code == 413


def test_batch_limits(client):
    ok = client.post("/v1/predict/batch", json={"transactions": [TX, NORMAL_TX]}, headers=HEADERS)
    assert ok.status_code == 200
    assert len(ok.json()["predictions"]) == 2
    too_many = {"transactions": [TX] * 101}
    assert client.post("/v1/predict/batch", json=too_many, headers=HEADERS).status_code == 422


def test_security_headers_on_every_response(client):
    for r in (client.get("/health"), client.post("/v1/predict", json=TX)):
        for header, value in SECURITY_HEADERS.items():
            assert r.headers[header] == value


def test_docs_disabled_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_model_info_exposes_provenance(client, model_dir):
    info = client.get("/v1/model", headers=HEADERS).json()
    assert len(info["sha256"]) == 64
    assert info["metrics"]["roc_auc"] > 0.9


def test_metrics_endpoint(client):
    client.post("/v1/predict", json=TX, headers=HEADERS)
    client.post("/v1/predict", json=TX)
    text = client.get("/metrics").text
    assert "fraudguard_predictions_total" in text
    assert "fraudguard_auth_failures_total 1.0" in text


def test_rate_limit(model_dir):
    with make_client(model_dir, rate=3) as c:
        codes = [c.post("/v1/predict", json=TX, headers=HEADERS).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_fails_closed_without_keys(model_dir):
    with make_client(model_dir, keys="") as c:
        assert c.post("/v1/predict", json=TX, headers=HEADERS).status_code == 401


def test_tampered_model_prevents_startup(model_dir, tmp_path):
    for f in model_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    (tmp_path / "model.skops").write_bytes((model_dir / "model.skops").read_bytes() + b"\0")
    with pytest.raises(Exception, match="sha256 mismatch"), make_client(tmp_path):
        pass


def test_keystore_rejects_malformed_digests():
    with pytest.raises(ValueError):
        KeyStore.from_env_value("not-a-digest")


def test_rate_limiter_isolates_clients():
    rl = RateLimiter(rate_per_minute=1)
    assert rl.allow("a")[0]
    assert not rl.allow("a")[0]
    assert rl.allow("b")[0]
