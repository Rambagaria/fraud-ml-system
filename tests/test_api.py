import json

import pytest
from fastapi.testclient import TestClient

from fraud.api import create_app
from fraud.config import DECISIONS
from fraud.feature_store import InMemoryFeatureStore
from fraud.model import FraudModel

TXN = {"txn_id": "t1", "card_id": "c1", "amount": 42.5, "merchant_category": "grocery",
       "location": "city_01", "ts": 1_767_300_000}


class BrokenStore(InMemoryFeatureStore):
    def get_history(self, *a, **k):
        raise TimeoutError("redis timeout")


@pytest.fixture
def make_client(model_dir, tmp_path):
    def _make(store=None):
        app = create_app(model=FraudModel(model_dir), store=store or InMemoryFeatureStore(),
                         pred_log_path=tmp_path / "pred.jsonl")
        return TestClient(app)
    return _make


def test_health_and_ready(make_client):
    with make_client() as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/ready").json()["feature_store"] is True


def test_predict_returns_valid_decision_and_logs(make_client, tmp_path):
    with make_client() as c:
        r = c.post("/predict", json=TXN)
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] in DECISIONS and 0 <= body["score"] <= 1 and not body["degraded"]
    rec = json.loads((tmp_path / "pred.jsonl").read_text().splitlines()[0])
    assert rec["txn_id"] == "t1" and set(rec["features"]) >= {"amount", "txn_count_1h"}


def test_rejects_bad_payload(make_client):
    with make_client() as c:
        assert c.post("/predict", json={**TXN, "amount": -5}).status_code == 422
        assert c.post("/predict", json={k: v for k, v in TXN.items() if k != "card_id"}).status_code == 422


def test_blocklisted_card_declined_without_model(make_client):
    store = InMemoryFeatureStore()
    store.block("c1")
    with make_client(store) as c:
        body = c.post("/predict", json=TXN).json()
    assert body["decision"] == "DECLINE" and body["score"] is None
    assert "rule:blocklisted_card" in body["reasons"]


def test_degraded_mode_when_feature_store_fails(make_client):
    with make_client(BrokenStore()) as c:
        small = c.post("/predict", json=TXN).json()
        big = c.post("/predict", json={**TXN, "txn_id": "t2", "amount": 900}).json()
    assert small["degraded"] and big["degraded"]
    assert big["decision"] in ("STEP_UP", "DECLINE") and "rule:degraded_mode" in big["reasons"]


def test_metrics_exposed_and_reload_protected(make_client):
    with make_client() as c:
        c.post("/predict", json=TXN)
        assert "fraud_predictions_total" in c.get("/metrics").text
        assert c.post("/admin/reload").status_code == 401
