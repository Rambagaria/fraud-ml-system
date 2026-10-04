"""Real-time fraud scoring API.

Hot path for one transaction:
  1. Validate input (Pydantic)            -> 422 on bad payloads
  2. One Redis read: card history + cached reference tables + blocklist
  3. compute_features (same code as training)
  4. XGBoost score -> tier -> post-inference guardrails
  5. Respond; prediction log is written AFTER the response (background task)

If Redis is slow or down (50 ms timeout), we do not fail the payment: we score with
empty history + default rates, flag the response as degraded, and step up larger
amounts. That's a deliberate fail-open-with-friction choice.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from fraud import config
from fraud.feature_store import RefTables, make_store
from fraud.features import compute_features
from fraud.model import FraudModel
from fraud.rules import post_inference, pre_inference

log = logging.getLogger("fraud.api")

PREDICTIONS = Counter("fraud_predictions_total", "Decisions returned", ["decision", "model_version"])
LATENCY = Histogram("fraud_request_latency_seconds", "End-to-end /predict latency",
                    buckets=[.001, .0025, .005, .01, .02, .05, .1, .25, .5])
SCORES = Histogram("fraud_score", "Model score distribution",
                   buckets=[.01, .05, .1, .25, .5, .75, .9, .95, .99, 1.0])
STORE_ERRORS = Counter("fraud_feature_store_errors_total", "Feature store failures (degraded mode)")
MODEL_INFO = Gauge("fraud_model_info", "Loaded model version", ["model_version"])


class Txn(BaseModel):
    txn_id: str = Field(min_length=1, max_length=64)
    card_id: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, le=1_000_000)
    merchant_category: str = Field(min_length=1, max_length=32)
    location: str = Field(min_length=1, max_length=64)
    ts: int | None = Field(default=None, description="Unix seconds; defaults to now")


class Decision(BaseModel):
    txn_id: str
    decision: str
    score: float | None
    model_version: str
    reasons: list[str]
    degraded: bool
    latency_ms: float


class PredictionLogger:
    """Append-only JSONL log of features + score. Feeds drift monitoring and, once
    labels arrive, becomes the next training set with exactly the features we served."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")


def create_app(model=None, store=None, pred_log_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or FraudModel(config.MODEL_DIR, config.MODEL_VERSION)
        app.state.store = store or make_store(config.REDIS_URL, config.REDIS_TIMEOUT_S)
        app.state.pred_log = PredictionLogger(pred_log_path or config.PRED_LOG_PATH)
        MODEL_INFO.labels(app.state.model.version).set(1)
        log.info("loaded model %s", app.state.model.version)
        yield

    app = FastAPI(title="Real-time fraud scoring", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        try:
            store_ok = app.state.store.ping()
        except Exception:
            store_ok = False
        return {"model_version": app.state.model.version, "feature_store": store_ok}

    @app.post("/predict", response_model=Decision)
    def predict(txn: Txn, background: BackgroundTasks):
        t0 = time.perf_counter()
        m, store = app.state.model, app.state.store
        ts = txn.ts or int(time.time())

        degraded = False
        try:
            blocked = store.is_blocked(txn.card_id)
            history = store.get_history(txn.card_id, ts)
            ref = store.get_reference()
        except Exception as e:  # timeout, connection refused, ...
            log.warning("feature store unavailable, degraded mode: %s", e)
            STORE_ERRORS.inc()
            degraded, blocked, history, ref = True, False, [], RefTables()

        decision, reasons = pre_inference(blocked=blocked)
        score, feats = None, None
        if decision is None:
            feats = compute_features(ts=ts, amount=txn.amount, location=txn.location, history=history,
                                     loc_rate=ref.loc(txn.location), mcc_rate=ref.mcc(txn.merchant_category))
            score = m.score(feats)
            decision, post = post_inference(m.tier(score), amount=txn.amount, feats=feats, degraded=degraded)
            reasons += post
            SCORES.observe(score)

        latency = time.perf_counter() - t0
        LATENCY.observe(latency)
        PREDICTIONS.labels(decision, m.version).inc()
        background.add_task(app.state.pred_log.write, {
            "ts": ts, "logged_at": int(time.time()), "txn_id": txn.txn_id, "card_id": txn.card_id,
            "model_version": m.version, "score": score, "decision": decision,
            "reasons": reasons, "degraded": degraded, "features": feats,
            "inputs": {"location": txn.location, "merchant_category": txn.merchant_category},
        })
        return Decision(txn_id=txn.txn_id, decision=decision, score=score, model_version=m.version,
                        reasons=reasons, degraded=degraded, latency_ms=round(latency * 1000, 3))

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reload")
    def reload(x_admin_token: str = Header(default="")):
        """Load whatever PRODUCTION points to now (promotion or rollback) without a restart."""
        if x_admin_token != config.ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="bad admin token")
        new = FraudModel(config.MODEL_DIR, config.MODEL_VERSION)
        old = app.state.model.version
        app.state.model = new  # atomic reference swap; in-flight requests finish on the old model
        MODEL_INFO.labels(old).set(0)
        MODEL_INFO.labels(new.version).set(1)
        return {"previous": old, "current": new.version}

    return app


app = create_app()
