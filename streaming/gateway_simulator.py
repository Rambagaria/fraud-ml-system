"""Payment gateway simulator: replays held-out transactions as live traffic.

For each transaction (like a real card network):
  1. Synchronous auth call: POST /predict, must answer within the latency budget.
  2. After the auth, emit a `transactions.completed` event (declines included) to Kafka. The feature
     updater consumes it and updates the card's history in Redis asynchronously.
     (No Kafka? --inline-redis writes the update directly, for quick local runs.)

Because replayed data has labels, it prints ONLINE recall/precision at the end:
if those disagree with offline test metrics, you have training-serving skew.

Drift demo: --amount-mult 3 --force-location city_99 shifts traffic so the drift
job alerts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud.config import DAY_S  # noqa: E402

TOPIC = "transactions.completed"


def make_emitter(args):
    if args.kafka:
        from confluent_kafka import Producer
        from confluent_kafka.admin import AdminClient, NewTopic

        AdminClient({"bootstrap.servers": args.kafka}).create_topics(
            [NewTopic(TOPIC, num_partitions=3, replication_factor=1)])  # ignore "already exists"
        # acks=all + idempotence: no lost or duplicated events from the producer side
        prod = Producer({"bootstrap.servers": args.kafka, "acks": "all", "enable.idempotence": True,
                         "linger.ms": 5})

        def emit(event: dict):
            # key = card_id -> all events of a card go to one partition -> ordered per card
            prod.produce(TOPIC, key=event["card_id"], value=json.dumps(event))
            prod.poll(0)
        return emit, prod.flush

    if args.inline_redis:
        from fraud.feature_store import make_store
        store = make_store(args.inline_redis, timeout_s=1)

        def emit(event: dict):
            store.record_txn(event["card_id"], event["txn_id"], event["ts"], event["amount"], event["location"])
        return emit, lambda: None

    return (lambda event: None), (lambda: None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--start-day", type=int, help="absolute day index; default = last 7 days")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--rate", type=float, default=0, help="target req/s (0 = as fast as possible)")
    ap.add_argument("--kafka", help="bootstrap servers, e.g. localhost:19092")
    ap.add_argument("--inline-redis", help="redis URL to update directly when Kafka isn't running")
    ap.add_argument("--amount-mult", type=float, default=1.0)
    ap.add_argument("--force-location")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    start = a.start_day if a.start_day is not None else int(df.ts.max() // DAY_S) - 6
    df = df[df.ts >= start * DAY_S].sort_values("ts").head(a.limit).copy()
    df["amount"] = (df.amount * a.amount_mult).round(2)
    if a.force_location:
        df["location"] = a.force_location

    emit, flush = make_emitter(a)
    decisions, lat, y_true, flagged = Counter(), [], [], []
    gap = 1 / a.rate if a.rate else 0
    with httpx.Client(base_url=a.api, timeout=2.0) as client:
        for r in df.itertuples(index=False):
            t0 = time.perf_counter()
            txn = {"txn_id": r.txn_id, "card_id": r.card_id, "amount": float(r.amount),
                   "merchant_category": r.merchant_category, "location": r.location, "ts": int(r.ts)}
            resp = client.post("/predict", json=txn)
            resp.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
            d = resp.json()["decision"]
            decisions[d] += 1
            y_true.append(int(r.is_fraud))
            flagged.append(d != "APPROVE")
            # Record every ATTEMPT, declined ones too: training history contains all
            # attempts, and a burst of declines is itself a strong fraud signal.
            emit({**txn, "decision": d})
            if gap:
                time.sleep(max(0.0, gap - (time.perf_counter() - t0)))
    flush()

    y, f = np.array(y_true), np.array(flagged)
    tp = int((y & f).sum())
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print(f"replayed {len(df):,} txns from day {start} | decisions {dict(decisions)}")
    print(f"client-side latency ms: p50={p50:.1f} p95={p95:.1f} p99={p99:.1f}")
    print(f"ONLINE recall={tp / max(int(y.sum()), 1):.3f} precision={tp / max(int(f.sum()), 1):.3f} "
          f"(fraud in sample: {int(y.sum())})")


if __name__ == "__main__":
    main()
