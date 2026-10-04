"""Kafka consumer that keeps the online feature store fresh.

Delivery semantics: at-least-once. We commit the offset only AFTER the Redis write
succeeds, so a crash replays a few events instead of losing them. Replays are safe
because writes are idempotent (the txn_id is part of the Redis set member).

Ordering: events are keyed by card_id, so each card's events land on one partition
and are processed in order. Scale out by adding consumers to the group (up to the
partition count).
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud.config import REDIS_URL  # noqa: E402
from fraud.feature_store import make_store  # noqa: E402

TOPIC = "transactions.completed"
running = True


def stop(*_):
    global running
    running = False


def main() -> None:
    from confluent_kafka import Consumer, KafkaError

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kafka", default="localhost:19092")
    ap.add_argument("--redis-url", default=REDIS_URL or "redis://localhost:6379/0")
    ap.add_argument("--group", default="feature-updater")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    store = make_store(a.redis_url, timeout_s=1)
    consumer = Consumer({"bootstrap.servers": a.kafka, "group.id": a.group,
                         "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    consumer.subscribe([TOPIC])
    processed, failed, last_report = 0, 0, time.time()
    print(f"consuming {TOPIC} from {a.kafka} -> {a.redis_url}", flush=True)
    try:
        while running:
            msg = consumer.poll(0.5)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"kafka error: {msg.error()}", flush=True)
                continue
            try:
                e = json.loads(msg.value())
                store.record_txn(e["card_id"], e["txn_id"], int(e["ts"]), float(e["amount"]), e["location"])
            except (ValueError, KeyError) as err:
                # Poison message: don't block the partition forever. In production this
                # goes to a dead-letter topic for inspection.
                failed += 1
                print(f"skipping bad event at offset {msg.offset()}: {err}", flush=True)
            consumer.commit(message=msg, asynchronous=False)
            processed += 1
            if time.time() - last_report > 10:
                print(f"processed={processed} failed={failed}", flush=True)
                last_report = time.time()
    finally:
        consumer.close()  # commits nothing extra; triggers a clean group rebalance


if __name__ == "__main__":
    main()
