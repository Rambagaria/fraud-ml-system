"""Concurrent load test for /predict. Reports throughput and latency percentiles.

  python scripts/load_test.py --requests 5000 --concurrency 32
"""
import argparse
import asyncio
import random
import time

import httpx
import numpy as np

MCCS = ["grocery", "gas", "restaurant", "retail", "travel", "online", "electronics", "gift_cards"]
REPLAY_TS = 1_771_804_800  # inside the replay window so cards have history


async def worker(client, queue, lat, errors):
    while True:
        i = await queue.get()
        if i is None:
            return
        txn = {"txn_id": f"load{i}", "card_id": f"c{random.randint(0, 2999):05d}",
               "amount": round(random.lognormvariate(3.7, 0.8), 2), "merchant_category": random.choice(MCCS),
               "location": f"city_{random.randint(0, 39):02d}", "ts": REPLAY_TS + random.randint(0, 86_400)}
        t0 = time.perf_counter()
        try:
            r = await client.post("/predict", json=txn)
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        except Exception:
            errors.append(i)


async def main(a):
    queue, lat, errors = asyncio.Queue(), [], []
    for i in range(a.requests):
        queue.put_nowait(i)
    for _ in range(a.concurrency):
        queue.put_nowait(None)
    limits = httpx.Limits(max_connections=a.concurrency)
    async with httpx.AsyncClient(base_url=a.api, timeout=5, limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client, queue, lat, errors) for _ in range(a.concurrency)))
        dur = time.perf_counter() - t0
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print(f"{len(lat):,} ok, {len(errors)} errors in {dur:.1f}s -> {len(lat) / dur:,.0f} req/s "
          f"at concurrency {a.concurrency}")
    print(f"latency ms: p50={p50:.1f} p95={p95:.1f} p99={p99:.1f} max={max(lat):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main(ap.parse_args()))
