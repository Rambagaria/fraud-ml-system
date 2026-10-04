"""Online feature store.

Per-card history lives in a Redis sorted set (score = timestamp), trimmed to 30 days.
Reference tables (location / merchant fraud rates) are loaded daily by the batch job.

Design notes worth knowing cold:
  * Writes are idempotent: the set member includes txn_id, so a Kafka redelivery
    (at-least-once) does not double count a transaction.
  * Reads use a strict upper bound "(ts" so a transaction never sees itself.
  * Reference tables are cached in-process for 60s: they change once a day, and
    skipping that Redis round trip keeps the hot path to one read.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from fraud.config import DEFAULT_PRIOR, WINDOW_30D_S
from fraud.features import PastTxn


@dataclass
class RefTables:
    global_rate: float = DEFAULT_PRIOR
    loc_rates: dict[str, float] = field(default_factory=dict)
    mcc_rates: dict[str, float] = field(default_factory=dict)
    as_of_day: int | None = None

    def loc(self, key: str) -> float:
        return self.loc_rates.get(key, self.global_rate)  # unseen -> prior (matches training)

    def mcc(self, key: str) -> float:
        return self.mcc_rates.get(key, self.global_rate)


def _member(txn_id: str, ts: int, amount: float, location: str) -> str:
    return json.dumps({"id": txn_id, "ts": int(ts), "a": round(float(amount), 2), "l": location},
                      sort_keys=True, separators=(",", ":"))


class RedisFeatureStore:
    REF_TTL_S = 60

    def __init__(self, client):
        self.r = client
        self._ref: RefTables | None = None
        self._ref_loaded_at = 0.0

    @staticmethod
    def _key(card_id: str) -> str:
        return f"card:{card_id}:txns"

    def ping(self) -> bool:
        return bool(self.r.ping())

    def get_history(self, card_id: str, ts: int) -> list[PastTxn]:
        raw = self.r.zrangebyscore(self._key(card_id), ts - WINDOW_30D_S, f"({ts}")
        out = []
        for m in raw:
            d = json.loads(m)
            out.append(PastTxn(d["ts"], d["a"], d["l"]))
        return out

    def record_txn(self, card_id: str, txn_id: str, ts: int, amount: float, location: str) -> None:
        key = self._key(card_id)
        p = self.r.pipeline(transaction=False)
        p.zadd(key, {_member(txn_id, ts, amount, location): int(ts)})
        p.zremrangebyscore(key, "-inf", f"({int(ts) - WINDOW_30D_S}")
        p.expire(key, WINDOW_30D_S + 86_400)
        p.execute()

    def bulk_record(self, rows, batch: int = 5000) -> int:
        """Pipelined backfill. rows: iterable of (card_id, txn_id, ts, amount, location)."""
        p, n = self.r.pipeline(transaction=False), 0
        for n, (card_id, txn_id, ts, amount, location) in enumerate(rows, 1):
            key = self._key(card_id)
            p.zadd(key, {_member(txn_id, ts, amount, location): int(ts)})
            p.expire(key, WINDOW_30D_S + 86_400)
            if n % batch == 0:
                p.execute()
        p.execute()
        return n

    def is_blocked(self, card_id: str) -> bool:
        return bool(self.r.sismember("ref:blocklist", card_id))

    def block(self, card_id: str) -> None:
        self.r.sadd("ref:blocklist", card_id)

    def load_reference(self, tables: dict) -> None:
        p = self.r.pipeline(transaction=True)  # atomic swap: readers never see half a table
        p.delete("ref:loc_rates", "ref:mcc_rates")
        if tables["loc_rates"]:
            p.hset("ref:loc_rates", mapping={k: str(v) for k, v in tables["loc_rates"].items()})
        if tables["mcc_rates"]:
            p.hset("ref:mcc_rates", mapping={k: str(v) for k, v in tables["mcc_rates"].items()})
        p.set("ref:global_rate", str(tables["global_rate"]))
        p.set("ref:as_of_day", str(tables["as_of_day"]))
        p.execute()
        self._ref = None

    def get_reference(self) -> RefTables:
        if self._ref is None or time.time() - self._ref_loaded_at > self.REF_TTL_S:
            g = self.r.get("ref:global_rate")
            as_of = self.r.get("ref:as_of_day")
            self._ref = RefTables(
                global_rate=float(g) if g else DEFAULT_PRIOR,
                loc_rates={_s(k): float(v) for k, v in self.r.hgetall("ref:loc_rates").items()},
                mcc_rates={_s(k): float(v) for k, v in self.r.hgetall("ref:mcc_rates").items()},
                as_of_day=int(as_of) if as_of else None,
            )
            self._ref_loaded_at = time.time()
        return self._ref


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else x


class InMemoryFeatureStore:
    """Same interface, no Redis. For tests and local dev only (not shared across processes)."""

    def __init__(self):
        self.txns: dict[str, dict[str, tuple[int, float, str]]] = {}
        self.blocklist: set[str] = set()
        self.ref = RefTables()

    def ping(self) -> bool:
        return True

    def get_history(self, card_id: str, ts: int) -> list[PastTxn]:
        return [PastTxn(t, a, l) for t, a, l in self.txns.get(card_id, {}).values()
                if ts - WINDOW_30D_S <= t < ts]

    def record_txn(self, card_id, txn_id, ts, amount, location) -> None:
        d = self.txns.setdefault(card_id, {})
        d[txn_id] = (int(ts), round(float(amount), 2), location)
        for k in [k for k, v in d.items() if v[0] < int(ts) - WINDOW_30D_S]:
            del d[k]

    def bulk_record(self, rows, batch: int = 5000) -> int:
        n = 0
        for n, row in enumerate(rows, 1):
            self.record_txn(*row)
        return n

    def is_blocked(self, card_id: str) -> bool:
        return card_id in self.blocklist

    def block(self, card_id: str) -> None:
        self.blocklist.add(card_id)

    def load_reference(self, tables: dict) -> None:
        self.ref = RefTables(tables["global_rate"], dict(tables["loc_rates"]),
                             dict(tables["mcc_rates"]), tables["as_of_day"])

    def get_reference(self) -> RefTables:
        return self.ref


def make_store(redis_url: str | None, timeout_s: float = 0.05):
    if not redis_url:
        return InMemoryFeatureStore()
    import redis
    client = redis.Redis.from_url(redis_url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s)
    return RedisFeatureStore(client)
