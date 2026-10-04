"""Feature logic shared by offline training and online serving.

`compute_features` is the ONLY place features are defined. Training replays history
through it row by row; the API calls it with history fetched from Redis. Same code
path means no training-serving skew (tests/test_features.py proves parity).
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd

from fraud.config import (
    DAY_S,
    DEFAULT_PRIOR,
    FEATURE_NAMES,
    LABEL_DELAY_DAYS,
    NO_HISTORY_GAP_S,
    SMOOTHING_M,
    WINDOW_30D_S,
)


@dataclass(frozen=True)
class PastTxn:
    ts: int
    amount: float
    location: str


def smoothed_rate(n: int, fraud: int, prior: float) -> float:
    """Bayesian-smoothed fraud rate. Rare categories shrink toward the prior."""
    return (fraud + SMOOTHING_M * prior) / (n + SMOOTHING_M)


def compute_features(
    *,
    ts: int,
    amount: float,
    location: str,
    history: list[PastTxn],
    loc_rate: float,
    mcc_rate: float,
) -> dict[str, float]:
    """Features for one transaction given the card's prior 30 days of transactions."""
    past = [p for p in history if ts - WINDOW_30D_S <= p.ts < ts]

    if past:
        last = max(past, key=lambda p: p.ts)
        secs_since_last = float(ts - last.ts)
        loc_changed = float(location != last.location)
        mean_30d = sum(p.amount for p in past) / len(past)
        amt_dev_30d = amount / mean_30d if mean_30d > 0 else 1.0
    else:
        secs_since_last = float(NO_HISTORY_GAP_S)
        loc_changed = 0.0
        amt_dev_30d = 1.0

    recent_24h = [p for p in past if ts - p.ts <= DAY_S]
    return {
        "amount": float(amount),
        "log_amount": math.log1p(amount),
        "hour_of_day": float((ts // 3600) % 24),
        "secs_since_last_txn": secs_since_last,
        "txn_count_1h": float(sum(1 for p in past if ts - p.ts <= 3600)),
        "txn_count_24h": float(len(recent_24h)),
        "amt_sum_24h": float(sum(p.amount for p in recent_24h)),
        "amt_dev_30d": float(amt_dev_30d),
        "loc_changed": loc_changed,
        "loc_fraud_rate": float(loc_rate),
        "mcc_fraud_rate": float(mcc_rate),
        "has_history": float(bool(past)),
    }


def build_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replay transactions in time order through `compute_features`.

    Target encodings (location / merchant category fraud rates) are point-in-time:
    a transaction on day d only sees labels from days < d - LABEL_DELAY_DAYS,
    which is exactly what the daily batch job gives the online system. No leakage.
    """
    df = df.sort_values(["ts", "txn_id"], kind="stable").reset_index(drop=True)

    card_hist: dict[str, deque[PastTxn]] = defaultdict(deque)
    rel_loc = defaultdict(lambda: [0, 0])  # location -> [n, fraud] (released labels)
    rel_mcc = defaultdict(lambda: [0, 0])
    rel_global = [0, 0]
    pending: deque[tuple[int, list[tuple[str, str, int]]]] = deque()  # (day, rows)

    rows = []
    for r in df.itertuples(index=False):
        ts, day = int(r.ts), int(r.ts) // DAY_S

        # Release labels that have "matured" by the start of today.
        cutoff = day - LABEL_DELAY_DAYS
        while pending and pending[0][0] < cutoff:
            _, items = pending.popleft()
            for loc, mcc, y in items:
                rel_loc[loc][0] += 1
                rel_loc[loc][1] += y
                rel_mcc[mcc][0] += 1
                rel_mcc[mcc][1] += y
                rel_global[0] += 1
                rel_global[1] += y
        prior = rel_global[1] / rel_global[0] if rel_global[0] else DEFAULT_PRIOR

        hist = card_hist[r.card_id]
        while hist and hist[0].ts < ts - WINDOW_30D_S:
            hist.popleft()

        feats = compute_features(
            ts=ts,
            amount=float(r.amount),
            location=r.location,
            history=list(hist),
            loc_rate=smoothed_rate(*rel_loc[r.location], prior),
            mcc_rate=smoothed_rate(*rel_mcc[r.merchant_category], prior),
        )
        feats.update(txn_id=r.txn_id, card_id=r.card_id, ts=ts, day=day, is_fraud=int(r.is_fraud),
                     location=r.location, merchant_category=r.merchant_category)
        rows.append(feats)

        hist.append(PastTxn(ts, float(r.amount), r.location))
        if pending and pending[-1][0] == day:
            pending[-1][1].append((r.location, r.merchant_category, int(r.is_fraud)))
        else:
            pending.append((day, [(r.location, r.merchant_category, int(r.is_fraud))]))

    out = pd.DataFrame(rows)
    return out[["txn_id", "card_id", "ts", "day", "is_fraud", "location", "merchant_category", *FEATURE_NAMES]]


def build_reference_tables(df: pd.DataFrame, as_of_day: int) -> dict:
    """Reference tables the online system uses on `as_of_day` (pandas version of the
    Spark batch job; tests check both produce the same numbers)."""
    day = df["ts"] // DAY_S
    labeled = df[day < as_of_day - LABEL_DELAY_DAYS]
    n, f = len(labeled), int(labeled["is_fraud"].sum())
    prior = f / n if n else DEFAULT_PRIOR

    def table(col: str) -> dict[str, float]:
        g = labeled.groupby(col)["is_fraud"].agg(["count", "sum"])
        return {k: smoothed_rate(int(c), int(s), prior) for k, (c, s) in g.iterrows()}

    return {
        "as_of_day": int(as_of_day),
        "global_rate": prior,
        "loc_rates": table("location"),
        "mcc_rates": table("merchant_category"),
    }
