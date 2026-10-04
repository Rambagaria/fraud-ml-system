"""Synthetic credit-card transactions with realistic fraud patterns.

Fraud comes in bursts (velocity), at odd hours, in risky locations and merchant
categories, with amounts far above the card's normal spend. ~20% of fraud events are
deliberately subtle so the model can't hit 100% recall.

Swap in a real dataset (e.g. the Kaggle "Sparkov" credit card fraud data) by mapping
its columns to: txn_id, card_id, ts, amount, merchant_category, location, is_fraud.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START_TS = 1_767_225_600  # 2026-01-01 00:00:00 UTC
LOCATIONS = [f"city_{i:02d}" for i in range(40)]
RISKY_LOCATIONS = LOCATIONS[35:]
MCCS = ["grocery", "gas", "restaurant", "retail", "travel", "online", "electronics", "gift_cards"]
NORMAL_MCC_P = [0.25, 0.15, 0.20, 0.20, 0.04, 0.10, 0.04, 0.02]
FRAUD_MCC_P = [0.03, 0.05, 0.04, 0.10, 0.10, 0.28, 0.20, 0.20]


def generate(n_cards: int = 3000, days: int = 60, fraud_events: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cards = [f"c{i:05d}" for i in range(n_cards)]
    home = rng.choice(LOCATIONS[:35], size=n_cards)
    rate = rng.uniform(0.5, 2.5, size=n_cards)               # txns per day
    spend = np.exp(rng.normal(np.log(40), 0.6, size=n_cards))  # typical ticket size

    parts = []
    for i, card in enumerate(cards):
        n = rng.poisson(rate[i] * days)
        if n == 0:
            continue
        day = rng.integers(0, days, size=n)
        hour = np.clip(rng.normal(15, 3.5, size=n), 0, 23.99)
        ts = START_TS + day * 86_400 + (hour * 3600).astype(int) + rng.integers(0, 60, size=n)
        travel = rng.random(n) < 0.05
        loc = np.where(travel, rng.choice(LOCATIONS, size=n), home[i])
        parts.append(pd.DataFrame({
            "card_id": card,
            "ts": ts,
            "amount": np.round(np.exp(rng.normal(np.log(spend[i]), 0.7, size=n)), 2),
            "merchant_category": rng.choice(MCCS, size=n, p=NORMAL_MCC_P),
            "location": loc,
            "is_fraud": 0,
        }))

    for _ in range(fraud_events):
        i = int(rng.integers(0, n_cards))
        subtle = rng.random() < 0.2
        k = 1 if subtle else int(rng.integers(2, 6))
        start = START_TS + int(rng.integers(3, days)) * 86_400
        start += int(rng.integers(0, 6) * 3600) if (rng.random() < 0.5 and not subtle) else int(rng.integers(8, 22) * 3600)
        gaps = np.cumsum(rng.exponential(300, size=k)).astype(int) + np.arange(k)
        if subtle:
            loc = np.array([home[i]])
            amt = np.round(np.exp(rng.normal(np.log(spend[i] * 1.5), 0.5, size=1)), 2)
            mcc = rng.choice(MCCS, size=1, p=NORMAL_MCC_P)
        else:
            far = rng.choice(RISKY_LOCATIONS) if rng.random() < 0.6 else rng.choice(LOCATIONS)
            loc = np.full(k, far if rng.random() < 0.6 else home[i])
            amt = np.round(spend[i] * np.exp(rng.normal(np.log(4), 0.6, size=k)), 2)
            if rng.random() < 0.3:
                amt[0] = round(float(rng.uniform(1, 5)), 2)  # card-testing probe
            mcc = rng.choice(MCCS, size=k, p=FRAUD_MCC_P)
        parts.append(pd.DataFrame({
            "card_id": cards[i], "ts": start + gaps, "amount": amt,
            "merchant_category": mcc, "location": loc, "is_fraud": 1,
        }))

    df = pd.concat(parts, ignore_index=True)
    df = df[df["ts"] < START_TS + days * 86_400]
    df = df.sort_values(["ts", "card_id"], kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", [f"t{j:08d}" for j in range(len(df))])
    df["ts"] = df["ts"].astype("int64")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/transactions.csv")
    ap.add_argument("--cards", type=int, default=3000)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--fraud-events", type=int, default=260)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    df = generate(a.cards, a.days, a.fraud_events, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} transactions, fraud rate {df.is_fraud.mean():.3%} -> {a.out}")


if __name__ == "__main__":
    main()
