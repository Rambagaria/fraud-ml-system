"""Push batch outputs into the online store (Redis).

  --ref-json      reference tables written by the Spark job (or computed here with pandas)
  --backfill-csv  bootstrap card histories: load the 30 days before --replay-start-day,
                  so the API has realistic history before live/replayed traffic starts
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from fraud.config import DAY_S, REDIS_URL, WINDOW_30D_S
from fraud.feature_store import make_store
from fraud.features import build_reference_tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redis-url", default=REDIS_URL or "redis://localhost:6379/0")
    ap.add_argument("--ref-json", help="output of batch/spark_reference_features.py")
    ap.add_argument("--backfill-csv", help="transactions CSV for history bootstrap")
    ap.add_argument("--replay-start-day", type=int, help="absolute day index (ts // 86400)")
    ap.add_argument("--block", nargs="*", default=[], help="card ids to blocklist")
    a = ap.parse_args()
    store = make_store(a.redis_url, timeout_s=5)

    if a.backfill_csv:
        df = pd.read_csv(a.backfill_csv)
        start = a.replay_start_day if a.replay_start_day is not None else int(df.ts.max() // DAY_S) - 6
        if not a.ref_json:  # no Spark output given: compute reference tables with pandas
            store.load_reference(build_reference_tables(df, start))
            print(f"loaded reference tables as of day {start} (pandas)")
        hist = df[(df.ts >= start * DAY_S - WINDOW_30D_S) & (df.ts < start * DAY_S)]
        cols = ["card_id", "txn_id", "ts", "amount", "location"]
        store.bulk_record(hist[cols].itertuples(index=False, name=None))
        print(f"backfilled {len(hist):,} transactions for {hist.card_id.nunique():,} cards "
              f"(replay starts day {start}, ts >= {start * DAY_S})")

    if a.ref_json:
        tables = json.loads(open(a.ref_json).read())
        store.load_reference(tables)
        print(f"loaded reference tables as of day {tables['as_of_day']}: "
              f"{len(tables['loc_rates'])} locations, {len(tables['mcc_rates'])} categories")

    for card in a.block:
        store.block(card)
    if a.block:
        print(f"blocklisted {len(a.block)} cards")


if __name__ == "__main__":
    main()
