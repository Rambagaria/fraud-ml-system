"""Daily PySpark batch job.

Computes, as of a given day:
  * location and merchant-category fraud rates (smoothed target encoding), using ONLY
    labels older than LABEL_DELAY_DAYS. Same formula as training -> no skew.
  * per-card 30-day aggregates (for analytics, monitoring and history backfill).

Outputs Parquet (the "offline store") plus a small JSON of the reference tables that
the loader pushes to Redis (the "online store").

Run:  spark-submit batch/spark_reference_features.py --input data/transactions.csv --as-of-day 20514
On Databricks/EMR the same script runs with s3:// paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud.config import DAY_S, DEFAULT_PRIOR, LABEL_DELAY_DAYS, SMOOTHING_M  # noqa: E402


def rate_table(labeled, col: str, prior: float):
    return (labeled.groupBy(col)
            .agg(F.count("*").alias("n"), F.sum("is_fraud").alias("fraud"))
            .withColumn("rate", (F.col("fraud") + F.lit(SMOOTHING_M * prior)) / (F.col("n") + F.lit(SMOOTHING_M))))


def run(spark: SparkSession, input_path: str, as_of_day: int, out_dir: str) -> dict:
    reader = spark.read.option("header", True).option("inferSchema", True)
    df = reader.parquet(input_path) if input_path.endswith(".parquet") else reader.csv(input_path)
    df = df.withColumn("day", F.floor(F.col("ts") / DAY_S).cast("long"))

    labeled = df.filter(F.col("day") < as_of_day - LABEL_DELAY_DAYS)
    g = labeled.agg(F.count("*").alias("n"), F.sum("is_fraud").alias("fraud")).collect()[0]
    prior = (g["fraud"] / g["n"]) if g["n"] else DEFAULT_PRIOR

    loc = rate_table(labeled, "location", prior)
    mcc = rate_table(labeled, "merchant_category", prior)

    card_30d = (df.filter((F.col("day") >= as_of_day - 30) & (F.col("day") < as_of_day))
                .groupBy("card_id")
                .agg(F.count("*").alias("txn_count_30d"),
                     F.avg("amount").alias("amt_mean_30d"),
                     F.stddev("amount").alias("amt_std_30d"),
                     F.max("ts").alias("last_ts"),
                     F.countDistinct("location").alias("distinct_locations_30d")))

    base = f"{out_dir}/as_of_day={as_of_day}"
    loc.write.mode("overwrite").parquet(f"{base}/location_rates")
    mcc.write.mode("overwrite").parquet(f"{base}/mcc_rates")
    card_30d.write.mode("overwrite").parquet(f"{base}/card_30d_stats")

    tables = {
        "as_of_day": as_of_day,
        "global_rate": prior,
        "loc_rates": {r["location"]: r["rate"] for r in loc.collect()},           # small: ~dozens
        "mcc_rates": {r["merchant_category"]: r["rate"] for r in mcc.collect()},  # of keys
    }
    Path(base).mkdir(parents=True, exist_ok=True)
    Path(f"{base}/ref_tables.json").write_text(json.dumps(tables, indent=2))
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/transactions.csv")
    ap.add_argument("--as-of-day", type=int, required=True, help="absolute day index = ts // 86400")
    ap.add_argument("--out", default="data/features")
    a = ap.parse_args()
    spark = SparkSession.builder.appName("fraud-reference-features").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    t = run(spark, a.input, a.as_of_day, a.out)
    print(f"as_of_day={t['as_of_day']} prior={t['global_rate']:.5f} "
          f"locations={len(t['loc_rates'])} categories={len(t['mcc_rates'])} -> {a.out}/as_of_day={a.as_of_day}/")
    spark.stop()


if __name__ == "__main__":
    main()
