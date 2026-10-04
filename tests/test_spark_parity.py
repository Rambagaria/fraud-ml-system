"""The Spark batch job and the training code must compute identical reference tables."""
import sys
from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "batch"))

from fraud.config import DAY_S  # noqa: E402
from fraud.features import build_reference_tables  # noqa: E402


@pytest.mark.slow
def test_spark_matches_pandas(small_df, tmp_path):
    from pyspark.sql import SparkSession
    from spark_reference_features import run

    csv = tmp_path / "txns.csv"
    small_df.to_csv(csv, index=False)
    as_of = int(small_df.ts.max() // DAY_S) - 3
    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    try:
        got = run(spark, str(csv), as_of, str(tmp_path / "out"))
    finally:
        spark.stop()
    want = build_reference_tables(small_df, as_of)
    assert got["global_rate"] == pytest.approx(want["global_rate"])
    for k in ("loc_rates", "mcc_rates"):
        assert got[k].keys() == want[k].keys()
        for key in want[k]:
            assert got[k][key] == pytest.approx(want[k][key])
