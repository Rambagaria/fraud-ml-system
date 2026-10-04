import json

import numpy as np

from fraud.drift import drift_report, psi, psi_categorical
from fraud.train import drift_baseline
import pandas as pd


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    base = drift_baseline(pd.Series(rng.lognormal(3, 0.7, 20_000)))
    assert psi(base, rng.lognormal(3, 0.7, 20_000)) < 0.02


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    base = drift_baseline(pd.Series(rng.lognormal(3, 0.7, 20_000)))
    assert psi(base, rng.lognormal(3, 0.7, 20_000) * 3) > 0.25


def test_report_flags_amount_drift(model_dir):
    version = (model_dir / "PRODUCTION").read_text().strip()
    meta = json.loads((model_dir / version / "metadata.json").read_text())
    feats = {f: 0.0 for f in meta["feature_names"]}
    records = [{"model_version": version, "decision": "APPROVE", "degraded": False, "score": 0.01,
                "features": {**feats, "amount": 5000.0, "log_amount": np.log1p(5000.0)},
                "inputs": {"location": "city_01", "merchant_category": "grocery"}}
               for _ in range(500)]
    report = drift_report(records, meta)
    assert report["features"]["amount"]["status"] == "ALERT" and report["overall"] == "ALERT"


def test_unseen_category_alerts():
    expected = {"grocery": 0.6, "gas": 0.4}
    assert psi_categorical(expected, ["grocery"] * 60 + ["gas"] * 40) < 0.01
    assert psi_categorical(expected, ["crypto_atm"] * 50 + ["grocery"] * 50) > 0.25


def test_binary_feature_drift_detected():
    """Regression test: 0/1 features used to fall into one bin and never alert."""
    base = drift_baseline(pd.Series([0.0] * 950 + [1.0] * 50))
    assert psi(base, np.array([1.0] * 1000)) > 0.25
