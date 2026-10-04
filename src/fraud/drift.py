"""Drift report: compares live traffic (prediction log) to the training baseline.

Why this matters for fraud: chargeback labels take weeks, so we can't watch recall
in real time. Feature drift, score drift and decision-rate shifts are the early
warning signals until labels catch up.

PSI rule of thumb: < 0.1 stable, 0.1-0.25 investigate, > 0.25 significant shift.
Exits with code 1 on ALERT so a scheduler (Airflow) marks the task failed and pages.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from fraud.config import MODEL_DIR, PRED_LOG_PATH

WARN, ALERT = 0.1, 0.25


def psi(baseline: dict, values: np.ndarray, eps: float = 1e-4) -> float:
    edges = np.asarray(baseline["edges"])
    expected = np.asarray(baseline["props"])
    idx = np.searchsorted(edges, values, side="left")  # must match train.drift_baseline
    actual = np.bincount(idx, minlength=len(edges) + 1) / max(len(values), 1)
    e, a = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: dict[str, float], values: list[str], eps: float = 1e-4) -> float:
    """PSI over category shares. Categories never seen in training land in '__unseen__'."""
    n = max(len(values), 1)
    actual = Counter(v if v in expected else "__unseen__" for v in values)
    keys = set(expected) | {"__unseen__"}
    total = 0.0
    for k in keys:
        e = max(expected.get(k, 0.0), eps)
        a = max(actual.get(k, 0) / n, eps)
        total += (a - e) * np.log(a / e)
    return float(total)


def status(v: float) -> str:
    return "ALERT" if v > ALERT else "WARN" if v > WARN else "OK"


def drift_report(records: list[dict], meta: dict) -> dict:
    scored = [r for r in records if r.get("features")]
    base = meta["drift_baseline"]
    feats = {}
    for name, b in base["features"].items():
        v = psi(b, np.asarray([r["features"][name] for r in scored], dtype=float))
        feats[name] = {"psi": round(v, 4), "status": status(v)}
    cats = {}
    for name, shares in base.get("categorical", {}).items():
        values = [r["inputs"][name] for r in scored if r.get("inputs")]
        if not values:  # older log format without raw inputs
            continue
        v = psi_categorical(shares, values)
        cats[name] = {"psi": round(v, 4), "status": status(v)}
    score_psi = psi(base["score"], np.asarray([r["score"] for r in scored], dtype=float))
    decisions = Counter(r["decision"] for r in records)
    n = max(len(records), 1)
    expected = meta["metrics_test"]["tier_share"]
    report = {
        "model_version": meta["version"],
        "n_records": len(records),
        "degraded_share": round(sum(r["degraded"] for r in records) / n, 4),
        "features": feats,
        "categorical": cats,
        "score": {"psi": round(score_psi, 4), "status": status(score_psi)},
        "decision_share": {d: round(decisions.get(d, 0) / n, 5) for d in expected},
        "decision_share_expected": expected,
    }
    worst = [s["status"] for s in [*feats.values(), *cats.values(), report["score"]]]
    report["overall"] = "ALERT" if "ALERT" in worst else "WARN" if "WARN" in worst else "OK"
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(PRED_LOG_PATH))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--last", type=int, default=20_000, help="most recent N predictions")
    ap.add_argument("--out", default="logs/drift_report.json")
    a = ap.parse_args()

    lines = Path(a.log).read_text().splitlines()[-a.last:]
    records = [json.loads(x) for x in lines]
    model_dir = Path(a.model_dir)
    # Compare each record against the baseline of the model that actually scored it.
    version = records[-1]["model_version"] if records else (model_dir / "PRODUCTION").read_text().strip()
    records = [r for r in records if r["model_version"] == version]
    meta = json.loads((model_dir / version / "metadata.json").read_text())

    report = drift_report(records, meta)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(f"model {version} | {report['n_records']:,} predictions | overall: {report['overall']}")
    for name, s in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
        print(f"  {name:<22} PSI={s['psi']:<8} {s['status']}")
    for name, s in report["categorical"].items():
        print(f"  {name + ' (mix)':<22} PSI={s['psi']:<8} {s['status']}")
    print(f"  {'SCORE':<22} PSI={report['score']['psi']:<8} {report['score']['status']}")
    print(f"  decisions live={report['decision_share']} expected={report['decision_share_expected']}")
    sys.exit(1 if report["overall"] == "ALERT" else 0)


if __name__ == "__main__":
    main()
