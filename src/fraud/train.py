"""Train, evaluate, and (if it beats the current model) promote a fraud model.

Pipeline:
  1. Point-in-time features via the shared feature code.
  2. Walk-forward (expanding window) validation to check stability over time.
  3. Final split: train | validation (early stopping + thresholds) | test (held out).
  4. Thresholds: STEP_UP at max F2 (recall-weighted); DECLINE at a high precision floor
     because a card lock is expensive for a legitimate customer.
  5. Champion/challenger: score the current production model on the SAME test window;
     promote only if the challenger is at least as good.
  6. Write versioned artifacts + drift baselines; update the PRODUCTION pointer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, fbeta_score, precision_recall_curve, roc_auc_score
from fraud.config import CATEGORICAL_INPUTS, DRIFT_SKIP_FEATURES, FEATURE_NAMES, MODEL_DIR
from fraud.features import build_training_features
# from src.fraud.config import CATEGORICAL_INPUTS, DRIFT_SKIP_FEATURES, FEATURE_NAMES, MODEL_DIR
# from src.fraud.features import build_training_features

PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, eval_metric="aucpr",
    early_stopping_rounds=40, tree_method="hist", n_jobs=4,
)


def fit(train: pd.DataFrame, val: pd.DataFrame) -> xgb.XGBClassifier:
    pos = int(train.is_fraud.sum())
    spw = (len(train) - pos) / max(pos, 1)  # cost-sensitive weighting for 0.2-0.3% positives
    model = xgb.XGBClassifier(scale_pos_weight=spw, **PARAMS)
    model.fit(train[FEATURE_NAMES], train.is_fraud,
              eval_set=[(val[FEATURE_NAMES], val.is_fraud)], verbose=False)
    return model


def walk_forward(feats: pd.DataFrame, first_train_days=21, val_days=7) -> list[dict]:
    d0, d_end = feats.day.min(), feats.day.max() + 1
    folds, cut = [], d0 + first_train_days
    while cut + val_days <= d_end - 7:  # keep the last week untouched for the final test
        tr, va = feats[feats.day < cut], feats[(feats.day >= cut) & (feats.day < cut + val_days)]
        m = fit(tr, va)
        p = m.predict_proba(va[FEATURE_NAMES])[:, 1]
        folds.append({"train_days": int(cut - d0), "pr_auc": round(average_precision_score(va.is_fraud, p), 4)})
        cut += val_days
    return folds


def pick_thresholds(y: np.ndarray, p: np.ndarray, beta=2.0, decline_precision=0.95) -> dict:
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-12, None)
    step_up = float(thr[np.argmax(f)])
    ok = np.where((prec >= decline_precision) & (thr >= step_up))[0]
    decline = float(thr[ok[0]]) if len(ok) else max(step_up, float(np.quantile(p, 0.9999)))
    return {"step_up": step_up, "decline": max(decline, step_up)}


def evaluate(y: np.ndarray, p: np.ndarray, th: dict) -> dict:
    flag, dec = p >= th["step_up"], p >= th["decline"]
    tp = int((flag & (y == 1)).sum())
    return {
        "pr_auc": round(average_precision_score(y, p), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "recall_at_flag": round(tp / max(int(y.sum()), 1), 4),
        "precision_at_flag": round(tp / max(int(flag.sum()), 1), 4),
        "f2_at_flag": round(fbeta_score(y, flag, beta=2), 4),
        "precision_at_decline": round(int((dec & (y == 1)).sum()) / max(int(dec.sum()), 1), 4),
        "tier_share": {
            "APPROVE": round(float((~flag).mean()), 5),
            "STEP_UP": round(float((flag & ~dec).mean()), 5),
            "DECLINE": round(float(dec.mean()), 5),
        },
        "false_positives": int((flag & (y == 0)).sum()),
        "n": int(len(y)), "positives": int(y.sum()),
        # Class weighting inflates scores: they rank well but are NOT probabilities.
        "mean_score": round(float(p.mean()), 5), "base_rate": round(float(y.mean()), 5),
    }


def drift_baseline(x: pd.Series, bins=10) -> dict:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    idx = np.searchsorted(edges, x.to_numpy(), side="left")  # "left": binary 0/1 must split
    props = np.bincount(idx, minlength=len(edges) + 1) / len(x)
    return {"edges": edges.tolist(), "props": props.tolist()}


def champion_pr_auc(model_dir: Path, test: pd.DataFrame) -> tuple[str | None, float | None]:
    ptr = model_dir / "PRODUCTION"
    if not ptr.exists():
        return None, None
    version = ptr.read_text().strip()
    meta = json.loads((model_dir / version / "metadata.json").read_text())
    if meta["feature_names"] != FEATURE_NAMES:
        return version, None  # schema changed; can't compare like-for-like
    booster = xgb.Booster()
    booster.load_model(model_dir / version / "model.json")
    p = booster.inplace_predict(test[FEATURE_NAMES].to_numpy(np.float32))
    return version, float(average_precision_score(test.is_fraud, p))


def train(df: pd.DataFrame, model_dir: Path = MODEL_DIR, min_pr_auc=0.5, tolerance=0.005,
          run_walk_forward=True) -> dict:
    t0 = time.time()
    feats = build_training_features(df)
    last = feats.day.max()
    test = feats[feats.day > last - 7]
    val = feats[(feats.day > last - 14) & (feats.day <= last - 7)]
    trn = feats[feats.day <= last - 14]

    folds = walk_forward(feats) if run_walk_forward else []
    model = fit(trn, val)
    # Keep ONLY the early-stopped trees and score through the same call the API uses.
    # (Bug found in live testing: evaluation used best_iteration via predict_proba,
    # but Booster.inplace_predict in serving used every tree.)
    booster = model.get_booster()
    if getattr(model, "best_iteration", None) is not None:
        booster = booster[: model.best_iteration + 1]

    def score(d: pd.DataFrame) -> np.ndarray:
        return booster.inplace_predict(d[FEATURE_NAMES].to_numpy(np.float32))

    p_val, p_test = score(val), score(test)
    th = pick_thresholds(val.is_fraud.to_numpy(), p_val)
    metrics = evaluate(test.is_fraud.to_numpy(), p_test, th)

    champ_version, champ_auc = champion_pr_auc(model_dir, test)
    promote = metrics["pr_auc"] >= min_pr_auc and (champ_auc is None or metrics["pr_auc"] >= champ_auc - tolerance)

    version = time.strftime("v%Y%m%d%H%M%S", time.gmtime())
    out = model_dir / version
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / "model.json")
    meta = {
        "version": version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_names": FEATURE_NAMES,
        "thresholds": th,
        "metrics_test": metrics,
        "walk_forward": folds,
        "champion": {"version": champ_version, "pr_auc_on_same_test": champ_auc},
        "promoted": promote,
        "data": {"rows": len(feats), "train_days": int(trn.day.nunique()),
                 "train_fraud_rate": round(float(trn.is_fraud.mean()), 5)},
        # Baseline = most recent pre-deployment window (validation), NOT the whole
        # training set: early training days include the cold-start period where
        # target encodings sit at the default prior, which made the first version
        # of this monitor fire false alarms on loc/mcc fraud rates.
        "drift_baseline": {
            "features": {f: drift_baseline(val[f]) for f in FEATURE_NAMES if f not in DRIFT_SKIP_FEATURES},
            "categorical": {c: val[c].value_counts(normalize=True).to_dict() for c in CATEGORICAL_INPUTS},
            "score": drift_baseline(pd.Series(p_val)),
        },
        "train_seconds": round(time.time() - t0, 1),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    if promote:
        (model_dir / "PRODUCTION").write_text(version)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--min-pr-auc", type=float, default=0.5)
    ap.add_argument("--skip-walk-forward", action="store_true")
    a = ap.parse_args()
    meta = train(pd.read_csv(a.data), Path(a.model_dir), a.min_pr_auc, run_walk_forward=not a.skip_walk_forward)
    m = meta["metrics_test"]
    print(f"version={meta['version']} promoted={meta['promoted']} ({meta['train_seconds']}s)")
    print(f"walk-forward PR-AUC: {[f['pr_auc'] for f in meta['walk_forward']]}")
    print(f"test PR-AUC={m['pr_auc']} ROC-AUC={m['roc_auc']} | at STEP_UP threshold: "
          f"recall={m['recall_at_flag']} precision={m['precision_at_flag']} | "
          f"precision at DECLINE={m['precision_at_decline']}")
    print(f"thresholds={meta['thresholds']} champion={meta['champion']}")
    print(f"calibration check: mean score {m['mean_score']} vs base rate {m['base_rate']}")


if __name__ == "__main__":
    main()
