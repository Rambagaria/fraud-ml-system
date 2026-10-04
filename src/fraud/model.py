"""Load the production model from the registry and turn scores into decisions.

The model artifact is versioned separately from the code: the container reads
MODEL_DIR/PRODUCTION (or a pinned MODEL_VERSION) at startup, so a rollback is
"point PRODUCTION at the previous version and reload", not a redeploy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb


class FraudModel:
    def __init__(self, model_dir: Path, version: str | None = None):
        model_dir = Path(model_dir)
        self.version = version or (model_dir / "PRODUCTION").read_text().strip()
        path = model_dir / self.version
        self.meta = json.loads((path / "metadata.json").read_text())
        self.feature_names: list[str] = self.meta["feature_names"]
        self.thresholds: dict = self.meta["thresholds"]
        self.booster = xgb.Booster()
        self.booster.load_model(path / "model.json")
        self.booster.set_param({"nthread": 1})  # one request = one row; avoid thread overhead

    def score(self, feats: dict[str, float]) -> float:
        x = np.asarray([[feats[n] for n in self.feature_names]], dtype=np.float32)
        return float(self.booster.inplace_predict(x)[0])

    def tier(self, score: float) -> str:
        if score >= self.thresholds["decline"]:
            return "DECLINE"
        if score >= self.thresholds["step_up"]:
            return "STEP_UP"
        return "APPROVE"
