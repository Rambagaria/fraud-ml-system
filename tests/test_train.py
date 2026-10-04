import json

from fraud.config import DRIFT_SKIP_FEATURES, FEATURE_NAMES
from fraud.train import train


def test_artifacts_and_registry(model_dir):
    version = (model_dir / "PRODUCTION").read_text().strip()
    meta = json.loads((model_dir / version / "metadata.json").read_text())
    assert (model_dir / version / "model.json").exists()
    assert meta["feature_names"] == FEATURE_NAMES
    assert meta["thresholds"]["decline"] >= meta["thresholds"]["step_up"]
    assert set(meta["drift_baseline"]["features"]) == set(FEATURE_NAMES) - DRIFT_SKIP_FEATURES
    assert set(meta["drift_baseline"]["categorical"]) == {"location", "merchant_category"}


def test_served_model_matches_evaluated_model(model_dir, small_df):
    """Regression test: the saved model must contain only the early-stopped trees."""
    import xgboost as xgb
    version = (model_dir / "PRODUCTION").read_text().strip()
    b = xgb.Booster()
    b.load_model(model_dir / version / "model.json")
    best = b.attr("best_iteration")
    assert best is None or b.num_boosted_rounds() == int(best) + 1


def test_champion_challenger_gate(small_df, tmp_path):
    first = train(small_df, tmp_path, min_pr_auc=0.0, run_walk_forward=False)
    assert first["promoted"] and first["champion"]["version"] is None
    # A challenger held to an impossible bar must NOT replace the champion.
    blocked = train(small_df, tmp_path, min_pr_auc=1.01, run_walk_forward=False)
    assert not blocked["promoted"]
    assert blocked["champion"]["version"] == first["version"]
    assert (tmp_path / "PRODUCTION").read_text().strip() == first["version"]
