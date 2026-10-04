import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.data import generate  # noqa: E402
from fraud.train import train  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    return generate(n_cards=400, days=40, fraud_events=120, seed=3)


@pytest.fixture(scope="session")
def model_dir(small_df, tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    train(small_df, d, min_pr_auc=0.0, run_walk_forward=False)
    return d
