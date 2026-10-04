"""Single source of truth for constants shared by training, serving, batch and streaming.

If training and serving disagree on any of these, you get training-serving skew,
so every component imports them from here instead of redefining them.
"""
import os
from pathlib import Path

DAY_S = 86_400
WINDOW_30D_S = 30 * DAY_S
NO_HISTORY_GAP_S = WINDOW_30D_S  # value for secs_since_last_txn when a card has no history

# Chargebacks arrive late. A label is only "known" this many days after the transaction,
# so target encodings may only use labels older than this. Same rule offline and online.
LABEL_DELAY_DAYS = 7
SMOOTHING_M = 50          # strength of the prior in smoothed target encoding
DEFAULT_PRIOR = 0.002     # used before any labels exist (cold start)

FEATURE_NAMES = [
    "amount",
    "log_amount",
    "hour_of_day",
    "secs_since_last_txn",
    "txn_count_1h",
    "txn_count_24h",
    "amt_sum_24h",
    "amt_dev_30d",
    "loc_changed",
    "loc_fraud_rate",
    "mcc_fraud_rate",
    "has_history",
]

DECISIONS = ["APPROVE", "STEP_UP", "DECLINE"]  # ordered by severity

# Drift monitoring: target-encoded rates take a handful of discrete values that shift
# slightly every day, so value-PSI on them is noise. Watch the raw categories instead.
DRIFT_SKIP_FEATURES = {"loc_fraud_rate", "mcc_fraud_rate"}
CATEGORICAL_INPUTS = ["location", "merchant_category"]

# Runtime settings (env-driven so the same image runs locally, in CI and on AWS)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION")  # optional pin; otherwise read PRODUCTION file
REDIS_URL = os.getenv("REDIS_URL")          # unset -> in-memory store (dev only)
REDIS_TIMEOUT_S = float(os.getenv("REDIS_TIMEOUT_S", "0.05"))
PRED_LOG_PATH = Path(os.getenv("PRED_LOG_PATH", "logs/predictions.jsonl"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
