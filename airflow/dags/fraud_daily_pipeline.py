"""Daily fraud ML pipeline (Airflow).

  spark_reference_features -> load_online_store -> drift_check   (alerts on PSI > 0.25)
                                               \\-> is_retrain_day -> train_and_gate -> reload_api

* Reference tables must refresh daily: they are features (target encodings), and the
  model was trained on encodings that update daily. Stale tables = skew.
* Retraining is weekly, AND gated: train.py promotes only if the challenger beats the
  champion on the same holdout. A failed gate leaves production untouched.
* Drift failing does NOT block retraining: drift is often the reason to retrain.
"""
from datetime import datetime, timedelta

from airflow import DAG

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.python import ShortCircuitOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import ShortCircuitOperator

REPO = "/opt/fraud-ml-system"
ENV = {"PYTHONPATH": f"{REPO}/src", "REDIS_URL": "redis://redis:6379/0", "PATH": "/usr/local/bin:/usr/bin:/bin"}
AS_OF_DAY = "{{ data_interval_end.int_timestamp // 86400 }}"  # absolute day index
DATA = f"{REPO}/data/transactions.csv"  # s3://bucket/transactions/ in production

default_args = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    # "on_failure_callback": notify_slack,   # page the on-call
}

with DAG(
    dag_id="fraud_daily_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["fraud", "ml"],
) as dag:
    spark_features = BashOperator(
        task_id="spark_reference_features",
        bash_command=f"spark-submit {REPO}/batch/spark_reference_features.py "
                     f"--input {DATA} --as-of-day {AS_OF_DAY} --out {REPO}/data/features",
        env=ENV,
    )
    load_online = BashOperator(
        task_id="load_online_store",
        bash_command=f"python -m fraud.load_online "
                     f"--ref-json {REPO}/data/features/as_of_day={AS_OF_DAY}/ref_tables.json",
        env=ENV,
    )
    drift_check = BashOperator(
        task_id="drift_check",
        bash_command=f"python -m fraud.drift --log {REPO}/logs/predictions.jsonl "
                     f"--model-dir {REPO}/artifacts/models --out {REPO}/logs/drift_{AS_OF_DAY}.json",
        env=ENV,
        retries=0,  # exit code 1 = ALERT; retrying won't change the data
    )
    is_retrain_day = ShortCircuitOperator(
        task_id="is_retrain_day",
        python_callable=lambda data_interval_end, **_: data_interval_end.day_of_week == 0,  # Monday
    )
    train_and_gate = BashOperator(
        task_id="train_and_gate",
        bash_command=f"python -m fraud.train --data {DATA} --model-dir {REPO}/artifacts/models",
        env=ENV,
        execution_timeout=timedelta(hours=2),
    )
    reload_api = BashOperator(
        task_id="reload_api",
        bash_command="curl -fsS -X POST http://api:8000/admin/reload -H \"x-admin-token: $ADMIN_TOKEN\"",
        env={**ENV, "ADMIN_TOKEN": "{{ var.value.fraud_admin_token }}"},
    )

    spark_features >> load_online >> [drift_check, is_retrain_day]
    is_retrain_day >> train_and_gate >> reload_api
