# Deploying to AWS

Two paths. Do **Path A** this week, it gets you a real public endpoint in about an hour.
Know **Path B** well enough to whiteboard it: that's what you'd build at their scale.

## Path A: EC2 + Docker Compose (about 1 hour, a few dollars)

1. **Launch an instance.** EC2 → Launch instance → Amazon Linux 2023, `t3.medium`
   (Redpanda + Redis + API need about 2 GB; `t3.micro` will run out of memory), 20 GB disk.
   Security group: SSH (22) and TCP 8000, source **My IP only**. Never 0.0.0.0/0.

2. **Install Docker.**
   ```bash
   ssh -i key.pem ec2-user@<public-ip>
   sudo dnf install -y docker git python3-pip && sudo systemctl enable --now docker
   sudo usermod -aG docker ec2-user && newgrp docker
   sudo mkdir -p /usr/local/lib/docker/cli-plugins
   sudo curl -sSL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
     -o /usr/local/lib/docker/cli-plugins/docker-compose && sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
   ```

3. **Get the code and a model there.** Either `git clone` your repo and run `make data train`
   on the box, or train locally and `scp -r artifacts data ec2-user@<ip>:~/fraud-ml-system/`.

4. **Run it.**
   ```bash
   export ADMIN_TOKEN=$(openssl rand -hex 16)
   docker compose up -d --build
   pip install -r requirements.txt
   PYTHONPATH=src python -m fraud.load_online --backfill-csv data/transactions.csv \
     --replay-start-day <START_DAY>          # computes reference tables with pandas
   PYTHONPATH=src python streaming/gateway_simulator.py --start-day <START_DAY> --kafka localhost:19092
   ```

5. **Hit it from your laptop.** `curl http://<public-ip>:8000/ready`, then run
   `scripts/load_test.py --api http://<public-ip>:8000` and write down the p50/p99
   **over the network**. That number is yours to quote.

6. **Tear down.** Stop or terminate the instance when you're done.

## Path B: production shape (whiteboard this)

| Concern | Service | Why |
|---|---|---|
| Image registry | ECR | CI pushes an image per commit SHA |
| API | ECS Fargate behind an ALB (or EKS), autoscaled on CPU / p99 | Stateless containers, rolling deploys |
| Online features | ElastiCache Redis (cluster mode, replica per AZ) | Sub-ms reads; failover |
| Event stream | MSK (Kafka), topic keyed by card_id | Per-card ordering, replay |
| Offline store | S3 (Parquet), Glue catalog | Spark/Databricks reads it for training and batch features |
| Batch jobs | EMR / Databricks, orchestrated by MWAA (managed Airflow) | Same DAG as `airflow/dags/` |
| Model registry | S3 versioned prefix + MLflow (or SageMaker Model Registry) | Promote / roll back = move a pointer |
| Secrets | Secrets Manager | No tokens in env files |
| Observability | CloudWatch / Prometheus + Grafana, alerts to PagerDuty | `monitoring/alerts.yml` |

**Rollout:** shadow first (new model scores live traffic, decisions not used, compare
offline), then a 5% canary split at the ALB, watching decline rate, p99 and score PSI,
then 100%. Rollback is pointing PRODUCTION at the previous version and calling `/admin/reload`.

**SageMaker or Lambda instead?** A SageMaker real-time endpoint gives managed autoscaling
and A/B production variants, but you still need Redis in front for features. Lambda is
cheap for spiky low volume but has cold starts (loading XGBoost and pandas costs seconds) and
needs VPC access to reach ElastiCache, so it's a poor fit for a strict p99 budget on payments.
