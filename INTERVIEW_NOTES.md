# Interview notes

Run everything yourself before using any of this. Every number and story below should
come from **your** terminal, so you can answer the fourth follow-up question.

## 60-second version

"At LTIMindtree I built the fraud model for a card client: point-in-time features, XGBoost
with cost-sensitive weighting, F-beta thresholds and three risk tiers. Afterwards I rebuilt
the system end to end myself, including the parts I hadn't owned. It has a FastAPI scorer
reading a Redis online feature store, a Kafka consumer keeping card history fresh, a daily
Spark job for target encodings, Airflow orchestration, a champion/challenger gate,
PSI-based drift monitoring, and CI that builds and smoke-tests the container. I replayed a
held-out week through the deployed API and got exactly the offline recall and precision,
which is how I verified there was no training-serving skew. Testing it live also surfaced
three real bugs."

## Design decisions (each one is a likely question)

**Why one `compute_features` function?** Training-serving skew is the most common silent
failure in production ML. Training replays history through the same function the API
calls. A parity test replays transactions through Redis and asserts every feature matches
to 1e-9.

**How do you avoid target-encoding leakage?** A transaction on day *d* only sees labels
from before *d − 7*. Chargebacks take time to arrive, so that's what production actually
knows. Encodings are smoothed toward the global prior with m=50, so rare locations
don't get extreme values.

**Why Redis sorted sets?** Score = timestamp gives O(log n) range reads for "last 1h / 24h
/ 30d". Each write trims the set to 30 days and sets a TTL so memory stays bounded.

**Why is the Kafka consumer idempotent?** It's at-least-once: the offset is committed after
the Redis write, so a crash means a replay, not data loss. The txn_id is part of the set member,
so a duplicate event is a no-op. Events are keyed by card_id, so they stay ordered per card.

**Why record declined attempts in history?** Training history contains all attempts, so
parity requires it. A burst of declines is also a strong card-testing signal.

**Why two thresholds?** STEP_UP maximizes F2, because missed fraud costs more than an OTP prompt.
DECLINE requires at least 95% precision, because locking a legitimate customer's card is
expensive. The thresholds are tuned on validation and reported on an untouched test week.

**Why PR-AUC?** At 0.2% positives, ROC-AUC looks great almost by default (0.98 here)
while PR-AUC (0.80) shows the real tradeoff.

**What does the deploy and rollback flow look like?** The model artifact is versioned separately from the code image.
Promotion writes the PRODUCTION pointer; `/admin/reload` swaps the model reference
atomically, with no restart. Rollback is the same operation with the previous version.

**Why does the gate compare models on the same holdout?** Comparing the challenger's test
score against the champion's old test score compares different time periods, which isn't a fair test.

## Failure modes, and what the system does

| Failure | Behavior |
|---|---|
| Redis slow or down | 50 ms timeout → degraded mode: score with empty history and default rates, step up amounts ≥ $200, counter metric, alert. Fail-open with friction. |
| Kafka consumer lag | Velocity features undercount (two fast transactions before the update lands). Monitor consumer lag; the inline write path is an option for velocity. |
| Bad model promoted | Gate blocks most cases. Decline-rate spike alert catches the rest; rollback in seconds. |
| Poison Kafka message | Skipped and logged (dead-letter topic in production) so the partition isn't blocked. |
| Stale reference tables | The daily Airflow job fails, so tables age. Alert on `ref:as_of_day` age. |
| Feature drift | PSI > 0.25 → Airflow task fails → page the on-call. Investigate, then retrain or adjust thresholds. |
| Labels delayed 30–90 days | Can't watch recall live. Use proxies: score PSI, decision rates, OTP-failure rate. |

## Bugs I found by running it live

1. **The served model wasn't the evaluated model.** Early stopping chose 324 trees;
   `Booster.inplace_predict` in serving used all 364. Found because the champion's PR-AUC
   on identical data didn't match. Fixed by saving only the best trees and scoring with the
   same call everywhere. A regression test guards it.
2. **Drift false alarms on target-encoded features.** The baseline included the cold-start
   days, and the encodings take a few discrete values that shift daily, so PSI was 5.0 on
   healthy traffic. Fixed by baselining on the most recent window and monitoring the raw
   location and merchant mix instead.
3. **Binary features were invisible to PSI.** With a single bin edge at 0, values 0 and 1
   landed in the same bin. Forcing every transaction to a new city showed PSI 0.0; after
   the fix it showed 10.4.
4. **Load-test p99 was 330 ms while compute was about 3 ms.** That was queueing at
   saturation: 16 in flight at about 240 req/s means ~67 ms average wait (Little's law),
   on 1 vCPU with the load generator competing. The fix is capacity (one worker per core,
   horizontal scaling), not code.
5. **Known gap:** with several uvicorn workers, each process has its own Prometheus
   registry, so a scrape sees one worker. Fix with `prometheus_client` multiprocess mode or
   one worker per container.

## Mapping to their ad platform

| This project | Their world |
|---|---|
| Fraud classifier at 0.2% positives | Click fraud / trust & safety; CTR prediction (~1% positives) |
| Calibration gap (mean score 0.023 vs 0.0021 base rate) | A pCTR feeds the bid: expected value = pCTR × value. Must be calibrated (isotonic / Platt on a holdout) |
| Card history in Redis, updated from Kafka | User and ad counters from impression/click streams |
| Delayed chargebacks | Delayed conversions (attribution windows) |
| Shadow → canary → full rollout | Online A/B on revenue, CTR and advertiser ROI, not just offline AUC |
| Movie recommender (retrieval + embeddings) | Ad ranking: candidate retrieval → ranking model → auction |

## Likely director questions

- *"Scale this to 1M QPS."* Stateless API horizontally scaled; Redis cluster sharded by
  key; reference tables cached in-process; precompute what you can; ONNX or a native model
  runtime; set a latency budget per stage; at worst, fall back to rules.
- *"How would you know the model is degrading?"* Score PSI, feature PSI, decision rates
  daily; label-based metrics once labels mature; business metrics such as chargeback
  rate and OTP abandonment.
- *"When do you retrain?"* On a schedule (weekly) plus a drift trigger, always through
  the gate. Retraining on drifted data without a gate can make things worse.
- *"What would you do differently?"* Calibrate scores, add a real feature store
  (Feast/Tecton) for point-in-time joins at scale, shadow deployments, and Prometheus multiprocess mode.
