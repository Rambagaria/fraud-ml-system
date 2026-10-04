export PYTHONPATH := src
REDIS ?= redis://localhost:6379/0
START_DAY ?= $(shell python -c "import pandas as pd; print(int(pd.read_csv('data/transactions.csv').ts.max()//86400)-6)" 2>/dev/null)

.PHONY: data train test spark load serve replay replay-kafka drift drift-demo load-test up down

data:            ## synthetic transactions -> data/transactions.csv
	python -m fraud.data
train:           ## walk-forward eval, thresholds, champion/challenger gate, registry
	python -m fraud.train
test:            ## unit + training/serving parity + Spark parity + API tests
	pytest -q
spark:           ## daily batch job (reference tables as of the replay start day)
	python batch/spark_reference_features.py --as-of-day $(START_DAY)
load:            ## push Spark output + 30-day history backfill into Redis
	python -m fraud.load_online --redis-url $(REDIS) --backfill-csv data/transactions.csv \
	  --replay-start-day $(START_DAY) --ref-json data/features/as_of_day=$(START_DAY)/ref_tables.json
serve:           ## API on :8000 against local Redis
	REDIS_URL=$(REDIS) uvicorn fraud.api:app --port 8000 --workers 1
replay:          ## replay held-out week as live traffic (no Kafka; updates Redis inline)
	python streaming/gateway_simulator.py --start-day $(START_DAY) --limit 100000 --inline-redis $(REDIS)
replay-kafka:    ## same, but feature updates flow through Kafka -> feature_updater
	python streaming/gateway_simulator.py --start-day $(START_DAY) --limit 100000 --kafka localhost:19092
drift:           ## PSI report on the prediction log (exit 1 on ALERT)
	python -m fraud.drift
drift-demo:      ## shift traffic (amounts x3, unseen city) then watch drift fire
	rm -f logs/predictions.jsonl
	python streaming/gateway_simulator.py --start-day $(START_DAY) --limit 3000 --amount-mult 3 --force-location city_99
	-python -m fraud.drift
load-test:       ## throughput + latency percentiles
	python scripts/load_test.py --requests 3000 --concurrency 16
up:              ## full stack in Docker (redis, redpanda, api, feature-updater, prometheus)
	docker compose up -d --build
down:
	docker compose down -v
