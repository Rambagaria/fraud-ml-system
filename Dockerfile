# One image for the API and the stream consumer. The MODEL is not baked in:
# it's mounted / pulled from the registry at startup (MODEL_DIR), so promoting or
# rolling back a model never requires rebuilding or redeploying code.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    MODEL_DIR=/app/artifacts/models PRED_LOG_PATH=/app/logs/predictions.jsonl \
    OMP_NUM_THREADS=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY streaming/ streaming/
RUN useradd --create-home app && mkdir -p logs artifacts && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fs localhost:8000/health || exit 1
# One worker per vCPU. Each worker is a separate process (no GIL contention).
CMD ["sh", "-c", "uvicorn fraud.api:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-2}"]
