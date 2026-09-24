# Anil2 credit origination platform — one image for API, worker and beat.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY alembic ./alembic
COPY config ./config
COPY rules ./rules
COPY artifacts ./artifacts
COPY docs/policies ./docs/policies
COPY scripts ./scripts
COPY app ./app

RUN useradd --create-home appuser && mkdir -p /app/data /app/logs && chown -R appuser /app
USER appuser

ENV REDIS_URL=redis://redis:6379/0 \
    TASK_QUEUE_BACKEND=celery

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD curl -fs http://localhost:8000/health/live || exit 1

# The same image runs the Celery worker/beat (compose overrides the command).
CMD ["sh", "-c", "alembic upgrade head && gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 --timeout 120"]
