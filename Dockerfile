# Akıllı Kredi Operasyon Ajanı — API image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first to leverage layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + rules.
COPY config ./config
COPY app ./app

# Docker topology: Celery over the compose Redis broker.
ENV REDIS_URL=redis://redis:6379/0 \
    RESULT_BACKEND=redis://redis:6379/0 \
    TASK_QUEUE_BACKEND=celery

EXPOSE 8000

# The same image runs both the API (this CMD) and the Celery worker
# (overridden by the compose ``command`` on the worker service).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
