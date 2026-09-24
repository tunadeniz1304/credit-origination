.PHONY: install lint format typecheck test cov gate train seed docs-samples run worker up down smoke llm-smoke

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy app

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest -q --cov=app --cov-report=term --cov-report=xml

# Quality gate run at the end of every phase.
gate: lint typecheck cov
	docker compose build

train:
	$(PY) scripts/generate_training_data.py
	$(PY) scripts/train_pd_model.py

docs-samples:
	$(PY) scripts/generate_sample_docs.py

seed:
	$(PY) scripts/seed_demo.py

run:
	uvicorn app.main:app --reload --port 8000

worker:
	celery -A app.worker.celery_app:celery_app worker --loglevel=info

up:
	docker compose up --build -d

down:
	docker compose down

smoke:
	$(PY) scripts/smoke.py

llm-smoke:
	$(PY) scripts/llm_smoke.py
