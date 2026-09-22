# Akıllı Kredi Operasyon Ajanı (Smart Credit Operations Agent)

A modular, logging-enabled Python pipeline that automates a bank's credit application
process end-to-end through three cooperating agents:

1. **Belge Kontrol Ajanı** (Document Control Agent) — compares submitted documents
   against the required policy document list, flags missing ones, and drafts a
   Turkish request template to send to the applicant.
2. **API Entegrasyon Ajanı** (API Integration Agent) — simulates external API calls
   (KKB credit bureau, e-Devlet employment records) and aggregates the applicant's
   financial data.
3. **Kredi Komitesi Ajanı** (Credit Committee Agent) — analyzes the aggregated data
   against committee thresholds and autonomously prepares a structured
   approve/reject rationale report for the human credit committee.

## Project Structure

```
Anil2/
├── README.md
├── requirements.txt            # stdlib-only runtime; pytest for tests
├── config/
│   └── config.json             # document policy, mock API, committee thresholds
├── src/
│   ├── __init__.py
│   ├── logger.py               # console + file logging setup
│   ├── models.py               # domain dataclasses
│   ├── integrations/
│   │   └── mock_providers.py   # KKB + e-Devlet mock clients
│   ├── agents/
│   │   ├── base.py
│   │   ├── document_agent.py
│   │   ├── api_agent.py
│   │   └── committee_agent.py
│   └── main.py                 # end-to-end pipeline runner
└── tests/
    └── test_flow.py
```

## Requirements

- Python 3.10+ (stdlib only for the runtime; no third-party packages needed).
- `pytest` (>=8,<9) for the test flow: `pip install -r requirements.txt`.

## Usage

Run the full demo pipeline (three scenarios: complete application, missing
documents, high-risk applicant):

```bash
python -m src.main
```

Run the test flow:

```bash
python -m pytest -v
```

Logs are written to `logs/` (console + rotating file handler).

## Configuration

All policies live in `config/config.json`:

- `document_policy.required_documents` — the mandatory document list;
- `api.kkb` / `api.edevlet` — mock API endpoints, timeouts, simulated latency;
- `committee` — credit committee thresholds (KBB score, debt/income, term, ...).
