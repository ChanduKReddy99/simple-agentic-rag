# Simple Agentic RAG

A production-style Agentic RAG starter project using FastAPI, LangGraph, ChromaDB, OpenAI, deterministic guardrails, RAGAS evaluation, Docker, and pytest.

The app serves a small RAG API over local markdown documents. The LangGraph workflow rewrites each question, retrieves context from Chroma, grades whether the retrieved context is strong enough, then either answers from context or returns a safe fallback path. Responses include sources and metadata for traceability.

## What Is Included

- FastAPI app with `/api/v1/health` and `/api/v1/chat`
- LangGraph workflow for query rewrite, retrieval, context answer, and fallback routing
- Local Chroma vector store persisted under `storage/chroma`
- OpenAI chat and embedding models configured through `.env`
- Deterministic input/output guardrails for prompt-injection and secret leakage checks
- Markdown document ingestion from `data/`
- RAGAS evaluation with local, CI, and nightly run modes
- Timestamped structured logs under `logs/`
- Docker image for serving the FastAPI app
- Unit tests for chunking, loading, and guardrails

## Project Structure

```text
simple-agentic-rag/
├── app/
│   ├── api/              # FastAPI routes
│   ├── agent/            # LangGraph workflow, state, and prompts
│   ├── core/             # Settings, logging, and guardrails
│   ├── rag/              # Loading, chunking, embeddings, vector store, retrieval
│   └── schemas/          # Request and response models
├── data/                 # Markdown source documents
├── evals/                # RAGAS evaluator and golden dataset
├── logs/                 # Timestamped runtime logs
├── reports/              # RAGAS JSON reports
├── scripts/              # Ingestion entrypoints
├── storage/chroma/       # Persisted local Chroma database
├── tests/                # Unit tests
├── Dockerfile
├── docker-compose.yml
├── local_workflow.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Setup

This project is configured for Python 3.12 and `uv`.

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY

uv sync
```

For test and evaluation dependencies:

```bash
uv sync --group test --group eval
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

The app reads settings from `.env`.

```env
APP_NAME=Simple Agentic Rag
APP_ENV=local
LOG_LEVEL=INFO
OPENAI_API_KEY=replace-me
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=storage/chroma
CHROMA_COLLECTION=agentic_rag_docs
CHUNK_SIZE=900
CHUNK_OVERLAP=150
RETRIEVAL_K=5
MIN_RETRIEVAL_SCORE=0.35
GUARDRAILS_ENABLED=true
```

## Ingest Documents

Load markdown files from `data/`, chunk them, embed them, and write them to the configured Chroma collection:

```bash
uv run python -m scripts.ingest
```

Equivalent direct script form:

```bash
uv run python scripts/ingest.py
```

## Run The API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Open the interactive docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Ask a question:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"How does FraudShield handle high-risk transactions?\",\"session_id\":\"local-demo\"}"
```

The chat response includes:

- `answer`: generated answer or guardrail/fallback message
- `route`: workflow route such as `rag_context`, `fallback_no_strong_context`, or a guardrail route
- `sources`: source file, chunk id, retrieval score, and preview text
- `metadata`: session id, rewritten question, and guardrail state

## Local Workflow

Run the full local flow:

```bash
uv sync --group eval
uv run python local_workflow.py
```

This runs document ingestion, guardrail smoke tests, and RAGAS evaluation. By default, the evaluation report is overwritten at:

```text
reports/ragas_report.json
```

Useful options:

```bash
uv run python local_workflow.py --skip-ragas
uv run python local_workflow.py --skip-ingest --skip-ragas
uv run python local_workflow.py --serve-api
uv run python local_workflow.py --run-type ci --ragas-limit 2
uv run python local_workflow.py --timestamped-report
```

## RAGAS Evaluation

The golden dataset lives at `evals/golden_dataset.json`.

```bash
uv sync --group eval
uv run python evals/evaluate_ragas.py --dataset evals/golden_dataset.json --run-type local
```

Use `--limit` for small CI runs and `--timestamped-report` when you want to keep historical artifacts:

```bash
uv run python evals/evaluate_ragas.py --run-type ci --limit 2
uv run python evals/evaluate_ragas.py --run-type nightly --timestamped-report
```

## Guardrails

Guardrails are enabled by default:

```env
GUARDRAILS_ENABLED=true
```

When enabled, the API blocks obvious prompt-injection attempts, requests to reveal secrets or system prompts, and generated responses that look like credentials.

## Tests

```bash
uv sync --group test
uv run pytest -q
```

## Logs And Reports

Entrypoints write timestamped logs under `logs/`, for example:

```text
logs/fastapi_app_20260608_224016.log
logs/local_workflow_20260608_221443.log
logs/ragas_evaluation_YYYYMMDD_HHMMSS.log
logs/document_ingestion_YYYYMMDD_HHMMSS.log
```

RAGAS reports are written under `reports/`. The default report path is `reports/ragas_report.json`.

## Docker

The Docker image serves the FastAPI app and installs the base application dependencies. It does not install test or RAGAS evaluation dependency groups.

```bash
docker build -t simple-agentic-rag .
docker run --env-file .env -p 8000:8000 simple-agentic-rag
```

With Docker Compose:

```bash
docker compose up --build
```
