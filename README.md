# Simple Agentic RAG

A production-style Agentic RAG starter project using FastAPI, LangGraph, ChromaDB, OpenAI, deterministic guardrails, RAGAS evaluation, Docker, and pytest.

The app serves a small RAG API over local markdown documents. The LangGraph workflow rewrites each question, retrieves context from Chroma, grades whether the retrieved context is strong enough, then either answers from context or returns a safe fallback path. Responses include sources and metadata for traceability.

## What Is Included

- FastAPI app with `/api/v1/health` and `/api/v1/chat`
- LangGraph workflow for query rewrite, retrieval, context answer, and fallback routing
- Local Chroma vector store persisted under `storage/chroma`
- OpenAI chat and embedding models configured through `.env`
- LLM Gateway (LiteLLM) integration with support for Redis Semantic Caching and fallback routing
- Observability and tracing via Langfuse Callback Handler
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

# LLM Gateway Config (Optional)
LLM_GATEWAY_URL=http://localhost:4000/v1

# Langfuse Config (Optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Redis Config (Optional, used by LiteLLM)
REDIS_URL=redis://localhost:6379/0
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

# Get litellm models from cli

```bash
$token = "sk-local-litellm-master-key"
Invoke-WebRequest `
  -Uri "http://localhost:4000/v1/models" `
  -Headers @{ Authorization = "Bearer $token" } |
  Select-Object -ExpandProperty Content
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

## LLM Gateway & Observability

This project incorporates a production-grade infrastructure pattern using an LLM Gateway (**LiteLLM**) and an Observability platform (**Langfuse**).

### What is an LLM Gateway?
An LLM Gateway acts as a proxy between your RAG application and LLM providers. Instead of calling OpenAI directly from code, all requests go through LiteLLM. This provides:
1. **Semantic Caching (`redis-semantic`)**: Converts prompt inputs into embedding vectors and queries Redis to find similar past questions. If a query matches with high similarity, the answer is returned instantly without hitting OpenAI, saving cost and latency.
2. **Resilience & Fallbacks**: If the primary LLM provider (e.g., OpenAI) goes down or hits rate limits, the gateway can automatically failover to backup models (e.g., Anthropic Claude, Azure OpenAI) without modifying application code.
3. **Decoupling**: Models, API keys, and load balancing are configured declaratively in `litellm-config.yaml` rather than being hardcoded in application settings.

### Why run LiteLLM as a Standalone Docker Proxy (Server) vs. Python SDK?
LiteLLM can be used either as an imported Python library (`pip install litellm`) or as a standalone Docker proxy server. This project uses the **Standalone Proxy Server** pattern for production-grade benefits:
1. **Security (Least Privilege)**: LLM API keys are isolated to the LiteLLM container. Your FastAPI application pods never see your raw API keys, reducing security attack vectors.
2. **Independent Scaling**: In systems like Kubernetes, your API application can scale out independently of the LLM Gateway depending on resource bottlenecks.
3. **Shared Caching**: Multiple distinct microservices inside a cluster can point to the same central gateway service and share the same Redis semantic cache instance, preventing duplicate LLM queries.
4. **Zero-Downtime Updates**: Changing backing models or adding fallbacks only requires a ConfigMap restart on the LiteLLM gateway, keeping the main application online and untouched.

### What is Langfuse?
Langfuse is an open-source LLM engineering and observability platform. 
1. **Tracing**: It captures the entire execution path of your LangGraph state graph. You can inspect the inputs and outputs of every single step (such as query rewriting, Chroma retrieval, and final context generation).
2. **Performance Tracking**: Automatically monitors latency, token counts, and API costs.
3. **Prompt Management & Evals**: Allows testing and versioning prompts outside the codebase and linking user feedback (like thumbs up/down) to traces.

To run LiteLLM alongside the RAG API, spin up the services using:
```bash
docker compose up --build -d
```

LiteLLM cache entries currently use a 5 minute Redis TTL via `litellm_settings.cache_params.default_redis_ttl`.

When running the API directly on your host machine, keep `LLM_GATEWAY_URL=http://localhost:4000/v1`.
Docker Compose overrides the API container to use `http://litellm-gateway:4000/v1`, because `localhost`
inside a container points to that container, not to the LiteLLM service.

Traces will be sent asynchronously to your configured `LANGFUSE_HOST` instance.


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
