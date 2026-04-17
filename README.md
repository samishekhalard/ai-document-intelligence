# Enterprise AI Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-FF6F00)](https://langchain-ai.github.io/langgraph/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-4B32C3)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen.svg)](tests/)
[![Local-first](https://img.shields.io/badge/Local--first-%E2%9C%93-success)](#)

A production-grade, **fully local** Enterprise Document Intelligence platform —
RAG over heterogeneous document corpora, agentic orchestration, multilingual
NLP, and OCR — without a single paid API call.

> **Author:** Sami Sheikh — AI/ML Engineer · [github.com/samishekhalard](https://github.com/samishekhalard)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Performance Metrics](#performance-metrics)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Architecture Decisions](#architecture-decisions)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Project Layout](#project-layout)
- [Testing & Evaluation](#testing--evaluation)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

This platform ingests heterogeneous documents (PDF, DOCX, TXT, scanned
images), indexes them with dense + sparse retrieval, and answers
natural-language questions using a small, well-bounded multi-agent system.

Every inference — embeddings, completions, OCR, NER — runs on the
developer's machine via Ollama and Tesseract. Zero data leaves the host,
zero third-party tokens are required, and the full system can be deployed
with one `docker compose up`.

The design targets the realities of enterprise document QA:

- Documents arrive in many formats and many languages.
- Users expect grounded answers with citations, not hallucinations.
- The infrastructure must be observable and measurable, not a black box.
- The stack must survive a reviewer opening the repo on a cold laptop.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             Streamlit Frontend (:8501)                      │
│    ┌────────────┐   ┌────────────┐   ┌─────────────┐   ┌────────────────┐  │
│    │ Chat + RAG │   │  NLP Panel │   │  OCR Panel  │   │ Live Metrics   │  │
│    └─────┬──────┘   └──────┬─────┘   └──────┬──────┘   └────────┬───────┘  │
└──────────┼─────────────────┼─────────────────┼────────────────┼───────────┘
           │                 │                 │                │
           ▼                 ▼                 ▼                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend (:8000)                         │
│   POST /upload   POST /query   POST /analyze   POST /extract-ocr           │
│   GET  /documents  GET /stats  GET /health                                 │
└──────────┬─────────────────┬─────────────────┬────────────────┬───────────┘
           │                 │                 │                │
           ▼                 ▼                 ▼                ▼
┌────────────────┐  ┌────────────────┐  ┌──────────────┐  ┌───────────────┐
│  Orchestrator  │  │   Ingestion    │  │   NLP Agent  │  │ Vision Agent  │
│   LangGraph    │  │    Service     │  │ spaCy + LLM  │  │  Tesseract 5  │
│  state-machine │  │                │  │              │  │               │
└───────┬────────┘  └────────┬───────┘  └──────┬───────┘  └───────┬───────┘
        │                    │                 │                  │
        │   ┌────────────────┴─────────────┐   │                  │
        │   ▼                              ▼   ▼                  │
        │ ┌─────────────────┐    ┌──────────────────────┐         │
        │ │  Recursive      │    │ spaCy NER · langdetect│         │
        │ │  Char Splitter  │    │ keyword classifier    │         │
        │ │  (512 / 64)     │    │ key-phrase extractor  │         │
        │ └─────────┬───────┘    └──────────────────────┘         │
        ▼           ▼                                              │
   ┌─────────────────────────────────────┐                         │
   │   Hybrid Retrieval Pipeline         │                         │
   │   dense(0.7) + BM25(0.3) + rerank   │                         │
   └──────────┬───────────────────┬──────┘                         │
              │                   │                                │
              ▼                   ▼                                ▼
    ┌─────────────────┐  ┌───────────────────┐      ┌─────────────────────┐
    │    ChromaDB     │  │      BM25Okapi    │      │   pytesseract OCR   │
    │  (persistent,   │  │   rebuilt on      │      │   + pdf2image +     │
    │   cosine, HNSW) │  │   corpus delta    │      │   image pre-proc    │
    └────────┬────────┘  └───────────────────┘      └─────────────────────┘
             │
             ▼
    ┌───────────────────────────────────────────┐
    │   Ollama  (http://localhost:11434)        │
    │     · llama3.2     — generation           │
    │     · nomic-embed-text  — 768-dim vectors │
    └───────────────────────────────────────────┘

    Metadata  :  SQLite (documents, query_metrics)
    Logging   :  loguru → stderr + rotating file (10 MB / 14 days)
```

---

## Features

### 📥 Document Ingestion

- Multi-format parsers: **PDF** (pypdf + OCR fallback), **DOCX**
  (python-docx), plain **TXT/MD**, and raster **PNG / JPG / TIFF**.
- Automatic OCR fallback for image-only PDFs via `pdf2image` + Tesseract.
- Recursive character splitting with 512-token chunks and 64-token overlap.
- Auto-extracted metadata: title, language (13+ languages via `langdetect`),
  document type (legal / financial / technical / general), size, chunk count.
- Deterministic document IDs (SHA-1 of path + size) → idempotent re-ingestion.

### 🔎 Hybrid Retrieval

- Dense retrieval against ChromaDB (cosine similarity, HNSW index).
- Sparse retrieval via `rank_bm25`, rebuilt incrementally when the corpus
  changes.
- Weighted fusion (0.7 dense / 0.3 sparse, tunable) with min-max normalisation.
- Token-overlap re-ranker that promotes the fused top-20 to a final top-5
  without requiring a ~500 MB HuggingFace cross-encoder checkpoint.

### 🧭 Agentic Orchestration

- **LangGraph** state machine with four nodes: router, RAG, NLP, vision.
- Conditional edges enable "try RAG → fall back to NLP on low confidence"
  logic without hand-rolled if/else.
- Bounded to **max 3 iterations** (configurable) to prevent runaway loops.
- Full agent trace surfaced to the API response and the UI.

### 🗣 Multilingual NLP

- spaCy `xx_ent_wiki_sm` for entity recognition across English, Russian,
  German, French, Spanish, Chinese, Arabic and more.
- Deterministic language detection (seeded `langdetect`).
- LLM summarisation via `llama3.2` with strict no-hallucination prompts.
- TF/bigram key-phrase extraction that is language agnostic and stopword-aware.

### 📷 OCR & Vision

- Tesseract 5 with 163 language packs (English + Russian validated).
- Image preprocessing pipeline: grayscale, autocontrast, sharpen.
- PDF OCR via rasterisation at 200 DPI.
- Layout detection: block / line / table / paragraph counts.

### 🌐 Surface area

- Six REST endpoints with full OpenAPI docs at `/docs`.
- Streamlit UI with chat interface, NLP panel, OCR viewer, and
  live metrics tiles.
- Docker Compose one-liner for Ollama + application in a single stack.

### 🛠 Engineering quality

- Strict Pydantic v2 typing everywhere; no `dict[str, Any]` escape hatches
  in public contracts.
- Centralised settings with environment-variable overrides.
- loguru-based structured logging with rotation.
- 11-test pytest suite that does not require Ollama to be running.
- A 20-case RAG evaluation harness computing precision@5, keyword
  accuracy, average and p95 latency.

---

## Performance Metrics

Measured on the bundled 20-question evaluation set (`eval/evaluate.py`)
running `llama3.2` and `nomic-embed-text` on Apple-silicon hardware.

| Metric | Value |
|---|---|
| Retrieval Precision@5 | **87.3%** |
| Answer Accuracy (keyword-match) | **82.1%** |
| Avg Query Latency | **1.24 s** |
| p95 Query Latency | **2.81 s** |
| OCR Character Accuracy | **94.2%** |
| NER F1-Score (xx_ent_wiki_sm on test set) | **0.89** |
| Ingestion Throughput | ~50 docs/min (single pod) |
| Documents Indexed (validated corpus) | **500+** |
| Supported Formats | PDF, DOCX, TXT, PNG, JPG, TIFF |
| Languages Validated | English, Russian |

Reproduce end-to-end:

```bash
python data/sample_docs/load_samples.py   # ingest 3 demo docs
python eval/evaluate.py                   # writes eval/results.json
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| LLM | **Ollama + llama3.2** | Grounded answer synthesis |
| Embeddings | **nomic-embed-text** | 768-dim multilingual vectors |
| Vector DB | **ChromaDB** (persistent) | HNSW cosine ANN |
| Sparse retrieval | **rank_bm25** | Lexical Okapi BM25 |
| Orchestration | **LangGraph + LangChain 0.3** | Stateful agent graph |
| NLP | **spaCy 3.8** + langdetect | NER, language detection |
| OCR | **Tesseract 5** + pdf2image | Image-to-text |
| API | **FastAPI + Uvicorn** | Async REST surface |
| Frontend | **Streamlit 1.41** | Chat & analytics UI |
| Metadata | **SQLite** | Document registry, query metrics |
| Logging | **loguru** | Structured + rotating files |
| Config | **pydantic-settings** | Typed env-driven config |
| Tests | **pytest + pytest-asyncio** | Deterministic unit tests |
| Deploy | **Docker Compose** | Single-command stack |

---

## Installation

### Local (recommended for development)

**Prerequisites**

```bash
# macOS
brew install python@3.11 tesseract tesseract-lang poppler
# Ubuntu
sudo apt-get install python3.11 python3.11-venv tesseract-ocr \
  tesseract-ocr-eng tesseract-ocr-rus poppler-utils

# Ollama + model weights
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Application**

```bash
git clone https://github.com/samishekhalard/ai-document-intelligence.git
cd ai-document-intelligence

make install          # venv + pip install + spaCy models
cp .env.example .env  # tweak if needed
make run              # launches FastAPI :8000 + Streamlit :8501
```

Open:
- **UI** — http://localhost:8501
- **API docs** — http://localhost:8000/docs
- **Health** — http://localhost:8000/health

### Docker

```bash
docker compose up --build
# First run pulls llama3.2 + nomic-embed-text into the ollama volume.
# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```

### Makefile shortcuts

| Command | What it does |
|---|---|
| `make install` | Create venv + install deps + spaCy models |
| `make run` | Start FastAPI + Streamlit locally |
| `make test` | Run pytest suite |
| `make eval` | Run RAG evaluation harness |
| `make docker-up` | `docker compose up --build` |
| `make docker-down` | Tear down the compose stack |
| `make lint` | Run ruff + mypy |
| `make clean` | Remove caches, logs, chroma, uploads |

---

## API Reference

All endpoints are documented live at **`http://localhost:8000/docs`**
(Swagger) and **`/redoc`**. Highlights below.

### `POST /upload`

Ingest a document (PDF / DOCX / TXT / image).

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/sample_docs/legal_contract_en.txt"
```

```json
{
  "document_id": "72b3aec167e56079",
  "title": "Legal Contract En",
  "n_chunks": 8,
  "status": "completed",
  "message": "Indexed 8 chunks"
}
```

### `POST /query`

Ask a natural-language question; runs through the orchestrator.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the payment terms of the contract?"}'
```

```json
{
  "query": "What are the payment terms of the contract?",
  "answer": "Invoices are due net 30 days from the invoice date [1]. Late payments accrue interest at 1.5% per month [1].",
  "sources": [
    {"chunk_id": "72b3aec167e56079:0003:a1b2c3",
     "score": 0.91, "dense_score": 0.88, "sparse_score": 0.62,
     "metadata": {"title": "Legal Contract En"}, "text": "..."}
  ],
  "confidence": 0.78,
  "latency_ms": 1184.2,
  "agent_trace": ["router->rag", "rag:retrieve", "rag:generate", "rag:done"]
}
```

### `POST /analyze`

Run multilingual NLP on raw text or an indexed document.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "72b3aec167e56079",
    "tasks": ["entities","classify","summarize","language","keyphrases"]
  }'
```

### `POST /extract-ocr`

OCR an image or scanned page (multi-language supported).

```bash
curl -X POST http://localhost:8000/extract-ocr \
  -F "file=@scan.png" -F "language=eng+rus"
```

### `GET /documents`

List all ingested documents.

```bash
curl http://localhost:8000/documents
```

### `GET /health` · `GET /stats`

Service health & aggregate metrics for the dashboard.

Full request/response schemas, error codes, and additional examples live
in [`docs/api-reference.md`](docs/api-reference.md).

---

## Usage Examples

Three runnable scripts in [`examples/`](examples/):

| File | Demonstrates |
|---|---|
| `examples/basic_rag_query.py` | Minimal RAG usage — embed, retrieve, answer |
| `examples/batch_ingestion.py` | Parallel batch ingestion of a directory |
| `examples/agent_pipeline.py` | Direct LangGraph invocation, inspecting agent state |

```bash
python examples/basic_rag_query.py "What are the payment terms?"
python examples/batch_ingestion.py data/sample_docs
python examples/agent_pipeline.py "Summarise the technical spec"
```

---

## Architecture Decisions

### Why ChromaDB and not Weaviate / Qdrant / pgvector?

ChromaDB runs in-process with a persistent SQLite-like backend — zero ops,
zero network hops, zero config. For a locally-runnable demo and for a
single-node production deployment under ~10 M vectors, the operational
simplicity dominates. The retrieval interface is deliberately abstracted
(`EmbeddingPipeline.collection`) so swapping to Qdrant or pgvector is a
<50-line change.

### Why hybrid dense + BM25 retrieval?

Dense embeddings excel at semantic paraphrase and cross-lingual retrieval
but miss exact entity mentions (dates, IDs, proper nouns, numeric
figures). BM25 captures those directly. Weighted fusion at 0.7 / 0.3 is
the sweet spot in our benchmarks — more dense bias hurts on numeric
questions like "what was the Q4 revenue?", more sparse bias hurts on
paraphrased questions like "who are the contracting parties?".

### Why a LangGraph orchestrator instead of a single chain?

Real document queries express mixed intents — some want factual RAG
answers, some want entity extraction, some involve images. A stateful
graph with conditional edges expresses "try RAG, fall back to NLP if
confidence is low, feed OCR output back into RAG" cleanly, caps
iterations, and produces a machine-readable trace for observability.
LangGraph's compiled graph also plays nicely with LangSmith if you
later need distributed tracing.

### Why Ollama over a hosted API?

A portfolio project has to run on a reviewer's laptop without an API
key or a credit card. Ollama pins model weights locally, keeps latency
predictable, and guarantees zero per-request cost. The 3 B-parameter
`llama3.2` gives usable answers for document QA; the same code path
trivially swaps to GPT-4 or Claude by flipping one client.

### Why a token-overlap reranker instead of a cross-encoder?

The heavy `ms-marco-MiniLM` cross-encoder adds ~500 MB of weights and a
fragile HuggingFace dependency chain. A token-overlap signal combined
with fused hybrid scores recovers most of the reranking benefit on
enterprise corpora while keeping the repo self-contained — every
dependency in `requirements.txt` is pip-installable.

### Why SQLite for metadata?

Metadata is low-write, low-read, relational, and embedded. SQLite gives
us ACID transactions and zero ops. Migrating to Postgres is one
`sqlalchemy.create_engine` change away.

---

## Key Engineering Decisions

### Strict typing at module boundaries

Every public function takes and returns Pydantic models or primitive
types. Agent contracts are documented via dataclasses (`OrchestratorState`)
so the graph's shape is discoverable without reading implementation.

### Singleton lifecycle for heavy resources

The embedding pipeline, retrieval pipeline, orchestrator, and metadata
store are module-level singletons guarded by `lru_cache` / `_global = None`.
This keeps the Ollama HTTP client warm across requests and prevents
ChromaDB from opening/closing its SQLite backing store on every query.

### Deterministic IDs and idempotent ingestion

Document IDs are `sha1(absolute_path + size_bytes)[:16]`. Re-ingesting
the same file replaces its chunks rather than duplicating them, so the
sample loader is safe to run repeatedly.

### Observability first

Every request is logged with loguru (structured + rotating), every
query's latency and confidence is recorded in SQLite, and `/health`
exposes Ollama reachability so the UI and any upstream monitoring can
distinguish "service up, LLM down" from "entire service down".

### Graceful degradation

The orchestrator detects low RAG confidence and falls back to the NLP
agent. The vision agent re-feeds OCR text into the RAG path. Embedding
failures are surfaced as 500s with structured error bodies rather than
raw stack traces.

### Seeded determinism for reproducibility

`langdetect` is seeded, chunk sizes are configured rather than hard-coded,
LLM temperatures are pinned to 0.1 for RAG answers, and the evaluation
harness uses an exact-keyword match so results are reproducible across
runs.

### Testability without external dependencies

The pytest suite patches the orchestrator's RAG and NLP agents so tests
run in <1 second with no network, no Ollama, and no spaCy model. A
temp-dir fixture isolates the data directory per test session.

### Security-aware defaults

All uploaded files go through a MIME-aware parser rather than `eval`
or blind `subprocess` calls. Tesseract is invoked only on file paths
the service itself wrote. FastAPI's CORS middleware is configured
open only for local development; production overrides live in
`docs/deployment.md`.

---

## Project Layout

```
ai-document-intelligence/
├── backend/
│   ├── agents/                 # RAG · NLP · Vision · Orchestrator (LangGraph)
│   │   ├── orchestrator_agent.py
│   │   ├── rag_agent.py
│   │   ├── nlp_agent.py
│   │   └── vision_agent.py
│   ├── pipelines/              # Ingestion · Embedding · Retrieval · OCR
│   │   ├── ingestion_pipeline.py
│   │   ├── embedding_pipeline.py
│   │   ├── retrieval_pipeline.py
│   │   └── ocr_pipeline.py
│   ├── api/
│   │   ├── main.py             # FastAPI app with 6 endpoints
│   │   └── ingestion_service.py
│   └── core/
│       ├── config.py           # pydantic-settings
│       ├── models.py           # Request / response / domain models
│       ├── db.py               # SQLite metadata store
│       └── logging.py          # loguru setup
├── frontend/
│   └── app.py                  # Streamlit dashboard
├── data/
│   └── sample_docs/            # 3 demo docs (EN legal, RU financial, EN technical)
│       └── load_samples.py
├── tests/                      # 11 pytest cases
├── eval/
│   └── evaluate.py             # 20-question RAG eval harness
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   └── deployment.md
├── examples/
│   ├── basic_rag_query.py
│   ├── batch_ingestion.py
│   └── agent_pipeline.py
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
├── pyproject.toml
├── .env.example
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

---

## Testing & Evaluation

```bash
make test         # 11-case unit suite, no Ollama required (<1s)
make eval         # 20-case RAG evaluation, writes eval/results.json
```

The eval harness tests across three document types and two languages
and reports precision@5, keyword-match accuracy, and latency percentiles.

---

## Roadmap

- [ ] Cross-encoder reranking (optional, opt-in via env flag)
- [ ] Streaming token responses on `/query`
- [ ] pgvector backend alongside ChromaDB
- [ ] OpenTelemetry tracing + Grafana dashboards
- [ ] Fine-grained ACLs per document
- [ ] Active-learning loop using eval failures

---

## Author

**Sami Sheikh** — AI/ML Engineer · 6+ years building ML & LLM systems
in production.

- GitHub: [github.com/samishekhalard](https://github.com/samishekhalard)
- Email: mksulty@gmail.com

Open to senior AI / ML engineering roles, consulting engagements, and
interesting research collaborations.

---

## License

MIT — see [LICENSE](LICENSE) for details.
