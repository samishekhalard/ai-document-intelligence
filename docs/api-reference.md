# API Reference

Base URL (local): **`http://localhost:8000`**

Interactive documentation is generated automatically by FastAPI:

- Swagger UI: **`/docs`**
- ReDoc: **`/redoc`**
- OpenAPI JSON: **`/openapi.json`**

All request / response bodies are JSON unless noted (`/upload` and
`/extract-ocr` use multipart/form-data).

---

## `GET /health`

Service liveness + Ollama reachability + index size.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "ollama_reachable": true,
  "llm_model": "llama3.2",
  "embed_model": "nomic-embed-text",
  "documents_indexed": 3,
  "app_version": "1.0.0"
}
```

| Field | Type | Meaning |
|---|---|---|
| `status` | `"ok" \| "degraded" \| "down"` | `degraded` when Ollama is unreachable but FastAPI is up |
| `ollama_reachable` | bool | Result of a 5-second Ollama `/api/tags` probe |
| `documents_indexed` | int | Row count in SQLite `documents` table |

**Recommended monitoring**: poll every 10 s; alert if `status != "ok"`.

---

## `GET /stats`

Aggregate metrics for the dashboard.

```json
{
  "documents_indexed": 3,
  "n_queries": 42,
  "avg_latency_ms": 1240.7,
  "chroma_chunks": 21
}
```

---

## `GET /documents`

List all ingested documents with metadata.

```json
[
  {
    "document_id": "72b3aec167e56079",
    "title": "Legal Contract En",
    "file_type": "txt",
    "language": "en",
    "doc_type": "legal",
    "n_chunks": 8,
    "created_at": "2026-04-17T19:45:08.642457",
    "status": "completed"
  }
]
```

`status` values: `pending`, `processing`, `completed`, `failed`.

---

## `POST /upload`

Ingest a document through the full pipeline: parse → chunk → embed → store.

**Content-Type**: `multipart/form-data`

**Fields**:

- `file` (required) — the document; one of PDF, DOCX, TXT, MD, PNG, JPG, TIFF.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/sample_docs/legal_contract_en.txt"
```

**Response 200**

```json
{
  "document_id": "72b3aec167e56079",
  "title": "Legal Contract En",
  "n_chunks": 8,
  "status": "completed",
  "message": "Indexed 8 chunks"
}
```

**Response 422** — the file contains no extractable text.

**Response 500** — the embedding backend failed.

Re-uploading the same file is idempotent; existing chunks are replaced.

---

## `POST /query`

Ask a natural-language question. Runs through the LangGraph orchestrator.

**Content-Type**: `application/json`

**Request schema**

```json
{
  "query": "What are the payment terms of the contract?",
  "top_k": 5,
  "document_ids": ["72b3aec167e56079"]
}
```

| Field | Required | Description |
|---|---|---|
| `query` | yes | 1 – 2000 character question |
| `top_k` | no | Override default rerank top-K (1–20) |
| `document_ids` | no | Restrict retrieval to these documents |

**Response**

```json
{
  "query": "What are the payment terms of the contract?",
  "answer": "Invoices are due net 30 days ... [1]",
  "sources": [
    {
      "chunk_id": "72b3aec167e56079:0003:a1b2c3",
      "document_id": "72b3aec167e56079",
      "text": "All invoices are due net thirty (30) days ...",
      "score": 0.91,
      "dense_score": 0.88,
      "sparse_score": 0.62,
      "metadata": {"title": "Legal Contract En", "doc_type": "legal"}
    }
  ],
  "confidence": 0.78,
  "latency_ms": 1184.2,
  "agent_trace": ["router->rag", "rag:retrieve", "rag:generate", "rag:done"]
}
```

`confidence` is derived from the retrieval score distribution, squashed
to `[0, 1]` via `0.5 + 0.5 * tanh(mean_score - 0.1)`. It is a relative
signal suitable for UI traffic-lighting, not an absolute probability.

---

## `POST /analyze`

Run NLP analysis on raw text or on an indexed document.

**Request schema**

```json
{
  "text": "Apple Inc. reported revenue of USD 96B ...",
  "document_id": null,
  "tasks": ["entities", "classify", "summarize", "language", "keyphrases"]
}
```

Provide **either** `text` **or** `document_id`. Tasks run in parallel.

**Response**

```json
{
  "entities": [
    {"text": "Apple Inc", "label": "ORG", "start": 0, "end": 9},
    {"text": "Tim Cook", "label": "PER", "start": 69, "end": 77}
  ],
  "classification": "financial",
  "summary": "Apple reported Q4 2025 revenue of 96 billion USD ...",
  "language": "en",
  "keyphrases": ["apple", "revenue", "usd", "billion", "ceo"]
}
```

Available tasks:

- `entities` — spaCy NER (PERSON, ORG, GPE, DATE, MONEY, PRODUCT, ...)
- `classify` — one of `legal`, `financial`, `technical`, `general`
- `summarize` — LLM-generated 4–6 sentence summary
- `language` — ISO 639-1 language code
- `keyphrases` — top-N TF-based unigrams + bigrams

---

## `POST /extract-ocr`

OCR an image or scanned page.

**Content-Type**: `multipart/form-data`

**Fields**:

- `file` (required) — PNG, JPG, JPEG, TIF, or TIFF.
- `language` (optional) — Tesseract language code, default `eng+rus`.

```bash
curl -X POST http://localhost:8000/extract-ocr \
  -F "file=@scan.png" -F "language=eng+rus"
```

**Response**

```json
{
  "text": "Enterprise AI Document Intelligence\ninvoice 2026-04-17 ...",
  "n_words": 14,
  "language": "eng+rus",
  "latency_ms": 8664.4
}
```

---

## Error shape

All 4xx / 5xx responses follow FastAPI's default schema:

```json
{"detail": "human-readable error message"}
```

## Rate Limiting

There is no built-in rate limiting. For production put FastAPI behind
a reverse proxy (nginx, Envoy, Traefik) with standard rate-limit rules.

## Authentication

The reference implementation is unauthenticated to keep the demo easy
to run. For production, wire in an `Authorization: Bearer <jwt>` check
via FastAPI dependencies — a 20-line change per endpoint. See
`docs/deployment.md`.
