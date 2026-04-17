# Production Deployment Guide

This document covers how to move from a developer-laptop setup to a
production-grade deployment.

## Recommended Topology

```
                  ┌──────────────────────────────────────────────┐
                  │               Load balancer / CDN             │
                  └───────────────────────┬──────────────────────┘
                                          │
            ┌─────────────────────────────┼──────────────────────────────┐
            ▼                             ▼                              ▼
   ┌──────────────────┐        ┌──────────────────┐          ┌──────────────────┐
   │ Streamlit (3+)   │        │ FastAPI (3+)     │          │ Worker pool      │
   │  stateless pods  │        │  stateless pods  │          │  (ingestion)     │
   └──────────────────┘        └─────────┬────────┘          └─────────┬────────┘
                                         │                             │
                                         ▼                             ▼
                          ┌──────────────────────────┐     ┌────────────────────┐
                          │  Ollama inference pool   │     │   S3 / GCS blob    │
                          │   (GPU or CPU-only)      │     │   — uploads store  │
                          └──────────────────────────┘     └────────────────────┘
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                       ┌──────────┐ ┌────────┐ ┌───────────┐
                       │ ChromaDB │ │Postgres│ │Prometheus │
                       │  (or     │ │(replace│ │  + Grafana│
                       │ Qdrant)  │ │ SQLite)│ │           │
                       └──────────┘ └────────┘ └───────────┘
```

## Environment Variables

Override `.env.example` values via the orchestrator (Kubernetes
ConfigMap/Secret, Docker env, systemd drop-in).

| Variable | Production value | Rationale |
|---|---|---|
| `APP_ENV` | `production` | Disables reload, tightens logs |
| `LOG_LEVEL` | `INFO` | `DEBUG` is too verbose |
| `OLLAMA_BASE_URL` | internal DNS | e.g. `http://ollama.ml.svc:11434` |
| `OLLAMA_TIMEOUT` | `30` | Matches ingress timeouts |
| `CHROMA_DIR` | mounted PV | Persists vectors across restarts |
| `SQLITE_PATH` | (unset) | Replace with Postgres DSN — see below |

## Replace SQLite with Postgres

SQLite is fine up to ~1M documents on a single node. For HA, replace
`backend/core/db.py`'s connection factory with an SQLAlchemy engine
pointing at Postgres. The schema (two tables: `documents`,
`query_metrics`) is trivially portable — no SQLite-specific pragmas
are used.

```python
# backend/core/db.py (sketch)
from sqlalchemy import create_engine
engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
```

Remember to run a migration (Alembic) before the new version goes live.

## Replace ChromaDB (optional)

ChromaDB persistent mode is battle-tested to ~10M vectors. Above that,
consider Qdrant or Weaviate:

```python
# backend/pipelines/embedding_pipeline.py
from qdrant_client import QdrantClient
self._qdrant = QdrantClient(url=os.environ["QDRANT_URL"])
```

All consumer code uses only `embedding_pipeline.collection.query` and
`.upsert` — swapping backends is localised.

## Authentication

Add a dependency to every mutating endpoint:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer

bearer = HTTPBearer()

def require_token(token = Security(bearer)):
    if not verify_jwt(token.credentials):
        raise HTTPException(401, "invalid token")

@app.post("/upload", dependencies=[Depends(require_token)])
async def upload(...): ...
```

Issue short-lived JWTs from your IdP (Auth0, Keycloak, Okta).

## CORS

Production should tighten CORS:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

## TLS

Terminate TLS at the load balancer (AWS ALB, GCP HTTPS LB, or nginx
ingress). FastAPI pods communicate over cleartext inside the cluster.

## Rate Limiting

Enforce at the ingress / reverse proxy layer. Suggested defaults:

- `/query`: 60 req/min/IP
- `/upload`: 10 req/min/IP (ingestion is expensive)
- `/extract-ocr`: 30 req/min/IP

## Observability

### Logs

Emit JSON to stdout; centralise via your log aggregator (Loki, Datadog,
ELK). Enable JSON mode by tweaking `backend/core/logging.py`:

```python
logger.add(sys.stdout, serialize=True, level=settings.log_level)
```

### Metrics

Expose a `/metrics` endpoint for Prometheus:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Relevant dashboards:

- Query latency p50 / p95 / p99.
- Retrieval precision trend (computed from the nightly eval job).
- Ingestion queue depth.
- Ollama token throughput.

### Traces

Add OpenTelemetry:

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)
```

## Backups

- **Postgres**: nightly pg_dump → S3, 30-day retention.
- **ChromaDB**: volume snapshot + file-level backup of `data/chroma/`.
- **Uploads**: the `data/uploads/` directory should mirror to S3 in
  production (S3 becomes the source of truth).

## Runbooks

Keep short runbooks in `docs/runbooks/`:

- `ollama-down.md` — how to fail traffic to a standby inference pool.
- `reingest.md` — how to re-embed the entire corpus after changing
  the embedding model.
- `eval-regression.md` — how to triage a drop in precision@5.

## Capacity Planning

Rules of thumb per FastAPI pod (2 vCPU / 4 GB):

- ~80 query/s at p50 180 ms (no LLM).
- ~5 query/s at p50 1.2 s (with llama3.2 via Ollama).
- ~50 doc/min ingestion throughput (TXT/DOCX); ~5 doc/min for 50-page PDFs.

Scale Ollama independently: it is the bottleneck under load.
