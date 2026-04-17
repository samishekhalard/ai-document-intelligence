# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-15

### Added
- Initial production release.
- RAG pipeline with hybrid search (dense ChromaDB vectors + sparse BM25).
- Weighted-score fusion (0.7 dense / 0.3 sparse) with min-max normalisation.
- Token-overlap reranker promoting fused top-20 to final top-5.
- Multi-agent orchestration via LangGraph state machine
  (router → RAG → NLP → vision → finalise) with bounded iterations.
- Automatic fallback from RAG to NLP when retrieval confidence is low.
- OCR support via Tesseract 5 with 163 language packs, image preprocessing,
  and PDF rasterisation fallback for scan-only PDFs.
- Multilingual NLP pipeline: spaCy NER (`xx_ent_wiki_sm`),
  `langdetect`-based language detection, LLM summarisation, keyword-based
  classification, and TF / bigram key-phrase extraction.
- FastAPI backend with six endpoints (`/upload`, `/query`, `/analyze`,
  `/extract-ocr`, `/documents`, `/health`) plus `/stats` for metrics.
- Streamlit dashboard with chat interface, NLP panel, OCR tab, and
  live metric tiles.
- Dockerfile and `docker-compose.yml` orchestrating Ollama + application
  in a single-command deployment.
- loguru-based structured logging with rotating file sink
  (10 MB rotation, 14-day retention).
- SQLite-backed metadata store for documents and query metrics.
- pytest suite with 11 deterministic test cases covering chunking,
  NLP heuristics, and orchestrator routing.
- RAG evaluation harness with 20 question/answer pairs reporting
  precision@5, answer accuracy, average and p95 latency.
- Comprehensive documentation: README, architecture deep-dive,
  API reference, deployment guide, CONTRIBUTING, and CHANGELOG.

### Changed
- Migrated all models from Pydantic v1 to v2 for 3× faster validation.
- Switched default chunker from sentence-based to recursive character
  splitting for more even chunk sizes across formats.

### Fixed
- Vector store no longer duplicates chunks when a document is re-ingested
  (deterministic SHA-1 IDs + upsert semantics).
- BM25 index now rebuilds incrementally when corpus size changes rather
  than on every query.

## [0.9.0] - 2025-08-20

### Added
- Beta release with end-to-end RAG functionality.
- Initial `OrchestratorAgent` / `RAGAgent` / `NLPAgent` / `VisionAgent`
  classes with shared state.
- Early BM25 sparse retrieval prototype.
- Streamlit UI scaffolding.

### Fixed
- Embedding pipeline memory leak on large (>100 MB) documents
  caused by retained PDF page objects.
- ChromaDB connection pooling issue where concurrent ingestion would
  hit "database is locked" under heavy load.
- `langdetect` non-deterministic output stabilised by seeding the
  underlying detector factory.

### Changed
- Standardised on `loguru` (previously mixed stdlib `logging` calls).
- Unified configuration via `pydantic-settings` singleton.

## [0.8.0] - 2025-07-10

### Added
- Initial project scaffolding: backend/pipelines, backend/agents,
  backend/core module layout.
- Basic document ingestion for PDF and DOCX.
- First version of the dense-only retrieval pipeline.
- Placeholder FastAPI app with `/health` endpoint.
- `.env.example` and environment-based configuration.

### Known Issues
- No OCR support yet.
- No multi-agent orchestration.
- No evaluation harness.
