"""FastAPI application exposing the document-intelligence platform."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.nlp_agent import NLPAgent
from backend.agents.orchestrator_agent import get_orchestrator
from backend.agents.vision_agent import VisionAgent
from backend.core.config import get_settings
from backend.core.db import get_store
from backend.core.logging import get_logger
from backend.core.models import (
    AnalyzeRequest,
    DocumentSummary,
    HealthResponse,
    NLPAnalysis,
    OCRResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from backend.api.ingestion_service import get_ingestion_service

log = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Enterprise-grade document intelligence: RAG + NLP + OCR agents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:  # pragma: no cover - side effect
    get_settings().ensure_directories()
    get_store()  # initialise SQLite schema
    log.info("API startup complete")


# --------- /health --------- #


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report local LLM readiness + index size."""
    llm_ready = settings.model_path.exists()
    if not llm_ready:
        log.warning(f"GGUF model missing at {settings.model_path}")

    store = get_store()
    return HealthResponse(
        status="ok" if llm_ready else "degraded",
        llm_ready=llm_ready,
        llm_model=str(settings.model_path.name),
        embed_model=settings.embedding_model,
        documents_indexed=store.count_documents(),
    )


# --------- /documents --------- #


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    """List every ingested document with its metadata."""
    return get_store().list_documents()


# --------- /upload --------- #


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a document and run the full ingestion pipeline synchronously."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    data = await file.read()
    try:
        return get_ingestion_service().ingest_bytes(file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.error(f"Ingestion failed: {exc}")
        raise HTTPException(status_code=500, detail="Ingestion pipeline failed")


# --------- /query --------- #


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer ``req.query`` via the orchestrator agent."""
    result = get_orchestrator().run(req.query)
    rag: QueryResponse | None = result.get("rag")
    if rag is None:
        # Orchestrator chose NLP path only — synthesise a QueryResponse.
        nlp: NLPAnalysis | None = result.get("nlp")
        if nlp is None:
            raise HTTPException(status_code=500, detail="No agent produced a result")
        answer_parts: list[str] = []
        if nlp.summary:
            answer_parts.append(nlp.summary)
        if nlp.entities:
            answer_parts.append(
                "Entities: " + ", ".join(f"{e.text} ({e.label})" for e in nlp.entities[:10])
            )
        if nlp.keyphrases:
            answer_parts.append("Key phrases: " + ", ".join(nlp.keyphrases))
        return QueryResponse(
            query=req.query,
            answer="\n".join(answer_parts) or "No result.",
            sources=[],
            confidence=0.4,
            latency_ms=0.0,
            agent_trace=result.get("trace", []),
        )
    rag.agent_trace = list(rag.agent_trace) + list(result.get("trace", []))
    get_store().record_query(req.query, rag.latency_ms, rag.confidence)
    return rag


# --------- /analyze --------- #


_nlp_agent: NLPAgent | None = None


def _nlp() -> NLPAgent:
    global _nlp_agent
    if _nlp_agent is None:
        _nlp_agent = NLPAgent()
    return _nlp_agent


@app.post("/analyze", response_model=NLPAnalysis)
def analyze(req: AnalyzeRequest) -> NLPAnalysis:
    """Run NLP analysis on raw text or on a previously-ingested document."""
    text = req.text
    if not text and req.document_id:
        emb = get_ingestion_service().embeddings
        data = emb.all_chunks(document_ids=[req.document_id])
        text = "\n\n".join(data.get("documents") or [])
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Either 'text' or 'document_id' with indexed chunks is required")
    return _nlp().analyze(text, tasks=list(req.tasks))


# --------- /extract-ocr --------- #


_vision_agent: VisionAgent | None = None


def _vision() -> VisionAgent:
    global _vision_agent
    if _vision_agent is None:
        _vision_agent = VisionAgent()
    return _vision_agent


@app.post("/extract-ocr", response_model=OCRResponse)
async def extract_ocr(
    file: UploadFile = File(...),
    language: str = Form(default="eng+rus"),
) -> OCRResponse:
    """Run OCR on an uploaded image."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return _vision().process_bytes(data, language=language)
    except Exception as exc:
        log.error(f"OCR failed: {exc}")
        raise HTTPException(status_code=500, detail="OCR pipeline failed")


# --------- /stats --------- #


@app.get("/stats")
def stats() -> dict[str, Any]:
    """Aggregate query metrics (used by the Streamlit dashboard)."""
    store = get_store()
    return {
        "documents_indexed": store.count_documents(),
        **store.query_stats(),
        "chroma_chunks": get_ingestion_service().embeddings.count(),
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
