"""Pydantic models shared by the API, agents, and pipelines."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Coarse document classification labels."""

    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    GENERAL = "general"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata persisted for every ingested document."""

    document_id: str
    title: str
    source_path: str
    file_type: str
    language: str | None = None
    doc_type: DocumentType = DocumentType.GENERAL
    n_chunks: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: IngestionStatus = IngestionStatus.PENDING
    size_bytes: int = 0


class Chunk(BaseModel):
    """A single chunk of text prepared for embedding/retrieval."""

    chunk_id: str
    document_id: str
    text: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedChunk]
    confidence: float
    latency_ms: float
    agent_trace: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    text: str | None = None
    document_id: str | None = None
    tasks: list[Literal["entities", "classify", "summarize", "language", "keyphrases"]] = (
        Field(default_factory=lambda: ["entities", "classify", "summarize", "language", "keyphrases"])
    )


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class NLPAnalysis(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    classification: DocumentType = DocumentType.GENERAL
    summary: str = ""
    language: str = "unknown"
    keyphrases: list[str] = Field(default_factory=list)


class OCRRequest(BaseModel):
    language: str = Field(default="eng")


class OCRResponse(BaseModel):
    text: str
    n_words: int
    language: str
    latency_ms: float


class UploadResponse(BaseModel):
    document_id: str
    title: str
    n_chunks: int
    status: IngestionStatus
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    ollama_reachable: bool
    llm_model: str
    embed_model: str
    documents_indexed: int
    app_version: str = "1.0.0"


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    file_type: str
    language: str | None
    doc_type: DocumentType
    n_chunks: int
    created_at: datetime
    status: IngestionStatus
