"""SQLite metadata store for ingested documents."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.models import (
    DocumentMetadata,
    DocumentSummary,
    DocumentType,
    IngestionStatus,
)

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    language TEXT,
    doc_type TEXT NOT NULL,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS query_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""


class MetadataStore:
    """Thread-safe SQLite wrapper for document metadata & metrics."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.Lock()
        self._settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self._settings.sqlite_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_document(self, meta: DocumentMetadata) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, title, source_path, file_type,
                        language, doc_type, n_chunks, created_at, status, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    file_type=excluded.file_type,
                    language=excluded.language,
                    doc_type=excluded.doc_type,
                    n_chunks=excluded.n_chunks,
                    status=excluded.status,
                    size_bytes=excluded.size_bytes
                """,
                (
                    meta.document_id,
                    meta.title,
                    meta.source_path,
                    meta.file_type,
                    meta.language,
                    meta.doc_type.value,
                    meta.n_chunks,
                    meta.created_at.isoformat(),
                    meta.status.value,
                    meta.size_bytes,
                ),
            )

    def update_status(self, document_id: str, status: IngestionStatus, n_chunks: int | None = None) -> None:
        with self._lock, self._connect() as conn:
            if n_chunks is None:
                conn.execute(
                    "UPDATE documents SET status=? WHERE document_id=?",
                    (status.value, document_id),
                )
            else:
                conn.execute(
                    "UPDATE documents SET status=?, n_chunks=? WHERE document_id=?",
                    (status.value, n_chunks, document_id),
                )

    def list_documents(self) -> list[DocumentSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [
            DocumentSummary(
                document_id=r["document_id"],
                title=r["title"],
                file_type=r["file_type"],
                language=r["language"],
                doc_type=DocumentType(r["doc_type"]),
                n_chunks=r["n_chunks"],
                created_at=datetime.fromisoformat(r["created_at"]),
                status=IngestionStatus(r["status"]),
            )
            for r in rows
        ]

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id=?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return DocumentMetadata(
            document_id=row["document_id"],
            title=row["title"],
            source_path=row["source_path"],
            file_type=row["file_type"],
            language=row["language"],
            doc_type=DocumentType(row["doc_type"]),
            n_chunks=row["n_chunks"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=IngestionStatus(row["status"]),
            size_bytes=row["size_bytes"],
        )

    def count_documents(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
        return int(row["c"])

    def record_query(self, query: str, latency_ms: float, confidence: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO query_metrics (query, latency_ms, confidence, created_at) VALUES (?, ?, ?, ?)",
                (query, latency_ms, confidence, datetime.utcnow().isoformat()),
            )

    def query_stats(self) -> dict[str, float]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(AVG(latency_ms), 0) AS avg_latency FROM query_metrics"
            ).fetchone()
        return {"n_queries": int(row["n"]), "avg_latency_ms": float(row["avg_latency"])}


_store: MetadataStore | None = None


def get_store() -> MetadataStore:
    """Return a process-wide :class:`MetadataStore` singleton."""
    global _store
    if _store is None:
        _store = MetadataStore()
    return _store
