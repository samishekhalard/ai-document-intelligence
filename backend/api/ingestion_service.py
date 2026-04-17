"""Business-logic facade for document ingestion used by the API layer."""
from __future__ import annotations

from pathlib import Path

from backend.core.config import get_settings
from backend.core.db import get_store
from backend.core.logging import get_logger
from backend.core.models import DocumentMetadata, IngestionStatus, UploadResponse
from backend.pipelines.embedding_pipeline import get_embedding_pipeline
from backend.pipelines.ingestion_pipeline import IngestionPipeline

log = get_logger(__name__)


class IngestionService:
    """Coordinates the ingestion pipeline, embedding store, and metadata DB."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline = IngestionPipeline()
        self.embeddings = get_embedding_pipeline()
        self.store = get_store()

    def ingest_path(self, path: Path) -> UploadResponse:
        """Run the full ingestion pipeline against a file on disk.

        Args:
            path: Filesystem path to the document to ingest.

        Returns:
            An :class:`UploadResponse` describing the final state of
            the document in the index (chunk count + status).

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file type is unsupported or contains
                no extractable text.
            ollama.ResponseError: If the embedding backend failed.
        """
        meta, chunks = self.pipeline.prepare(path)

        # Replace any previous chunks for this document before re-indexing.
        self.embeddings.delete_document(meta.document_id)
        self.store.upsert_document(meta)

        try:
            n = self.embeddings.add_chunks(chunks)
            meta.n_chunks = n
            meta.status = IngestionStatus.COMPLETED
            self.store.update_status(meta.document_id, meta.status, n_chunks=n)
        except Exception as exc:
            log.error(f"Embedding failed for {path.name}: {exc}")
            self.store.update_status(meta.document_id, IngestionStatus.FAILED)
            raise

        return UploadResponse(
            document_id=meta.document_id,
            title=meta.title,
            n_chunks=meta.n_chunks,
            status=meta.status,
            message=f"Indexed {meta.n_chunks} chunks",
        )

    def ingest_bytes(self, filename: str, data: bytes) -> UploadResponse:
        """Persist uploaded bytes under ``UPLOAD_DIR`` then ingest them.

        Args:
            filename: Original client-provided filename (used for the
                on-disk path and for format sniffing).
            data: Raw file bytes as read from the multipart upload.

        Returns:
            The :class:`UploadResponse` returned by :meth:`ingest_path`.
        """
        dst = self.settings.upload_dir / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return self.ingest_path(dst)


_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    """Return a process-wide :class:`IngestionService` singleton."""
    global _service
    if _service is None:
        _service = IngestionService()
    return _service
