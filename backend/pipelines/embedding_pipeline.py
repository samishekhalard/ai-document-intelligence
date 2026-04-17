"""Embedding generation (via sentence-transformers) and ChromaDB persistence."""
from __future__ import annotations

import time
from typing import Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.core.config import get_settings
from backend.core.embedding_client import embed as _embed
from backend.core.logging import get_logger
from backend.core.models import Chunk

log = get_logger(__name__)


class EmbeddingPipeline:
    """Generates embeddings with sentence-transformers and persists them into ChromaDB.

    The ChromaDB client is persistent (on-disk) and the collection uses
    cosine similarity. Embeddings are produced by a local, CPU-only
    SentenceTransformer model loaded via the singleton client.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._chroma = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._chroma.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        """Return the underlying ChromaDB collection."""
        return self._collection

    # --------- embedding --------- #

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute dense vectors for a batch of texts.

        Args:
            texts: Raw strings to embed. Order is preserved in the result.

        Returns:
            A list of float vectors (one per input string). Vector
            dimensionality depends on the configured SentenceTransformer
            model (384 for the default ``all-MiniLM-L6-v2``).
        """
        if not texts:
            return []
        return _embed(texts)

    # --------- persistence --------- #

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Embed ``chunks`` and upsert them into the vector store.

        Args:
            chunks: Iterable of :class:`Chunk` objects produced by the
                ingestion pipeline. Each chunk's ``chunk_id`` is used
                as the ChromaDB primary key, so re-indexing the same
                chunk is idempotent.

        Returns:
            The number of chunks persisted.
        """
        chunks = list(chunks)
        if not chunks:
            return 0
        start = time.perf_counter()
        texts = [c.text for c in chunks]
        vectors = self.embed(texts)
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[c.metadata for c in chunks],
        )
        log.info(
            f"Indexed {len(chunks)} chunks in "
            f"{time.perf_counter() - start:.2f}s"
        )
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk belonging to a given document."""
        self._collection.delete(where={"document_id": document_id})

    def all_chunks(self, document_ids: list[str] | None = None) -> dict:
        """Return every chunk (id/text/metadata) in the collection.

        Used to rebuild BM25 and for evaluation.
        """
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}
        return self._collection.get(include=["documents", "metadatas"], where=where)

    def count(self) -> int:
        return int(self._collection.count())


_pipeline: EmbeddingPipeline | None = None


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Return a process-wide :class:`EmbeddingPipeline` singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = EmbeddingPipeline()
    return _pipeline
