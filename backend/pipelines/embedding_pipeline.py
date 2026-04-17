"""Embedding generation (via Ollama) and ChromaDB persistence."""
from __future__ import annotations

import time
from typing import Iterable

import chromadb
import ollama
from chromadb.config import Settings as ChromaSettings

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.models import Chunk

log = get_logger(__name__)


class EmbeddingPipeline:
    """Generates embeddings with Ollama and persists them into ChromaDB.

    The ChromaDB client is persistent (on-disk) and the collection uses
    cosine similarity. A single :class:`ollama.Client` instance is reused
    to keep the HTTP connection warm.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._ollama = ollama.Client(
            host=self.settings.ollama_base_url,
            timeout=self.settings.ollama_timeout,
        )
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
            A list of 768-dim float vectors (one per input string).

        Raises:
            ollama.ResponseError: If the embedding backend is unreachable
                or the configured embedding model is not installed.
        """
        vectors: list[list[float]] = []
        for t in texts:
            resp = self._ollama.embeddings(
                model=self.settings.ollama_embed_model,
                prompt=t,
            )
            vectors.append(list(resp["embedding"]))
        return vectors

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

        Raises:
            ollama.ResponseError: If the embedding backend fails mid-batch.
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
