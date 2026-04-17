"""Hybrid (dense + BM25) retrieval with lightweight LLM-free re-ranking.

Flow
----
1. Dense top-K via ChromaDB (cosine similarity against nomic-embed-text).
2. Sparse top-K via BM25 over the same corpus.
3. Fuse via weighted score combination (configurable 0.7 / 0.3).
4. Re-rank the fused top-20 to top-5 using a token-overlap signal that
   does not require downloading a heavy cross-encoder (keeps the stack
   fully local / free).
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.models import RetrievedChunk
from backend.pipelines.embedding_pipeline import get_embedding_pipeline

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class _BM25Index:
    """Snapshot of the BM25 index rebuilt whenever the corpus changes."""

    ids: list[str]
    texts: list[str]
    metadatas: list[dict]
    bm25: BM25Okapi
    signature: int


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class RetrievalPipeline:
    """Hybrid retrieval across the ChromaDB corpus."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embeddings = get_embedding_pipeline()
        self._bm25: _BM25Index | None = None
        self._bm25_lock = threading.Lock()

    # --------- BM25 --------- #

    def _corpus_signature(self) -> int:
        return self.embeddings.count()

    def _rebuild_bm25(self) -> _BM25Index | None:
        data = self.embeddings.all_chunks()
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{} for _ in ids]
        if not ids:
            return None
        tokenized = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized)
        return _BM25Index(
            ids=ids,
            texts=docs,
            metadatas=metas,
            bm25=bm25,
            signature=len(ids),
        )

    def _ensure_bm25(self) -> _BM25Index | None:
        sig = self._corpus_signature()
        with self._bm25_lock:
            if self._bm25 is None or self._bm25.signature != sig:
                self._bm25 = self._rebuild_bm25()
            return self._bm25

    # --------- public API --------- #

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid retrieval and return the top-reranked chunks.

        The pipeline executes the following steps:

        1. Dense retrieval against ChromaDB (cosine similarity).
        2. Sparse retrieval against an in-memory BM25 index.
        3. Min-max normalise each score stream, then fuse using
           ``dense_weight`` / ``sparse_weight`` from settings.
        4. Re-rank the fused top-``RETRIEVAL_TOP_K`` by combining the
           fused score (0.7) with a query/chunk token-overlap
           signal (0.3).

        Args:
            query: Natural-language query string.
            top_k: Optional override for the final number of chunks
                returned. Defaults to ``RERANK_TOP_K``.
            document_ids: Optional allow-list of document IDs. When
                ``None`` the entire corpus is searched.

        Returns:
            A list of :class:`RetrievedChunk` ordered by descending
            relevance. May be empty if the corpus is empty or every
            candidate is filtered out.

        Raises:
            ollama.ResponseError: If the embedding backend is unreachable.
        """
        top_k = top_k or self.settings.rerank_top_k
        fetch_k = max(self.settings.retrieval_top_k, top_k * 4)
        start = time.perf_counter()

        # --- dense ---
        dense_results: dict[str, float] = {}
        dense_meta: dict[str, dict] = {}
        dense_text: dict[str, str] = {}
        query_vec = self.embeddings.embed([query])[0]
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        res = self.embeddings.collection.query(
            query_embeddings=[query_vec],
            n_results=fetch_k,
            where=where,
        )
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        for i, cid in enumerate(ids):
            sim = 1.0 - float(distances[i])  # cosine distance -> similarity
            dense_results[cid] = sim
            dense_text[cid] = documents[i]
            dense_meta[cid] = metadatas[i] or {}

        # --- sparse ---
        sparse_results: dict[str, float] = {}
        index = self._ensure_bm25()
        if index is not None:
            scores = index.bm25.get_scores(_tokenize(query))
            order = np.argsort(scores)[::-1][:fetch_k]
            for pos in order:
                cid = index.ids[int(pos)]
                if document_ids and index.metadatas[int(pos)].get("document_id") not in document_ids:
                    continue
                sparse_results[cid] = float(scores[int(pos)])
                dense_text.setdefault(cid, index.texts[int(pos)])
                dense_meta.setdefault(cid, index.metadatas[int(pos)] or {})

        candidates = set(dense_results) | set(sparse_results)
        if not candidates:
            return []

        dense_arr = np.array([dense_results.get(c, 0.0) for c in candidates])
        sparse_arr = np.array([sparse_results.get(c, 0.0) for c in candidates])
        dense_n = _minmax(dense_arr)
        sparse_n = _minmax(sparse_arr)
        fused = (
            self.settings.dense_weight * dense_n
            + self.settings.sparse_weight * sparse_n
        )

        fused_ranked = sorted(
            zip(candidates, fused, dense_n, sparse_n),
            key=lambda x: x[1],
            reverse=True,
        )[: self.settings.retrieval_top_k]

        # --- re-rank via token overlap (proxy for cross-encoder) --- #
        q_tokens = set(_tokenize(query))
        reranked: list[tuple[str, float, float, float]] = []
        for cid, fused_score, dn, sn in fused_ranked:
            c_tokens = set(_tokenize(dense_text[cid]))
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            # 70% fused + 30% overlap boost
            final = 0.7 * fused_score + 0.3 * overlap
            reranked.append((cid, final, dn, sn))
        reranked.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievedChunk] = []
        for cid, score, dn, sn in reranked[:top_k]:
            meta = dense_meta.get(cid, {})
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=meta.get("document_id", "unknown"),
                    text=dense_text[cid],
                    score=float(score),
                    dense_score=float(dn),
                    sparse_score=float(sn),
                    metadata=meta,
                )
            )

        log.info(
            f"Hybrid retrieval '{query[:60]}...' "
            f"-> {len(results)} chunks in {time.perf_counter() - start:.2f}s"
        )
        return results


_pipeline: RetrievalPipeline | None = None


def get_retrieval_pipeline() -> RetrievalPipeline:
    """Return a process-wide :class:`RetrievalPipeline` singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline()
    return _pipeline
