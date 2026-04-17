"""RAG agent: retrieves evidence, then asks the local LLM to synthesise the answer."""
from __future__ import annotations

import math
import time
from textwrap import dedent

from backend.core.config import get_settings
from backend.core.llm_client import chat
from backend.core.logging import get_logger
from backend.core.models import QueryResponse, RetrievedChunk
from backend.pipelines.retrieval_pipeline import get_retrieval_pipeline

log = get_logger(__name__)

_SYSTEM_PROMPT = dedent(
    """
    You are an enterprise document-analysis assistant. Answer the user's
    question ONLY from the supplied context. If the context is insufficient,
    say you do not have enough information. Be concise, factual, and cite
    which source snippets support each claim using bracketed numbers like
    [1], [2] that match the context list below.
    """
).strip()


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        title = c.metadata.get("title", "Untitled")
        lines.append(f"[{i}] ({title})\n{c.text.strip()}")
    return "\n\n".join(lines)


def _confidence(chunks: list[RetrievedChunk]) -> float:
    """Heuristic confidence: mean of top-k fused scores, squashed to [0,1]."""
    if not chunks:
        return 0.0
    scores = [c.score for c in chunks]
    mean = sum(scores) / len(scores)
    # Clamp to a reasonable reported range
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(mean - 0.1)))


class RAGAgent:
    """Retrieval-Augmented Generation agent.

    Combines hybrid retrieval (:class:`RetrievalPipeline`) with the local
    llama.cpp CPU runtime to synthesise grounded answers with inline
    citations.

    Attributes:
        settings: Cached application settings.
        retriever: Singleton hybrid retrieval pipeline.
    """

    def __init__(self) -> None:
        """Create a RAG agent bound to the singleton llama.cpp client."""
        self.settings = get_settings()
        self.retriever = get_retrieval_pipeline()

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> QueryResponse:
        """Retrieve supporting chunks and generate a grounded answer.

        Args:
            query: Natural-language question (1–2000 characters).
            top_k: Optional override for the final number of source
                chunks returned. Defaults to ``RERANK_TOP_K``.
            document_ids: Optional restriction to specific document IDs.
                When ``None`` the full corpus is searched.

        Returns:
            A :class:`QueryResponse` containing the generated answer,
            source chunks, heuristic confidence, and the agent trace.

        Raises:
            This method never raises — all downstream failures (empty
            index, unreachable LLM) are captured and surfaced in the
            response ``answer`` / ``agent_trace`` fields.
        """
        start = time.perf_counter()
        trace: list[str] = ["rag:retrieve"]
        chunks = self.retriever.retrieve(query, top_k=top_k, document_ids=document_ids)

        if not chunks:
            latency = (time.perf_counter() - start) * 1000
            return QueryResponse(
                query=query,
                answer="I could not find any relevant documents in the knowledge base.",
                sources=[],
                confidence=0.0,
                latency_ms=latency,
                agent_trace=trace + ["rag:no-context"],
            )

        context = _format_context(chunks)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer (cite sources inline like [1], [2]):"
        )

        trace.append("rag:generate")
        try:
            answer = chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.settings.llm_max_tokens,
                temperature=0.1,
            )
        except Exception as exc:
            log.error(f"LLM generation failed: {exc}")
            answer = (
                "I retrieved relevant passages but the local LLM is currently "
                "unavailable. Please retry."
            )
            trace.append("rag:llm-error")

        latency = (time.perf_counter() - start) * 1000
        return QueryResponse(
            query=query,
            answer=answer,
            sources=chunks,
            confidence=_confidence(chunks),
            latency_ms=latency,
            agent_trace=trace + ["rag:done"],
        )
