"""LangGraph-based orchestrator that routes queries to specialised agents.

The graph is deliberately small:

    router ──▶ rag ───▶ finalise
          │
          └──▶ nlp ───▶ finalise
          │
          └──▶ vision ─▶ finalise

Each node mutates a shared :class:`OrchestratorState`. The router inspects
the user query, picks an initial agent, and may request a follow-up (up to
``agent_max_iterations``) if the primary agent failed. This gives us
fallback behaviour without an unbounded loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from backend.agents.nlp_agent import NLPAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.vision_agent import VisionAgent
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.models import NLPAnalysis, OCRResponse, QueryResponse

log = get_logger(__name__)

Route = Literal["rag", "nlp", "vision", "finalise"]


@dataclass
class OrchestratorState:
    """Shared state that flows through the LangGraph nodes."""

    query: str
    image_path: str | None = None
    route: Route = "rag"
    rag_response: QueryResponse | None = None
    nlp_response: NLPAnalysis | None = None
    vision_response: OCRResponse | None = None
    trace: list[str] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None


_NLP_PATTERNS = [
    r"\b(entit(ies|y)|ner|named entities)\b",
    r"\b(summari[sz]e|summary|tl;dr)\b",
    r"\b(classif(y|ication)|what kind of document)\b",
    r"\b(key[- ]?phrases|keywords)\b",
    r"\b(language|какой язык)\b",
]


def _decide_route(query: str, image_path: str | None) -> Route:
    if image_path:
        return "vision"
    q = query.lower()
    for pat in _NLP_PATTERNS:
        if re.search(pat, q):
            return "nlp"
    return "rag"


class OrchestratorAgent:
    """Public entry point that builds and runs the LangGraph."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.rag = RAGAgent()
        self.nlp = NLPAgent()
        self.vision = VisionAgent()
        self._graph = self._build_graph()

    # --------- graph nodes --------- #

    def _router(self, state: OrchestratorState) -> OrchestratorState:
        state.iterations += 1
        state.route = _decide_route(state.query, state.image_path)
        state.trace.append(f"router->{state.route}")
        return state

    def _run_rag(self, state: OrchestratorState) -> OrchestratorState:
        try:
            state.rag_response = self.rag.answer(state.query)
            state.trace.append("rag:ok")
            if (
                state.rag_response.confidence < 0.15
                and state.iterations < self.settings.agent_max_iterations
            ):
                # Low confidence -> fall back to NLP for extractive info
                state.route = "nlp"
                state.trace.append("rag:low-conf->nlp")
                return state
        except Exception as exc:
            log.error(f"RAG agent failed: {exc}")
            state.error = f"rag: {exc}"
            state.trace.append("rag:error")
            if state.iterations < self.settings.agent_max_iterations:
                state.route = "nlp"
                return state
        state.route = "finalise"
        return state

    def _run_nlp(self, state: OrchestratorState) -> OrchestratorState:
        try:
            state.nlp_response = self.nlp.analyze(
                state.query,
                tasks=["entities", "classify", "language", "keyphrases"],
            )
            state.trace.append("nlp:ok")
        except Exception as exc:
            log.error(f"NLP agent failed: {exc}")
            state.error = (state.error or "") + f" | nlp: {exc}"
            state.trace.append("nlp:error")
        state.route = "finalise"
        return state

    def _run_vision(self, state: OrchestratorState) -> OrchestratorState:
        if not state.image_path:
            state.trace.append("vision:no-path")
            state.route = "rag"
            return state
        try:
            state.vision_response = self.vision.process_path(Path(state.image_path))
            state.trace.append("vision:ok")
            # Feed the OCR text back into RAG for richer answers.
            if state.vision_response.text and state.iterations < self.settings.agent_max_iterations:
                state.query = f"{state.query}\n\nDocument text: {state.vision_response.text[:2000]}"
                state.route = "rag"
                state.image_path = None
                return state
        except Exception as exc:
            log.error(f"Vision agent failed: {exc}")
            state.error = f"vision: {exc}"
            state.trace.append("vision:error")
        state.route = "finalise"
        return state

    def _finalise(self, state: OrchestratorState) -> OrchestratorState:
        state.trace.append("finalise")
        return state

    # --------- graph assembly --------- #

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("router", self._router)
        graph.add_node("rag", self._run_rag)
        graph.add_node("nlp", self._run_nlp)
        graph.add_node("vision", self._run_vision)
        graph.add_node("finalise", self._finalise)

        graph.set_entry_point("router")
        graph.add_conditional_edges("router", lambda s: s.route,
                                    {"rag": "rag", "nlp": "nlp", "vision": "vision"})
        graph.add_conditional_edges("rag", lambda s: s.route,
                                    {"nlp": "nlp", "finalise": "finalise"})
        graph.add_conditional_edges("nlp", lambda s: s.route,
                                    {"finalise": "finalise"})
        graph.add_conditional_edges("vision", lambda s: s.route,
                                    {"rag": "rag", "finalise": "finalise"})
        graph.add_edge("finalise", END)
        return graph.compile()

    # --------- public entry points --------- #

    def run(self, query: str, image_path: str | None = None) -> dict[str, Any]:
        """Execute the LangGraph agent pipeline for a single request.

        The router decides which agent runs first. Conditional edges
        allow fallback: e.g. RAG → NLP when retrieval confidence is
        low, or Vision → RAG after OCR extracts text from an image.

        Args:
            query: Natural-language input from the user.
            image_path: Optional path to an image to route through the
                vision agent first. When ``None`` the query is routed
                to the NLP or RAG agent based on keyword heuristics.

        Returns:
            A dict with the following keys, any of which may be ``None``
            if that agent did not run:
                - ``rag`` (:class:`QueryResponse` | None)
                - ``nlp`` (:class:`NLPAnalysis` | None)
                - ``vision`` (:class:`OCRResponse` | None)
                - ``trace`` (``list[str]``) — ordered list of node visits.
                - ``iterations`` (``int``) — number of router passes.
                - ``error`` (``str`` | None) — concatenated failure reasons.
        """
        state = OrchestratorState(query=query, image_path=image_path)
        # LangGraph returns a dict, not the dataclass we passed in.
        final_state: Any = self._graph.invoke(state)

        def _get(attr: str, default=None):
            if isinstance(final_state, dict):
                return final_state.get(attr, default)
            return getattr(final_state, attr, default)

        return {
            "rag": _get("rag_response"),
            "nlp": _get("nlp_response"),
            "vision": _get("vision_response"),
            "trace": _get("trace", []),
            "iterations": _get("iterations", 0),
            "error": _get("error"),
        }


_orchestrator: OrchestratorAgent | None = None


def get_orchestrator() -> OrchestratorAgent:
    """Return a process-wide :class:`OrchestratorAgent` singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator
