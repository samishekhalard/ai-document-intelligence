"""Orchestrator routing tests — use monkeypatched agents to avoid LLM calls."""
from __future__ import annotations

from backend.agents.orchestrator_agent import OrchestratorAgent, _decide_route
from backend.core.models import NLPAnalysis, QueryResponse


def test_route_for_nlp_query():
    assert _decide_route("summarize this document", None) == "nlp"
    assert _decide_route("extract the entities", None) == "nlp"
    assert _decide_route("classify this document", None) == "nlp"


def test_route_for_vision_with_image():
    assert _decide_route("what is in this file?", "/tmp/x.png") == "vision"


def test_route_default_rag():
    assert _decide_route("what are the payment terms?", None) == "rag"


def test_orchestrator_routes_to_nlp(monkeypatch):
    orch = OrchestratorAgent()

    def fake_analyze(self, text, tasks):  # noqa: ANN001 - test double
        return NLPAnalysis(summary="mocked", language="en")

    def fake_answer(self, query, top_k=None, document_ids=None):  # noqa: ANN001
        return QueryResponse(
            query=query, answer="should not be called",
            sources=[], confidence=0.0, latency_ms=0.0,
        )

    monkeypatch.setattr(orch.nlp, "analyze", fake_analyze.__get__(orch.nlp))
    monkeypatch.setattr(orch.rag, "answer", fake_answer.__get__(orch.rag))

    result = orch.run("please summarize the contract")
    assert result["nlp"] is not None
    assert result["nlp"].summary == "mocked"
