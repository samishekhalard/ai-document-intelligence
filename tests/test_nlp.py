"""Unit tests for the NLP agent — do not hit the LLM."""
from __future__ import annotations

from backend.agents.nlp_agent import NLPAgent, _keyphrases
from backend.core.models import DocumentType


LEGAL_SNIPPET = (
    "This Master Services Agreement is entered into by and between the parties "
    "hereby. WHEREAS each party desires to set forth the terms of the agreement, "
    "each clause is governed by the jurisdiction of New York."
)

FINANCIAL_RU = (
    "Общая выручка за 2025 год составила 1 284 500 000 рублей. "
    "Чистая прибыль достигла 162 300 000 рублей. EBITDA 247 миллионов."
)


def test_language_detection_english():
    assert NLPAgent().detect_language(LEGAL_SNIPPET) == "en"


def test_language_detection_russian():
    assert NLPAgent().detect_language(FINANCIAL_RU) == "ru"


def test_classification_legal():
    assert NLPAgent().classify(LEGAL_SNIPPET) == DocumentType.LEGAL


def test_classification_financial_ru():
    assert NLPAgent().classify(FINANCIAL_RU) == DocumentType.FINANCIAL


def test_keyphrases_returns_non_empty():
    out = _keyphrases(LEGAL_SNIPPET, top_n=5)
    assert len(out) > 0
    # None of the returned key phrases should be a stopword.
    assert all(p not in {"the", "a", "an"} for p in out)
