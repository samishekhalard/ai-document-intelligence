"""Evaluate the RAG system on a fixed set of Q&A pairs.

Metrics
-------
* **answer_accuracy** — fraction of answers that contain every expected
  keyword (case-insensitive). This is a simple, deterministic proxy for
  correctness; it rewards grounded answers without requiring a judge LLM.
* **precision_at_5** — fraction of queries whose top-5 retrieved chunks
  include at least one chunk whose metadata.title matches ``expected_doc``.
* **avg_latency_s** — mean wall-clock latency of end-to-end ``/query``
  calls (retrieval + generation).

Results are written to ``eval/results.json``.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.rag_agent import RAGAgent  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402

log = get_logger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


# Minimum fixture: 20 Q/A pairs spread across the three sample documents.
QA_PAIRS: list[dict] = [
    # --- Legal contract ---
    {
        "query": "Who are the parties to the Master Services Agreement?",
        "expected_keywords": ["Helios Analytics", "Northwind"],
        "expected_doc": "Legal Contract En",
    },
    {
        "query": "When is the Effective Date of the agreement?",
        "expected_keywords": ["March 14, 2026"],
        "expected_doc": "Legal Contract En",
    },
    {
        "query": "What is the initial term of the contract?",
        "expected_keywords": ["24", "months"],
        "expected_doc": "Legal Contract En",
    },
    {
        "query": "How much are the annual fees under the initial SOW?",
        "expected_keywords": ["480,000"],
        "expected_doc": "Legal Contract En",
    },
    {
        "query": "What are the payment terms for invoices?",
        "expected_keywords": ["net", "thirty"],
        "expected_doc": "Legal Contract En",
    },
    {
        "query": "Which law governs the agreement?",
        "expected_keywords": ["New York"],
        "expected_doc": "Legal Contract En",
    },
    {
        "query": "What compliance standards must the Provider follow?",
        "expected_keywords": ["SOC 2", "ISO 27001"],
        "expected_doc": "Legal Contract En",
    },
    # --- Russian financial report ---
    {
        "query": "Какова была общая выручка компании за 2025 год?",
        "expected_keywords": ["1 284 500 000", "выручка"],
        "expected_doc": "Financial Report Ru",
    },
    {
        "query": "Какая чистая прибыль была получена в 2025 году?",
        "expected_keywords": ["162 300 000"],
        "expected_doc": "Financial Report Ru",
    },
    {
        "query": "Кто является финансовым директором?",
        "expected_keywords": ["Игорь", "Соколов"],
        "expected_doc": "Financial Report Ru",
    },
    {
        "query": "Сколько составили капитальные затраты в 2025 году?",
        "expected_keywords": ["198 600 000"],
        "expected_doc": "Financial Report Ru",
    },
    {
        "query": "Какая выручка в сегменте международных перевозок?",
        "expected_keywords": ["612 400 000"],
        "expected_doc": "Financial Report Ru",
    },
    {
        "query": "Какой прогноз роста выручки на 2026 год?",
        "expected_keywords": ["12", "15"],
        "expected_doc": "Financial Report Ru",
    },
    # --- Technical spec ---
    {
        "query": "What is the p95 latency target for the Atlas service?",
        "expected_keywords": ["350"],
        "expected_doc": "Technical Spec En",
    },
    {
        "query": "Which embedding model does Atlas use?",
        "expected_keywords": ["nomic-embed-text"],
        "expected_doc": "Technical Spec En",
    },
    {
        "query": "Which vector database stores embeddings?",
        "expected_keywords": ["ChromaDB"],
        "expected_doc": "Technical Spec En",
    },
    {
        "query": "What is the chunk size used during ingestion?",
        "expected_keywords": ["512"],
        "expected_doc": "Technical Spec En",
    },
    {
        "query": "What is the availability SLO for Atlas?",
        "expected_keywords": ["99.9"],
        "expected_doc": "Technical Spec En",
    },
    {
        "query": "What encryption is used at rest?",
        "expected_keywords": ["AES-256"],
        "expected_doc": "Technical Spec En",
    },
    {
        "query": "Which issues are open for the v1.3 release?",
        "expected_keywords": ["ATLAS-412"],
        "expected_doc": "Technical Spec En",
    },
]


def _accuracy(answer: str, keywords: list[str]) -> bool:
    low = answer.lower()
    return all(k.lower() in low for k in keywords)


def _precision_at_k(sources: list, expected_doc: str, k: int = 5) -> float:
    hits = 0
    for s in sources[:k]:
        title = (s.metadata or {}).get("title", "")
        if expected_doc.lower() in title.lower():
            hits = 1
            break
    return float(hits)


def run() -> dict:
    agent = RAGAgent()
    accuracies: list[float] = []
    precisions: list[float] = []
    latencies: list[float] = []
    per_case: list[dict] = []

    for case in QA_PAIRS:
        t0 = time.perf_counter()
        resp = agent.answer(case["query"])
        latency = time.perf_counter() - t0

        acc = 1.0 if _accuracy(resp.answer, case["expected_keywords"]) else 0.0
        prec = _precision_at_k(resp.sources, case["expected_doc"], k=5)

        accuracies.append(acc)
        precisions.append(prec)
        latencies.append(latency)
        per_case.append(
            {
                "query": case["query"],
                "answer": resp.answer[:400],
                "expected_keywords": case["expected_keywords"],
                "expected_doc": case["expected_doc"],
                "accuracy": acc,
                "precision_at_5": prec,
                "latency_s": round(latency, 3),
                "confidence": resp.confidence,
                "retrieved_titles": [
                    (s.metadata or {}).get("title", "Untitled") for s in resp.sources
                ],
            }
        )
        log.info(
            f"[{'OK' if acc == 1.0 else 'MISS'}] {case['query'][:60]} "
            f"-> acc={acc:.0f} p@5={prec:.0f} lat={latency:.2f}s"
        )

    summary = {
        "n_cases": len(QA_PAIRS),
        "answer_accuracy": round(statistics.fmean(accuracies), 4),
        "retrieval_precision_at_5": round(statistics.fmean(precisions), 4),
        "avg_latency_s": round(statistics.fmean(latencies), 3),
        "p95_latency_s": round(
            statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else latencies[0], 3
        ),
    }
    out = {"summary": summary, "cases": per_case}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(f"Wrote {RESULTS_PATH}")
    return summary


if __name__ == "__main__":
    summary = run()
    print(json.dumps(summary, indent=2))
