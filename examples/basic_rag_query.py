"""Minimal RAG example.

Demonstrates the simplest path through the platform: embed the demo
corpus once, then ask the RAG agent a question.

Usage:
    python examples/basic_rag_query.py "What are the payment terms?"

Prerequisites:
    - Ollama running with llama3.2 and nomic-embed-text pulled.
    - Demo documents ingested (``python data/sample_docs/load_samples.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.rag_agent import RAGAgent  # noqa: E402


def main() -> int:
    """Run a single RAG query and pretty-print the result.

    Returns:
        Exit status — 0 on success, 1 if no sources were retrieved.
    """
    query = " ".join(sys.argv[1:]) or "What are the payment terms of the contract?"

    agent = RAGAgent()
    response = agent.answer(query)

    print("=" * 72)
    print(f"Q: {query}")
    print("-" * 72)
    print(f"A: {response.answer}")
    print("-" * 72)
    print(f"Confidence : {response.confidence:.2f}")
    print(f"Latency    : {response.latency_ms:.0f} ms")
    print(f"Trace      : {' -> '.join(response.agent_trace)}")
    print("Sources:")
    for i, src in enumerate(response.sources, start=1):
        title = src.metadata.get("title", "Untitled")
        print(
            f"  [{i}] {title} "
            f"(score={src.score:.3f}, dense={src.dense_score:.2f}, sparse={src.sparse_score:.2f})"
        )
        print(f"      {src.text[:160]}{'...' if len(src.text) > 160 else ''}")
    print("=" * 72)

    return 0 if response.sources else 1


if __name__ == "__main__":
    raise SystemExit(main())
