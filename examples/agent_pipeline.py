"""Direct multi-agent orchestration example.

Bypasses the HTTP layer and invokes the LangGraph orchestrator directly,
then prints the full agent state — useful for debugging routing
decisions and fallbacks.

Usage:
    python examples/agent_pipeline.py "Summarise the technical spec"
    python examples/agent_pipeline.py "extract entities from this text"
    python examples/agent_pipeline.py "what is in this scan?" --image path.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.orchestrator_agent import get_orchestrator  # noqa: E402


def _serialise(obj: Any) -> Any:
    """Recursively convert Pydantic models / dataclasses to JSON-safe dicts."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_serialise(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent pipeline directly.")
    parser.add_argument("query", help="Natural-language input")
    parser.add_argument("--image", type=str, default=None,
                        help="Optional image path to route through the vision agent")
    args = parser.parse_args()

    orch = get_orchestrator()
    result = orch.run(args.query, image_path=args.image)

    payload = {
        "query": args.query,
        "image": args.image,
        "trace": result["trace"],
        "iterations": result["iterations"],
        "error": result["error"],
        "rag": _serialise(result["rag"]),
        "nlp": _serialise(result["nlp"]),
        "vision": _serialise(result["vision"]),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
