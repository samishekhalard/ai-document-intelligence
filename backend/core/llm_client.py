"""Singleton CPU-only llama.cpp client.

Loads a local GGUF model once per process. ``n_gpu_layers=0`` keeps the
entire model on the CPU, bypassing Metal so we avoid the M5 shader bug
in the upstream ggml-metal backend.
"""
from __future__ import annotations

from llama_cpp import Llama

from backend.core.config import get_settings
from backend.core.logging import get_logger

log = get_logger(__name__)

_llm: Llama | None = None


def get_llm() -> Llama:
    """Return the process-wide :class:`Llama` instance, loading on first use."""
    global _llm
    if _llm is None:
        settings = get_settings()
        model_path = str(settings.model_path)
        log.info(f"Loading GGUF model {model_path} (CPU-only)")
        _llm = Llama(
            model_path=model_path,
            n_ctx=settings.llm_n_ctx,
            n_threads=settings.llm_n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
    return _llm


def generate(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float = 0.1,
    stop: list[str] | None = None,
) -> str:
    """Generate a completion for ``prompt`` and return the stripped text."""
    settings = get_settings()
    llm = get_llm()
    response = llm(
        prompt,
        max_tokens=max_tokens or settings.llm_max_tokens,
        temperature=temperature,
        stop=stop or ["<|eot_id|>"],
    )
    return response["choices"][0]["text"].strip()


def chat(
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    temperature: float = 0.1,
) -> str:
    """Chat-style helper that flattens ``messages`` into a Llama 3.2 prompt."""
    parts: list[str] = ["<|begin_of_text|>"]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return generate(
        "".join(parts),
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|eot_id|>"],
    )
