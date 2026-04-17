"""Singleton sentence-transformers embedding client (CPU, local, free)."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.core.logging import get_logger

log = get_logger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the process-wide :class:`SentenceTransformer`, loading on first use."""
    global _model
    if _model is None:
        name = get_settings().embedding_model
        log.info(f"Loading embedding model {name}")
        _model = SentenceTransformer(name, device="cpu")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` and return a list of float vectors, order preserved."""
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


def embedding_dim() -> int:
    """Report the output vector dimensionality of the loaded model."""
    return int(get_embedding_model().get_sentence_embedding_dimension())
