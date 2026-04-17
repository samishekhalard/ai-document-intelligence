"""Tests for chunking + hybrid retrieval wiring (no LLM calls)."""
from __future__ import annotations

from backend.core.models import Chunk
from backend.pipelines.ingestion_pipeline import chunk_text


SAMPLE = (
    "Helios Analytics provides document intelligence. "
    "It uses retrieval augmented generation. " * 30
)


def test_chunk_text_produces_multiple_chunks():
    chunks = chunk_text(SAMPLE, document_id="doc-test")
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.document_id == "doc-test" for c in chunks)
    # All chunks respect (approximately) the configured size.
    assert all(len(c.text) <= 600 for c in chunks)


def test_chunks_have_stable_ids():
    chunks_a = chunk_text(SAMPLE, document_id="doc-a")
    chunks_b = chunk_text(SAMPLE, document_id="doc-a")
    # Chunk prefixes up to the uuid suffix should match across runs.
    prefixes_a = [c.chunk_id.rsplit(":", 1)[0] for c in chunks_a]
    prefixes_b = [c.chunk_id.rsplit(":", 1)[0] for c in chunks_b]
    assert prefixes_a == prefixes_b
