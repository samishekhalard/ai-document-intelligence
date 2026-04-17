"""Vision agent: OCR + layout detection for images & scanned PDFs."""
from __future__ import annotations

import time
from pathlib import Path

from backend.core.logging import get_logger
from backend.core.models import OCRResponse
from backend.pipelines.ocr_pipeline import detect_layout, ocr_bytes, ocr_image

log = get_logger(__name__)


class VisionAgent:
    """Thin wrapper around :mod:`backend.pipelines.ocr_pipeline`."""

    def process_path(self, path: Path, language: str = "eng+rus") -> OCRResponse:
        start = time.perf_counter()
        text = ocr_image(path, language=language)
        return OCRResponse(
            text=text,
            n_words=len(text.split()),
            language=language,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def process_bytes(self, data: bytes, language: str = "eng+rus") -> OCRResponse:
        start = time.perf_counter()
        text = ocr_bytes(data, language=language)
        return OCRResponse(
            text=text,
            n_words=len(text.split()),
            language=language,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def layout(self, path: Path) -> dict[str, int]:
        return detect_layout(path)
