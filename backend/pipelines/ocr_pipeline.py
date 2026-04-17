"""OCR primitives used by the vision agent and the ingestion pipeline."""
from __future__ import annotations

import time
from pathlib import Path

from backend.core.logging import get_logger

log = get_logger(__name__)


def _preprocess(image):
    """Convert image to grayscale and threshold for better OCR accuracy."""
    from PIL import Image, ImageFilter, ImageOps

    if image.mode != "L":
        image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def ocr_image(path: Path, language: str = "eng+rus") -> str:
    """Run tesseract OCR on an image file.

    Args:
        path: Filesystem path to a PNG / JPG / TIFF file.
        language: Tesseract language string. Combine multiple with
            ``+``, e.g. ``"eng+rus"``. Falls back to ``"eng"`` if a
            requested language pack is missing.

    Returns:
        The extracted text, stripped of leading/trailing whitespace.
    """
    import pytesseract
    from PIL import Image

    start = time.perf_counter()
    with Image.open(path) as img:
        processed = _preprocess(img)
        try:
            text = pytesseract.image_to_string(processed, lang=language)
        except pytesseract.TesseractError:
            # fall back to english-only if a language pack is missing
            text = pytesseract.image_to_string(processed, lang="eng")
    log.debug(f"OCR {path.name} took {time.perf_counter() - start:.2f}s")
    return text.strip()


def ocr_bytes(data: bytes, language: str = "eng+rus") -> str:
    """Run tesseract OCR on in-memory image bytes.

    Args:
        data: Raw image bytes (any format supported by Pillow).
        language: Tesseract language string.

    Returns:
        The extracted text, stripped.
    """
    import io

    import pytesseract
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        processed = _preprocess(img)
        try:
            return pytesseract.image_to_string(processed, lang=language).strip()
        except pytesseract.TesseractError:
            return pytesseract.image_to_string(processed, lang="eng").strip()


def ocr_pdf(path: Path, language: str = "eng+rus") -> str:
    """Convert each PDF page to an image and OCR it.

    Uses ``pdf2image`` which delegates to ``poppler``.
    Returns an empty string if poppler is not installed.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:  # pragma: no cover
        log.warning("pdf2image not available, skipping OCR fallback")
        return ""
    import pytesseract

    try:
        pages = convert_from_path(str(path), dpi=200)
    except Exception as exc:
        log.warning(f"pdf2image failed for {path.name}: {exc}")
        return ""

    parts = []
    for i, img in enumerate(pages):
        processed = _preprocess(img)
        try:
            parts.append(pytesseract.image_to_string(processed, lang=language))
        except pytesseract.TesseractError:
            parts.append(pytesseract.image_to_string(processed, lang="eng"))
    return "\n\n".join(parts).strip()


def detect_layout(path: Path) -> dict[str, int]:
    """Very lightweight layout descriptor: counts of tables, headers, paragraphs.

    Based on tesseract block-level output.
    """
    import pytesseract
    from PIL import Image

    with Image.open(path) as img:
        data = pytesseract.image_to_data(
            _preprocess(img), output_type=pytesseract.Output.DICT
        )

    blocks = set()
    for i, text in enumerate(data.get("text", [])):
        if text.strip():
            blocks.add(data["block_num"][i])

    lines = {(data["block_num"][i], data["line_num"][i])
             for i, t in enumerate(data.get("text", [])) if t.strip()}

    # Simple heuristic: block containing >3 tab-separated tokens -> table-like.
    table_blocks = 0
    for b in blocks:
        tokens = [data["text"][i] for i, bb in enumerate(data["block_num"])
                  if bb == b and data["text"][i].strip()]
        if len(tokens) > 20 and any("\t" in t for t in tokens):
            table_blocks += 1

    return {
        "blocks": len(blocks),
        "lines": len(lines),
        "tables": table_blocks,
        "paragraphs": max(0, len(blocks) - table_blocks),
    }
