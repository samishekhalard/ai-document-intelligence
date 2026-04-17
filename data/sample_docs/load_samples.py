"""Ingest every file in ``data/sample_docs/`` into the platform.

Run as: ``python -m data.sample_docs.load_samples``.
Safe to re-run — ingestion uses deterministic IDs and upserts.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when running as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.ingestion_service import get_ingestion_service  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402

log = get_logger(__name__)


SAMPLE_DIR = Path(__file__).resolve().parent


def main() -> int:
    service = get_ingestion_service()
    files = [p for p in sorted(SAMPLE_DIR.iterdir())
             if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}]
    if not files:
        log.warning("No sample files found")
        return 1

    failures = 0
    for path in files:
        try:
            res = service.ingest_path(path)
            log.info(f"Ingested {path.name}: {res.n_chunks} chunks (id={res.document_id})")
        except Exception as exc:
            log.error(f"Failed to ingest {path.name}: {exc}")
            failures += 1
    log.info(f"Sample ingestion finished: {len(files) - failures}/{len(files)} successful")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
