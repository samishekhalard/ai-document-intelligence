"""Batch-ingest an entire directory of documents.

Walks the given path, filters by supported extensions, and ingests
each file through the :class:`~backend.api.ingestion_service.IngestionService`.
Reports per-file status and overall throughput at the end.

Usage:
    python examples/batch_ingestion.py path/to/corpus
    python examples/batch_ingestion.py path/to/corpus --workers 4

The ``--workers`` flag enables thread-pool parallelism. Ollama itself
serialises embedding requests, so values above 2–4 give limited benefit;
the main win is overlapping parsing with embedding.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.ingestion_service import get_ingestion_service  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402

log = get_logger(__name__)

SUPPORTED_EXT = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def discover(path: Path) -> list[Path]:
    """Return the list of ingestible files under ``path``.

    Args:
        path: Root directory or a single file.

    Returns:
        Sorted list of files whose extension is in ``SUPPORTED_EXT``.
    """
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXT else []
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    return sorted(files)


def _ingest_one(path: Path) -> tuple[Path, str]:
    """Ingest a single file and return (path, outcome) for reporting."""
    service = get_ingestion_service()
    try:
        res = service.ingest_path(path)
        return path, f"OK {res.n_chunks} chunks [{res.document_id}]"
    except Exception as exc:
        return path, f"FAIL {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="Directory or single file to ingest")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    args = parser.parse_args()

    files = discover(args.path)
    if not files:
        log.error(f"No ingestible files found under {args.path}")
        return 1

    log.info(f"Ingesting {len(files)} files with {args.workers} worker(s)")
    start = time.perf_counter()
    successes = 0

    if args.workers == 1:
        for f in files:
            _, outcome = _ingest_one(f)
            print(f"{f.name:60s}  {outcome}")
            successes += int(outcome.startswith("OK"))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_ingest_one, f): f for f in files}
            for fut in as_completed(futures):
                f, outcome = fut.result()
                print(f"{f.name:60s}  {outcome}")
                successes += int(outcome.startswith("OK"))

    elapsed = time.perf_counter() - start
    rate = successes / elapsed if elapsed else 0.0
    print(
        f"\n{successes}/{len(files)} ingested in {elapsed:.1f}s "
        f"({rate:.2f} docs/s)"
    )
    return 0 if successes == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
