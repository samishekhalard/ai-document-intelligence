"""Shared pytest fixtures.

All tests run against a temporary data directory so they do not touch
real ingested documents.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _isolate_data_dir():
    """Redirect data/* and the SQLite file to a temp dir for the test session."""
    tmp = Path(tempfile.mkdtemp(prefix="docintel-test-"))
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["UPLOAD_DIR"] = str(tmp / "uploads")
    os.environ["CHROMA_DIR"] = str(tmp / "chroma")
    os.environ["SQLITE_PATH"] = str(tmp / "metadata.db")
    # Force re-evaluation of the cached Settings singleton.
    from backend.core import config as cfg

    cfg.get_settings.cache_clear()
    cfg.get_settings()
    yield
    shutil.rmtree(tmp, ignore_errors=True)
