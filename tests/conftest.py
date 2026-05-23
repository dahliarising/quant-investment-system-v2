"""Pytest fixtures for Corvin tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """tmp_path 안에 timeseries.db 경로 반환 (파일은 미생성)."""
    return tmp_path / "timeseries.db"
