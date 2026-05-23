"""Tests for corvin_jarvis.earnings module."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from corvin_jarvis import earnings


@pytest.mark.unit
def test_init_earnings_table_creates_table(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='earnings_calendar'"
        )
        assert cur.fetchone() is not None


@pytest.mark.unit
def test_init_earnings_table_creates_index(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='earnings_calendar'"
        )
        names = {row[0] for row in cur.fetchall()}
    assert "idx_ec_symbol_date" in names


@pytest.mark.unit
def test_init_earnings_table_idempotent(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    earnings.init_earnings_table(tmp_db_path)
    assert tmp_db_path.exists()
