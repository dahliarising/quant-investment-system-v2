"""Tests for corvin_jarvis.timeseries module."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corvin_jarvis import timeseries


@pytest.mark.unit
def test_init_db_creates_quote_history_table(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='quote_history'"
        )
        assert cur.fetchone() is not None


@pytest.mark.unit
def test_init_db_creates_indexes(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='quote_history'"
        )
        names = {row[0] for row in cur.fetchall()}
    assert "idx_qh_symbol_ts" in names
    assert "idx_qh_category_ts" in names


@pytest.mark.unit
def test_init_db_is_idempotent(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    timeseries.init_db(tmp_db_path)
    assert tmp_db_path.exists()
