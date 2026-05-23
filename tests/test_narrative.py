"""Tests for corvin_jarvis.narrative readonly adapter."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corvin_jarvis import narrative

SCHEMA = """
CREATE TABLE signals (
    date TEXT NOT NULL,
    session TEXT NOT NULL,
    vix REAL NOT NULL,
    foreign_net_buy INTEGER NOT NULL,
    sentiment_tone REAL NOT NULL,
    market TEXT NOT NULL DEFAULT 'KR',
    PRIMARY KEY (date, session, market)
);
"""


@pytest.fixture
def fake_signals_db(tmp_path: Path) -> Path:
    db = tmp_path / "signals.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(SCHEMA)
        rows = [
            ("2026-05-19", "AM", 22.0, 100000, -0.5, "KR"),
            ("2026-05-19", "PM", 23.0, -50000, -0.7, "KR"),
            ("2026-05-20", "AM", 22.5, 80000, -0.3, "KR"),
            ("2026-05-20", "PM", 24.0, -120000, -0.9, "KR"),
            ("2026-05-21", "AM", 25.0, 200000, -1.2, "KR"),
        ]
        conn.executemany(
            "INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?)", rows
        )
    return db


@pytest.mark.unit
def test_latest_signal_returns_most_recent_row(fake_signals_db: Path) -> None:
    row = narrative.latest_signal(fake_signals_db, market="KR")
    assert row is not None
    assert row["date"] == "2026-05-21"
    assert row["session"] == "AM"
    assert row["vix"] == 25.0


@pytest.mark.unit
def test_latest_signal_returns_none_for_missing_market(fake_signals_db: Path) -> None:
    row = narrative.latest_signal(fake_signals_db, market="US")
    assert row is None


@pytest.mark.unit
def test_latest_signal_returns_none_for_missing_db(tmp_path: Path) -> None:
    row = narrative.latest_signal(tmp_path / "nope.db", market="KR")
    assert row is None
