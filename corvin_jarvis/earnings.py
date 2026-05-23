"""Corvin Jarvis — Earnings Calendar (Tier 1.3)

yfinance.Ticker.calendar로 어닝 발표일 수집 → SQLite 저장 → D-7/D-3/D-1 alert.

기존 timeseries.db에 별도 테이블 earnings_calendar로 저장.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.earnings")

SCHEMA = """
CREATE TABLE IF NOT EXISTS earnings_calendar (
    symbol TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    eps_avg REAL,
    revenue_avg REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, earnings_date)
);
CREATE INDEX IF NOT EXISTS idx_ec_symbol_date ON earnings_calendar (symbol, earnings_date);
"""

ALERT_OFFSETS = (7, 3, 1)


def init_earnings_table(db_path: Path) -> None:
    """earnings_calendar 테이블 생성 (idempotent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
    log.info("earnings_calendar table ready at %s", db_path)
