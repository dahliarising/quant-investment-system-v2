"""Corvin Jarvis — Time-Series DB (Phase 5)

매 pulse마다 quote를 SQLite에 누적 저장. predictive/what-if/attribution 기능의 토대.

스키마: 단일 정규화 테이블 quote_history. category 컬럼으로 index/commodity/fx/portfolio/watchlist 분리.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.timeseries")

SCHEMA = """
CREATE TABLE IF NOT EXISTS quote_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    ts_kst TEXT NOT NULL,
    category TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price REAL,
    pct_change REAL,
    pnl_pct REAL,
    market_value REAL,
    shares REAL,
    source TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_qh_symbol_ts ON quote_history (symbol, ts_utc);
CREATE INDEX IF NOT EXISTS idx_qh_category_ts ON quote_history (category, ts_utc);
"""


def init_db(db_path: Path) -> None:
    """DB 파일 + schema 생성 (idempotent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
    log.info("timeseries DB initialized at %s", db_path)
