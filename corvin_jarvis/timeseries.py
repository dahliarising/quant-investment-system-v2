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


def write_snapshot(db_path: Path, snapshot: dict[str, Any]) -> int:
    """Snapshot dict를 quote_history에 누적 저장.

    price가 None인 row는 skip (수집 실패). 반환: inserted row 수.
    """
    init_db(db_path)
    ts_utc = snapshot["timestamp_utc"]
    ts_kst = snapshot["timestamp_kst"]
    rows: list[tuple[Any, ...]] = []

    cat_map = {"indices": "index", "commodities": "commodity", "fx": "fx"}
    for category, cat_name in cat_map.items():
        for symbol, q in snapshot.get(category, {}).items():
            if q.get("price") is None:
                continue
            rows.append((
                ts_utc, ts_kst, cat_name, symbol,
                q.get("price"), q.get("pct_change"),
                None, None, None,
                q.get("source"), q.get("error"),
            ))

    for pos in snapshot.get("portfolio", []):
        if pos.get("current_price") is None:
            continue
        rows.append((
            ts_utc, ts_kst, "portfolio", pos["symbol"],
            pos.get("current_price"), None,
            pos.get("pnl_pct"), pos.get("market_value"), pos.get("shares"),
            None, pos.get("error"),
        ))

    for w in snapshot.get("watchlist", []):
        if w.get("price") is None:
            continue
        rows.append((
            ts_utc, ts_kst, "watchlist", w["symbol"],
            w.get("price"), w.get("pct_change"),
            None, None, None,
            w.get("source"), w.get("error"),
        ))

    if not rows:
        return 0

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO quote_history
               (ts_utc, ts_kst, category, symbol, price, pct_change,
                pnl_pct, market_value, shares, source, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    log.info("timeseries wrote %d rows for ts=%s", len(rows), ts_kst)
    return len(rows)
