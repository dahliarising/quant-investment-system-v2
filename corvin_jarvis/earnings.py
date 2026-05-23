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


def upsert_earnings(
    db_path: Path,
    symbol: str,
    dates: list[date],
    eps_avg: float | None = None,
    revenue_avg: float | None = None,
) -> int:
    """주어진 (symbol, date) 쌍 upsert. 동일 PK는 UPDATE. 반환: row 수."""
    init_earnings_table(db_path)
    if not dates:
        return 0
    now_iso = datetime.now().isoformat(timespec="seconds")
    rows = [
        (symbol, d.isoformat(), eps_avg, revenue_avg, now_iso)
        for d in dates
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO earnings_calendar
                 (symbol, earnings_date, eps_avg, revenue_avg, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(symbol, earnings_date) DO UPDATE SET
                 eps_avg = excluded.eps_avg,
                 revenue_avg = excluded.revenue_avg,
                 updated_at = excluded.updated_at""",
            rows,
        )
    return len(rows)


def pending_earnings(
    db_path: Path,
    today: date,
    days_ahead: int = 14,
) -> list[dict[str, Any]]:
    """today 이상 ~ today+days_ahead 이내의 어닝 row 반환 (오름차순)."""
    if not db_path.exists():
        return []
    end = today + timedelta(days=days_ahead)
    sql = (
        "SELECT symbol, earnings_date, eps_avg, revenue_avg "
        "FROM earnings_calendar "
        "WHERE earnings_date >= ? AND earnings_date <= ? "
        "ORDER BY earnings_date ASC"
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, (today.isoformat(), end.isoformat())).fetchall()
    return [dict(r) for r in rows]
