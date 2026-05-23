"""Corvin Jarvis — Narrative DB Adapter (Tier 1.4)

narrative-shift-detector의 signals.db readonly 어댑터.

스키마 (signals 테이블):
    date, session, vix, foreign_net_buy, sentiment_tone, market

KR 시장 레벨만 제공 (종목별 narrative는 미지원). |Z| > 2σ alert 생성.

⚠️ readonly only — 절대 write 금지.
"""
from __future__ import annotations

import logging
import sqlite3
import statistics
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.narrative")

DEFAULT_SIGNALS_DB = Path("/Users/thethethe/Claude/narrative-shift-detector/data/signals.db")


def _readonly_connect(db_path: Path) -> sqlite3.Connection | None:
    """signals.db readonly 모드로 connect. URI scheme 사용."""
    if not db_path.exists():
        return None
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def latest_signal(db_path: Path, market: str = "KR") -> dict[str, Any] | None:
    """가장 최근 (date DESC, session DESC) 행 반환. 없으면 None."""
    conn = _readonly_connect(db_path)
    if conn is None:
        return None
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT date, session, vix, foreign_net_buy, sentiment_tone, market "
            "FROM signals WHERE market = ? "
            "ORDER BY date DESC, session DESC LIMIT 1",
            (market,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
