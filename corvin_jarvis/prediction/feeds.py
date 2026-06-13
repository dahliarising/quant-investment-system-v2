# corvin_jarvis/prediction/feeds.py
"""외부 신호 → 예측 모듈 payload 배선 (헤드리스 cron용, MCP 없이).

narrative-shift-detector signals.db의 GDELT sentiment_tone을 직접 조회해
m_sentiment payload로 변환. geo risk_score는 signals.db에 없어 미배선(정직).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_SIGNALS_DB = Path.home() / "Claude" / "narrative-shift-detector" / "data" / "signals.db"


def fetch_sentiment(db_path: Path = _SIGNALS_DB) -> dict[str, Any] | None:
    """signals.db 최신 sentiment_tone(GDELT TimelineTone) → m_sentiment payload.

    trend = 최신 tone vs 최근 평균 (개선/악화/보합). 없거나 비면 None → 모듈은 보류.
    """
    if not Path(db_path).exists():
        return None
    try:
        with sqlite3.connect(db_path) as c:
            rows = c.execute(
                "SELECT date, sentiment_tone FROM signals "
                "ORDER BY date DESC, rowid DESC LIMIT 10").fetchall()
    except sqlite3.Error:
        return None
    if not rows:
        return None
    latest = float(rows[0][1])
    recent = [float(r[1]) for r in rows]
    avg = sum(recent) / len(recent)
    trend = "개선" if latest > avg else ("악화" if latest < avg else "보합")
    return {"tone": latest, "article_volume": None, "trend": trend}
