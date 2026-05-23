"""Corvin Jarvis — Q&A Retrieval Toolkit (Tier 1.5)

Discord 질문 응답을 위한 RAG-style retrieval 함수들.

- recent_history_summary: timeseries.db의 종목 N일 통계
- relative_strength: symbol vs benchmark 상대 성과
- wiki_search: corvin-sessions/*.md 텍스트 검색
- explain_move: 위 셋 + earnings + narrative 종합 dict

LLM(Corvin 본인)이 답변 합성 — 이 모듈은 raw retrieval만 담당.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from corvin_jarvis import timeseries

log = logging.getLogger("corvin.qa")

DEFAULT_WIKI_DIR = Path("/Users/thethethe/Claude/llm-wiki/wiki/corvin-sessions")


def recent_history_summary(
    db_path: Path,
    symbol: str,
    days: int = 7,
) -> dict[str, Any]:
    """timeseries.db에서 symbol의 최근 N일 (날 기준 limit=days*24) 통계 dict 반환.

    반환: {symbol, n, start_price, end_price, min_price, max_price, pct_change}
    """
    rows = timeseries.read_history(db_path, symbol=symbol, limit=days * 24)
    if not rows:
        return {
            "symbol": symbol, "n": 0,
            "start_price": None, "end_price": None,
            "min_price": None, "max_price": None,
            "pct_change": None,
        }
    prices = [r["price"] for r in rows if r["price"] is not None]
    if not prices:
        return {
            "symbol": symbol, "n": 0,
            "start_price": None, "end_price": None,
            "min_price": None, "max_price": None,
            "pct_change": None,
        }
    start, end = prices[0], prices[-1]
    pct = (end / start - 1) * 100 if start else 0.0
    return {
        "symbol": symbol,
        "n": len(prices),
        "start_price": start,
        "end_price": end,
        "min_price": min(prices),
        "max_price": max(prices),
        "pct_change": round(pct, 3),
    }
