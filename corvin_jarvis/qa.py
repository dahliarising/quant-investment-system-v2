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


def relative_strength(
    db_path: Path,
    symbol: str,
    benchmark: str = "sp500",
    days: int = 7,
) -> dict[str, Any]:
    """symbol vs benchmark 상대 성과 비교.

    verdict: outperform (+1pp 이상) / underperform (-1pp 이하) / neutral / unknown
    """
    s = recent_history_summary(db_path, symbol=symbol, days=days)
    b = recent_history_summary(db_path, symbol=benchmark, days=days)
    if s["pct_change"] is None or b["pct_change"] is None:
        return {
            "symbol": symbol, "benchmark": benchmark,
            "symbol_pct": None, "benchmark_pct": None,
            "relative_pp": None, "verdict": "unknown",
        }
    rel = s["pct_change"] - b["pct_change"]
    if rel > 1.0:
        verdict = "outperform"
    elif rel < -1.0:
        verdict = "underperform"
    else:
        verdict = "neutral"
    return {
        "symbol": symbol,
        "benchmark": benchmark,
        "symbol_pct": s["pct_change"],
        "benchmark_pct": b["pct_change"],
        "relative_pp": round(rel, 3),
        "verdict": verdict,
    }


def wiki_search(
    query: str,
    wiki_dir: Path = DEFAULT_WIKI_DIR,
    top_k: int = 5,
    snippet_chars: int = 200,
) -> list[dict[str, Any]]:
    """corvin-sessions/*.md 안에서 query를 case-insensitive grep.

    파일별로 첫 매칭 라인 주변 snippet 반환. 최근 파일(파일명 정렬 DESC) 우선.
    """
    if not wiki_dir.exists() or not wiki_dir.is_dir():
        return []
    q = query.lower()
    hits: list[dict[str, Any]] = []
    for md in sorted(wiki_dir.glob("*.md"), reverse=True):
        try:
            text = md.read_text(errors="ignore")
        except OSError:
            continue
        idx = text.lower().find(q)
        if idx < 0:
            continue
        start = max(0, idx - snippet_chars // 2)
        end = min(len(text), idx + snippet_chars // 2)
        snippet = text[start:end].strip().replace("\n", " ")
        hits.append({"path": md, "snippet": snippet})
        if len(hits) >= top_k:
            break
    return hits
