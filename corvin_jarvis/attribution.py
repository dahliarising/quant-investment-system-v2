"""Corvin Jarvis — Outcome Attribution (Tier 2.4)

지난 N일 wiki sessions에서 언급된 종목 → 실제 가격 변동 추적.
LLM(Corvin)이 raw data를 받아 자기신뢰도 calibration.

⚠️ 텍스트 파싱은 휴리스틱 — 100% 정확하지 않음. 추세/방향성 측정용.
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from corvin_jarvis import timeseries

log = logging.getLogger("corvin.attribution")

# 일반 영어 단어 / 한국어 noise → ticker 오인 방지
_STOP_WORDS = {
    # 영어 일반어
    "THE", "AT", "IN", "ON", "FOR", "AND", "OR", "BUT", "TO", "OF",
    "A", "AN", "IS", "IT", "BE", "BY", "AS", "SO", "IF", "DO",
    "WITH", "FROM", "INTO", "OVER", "UP", "DOWN", "NEW", "OLD",
    "ABOUT", "RISK", "BUY", "SELL", "HOLD", "STRONG", "HIGH", "LOW",
    "FRESH", "STALE", "OK", "OFF", "ON",
    # 화폐/지표
    "KST", "KRW", "USD", "EUR", "JPY", "FX", "VIX", "GDP", "CPI",
    "RSI", "MA", "MA20", "MA50", "EPS", "EBITDA", "ROE", "PER", "PBR",
    # 약어
    "API", "ETF", "ESG", "MMF", "MMU", "DB", "DESC", "ASC",
    "MCP", "AI", "LLM", "RAG", "TDD", "URI", "URL", "JSON", "SQL",
    "IRP", "ISA", "IRA", "CMA", "MMDA", "HSA",
    "FDR", "FRD", "FNB", "FN", "KR", "US",
    # 지수
    "KOSPI", "KOSDAQ", "NDX", "DJI", "DOW", "SPX", "SP",
    # 그 외
    "META", "MSFT", "NVDA", "TSLA", "AAPL",  # — wait, 이건 실제 ticker
}
# Real US tickers shouldn't be in stop list — remove them
_STOP_WORDS -= {"META", "MSFT", "NVDA", "TSLA", "AAPL"}

# US ticker 패턴: 1-5 대문자 (단어 경계)
_US_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
# Korean stock code: 6 digits
_KR_CODE_RE = re.compile(r"\b(\d{6})\b")


def session_symbols(text: str) -> set[str]:
    """text에서 가능성 있는 ticker 추출 (휴리스틱)."""
    syms: set[str] = set()
    for m in _US_TICKER_RE.findall(text):
        if m in _STOP_WORDS:
            continue
        syms.add(m)
    for m in _KR_CODE_RE.findall(text):
        syms.add(m)
    return syms


def actual_moves(
    symbols: list[str],
    db_path: Path,
    days_back: int = 7,
) -> list[dict[str, Any]]:
    """각 symbol의 timeseries에서 N일치 가격 fetch → pct change.

    반환: [{symbol, pct_change, n}].
    """
    out: list[dict[str, Any]] = []
    for sym in symbols:
        rows = timeseries.read_history(db_path, symbol=sym, limit=days_back * 24)
        prices = [r["price"] for r in rows if r["price"] is not None and r["price"] > 0]
        if len(prices) < 2:
            out.append({"symbol": sym, "pct_change": None, "n": len(prices)})
            continue
        start, end = prices[0], prices[-1]
        pct = (end / start - 1) * 100 if start else None
        out.append({
            "symbol": sym,
            "pct_change": round(pct, 3) if pct is not None else None,
            "n": len(prices),
        })
    return out


def weekly_report(
    wiki_dir: Path,
    db_path: Path,
    today: date,
    lookback_days: int = 7,
) -> dict[str, Any]:
    """지난 lookback_days의 wiki sessions → 각 session에서 mention된 symbol → 실제 move.

    반환: {today, sessions_analyzed, entries: [{file, date, symbols, moves}]}
    """
    if not wiki_dir.exists():
        return {"today": today.isoformat(), "sessions_analyzed": 0, "entries": []}

    cutoff = today - timedelta(days=lookback_days)
    entries: list[dict[str, Any]] = []

    for md in sorted(wiki_dir.glob("*.md"), reverse=True):
        # 파일명에서 날짜 prefix 파싱 (YYYY-MM-DD-*.md)
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})-", md.name)
        if not m:
            continue
        session_date = date(int(m[1]), int(m[2]), int(m[3]))
        if session_date < cutoff or session_date > today:
            continue
        try:
            text = md.read_text(errors="ignore")
        except OSError:
            continue
        syms = session_symbols(text)
        if not syms:
            continue
        moves = actual_moves(sorted(syms), db_path, days_back=lookback_days)
        entries.append({
            "file": md.name,
            "date": session_date.isoformat(),
            "symbols": sorted(syms),
            "moves": moves,
        })

    return {
        "today": today.isoformat(),
        "sessions_analyzed": len(entries),
        "entries": entries,
    }
