"""Corvin Jarvis — 시장 영업시간 기반 alert 억제 (2026-05-29 폐하 지시).

장마감 시장의 종목 alert은 값이 안 변해 반복 푸시 = 노이즈. urgent 푸시에서
"alert이 가리키는 시장이 전부 휴장이면 억제". 거시(FX·원자재·VIX·지수外)는 시장 무관 → 항상 발송.
일일 다이제스트(16:00)는 이 억제를 적용하지 않음 (전체 요약 유지).
"""
from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
NY = ZoneInfo("America/New_York")

BASE_DIR = Path(__file__).resolve().parent
MONITORED_UNIVERSE = BASE_DIR / "monitored_universe.json"

# 거시/24h 카테고리 — 시장 영업시간과 무관, 항상 발송.
_MACRO_CATEGORIES = frozenset(
    {"fx", "commodity", "risk", "narrative", "earnings", "regime", "acceleration"}
)


def is_kr_open(now: datetime) -> bool:
    """KOSPI/KOSDAQ 정규장: 평일 09:00–15:30 KST."""
    k = now.astimezone(KST)
    if k.weekday() >= 5:
        return False
    return time(9, 0) <= k.time() <= time(15, 30)


def is_us_open(now: datetime) -> bool:
    """US 정규장: 평일 09:30–16:00 ET (DST는 ZoneInfo가 처리)."""
    n = now.astimezone(NY)
    if n.weekday() >= 5:
        return False
    return time(9, 30) <= n.time() <= time(16, 0)


def is_market_open(market: str, now: datetime) -> bool:
    return is_kr_open(now) if market == "KR" else is_us_open(now)


def _market_of_symbol(sym: str) -> str:
    return "KR" if sym.isdigit() and len(sym) == 6 else "US"


def _sector_name(metric: str) -> str:
    s = metric[len("sector_"):] if metric.startswith("sector_") else metric
    for suf in ("_provisional", "_confirmed"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def load_sector_markets(path: Path | None = None) -> dict[str, set[str]]:
    """monitored_universe.json → {sector: {시장...}}. 멤버 심볼 형식으로 시장 판정."""
    p = path or MONITORED_UNIVERSE
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    tickers = data.get("tickers", []) if isinstance(data, dict) else data
    out: dict[str, set[str]] = {}
    for t in tickers:
        sector = t.get("sector")
        sym = str(t.get("symbol", ""))
        if not sector or not sym:
            continue
        out.setdefault(sector, set()).add(_market_of_symbol(sym))
    return out


def alert_markets(alert: dict[str, Any], sector_markets: dict[str, set[str]]) -> set[str]:
    """alert이 가리키는 시장 집합. 빈 집합 = 거시(시장 무관)."""
    cat = alert.get("category", "")
    metric = alert.get("metric", "")
    if cat in ("universe", "leading_rs"):
        parts = metric.split("_")
        sym = parts[1] if len(parts) > 1 else ""
        return {_market_of_symbol(sym)} if sym else set()
    if cat == "portfolio":
        sym = metric.split("_")[-1]
        return {_market_of_symbol(sym)} if sym else set()
    if cat == "index":
        return {"KR"} if metric in ("kospi", "kosdaq") else {"US"}
    if cat == "sector":
        return set(sector_markets.get(_sector_name(metric), set()))
    if cat in _MACRO_CATEGORIES:
        return set()
    return set()  # 미지 카테고리는 보수적으로 항상 발송


def should_suppress(alert: dict[str, Any], now: datetime, sector_markets: dict[str, set[str]]) -> bool:
    """alert이 가리키는 시장이 모두 휴장이면 True. 거시(빈 집합)는 항상 False(발송)."""
    markets = alert_markets(alert, sector_markets)
    if not markets:
        return False
    return all(not is_market_open(m, now) for m in markets)
