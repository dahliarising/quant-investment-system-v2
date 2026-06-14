"""Corvin Jarvis — 시장 영업시간 기반 alert 억제 (2026-05-29 폐하 지시).

장마감 시장의 종목 alert은 값이 안 변해 반복 푸시 = 노이즈. urgent 푸시에서
"alert이 가리키는 시장이 전부 휴장이면 억제". 거시(FX·원자재·VIX·지수外)는 시장 무관 → 항상 발송.
일일 다이제스트(16:00)는 이 억제를 적용하지 않음 (전체 요약 유지).
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
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
    if cat == "predictive":
        # predictive alert의 metric은 종목 심볼 자체 (예: "META", "005930").
        return {_market_of_symbol(metric)} if metric else set()
    if cat == "early_warning":
        # 포지션성 하드스톱 등 — metric 끝 토큰이 심볼 (예: "hardstop_012450").
        sym = metric.split("_")[-1]
        return {_market_of_symbol(sym)} if sym else set()
    if cat == "portfolio":
        sym = metric.split("_")[-1]
        return {_market_of_symbol(sym)} if sym else set()
    if cat == "index":
        return {"KR"} if metric in ("kospi", "kosdaq") else {"US"}
    if cat == "sector":
        # 시장별 분리 alert은 explicit market 필드로 정밀 판정 (구버전은 멤버 기반 폴백)
        mkt = alert.get("market")
        if mkt:
            return {mkt}
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


def partition_alerts(
    alerts: list[dict[str, Any]],
    now: datetime,
    sector_markets: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """alert을 (live, stale)로 분리.

    stale = 가리키는 시장이 전부 휴장이라 값이 갱신될 수 없는 신호(지난 거래일 마감 시점).
    브리핑에서 stale을 가짜 CRITICAL로 재노출하지 않고 '참고' 섹션으로 접기 위함.
    거시(시장 무관)는 항상 live.
    """
    live: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for a in alerts:
        (stale if should_suppress(a, now, sector_markets) else live).append(a)
    return live, stale


# ---------------------------------------------------------------------------
# 거래일 캘린더 — 브리핑 시점 라벨링용.
# 주의: 공휴일은 미반영(주말만 처리). 휴장일 정밀화가 필요하면 거래소 캘린더 도입.
# ---------------------------------------------------------------------------
def _tz(market: str) -> ZoneInfo:
    return KST if market == "KR" else NY


def _close_time(market: str) -> time:
    return time(15, 30) if market == "KR" else time(16, 0)


def _open_time(market: str) -> time:
    return time(9, 0) if market == "KR" else time(9, 30)


def last_close_date(now: datetime, market: str) -> date:
    """가장 최근 '마감이 완료된' 정규장 날짜 (해당 시장 현지 기준)."""
    local = now.astimezone(_tz(market))
    day = local.date()
    closed_today = day.weekday() < 5 and local.time() >= _close_time(market)
    if not closed_today:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def next_open_date(now: datetime, market: str) -> date:
    """다음 정규장 개장 날짜 (해당 시장 현지 기준). 평일 개장 전이면 당일."""
    local = now.astimezone(_tz(market))
    day = local.date()
    if day.weekday() < 5 and local.time() < _open_time(market):
        return day
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def market_status_label(now: datetime) -> str:
    """브리핑 헤더용 시장 상태 1줄. 휴장 시 데이터 시점/다음 개장을 명시."""
    kr_open = is_kr_open(now)
    us_open = is_us_open(now)
    if kr_open or us_open:
        parts = []
        parts.append("KR 정규장" if kr_open else "KR 휴장")
        parts.append("US 정규장" if us_open else "US 휴장")
        return "🟢 " + " · ".join(parts) + " — 일부 실시간"
    kr_close = last_close_date(now, "KR")
    us_close = last_close_date(now, "US")
    kr_next = next_open_date(now, "KR")
    return (
        f"🔴 휴장 — 시세는 마지막 거래일 종가 기준 "
        f"(KR {kr_close:%m-%d} · US {us_close:%m-%d}) · 다음 개장 KR {kr_next:%m-%d}"
    )
