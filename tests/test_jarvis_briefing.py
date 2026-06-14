"""Tests for jarvis briefing weekend/freshness rendering (2026-06-14 일).

A: 시점 라벨 — 휴장 시 '종가' 표기 + 헤더 시장상태.
B: 휴장 시 시세/종목 알람을 '지난 거래일 마감 시점(참고)'로 접고, 거시만 top 노출.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from corvin_jarvis import jarvis

KST = ZoneInfo("Asia/Seoul")
SUN_0500 = datetime(2026, 6, 14, 5, 0, tzinfo=KST)   # 일요일 — 양 시장 휴장
FRI_1000 = datetime(2026, 6, 12, 10, 0, tzinfo=KST)  # 금요일 KR 장중

LATEST = {
    "timestamp_kst": "2026-06-14T05:00:03+09:00",
    "indices": {"kospi": {"price": 8123.62, "pct_change": 4.63}},
    "commodities": {"gold": {"price": 4215.0, "pct_change": 3.05}},
    "fx": {"usd_krw": {"price": 1517.89, "pct_change": -0.47}},
    "portfolio": [
        {"symbol": "META", "shares": 7, "avg_price": 597.61, "current_price": 566.98,
         "pnl_pct": -5.13, "market_value": 3968.86, "currency": "USD"},
    ],
    "portfolio_summary": {
        "total_value_krw": 9492000.0, "total_value_usd": 12574.76,
        "position_count": 1, "winners_count": 0, "losers_count": 1,
        "best_pnl_pct": -5.13, "worst_pnl_pct": -5.13, "stale_as_of": "2026-06-05",
    },
}

ALERTS = [
    {"category": "fx", "metric": "usd_krw_level", "severity": "low", "message": "USD/KRW 1517"},
    {"category": "index", "metric": "kospi", "severity": "critical", "message": "KOSPI 급등 +4.63%"},
    {"category": "predictive", "metric": "META", "severity": "critical", "message": "META 도달확률 98%"},
]


# ---- A: 시점 라벨 ----
def test_market_section_relabels_close_on_weekend():
    out = jarvis._format_market_section(LATEST, now=SUN_0500)
    assert "지난 거래일" in out
    assert "06-12" in out


def test_market_section_no_close_label_during_open():
    out = jarvis._format_market_section(LATEST, now=FRI_1000)
    assert "지난 거래일" not in out


def test_portfolio_uses_close_header_on_weekend():
    out = jarvis._format_portfolio_section(LATEST, now=SUN_0500)
    assert "종가" in out
    assert "| 현재가 |" not in out  # 휴장 땐 '현재가' 금지


def test_portfolio_uses_current_header_during_open():
    out = jarvis._format_portfolio_section(LATEST, now=FRI_1000)
    assert "| 현재가 |" in out


# ---- B: 휴장 시 stale 알람 접기 ----
def test_alerts_collapse_stale_on_weekend():
    out = jarvis._format_alerts_section(ALERTS, now=SUN_0500)
    # 거시(fx)는 top 정상 노출
    assert "USD/KRW 1517" in out
    # stale(KOSPI/META)은 '참고/지난 거래일' 섹션으로 분리
    assert "지난 거래일" in out
    # KOSPI가 top CRITICAL 헤더 직후가 아니라 참고 섹션에 있어야 한다:
    # CRITICAL 카운트에 stale이 안 잡혀야 함 → top에 critical 0건이면 'CRITICAL' 헤더 없음
    top = out.split("지난 거래일")[0]
    assert "KOSPI 급등" not in top


def test_alerts_no_collapse_during_open_session():
    # KR 장중엔 KR/거시 알람은 stale이 아니다 (US는 이 시각 휴장이라 제외).
    kr_macro = [a for a in ALERTS if a["category"] in ("fx", "index")]
    out = jarvis._format_alerts_section(kr_macro, now=FRI_1000)
    assert "지난 거래일" not in out
    assert "KOSPI 급등" in out


# ---- A: 헤더 시장상태 ----
def test_compose_briefing_header_has_market_status(monkeypatch):
    def fake_load(p):
        if p == jarvis.LATEST_FILE:
            return LATEST
        if p == jarvis.ALERTS_FILE:
            return {"alerts": ALERTS}
        return None
    monkeypatch.setattr(jarvis, "_load", fake_load)
    out = jarvis.compose_briefing(now=SUN_0500)
    assert "휴장" in out
    assert "다음 개장" in out
    # 회귀: 헤더 시장상태가 본문 위에 위치
    assert out.index("Market") < out.index("시장 스냅샷")
