"""🅑: 휴장 시 verdict rationale의 '오늘 이미 +N%' → '지난 거래일' 라벨.

미보유 급등 종목의 추격위험 판정 문구가 주말에 금요일 움직임을 '오늘'로
오기하던 가짜 신선도 문제 수정.
"""
from __future__ import annotations

from corvin_jarvis.signals import verdict


def _spike_ctx(market_closed: bool) -> dict:
    # 미보유 + 당일 +10% 급등 → spike 분기(추격위험) 진입
    return {
        "symbol": "035420", "held": False, "pct_today": 10.0,
        "dca_score": 0, "rs": None, "theme_alive": False,
        "high_vol": False, "market_closed": market_closed,
    }


def test_spike_label_weekday_uses_today():
    v = verdict.decide(_spike_ctx(market_closed=False))
    assert v.action == "관망"
    assert "오늘 이미" in v.rationale


def test_spike_label_weekend_uses_last_session():
    v = verdict.decide(_spike_ctx(market_closed=True))
    assert v.action == "관망"
    assert "지난 거래일" in v.rationale
    assert "오늘 이미" not in v.rationale
