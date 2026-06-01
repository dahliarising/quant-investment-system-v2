"""Corvin 선행 인텔리전스 — Pillar 4: 교차시장 (advisory only).

시간대 시차 활용한 참고용 방향 힌트:
- KR 종목: LMT/RTX/ITA 야간 마감(미국장이 KR보다 먼저 닫힘) → 012450 힌트
- US 종목: NQ/ES 선물 → META/MSFT/NVDA/TSLA 힌트

⚠️ deep-research 검증 약함(0-3 기각) → advisory=True 강제 + confidence 30% 페널티.
순수 함수. 라이브 프록시 가격은 호출부 책임.
"""
from __future__ import annotations

from corvin_jarvis.signals.leading_signal import LeadingSignal

# 검증 약함 반영: 신뢰도 30% 페널티
CONFIDENCE_PENALTY = 0.7
# 방향 판정 임계 (프록시 %change)
NEUTRAL_BAND = 0.5


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def build_cross_market_signal(
    symbol: str, proxy_name: str, proxy_pct: float
) -> LeadingSignal:
    """프록시 %change → advisory 교차시장 신호. confidence 페널티 적용."""
    if proxy_pct >= NEUTRAL_BAND:
        direction = "bull"
    elif proxy_pct <= -NEUTRAL_BAND:
        direction = "bear"
    else:
        direction = "neutral"
    base_conf = _clamp(abs(proxy_pct) * 10.0)
    confidence = round(base_conf * CONFIDENCE_PENALTY, 2)
    label = {"bull": "강세", "bear": "약세", "neutral": "중립"}[direction]
    return LeadingSignal(
        pillar="cross_market", symbol=symbol, direction=direction,
        confidence=confidence, score=None, horizon="intraday",
        advisory=True,
        message=f"🌐 {symbol} {proxy_name} {proxy_pct:+.1f}% → {label} 힌트",
        evidence={"proxy_name": proxy_name, "proxy_pct": proxy_pct,
                  "penalized": True},
    )
