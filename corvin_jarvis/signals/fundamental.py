"""Corvin 선행 인텔리전스 — Pillar 1: 펀더멘털 성장 엔진.

종목 성장 방향성을 0-100 Growth Score로 산출.
구성: 재무 수치(50%) + Claude 정성(30%) + 뉴스 감정(20%).

모든 점수 함수는 순수(입력 dict → float). 라이브 fetch는 호출부 책임.
Claude 정성 점수 없으면 재무+뉴스로 graceful degrade.
"""
from __future__ import annotations

from typing import Any

from corvin_jarvis.signals.leading_signal import LeadingSignal

NEUTRAL = 50.0

W_FINANCIAL = 0.5
W_QUALITATIVE = 0.3
W_NEWS = 0.2


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _score_eps_growth(g: float) -> float:
    # -20% → ~23, 0% → 50, +30%+ → ~90
    return _clamp(50.0 + g * 133.0)


def _score_margin_trend(t: float) -> float:
    # 마진 개선폭 ±5%p 기준
    return _clamp(50.0 + t * 600.0)


def _score_debt_ratio(d: float) -> float:
    # 부채비율 0 → 90, 1.0 → 58, 2.5+ → 10
    return _clamp(90.0 - d * 32.0)


def _score_revenue_growth(g: float) -> float:
    return _clamp(50.0 + g * 150.0)


def financial_growth_score(metrics: dict[str, Any]) -> float:
    """재무 4지표 → 0-100. 가용 지표만 평균. 전무하면 NEUTRAL."""
    parts: list[float] = []
    if "eps_growth_yoy" in metrics:
        parts.append(_score_eps_growth(float(metrics["eps_growth_yoy"])))
    if "op_margin_trend" in metrics:
        parts.append(_score_margin_trend(float(metrics["op_margin_trend"])))
    if "debt_ratio" in metrics:
        parts.append(_score_debt_ratio(float(metrics["debt_ratio"])))
    if "revenue_growth_yoy" in metrics:
        parts.append(_score_revenue_growth(float(metrics["revenue_growth_yoy"])))
    if not parts:
        return NEUTRAL
    return round(sum(parts) / len(parts), 2)


def news_sentiment_score(tone: float | None) -> float:
    """뉴스 감정 tone(-1~+1) → 0-100. None이면 NEUTRAL."""
    if tone is None:
        return NEUTRAL
    return _clamp(50.0 + float(tone) * 50.0)


def combine_growth_score(
    financial: float, qualitative: float | None, news: float
) -> float:
    """가중 평균 Growth Score. qualitative None이면 가중치 재정규화."""
    pairs: list[tuple[float, float]] = [(financial, W_FINANCIAL), (news, W_NEWS)]
    if qualitative is not None:
        pairs.append((qualitative, W_QUALITATIVE))
    total_w = sum(w for _, w in pairs)
    return round(sum(v * w for v, w in pairs) / total_w, 2)


def growth_to_direction(score: float) -> str:
    """Growth Score → bull/neutral/bear."""
    if score >= 70.0:
        return "bull"
    if score < 40.0:
        return "bear"
    return "neutral"


def _confidence_from_score(score: float) -> float:
    """Growth Score가 중립(50)에서 멀수록 confidence 높게. |score-50|*2 → 0-100."""
    return _clamp(abs(score - NEUTRAL) * 2.0)


def build_fundamental_signal(
    symbol: str,
    financial: float,
    qualitative: float | None,
    news: float,
) -> LeadingSignal:
    """sub-score → Growth Score → LeadingSignal."""
    growth = combine_growth_score(financial, qualitative, news)
    direction = growth_to_direction(growth)
    confidence = _confidence_from_score(growth)
    label = {"bull": "성장", "bear": "둔화", "neutral": "중립"}[direction]
    return LeadingSignal(
        pillar="fundamental", symbol=symbol, direction=direction,
        confidence=confidence, score=growth, horizon="weeks",
        advisory=False,
        message=f"📊 {symbol} 펀더멘털 {label} (Growth {growth:.0f}/100)",
        evidence={
            "financial": financial, "news": news,
            "qualitative": qualitative,
            "qualitative_used": qualitative is not None,
            "growth_score": growth,
        },
    )
