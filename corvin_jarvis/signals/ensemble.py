"""Corvin 선행 인텔리전스 — Pillar 3: 앙상블 스크리너.

여러 방법론의 복합 점수 (단일 신호 아님). xang1234/stock-screener 패턴 차용:
- Minervini 추세 템플릿 (MA 정렬 + 52주 위치)
- 거래량 돌파
- 멀티타임프레임 상대강도(RS)

순수 함수 (지표 입력 → 0-100). 라이브 계산은 호출부(dca_timing 헬퍼 재사용).
"""
from __future__ import annotations

from corvin_jarvis.signals.leading_signal import LeadingSignal

NEUTRAL = 50.0

W_MINERVINI = 0.4
W_RS = 0.4
W_VOLUME = 0.2


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def minervini_trend_score(
    price: float,
    ma50: float | None,
    ma150: float | None,
    ma200: float | None,
    low_52w: float | None,
    high_52w: float | None,
) -> float:
    """Minervini 추세 템플릿 6기준 통과율 → 0-100. 지표 전무하면 NEUTRAL."""
    criteria: list[bool] = []
    if ma150 is not None and ma200 is not None:
        criteria.append(price > ma150 and price > ma200)  # 1. price > MA150/200
        criteria.append(ma150 > ma200)                    # 2. MA150 > MA200
    if ma50 is not None and ma150 is not None and ma200 is not None:
        criteria.append(ma50 > ma150 > ma200)             # 3. MA 정배열
    if ma50 is not None:
        criteria.append(price > ma50)                     # 4. price > MA50
    if low_52w is not None and low_52w > 0:
        criteria.append(price >= low_52w * 1.30)          # 5. 52주 저가 +30%↑
    if high_52w is not None and high_52w > 0:
        criteria.append(price >= high_52w * 0.75)         # 6. 52주 고가 -25%내
    if not criteria:
        return NEUTRAL
    return round(sum(criteria) / len(criteria) * 100.0, 2)


def volume_breakthrough_score(vol: float | None, vol_avg: float | None) -> float:
    """거래량/평균 비율 → 0-100. 1배=50, 2배+=100. None이면 NEUTRAL."""
    if vol is None or vol_avg is None or vol_avg <= 0:
        return NEUTRAL
    ratio = vol / vol_avg
    return _clamp(50.0 + (ratio - 1.0) * 50.0)


def multi_timeframe_rs_score(rs_by_window: dict[str, float]) -> float:
    """타임프레임별 상대강도(%p) 평균 → 0-100. ±10%p 기준. 비면 NEUTRAL."""
    if not rs_by_window:
        return NEUTRAL
    avg_rs = sum(rs_by_window.values()) / len(rs_by_window)
    return _clamp(50.0 + avg_rs * 4.0)


_RS_WINDOWS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}


def _ma(closes: list[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def indicators_from_closes(closes: list[float]) -> dict[str, float | None]:
    """종가 시계열 → Minervini 입력 지표. 부족하면 해당 항목 None."""
    if not closes:
        return {"price": None, "ma50": None, "ma150": None,
                "ma200": None, "low_52w": None, "high_52w": None}
    window_52w = closes[-252:] if len(closes) >= 252 else closes
    return {
        "price": closes[-1],
        "ma50": _ma(closes, 50),
        "ma150": _ma(closes, 150),
        "ma200": _ma(closes, 200),
        "low_52w": min(window_52w),
        "high_52w": max(window_52w),
    }


def _pct_return(closes: list[float], window: int) -> float | None:
    if len(closes) <= window:
        return None
    past = closes[-window - 1]
    if past == 0:
        return None
    return (closes[-1] - past) / past * 100.0


def rs_windows_from_closes(
    closes: list[float], bench_closes: list[float]
) -> dict[str, float]:
    """타임프레임별 (종목수익률 - 벤치수익률) %p. 데이터 부족 윈도우는 제외."""
    out: dict[str, float] = {}
    for name, w in _RS_WINDOWS.items():
        sym_r = _pct_return(closes, w)
        bench_r = _pct_return(bench_closes, w)
        if sym_r is not None and bench_r is not None:
            out[name] = round(sym_r - bench_r, 4)
    return out


def ensemble_score(minervini: float, rs: float, volume: float) -> float:
    """3방법론 가중 합산 → 0-100."""
    return round(
        minervini * W_MINERVINI + rs * W_RS + volume * W_VOLUME, 2
    )


def _direction(score: float) -> str:
    if score >= 70.0:
        return "bull"
    if score < 40.0:
        return "bear"
    return "neutral"


def build_ensemble_signal(
    symbol: str, minervini: float, rs: float, volume: float
) -> LeadingSignal:
    """앙상블 복합 점수 → LeadingSignal."""
    score = ensemble_score(minervini, rs, volume)
    direction = _direction(score)
    confidence = _clamp(abs(score - NEUTRAL) * 2.0)
    label = {"bull": "강세", "bear": "약세", "neutral": "중립"}[direction]
    return LeadingSignal(
        pillar="ensemble", symbol=symbol, direction=direction,
        confidence=confidence, score=score, horizon="days",
        advisory=False,
        message=f"🎯 {symbol} 앙상블 {label} (복합 {score:.0f}/100)",
        evidence={"minervini": minervini, "rs": rs, "volume": volume,
                  "composite": score},
    )
