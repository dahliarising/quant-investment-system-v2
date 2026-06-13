# corvin_jarvis/prediction/m_momentum.py
"""시스템3 어댑터 — 이동평균 교차 추세 신호."""
from __future__ import annotations

from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def _sma(values: list[float], window: int) -> float:
    return sum(values[-window:]) / window


def run_symbol(symbol: str, closes: list[float], short: int = 20,
               long: int = 120, min_days: int = 120) -> PredictionResult:
    if short >= long:
        return insufficient("momentum", symbol, f"short({short}) >= long({long}) 파라미터 오류")
    if len(closes) < min_days:
        return insufficient("momentum", symbol, f"일봉 {len(closes)} < {min_days}")
    sma_s = _sma(closes, short)
    sma_l = _sma(closes, long)
    if sma_s > sma_l:
        signal, verdict = "bullish", f"단기 MA({short}) > 장기 MA({long}) → 추세 상방"
    elif sma_s < sma_l:
        signal, verdict = "bearish", f"단기 MA({short}) < 장기 MA({long}) → 추세 하방"
    else:
        signal, verdict = "neutral", "MA 교차 중립"
    gap = (sma_s - sma_l) / sma_l if sma_l else 0.0
    conf = round(min(90.0, 55 + abs(gap) * 300), 1)
    return PredictionResult("momentum", symbol, verdict, conf,
                            {"signal": signal, "sma_short": sma_s, "sma_long": sma_l,
                             "gap_pct": gap * 100}, data_ok=True)
