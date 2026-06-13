# corvin_jarvis/prediction/m_velocity.py
"""시스템1 어댑터 — predictive_engine.evaluate_velocity 재사용."""
from __future__ import annotations

from typing import Any

from corvin_jarvis import predictive_engine
from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def run(holdings: list[dict[str, Any]], stops: dict[str, float],
        closes_by_sym: dict[str, list[float]], regime_trend: str | None = "down"
        ) -> list[PredictionResult]:
    signals = predictive_engine.evaluate_velocity(
        holdings, stops, closes_by_sym, regime_trend=regime_trend)
    by_sym = {s.symbol for s in signals}
    out: list[PredictionResult] = []
    for s in signals:
        out.append(PredictionResult(
            system="velocity", scope=s.symbol,
            verdict=f"손절선 도달 예상 {s.horizon_days}일 이내" if s.horizon_days else "하락속도 경보",
            confidence=s.confidence, evidence=s.to_dict(), data_ok=True))
    # stop 없는 보유는 '보류'로 가시화
    for pos in holdings:
        sym = str(pos.get("symbol", ""))
        if sym not in by_sym and sym not in stops:
            out.append(insufficient("velocity", sym, "손절선 미설정"))
    return out
