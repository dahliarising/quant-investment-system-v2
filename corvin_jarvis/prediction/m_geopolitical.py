# corvin_jarvis/prediction/m_geopolitical.py
"""시스템4 어댑터 — Geopolitical MCP risk score → PredictionResult.

MCP 호출은 오케스트레이터 담당. 여기선 dict payload만 변환(주입식, 테스트 가능).
payload 예: {"risk_score": int, "trend": str, "top_event": str}
"""
from __future__ import annotations

from typing import Any

from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def run(payload: dict[str, Any] | None) -> PredictionResult:
    score = payload.get("risk_score") if payload else None
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return insufficient("geopolitical", "market", "MCP 응답 없음/타입 오류")
    trend = payload.get("trend", "?")
    event = payload.get("top_event", "")
    if score >= 65:
        verdict = f"지정학 리스크 높음({score}) — 변동성 경계. 핵심: {event}"
    elif score >= 40:
        verdict = f"지정학 리스크 보통({score}, {trend})"
    else:
        verdict = f"지정학 리스크 낮음({score})"
    conf = round(min(95.0, 50 + abs(score - 50)), 1)
    return PredictionResult("geopolitical", "market", verdict, conf,
                            {"risk_score": score, "trend": trend, "top_event": event},
                            data_ok=True)
