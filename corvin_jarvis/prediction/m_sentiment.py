# corvin_jarvis/prediction/m_sentiment.py
"""시스템8 — GDELT 뉴스 센티먼트 → PredictionResult (주입식, 의존성 0).

MCP 호출은 오케스트레이터 담당. 여기선 dict payload만 변환(테스트 가능).
payload 예: {"tone": float, "article_volume": int, "trend": str}
tone = GDELT 평균 감정톤(음수=부정). |tone|>=0.5 방향 신호, 그 미만 중립.
"""
from __future__ import annotations

from typing import Any

from corvin_jarvis.prediction.contract import PredictionResult, insufficient

_THRESH = 0.5


def run(payload: dict[str, Any] | None) -> PredictionResult:
    if not payload or "tone" not in payload:
        return insufficient("sentiment", "market", "GDELT 응답 없음")
    tone = float(payload["tone"])
    vol = payload.get("article_volume")
    trend = payload.get("trend", "?")
    if tone >= _THRESH:
        verdict = f"뉴스 긍정 우위(tone {tone:+.1f}) → 심리 상방"
    elif tone <= -_THRESH:
        verdict = f"뉴스 부정 우위(tone {tone:+.1f}) → 심리 하방"
    else:
        verdict = f"뉴스 심리 중립(tone {tone:+.1f})"
    conf = round(min(85.0, 50 + abs(tone) * 12), 1)
    return PredictionResult("sentiment", "market", verdict, conf,
                            {"tone": tone, "article_volume": vol, "trend": trend},
                            data_ok=True)
