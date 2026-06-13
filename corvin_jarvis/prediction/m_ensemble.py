# corvin_jarvis/prediction/m_ensemble.py
"""시스템10 — 방법론 합의 앙상블 (백테스트 게이트 존중).

시장 방향 신호를 가진 결과들에서 방향표(+1/-1)를 추출해 가중 합의.
게이트 대상(logistic/montecarlo)은 백테스트 통과분만 투표(검증 원칙).
"""
from __future__ import annotations

from typing import Any

from corvin_jarvis.prediction import backtest
from corvin_jarvis.prediction.contract import PredictionResult, insufficient

_NEUTRAL_BAND = 0.15
_SENTIMENT_NEUTRAL = 0.5   # m_sentiment._THRESH와 일치 — 중립 tone은 합의에서 기권


def consensus(votes: list[tuple[str, int, float]]) -> dict[str, Any]:
    """votes = [(name, direction±1, weight)]. 가중 net 점수 → 방향."""
    elig = [(n, d, w) for n, d, w in votes if d != 0 and w > 0]
    if not elig:
        return {"direction": "neutral", "net": 0.0, "agreement": 0.0,
                "n": 0, "up": 0}
    total = sum(w for _, _, w in elig)
    score = sum(d * w for _, d, w in elig) / total if total else 0.0
    up = sum(1 for _, d, _ in elig if d > 0)
    n = len(elig)
    direction = ("up" if score > _NEUTRAL_BAND
                 else "down" if score < -_NEUTRAL_BAND else "neutral")
    return {"direction": direction, "net": score,
            "agreement": max(up, n - up) / n, "n": n, "up": up}


def _vote(r: PredictionResult, gate: dict | None) -> tuple[str, int, float] | None:
    if not r.data_ok:
        return None
    s = r.system
    if s == "vector_analog":
        return ("vector", 1 if r.evidence.get("mean", 0) > 0 else -1, 1.0)
    if s == "logistic":
        if gate is not None and not backtest.passed(gate, "logistic"):
            return None
        p = r.evidence.get("prob_up", 0.5)
        return ("logistic", 1 if p >= 0.5 else -1, min(abs(p - 0.5) * 2, 1.0))
    if s == "montecarlo" and r.scope == "market":
        if gate is not None and not backtest.passed(gate, "montecarlo"):
            return None
        p50 = r.evidence.get("p50", 0.0)
        return ("montecarlo", 1 if p50 > 0 else -1, min(abs(p50) / 5.0, 1.0))
    if s == "sentiment":
        t = r.evidence.get("tone", 0.0)
        if abs(t) < _SENTIMENT_NEUTRAL:   # 중립은 기권 (모델 판정과 일치)
            return None
        return ("sentiment", 1 if t > 0 else -1, min(abs(t) / 3.0, 1.0))
    if s == "geopolitical":
        sc = r.evidence.get("risk_score", 50)
        return ("geo", -1 if sc >= 65 else (1 if sc < 40 else 0), 0.5)
    return None


def run(results: list[PredictionResult], gate: dict | None = None,
        min_votes: int = 2) -> PredictionResult:
    votes = [v for v in (_vote(r, gate) for r in results) if v and v[1] != 0]
    if len(votes) < min_votes:
        # 1표는 '합의'가 아니다 — 과신 방지(meaningful-metrics)
        return insufficient("ensemble", "market",
                            f"합의 표본 {len(votes)} < 최소 {min_votes}")
    c = consensus(votes)
    label = {"up": "상승 우위", "down": "하락 우위", "neutral": "중립"}[c["direction"]]
    verdict = (f"합의: {label} ({c['up']}/{c['n']} 동의, 강도 {c['net']:+.2f})")
    conf = round(min(92.0, 50 + abs(c["net"]) * 45), 1)
    return PredictionResult("ensemble", "market", verdict, conf,
                            {**c, "votes": [n for n, _, _ in votes]}, data_ok=True)
