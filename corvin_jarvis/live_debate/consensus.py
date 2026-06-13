"""신뢰도 가중 종합 — track record 높은 페르소나 의견에 더 무게.

동등 가중(그룹씽크)의 한계 보완: 퀀트(92)·가치(88)가 성장(52)보다 무겁다.
votes: [{persona, credibility, stance}] — stance ∈ [-1(매도) .. +1(매수)].
"""
from __future__ import annotations


def weighted_consensus(votes: list[dict]) -> dict:
    """신뢰도 가중 평균 스탠스 → 점수·라벨."""
    total_w = sum(v.get("credibility", 0) for v in votes)
    if not total_w:
        return {"score": 0.0, "label": "중립/데이터 없음", "weight": 0}
    score = sum(v["stance"] * v.get("credibility", 0) for v in votes) / total_w
    if score > 0.2:
        label = "매수 우위"
    elif score < -0.2:
        label = "매도 우위"
    else:
        label = "중립/혼조"
    return {"score": round(score, 3), "label": label, "weight": total_w}
