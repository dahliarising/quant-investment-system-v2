"""Phase 3 — 적대적 검증.

토론/전략 결론을 *반증*하려는 4렌즈 회의론자 패널. 과반이 반증하면 결론 폐기 →
그룹씽크(같은 데이터·같은 프레임 합의=착시) 차단. 각 렌즈 판정은 LLM/룰 DI.

4 렌즈 = 이번 세션 허점 분석에서 도출:
 freshness   — 데이터 신선도/장중 미완성봉으로 판단했나
 correlation — 상관관계/가짜 분산을 무시했나
 unvalidated — 백테스트 안 된 임계로 판단했나
 cause       — 촉발 원인을 규명 안 했나
"""
from __future__ import annotations

from typing import Callable

LENSES = [
    {"id": "freshness", "name": "데이터 신선도",
     "prompt": "이 결론이 장중 미완성봉/저장 stale 데이터에 기댔는지 반증하라."},
    {"id": "correlation", "name": "상관관계",
     "prompt": "보유가 고상관(가짜 분산)이라 결론이 위험한지 반증하라."},
    {"id": "unvalidated", "name": "임계 미검증",
     "prompt": "결론의 근거 신호가 백테스트로 검증됐는지, 안 됐으면 반증하라."},
    {"id": "cause", "name": "원인 부재",
     "prompt": "급락의 촉발 원인을 모르고 패닉/추세를 단정했는지 반증하라."},
]


def tally(verdicts: list[dict]) -> dict:
    """렌즈 판정 집계 — 과반 반증이면 survives=False, 최강 반증(weakest) 표시."""
    refuted = [v for v in verdicts if v.get("refuted")]
    total = len(verdicts)
    survives = len(refuted) <= total // 2          # 과반 미달이어야 생존
    weakest = max(refuted, key=lambda v: v.get("confidence", 0.0)) if refuted else None
    return {"survives": survives, "refuted_count": len(refuted),
            "total": total, "weakest": weakest}


def verify_conclusion(conclusion: str,
                      lens_runner: Callable[[dict, str], dict]) -> dict:
    """4렌즈로 결론 반증 시도 → 집계. lens_runner(lens, conclusion)=DI."""
    votes = []
    for lens in LENSES:
        v = lens_runner(lens, conclusion)
        votes.append({"lens": lens["id"], **v})
    result = tally(votes)
    result["votes"] = votes
    result["conclusion"] = conclusion
    return result
