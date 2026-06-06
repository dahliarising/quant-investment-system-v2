"""Phase 2 — 촉발원인 규명 테스트.

멀티소스 당일 이상치 수집(DI) → 크기순 랭킹 → top-N 원인 메시지. 네트워크 없음.
"""
from corvin_jarvis import cause_attribution as ca


def test_gather_candidates_collects_nonempty_di():
    def rates():
        return {"factor": "금리", "magnitude": 2.3, "detail": "10Y +12bp"}
    def fx():
        return None                       # 이상 없음 → 스킵
    def broken():
        raise RuntimeError("source down")  # 격리 → 스킵
    cands = ca.gather_candidates({"rates": rates, "fx": fx, "broken": broken})
    assert len(cands) == 1
    assert cands[0]["factor"] == "금리"


def test_rank_causes_by_magnitude():
    cands = [
        {"factor": "환율", "magnitude": 1.1, "detail": "원/달러 +0.4%"},
        {"factor": "반도체", "magnitude": 3.5, "detail": "SOXX -5.2%"},
        {"factor": "금리", "magnitude": -2.0, "detail": "10Y -8bp"},
    ]
    ranked = ca.rank_causes(cands)
    assert [c["factor"] for c in ranked] == ["반도체", "금리", "환율"]   # |magnitude| 내림차순


def test_attribution_message_topn():
    ranked = [
        {"factor": "반도체", "magnitude": 3.5, "detail": "SOXX -5.2%"},
        {"factor": "금리", "magnitude": -2.0, "detail": "10Y -8bp"},
        {"factor": "환율", "magnitude": 1.1, "detail": "원/달러 +0.4%"},
    ]
    msg = ca.attribution_message(ranked, top_n=2)
    assert "반도체" in msg and "SOXX -5.2%" in msg
    assert "금리" in msg
    assert "환율" not in msg               # top_n=2라 제외


def test_attribution_message_empty():
    assert "원인 불명" in ca.attribution_message([], top_n=3)
