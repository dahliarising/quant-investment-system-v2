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


# ── candidates_from_snapshot: jarvis 스냅샷 → 원인 후보 (순수, 무네트워크) ──
def _snap():
    return {
        "indices": {"vix": {"price": 21.5, "pct_change": 40.0},
                    "sp500": {"pct_change": -2.6}, "kospi": {"pct_change": -5.5}},
        "universe": [
            {"sector": "반도체", "pct_change": -5.0},
            {"sector": "반도체", "pct_change": -5.4},
            {"sector": "방어주", "pct_change": -0.2},
            {"sector": "방어주", "pct_change": 0.1},
        ],
    }


def test_candidates_from_snapshot_finds_worst_sector_and_vix():
    cands = ca.candidates_from_snapshot(_snap())
    sem = next((c for c in cands if "반도체" in c["factor"]), None)
    assert sem is not None and sem["magnitude"] < -4          # 반도체 평균 -5.2
    assert any("변동성" in c["factor"] for c in cands)         # VIX +40% → 변동성 후보


def test_candidates_from_snapshot_quiet_market_few_causes():
    quiet = {"indices": {"vix": {"price": 14.0, "pct_change": 1.0}},
             "universe": [{"sector": "반도체", "pct_change": 0.3}]}
    cands = ca.candidates_from_snapshot(quiet)
    # 평온하면 강한 원인 후보 없음(이상치 임계 미달)
    assert all(abs(c["magnitude"]) < 2 for c in cands) or cands == []
