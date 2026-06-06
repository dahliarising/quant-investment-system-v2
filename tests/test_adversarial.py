"""Phase 3 — 적대적 검증 테스트.

결론을 *반증*하려는 4렌즈 회의론자 → 과반 반증 시 결론 폐기. LLM=DI.
허점①(그룹씽크) 대응 — 같은 프레임 합의를 깨는 독립 시선.
"""
from corvin_jarvis import adversarial as adv


def test_lenses_cover_four_blind_spots():
    ids = {lens["id"] for lens in adv.LENSES}
    assert ids == {"freshness", "correlation", "unvalidated", "cause"}


def test_tally_survives_on_minority_refute():
    verdicts = [
        {"lens": "freshness", "refuted": False},
        {"lens": "correlation", "refuted": True, "confidence": 0.6, "reason": "고상관"},
        {"lens": "unvalidated", "refuted": False},
        {"lens": "cause", "refuted": False},
    ]
    r = adv.tally(verdicts)
    assert r["survives"] is True            # 1/4 반증 = 과반 미달
    assert r["refuted_count"] == 1


def test_tally_killed_on_majority_refute():
    verdicts = [
        {"lens": "freshness", "refuted": True, "confidence": 0.5, "reason": "장중 미완성봉"},
        {"lens": "correlation", "refuted": True, "confidence": 0.8, "reason": "가짜 분산"},
        {"lens": "unvalidated", "refuted": True, "confidence": 0.7, "reason": "임계 미검증"},
        {"lens": "cause", "refuted": False},
    ]
    r = adv.tally(verdicts)
    assert r["survives"] is False           # 3/4 반증 = 과반
    assert r["weakest"]["reason"] == "가짜 분산"   # 가장 강한 반증(confidence 최고)


def test_verify_conclusion_runs_all_lenses():
    def fake_runner(lens, conclusion):
        return {"refuted": True, "confidence": 0.9, "reason": f"{lens['id']} 반박"}
    r = adv.verify_conclusion("전량 매도하자", fake_runner)
    assert r["survives"] is False
    assert len(r["votes"]) == 4


def test_live_lens_runner_parses_cli_verdict():
    lens = adv.LENSES[0]
    # CLI=DI: "반증:예" → refuted True, 이유 파싱
    refuted = adv.live_lens_runner(lens, "전량 매도",
              cli=lambda p, **k: "반증:예\n장중 미완성봉으로 판단함")
    assert refuted["refuted"] is True
    assert "미완성봉" in refuted["reason"]
    # "반증:아니오" → refuted False
    ok = adv.live_lens_runner(lens, "코어 홀드",
         cli=lambda p, **k: "반증:아니오\n근거 충분함")
    assert ok["refuted"] is False
