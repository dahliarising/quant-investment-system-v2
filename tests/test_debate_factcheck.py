"""실시간 토론 툴 — Phase C: fact-check 패스 테스트.

에이전트 발언의 수치를 검증 컨텍스트와 대조 → 환각/불일치 flag.
"""
from corvin_jarvis.live_debate import factcheck as fc


def test_extract_claims_finds_symbol_pct():
    claims = fc.extract_claims("TSLA −10.5% 손절하자, MSFT +8.7% 보유")
    pairs = {(c["symbol"], c["value"]) for c in claims}
    assert ("TSLA", -10.5) in pairs
    assert ("MSFT", 8.7) in pairs


def test_verify_turn_ok_when_matches_context():
    facts = {("TSLA", "live_pnl_pct", -10.5)}
    v = fc.verify_turn("TSLA −10.5% 손절선 돌파", facts, tol=0.5)
    assert v["ok"] is True
    assert v["flags"] == []


def test_verify_turn_flags_mismatch():
    facts = {("TSLA", "live_pnl_pct", -10.5)}
    v = fc.verify_turn("TSLA −5.0% 별거 아냐", facts, tol=0.5)
    assert v["ok"] is False
    assert any(f["kind"] == "mismatch" and f["symbol"] == "TSLA" for f in v["flags"])


def test_verify_turn_flags_unverified_number():
    v = fc.verify_turn("ABCD +99% 가즈아", set(), tol=0.5)
    assert v["ok"] is False
    assert any(f["kind"] == "unverified" for f in v["flags"])


def test_annotate_attaches_badge():
    assert fc.annotate({"text": "x"}, {"ok": True, "flags": []})["badge"] == "verified"
    assert fc.annotate({"text": "x"}, {"ok": False,
            "flags": [{"kind": "mismatch"}]})["badge"] == "mismatch"
    assert fc.annotate({"text": "x"}, {"ok": False,
            "flags": [{"kind": "unverified"}]})["badge"] == "unverified"
