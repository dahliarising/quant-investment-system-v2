"""실시간 토론 툴 — Phase D: SSE 스트림 순수 로직 테스트.

debate 턴마다 fact-check → SSE 이벤트 emit. emit은 DI(소켓 없음).
"""
import json

from corvin_jarvis.live_debate import stream


def test_sse_event_format():
    s = stream.sse_event({"text": "hi", "badge": "verified"})
    assert s.startswith("data: ") and s.endswith("\n\n")
    payload = json.loads(s[len("data: "):].strip())
    assert payload["text"] == "hi" and payload["badge"] == "verified"


def test_run_stream_factchecks_each_turn():
    ctx = {"holdings": [
        {"symbol": "TSLA", "live_pnl_pct": -10.5, "price": {"value": 391.0}},
    ]}
    # value 페르소나가 틀린 수치 인용, 나머진 무난
    seq = iter(["TSLA −5.0% 별거 아냐"])
    def fake_llm(prompt):
        try:
            return next(seq)
        except StopIteration:
            return "데이터 보고 판단하자"
    emitted = []
    stream.run_stream("style", ctx, fake_llm, emitted.append, rounds=["opening"])

    payloads = [json.loads(e[6:].strip()) for e in emitted if e.startswith("data: ")]
    turns = [p for p in payloads if p.get("kind") == "turn"]
    assert len(turns) == 5                       # 5 style 페르소나, opening
    assert "mismatch" in [t["badge"] for t in turns]   # 틀린 TSLA 인용 잡힘
    assert payloads[-1]["kind"] == "done"        # 마지막 done 이벤트


def test_run_reply_stream_emits_user_replies():
    ctx = {"holdings": [{"symbol": "TSLA", "live_pnl_pct": -10.5, "price": {"value": 391.0}}]}
    emitted = []
    stream.run_reply_stream("style", ctx, "TSLA 더 살까?", lambda p: "답변", emitted.append)
    payloads = [json.loads(e[6:].strip()) for e in emitted if e.startswith("data: ")]
    turns = [p for p in payloads if p.get("kind") == "turn"]
    assert len(turns) == 5                        # 5 페르소나가 폐하께 응답
    assert all(t["round"] == "reply" for t in turns)
    assert payloads[-1]["kind"] == "done"
