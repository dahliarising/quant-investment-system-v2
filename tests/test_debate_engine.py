"""실시간 토론 툴 — Phase B: 페르소나 + 토론 엔진 테스트.

LLM은 DI(fake) — 네트워크 없이 라운드 오케스트레이션 검증.
"""
from corvin_jarvis.live_debate import engine, personas


def test_two_persona_modes():
    assert {p["id"] for p in personas.STANCE} == {"cut", "buy", "hold"}
    assert {p["id"] for p in personas.STYLE} == {"value", "growth", "trend", "macro", "quant"}
    for p in personas.STANCE + personas.STYLE:
        assert p["name"] and p["avatar"] and p["system"]   # 필수 필드


def test_run_round_one_turn_per_persona():
    calls = []
    def fake_llm(prompt):
        calls.append(prompt)
        return "테스트 발언"
    ps = personas.STANCE
    turns = engine.run_round(ps, {"x": 1}, "opening", fake_llm)
    assert len(turns) == len(ps) == len(calls)
    assert {t["persona"] for t in turns} == {p["id"] for p in ps}
    assert all(t["round"] == "opening" and t["text"] == "테스트 발언" for t in turns)


def test_debate_yields_rounds_in_order():
    turns = list(engine.debate("style", {"c": 1}, lambda p: "x",
                               rounds=["opening", "rebuttal"]))
    rounds = [t["round"] for t in turns]
    assert rounds.count("opening") == 5 and rounds.count("rebuttal") == 5
    last_opening = max(i for i, r in enumerate(rounds) if r == "opening")
    assert rounds.index("rebuttal") > last_opening      # 모든 opening 후 rebuttal


def test_respond_to_user_replies_per_persona_with_question():
    captured = []
    def fake_llm(prompt):
        captured.append(prompt)
        return "폐하께 답변"
    turns = list(engine.respond_to_user("style", {"c": 1}, "TSLA 더 사도 돼?", fake_llm))
    assert len(turns) == 5                                 # 페르소나마다 1 응답
    assert all(t["round"] == "reply" for t in turns)
    assert all("TSLA 더 사도 돼?" in p for p in captured)   # 질문이 프롬프트에 주입됨
    assert all("폐하" in p for p in captured)              # 폐하 발언으로 제시


def test_prompt_includes_context_and_persona():
    captured = {}
    def fake_llm(prompt):
        captured["p"] = prompt
        return "ok"
    engine.run_round([personas.STYLE[0]], {"note": "코스피 -5.5%"}, "opening", fake_llm)
    assert "코스피 -5.5%" in captured["p"]              # 컨텍스트 주입
    assert personas.STYLE[0]["system"][:10] in captured["p"]  # 페르소나 주입
