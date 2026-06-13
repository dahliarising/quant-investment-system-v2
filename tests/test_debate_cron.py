"""선제 토론 크론 — 폐하가 물을 법한 질문에 미리 토론 → Telegram 다이제스트.

순수 코어(run_anticipatory·format_digest)는 LLM=DI, 실발송 없음.
"""
from corvin_jarvis import debate_cron


def test_likely_questions_nonempty():
    assert len(debate_cron.LIKELY_QUESTIONS) >= 3
    assert all(isinstance(q, str) and q for q in debate_cron.LIKELY_QUESTIONS)


def test_run_anticipatory_sorts_replies_by_credibility():
    qs = ["지금 매수?", "손절?"]
    res = debate_cron.run_anticipatory(qs, {"c": 1}, lambda p: "내 답변", mode="style")
    assert [r["question"] for r in res] == qs
    creds = [rep["credibility"] for rep in res[0]["replies"]]
    assert creds == sorted(creds, reverse=True)         # 신뢰도 높은 순(가장 믿을 의견 먼저)


def test_format_digest_scannable_and_bounded():
    results = [{"question": "지금 매수?", "replies": [
        {"name": "퀀트", "tag": "시먼스", "credibility": 92, "text": "vix_term만 믿어라"},
        {"name": "가치투자자", "tag": "버핏", "credibility": 88, "text": "우량주는 보유"},
        {"name": "성장투자자", "tag": "우드", "credibility": 52, "text": "줍줍 기회"},
    ]}]
    d = debate_cron.format_digest(results, top_n=2)
    assert "지금 매수?" in d
    assert "퀀트" in d and "92" in d                     # 최고신뢰 의견 포함
    assert "우드" not in d                               # top_n=2라 제외
    assert len(d) < 4000                                 # Telegram 한계 내
