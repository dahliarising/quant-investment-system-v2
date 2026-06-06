"""신뢰도 가중 종합 테스트 — 높은 track record 페르소나 의견에 더 무게."""
from corvin_jarvis.live_debate import consensus


def test_weighted_consensus_high_cred_dominates():
    votes = [
        {"persona": "quant", "credibility": 92, "stance": -1.0},   # 강한 매도
        {"persona": "growth", "credibility": 52, "stance": 1.0},   # 강한 매수
    ]
    r = consensus.weighted_consensus(votes)
    assert r["score"] < 0                         # 고신뢰 퀀트(매도) 쪽으로 기움
    assert "매도" in r["label"]


def test_weighted_consensus_neutral_when_balanced():
    votes = [
        {"persona": "a", "credibility": 70, "stance": 1.0},
        {"persona": "b", "credibility": 70, "stance": -1.0},
    ]
    assert consensus.weighted_consensus(votes)["label"].startswith("중립")


def test_weighted_consensus_empty():
    assert consensus.weighted_consensus([])["score"] == 0.0
