# tests/test_prediction_ensemble.py
"""시스템10 — 백테스트 게이트 존중 합의 앙상블."""
from corvin_jarvis.prediction import m_ensemble
from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def test_consensus_all_up():
    c = m_ensemble.consensus([("a", 1, 1.0), ("b", 1, 1.0), ("c", 1, 1.0)])
    assert c["direction"] == "up"
    assert c["agreement"] == 1.0
    assert c["n"] == 3


def test_consensus_split_is_neutral():
    c = m_ensemble.consensus([("a", 1, 1.0), ("b", -1, 1.0)])
    assert c["direction"] == "neutral"


def test_run_builds_market_consensus():
    results = [
        PredictionResult("vector_analog", "market", "상승 우위", 70,
                         {"mean": 0.02}, True),
        PredictionResult("logistic", "market", "상승확률 65%", 65,
                         {"prob_up": 0.65}, True),
        PredictionResult("sentiment", "market", "심리 상방", 60, {"tone": 1.5}, True),
    ]
    r = m_ensemble.run(results, gate={"logistic": {"passed": True}})
    assert r.system == "ensemble"
    assert r.scope == "market"
    assert r.data_ok is True
    assert r.evidence["direction"] == "up"


def test_gated_system_excluded_when_not_passed():
    results = [
        PredictionResult("logistic", "market", "상승확률 90%", 90,
                         {"prob_up": 0.9}, True),
    ]
    # 게이트 미통과 → logistic 표 제외 → 합의 불가
    r = m_ensemble.run(results, gate={"logistic": {"passed": False}})
    assert r.data_ok is False


def test_run_insufficient_when_no_votes():
    r = m_ensemble.run([], gate={})
    assert r.data_ok is False


def test_neutral_sentiment_abstains():
    """중립 sentiment(|tone|<0.5)는 합의에서 기권 — 가짜 합의 방지."""
    results = [
        PredictionResult("vector_analog", "market", "상승", 70, {"mean": 0.02}, True),
        PredictionResult("sentiment", "market", "심리 중립", 53, {"tone": 0.2}, True),
    ]
    r = m_ensemble.run(results, gate={})
    assert r.data_ok is False   # vector 1표만 남아 합의 불가


def test_strong_sentiment_votes():
    """뚜렷한 sentiment(|tone|>=0.5)는 정상 투표."""
    results = [
        PredictionResult("vector_analog", "market", "상승", 70, {"mean": 0.02}, True),
        PredictionResult("sentiment", "market", "심리 상방", 60, {"tone": 0.9}, True),
    ]
    r = m_ensemble.run(results, gate={})
    assert r.data_ok is True
    assert r.evidence["direction"] == "up"


def test_single_vote_is_not_consensus():
    """1표는 합의 아님 — 과신 방지."""
    results = [PredictionResult("vector_analog", "market", "상승", 70,
                                {"mean": 0.02}, True)]
    r = m_ensemble.run(results, gate={})
    assert r.data_ok is False
