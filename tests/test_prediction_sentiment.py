# tests/test_prediction_sentiment.py
"""시스템8 — GDELT 센티먼트 주입식 변환 (Phase 1 geo 패턴)."""
from corvin_jarvis.prediction import m_sentiment
from corvin_jarvis.prediction.contract import PredictionResult


def test_none_payload_insufficient():
    assert m_sentiment.run(None).data_ok is False
    assert m_sentiment.run({}).data_ok is False


def test_positive_tone_bullish():
    r = m_sentiment.run({"tone": 2.5, "article_volume": 1200, "trend": "rising"})
    assert isinstance(r, PredictionResult)
    assert r.scope == "market"
    assert r.evidence["tone"] == 2.5
    assert "긍정" in r.verdict or "상방" in r.verdict


def test_negative_tone_bearish():
    r = m_sentiment.run({"tone": -3.0, "article_volume": 900})
    assert "부정" in r.verdict or "하방" in r.verdict


def test_neutral_tone():
    r = m_sentiment.run({"tone": 0.1})
    assert r.data_ok is True
    assert "중립" in r.verdict
