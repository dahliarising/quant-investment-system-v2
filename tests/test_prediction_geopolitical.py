# tests/test_prediction_geopolitical.py
from corvin_jarvis.prediction import m_geopolitical
from corvin_jarvis.prediction.contract import PredictionResult


def test_high_risk_maps_to_caution_verdict():
    payload = {"risk_score": 72, "trend": "rising", "top_event": "중동 긴장"}
    r = m_geopolitical.run(payload)
    assert isinstance(r, PredictionResult)
    assert r.scope == "market"
    assert r.evidence["risk_score"] == 72
    assert "주의" in r.verdict or "경계" in r.verdict


def test_none_payload_insufficient():
    r = m_geopolitical.run(None)
    assert r.data_ok is False
