# tests/test_prediction_contract.py
from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def test_result_fields_and_to_dict():
    r = PredictionResult(system="velocity", scope="META", verdict="손절 근접",
                         confidence=65.0, evidence={"days_to_stop": 4}, data_ok=True)
    d = r.to_dict()
    assert d["system"] == "velocity"
    assert d["scope"] == "META"
    assert d["data_ok"] is True
    assert d["evidence"]["days_to_stop"] == 4


def test_insufficient_helper_sets_data_ok_false():
    r = insufficient(system="vector_analog", scope="market", reason="과거 250일 미만")
    assert r.data_ok is False
    assert r.confidence == 0.0
    assert "데이터 부족" in r.verdict
    assert r.evidence["reason"] == "과거 250일 미만"
