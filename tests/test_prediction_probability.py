# tests/test_prediction_probability.py
import statistics
from corvin_jarvis.prediction import m_probability
from corvin_jarvis.prediction.contract import PredictionResult


def test_probability_result_for_symbol_with_history():
    closes = [100, 101, 99, 102, 98, 103, 97, 104]  # 변동 있음
    r = m_probability.run_symbol("AAA", closes, stop=90.0, horizon_days=5, min_days=5)
    assert isinstance(r, PredictionResult)
    assert r.system == "probability"
    assert r.data_ok is True
    assert 0.0 <= r.evidence["prob_below_stop"] <= 1.0


def test_probability_insufficient_when_short():
    r = m_probability.run_symbol("BBB", [100], stop=90.0, horizon_days=5, min_days=5)
    assert r.data_ok is False


def test_zero_stop_insufficient():
    r = m_probability.run_symbol("AAA", [100, 101, 99, 102, 98, 103], stop=0.0)
    assert r.data_ok is False
