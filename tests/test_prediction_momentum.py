# tests/test_prediction_momentum.py
from corvin_jarvis.prediction import m_momentum
from corvin_jarvis.prediction.contract import PredictionResult


def test_uptrend_signals_bullish():
    closes = list(range(1, 200))  # 꾸준한 상승
    r = m_momentum.run_symbol("AAA", closes, short=20, long=120, min_days=120)
    assert isinstance(r, PredictionResult)
    assert r.evidence["signal"] == "bullish"
    assert r.data_ok is True


def test_short_history_insufficient():
    r = m_momentum.run_symbol("BBB", [1, 2, 3], short=20, long=120, min_days=120)
    assert r.data_ok is False
