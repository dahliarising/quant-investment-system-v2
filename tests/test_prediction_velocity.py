# tests/test_prediction_velocity.py
from corvin_jarvis.prediction import m_velocity
from corvin_jarvis.prediction.contract import PredictionResult


def test_downtrend_produces_velocity_result():
    holdings = [{"symbol": "AAA", "price": 100.0}]
    stops = {"AAA": 95.0}
    # 명확한 하락 추세 closes
    closes = {"AAA": [120, 116, 112, 108, 104, 100]}
    results = m_velocity.run(holdings, stops, closes, regime_trend="down")
    assert all(isinstance(r, PredictionResult) for r in results)
    assert any(r.system == "velocity" and r.scope == "AAA" for r in results)


def test_no_stop_yields_insufficient_per_symbol():
    holdings = [{"symbol": "BBB", "price": 50.0}]
    results = m_velocity.run(holdings, stops={}, closes_by_sym={"BBB": [50, 50]},
                             regime_trend="down")
    assert results and results[0].data_ok is False
