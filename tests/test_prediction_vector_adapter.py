# tests/test_prediction_vector_adapter.py
import numpy as np
from corvin_jarvis.prediction import m_vector
from corvin_jarvis.prediction.contract import PredictionResult

FEATURES = ["kospi", "nasdaq", "vix"]


def _series(db_like, sym, vals):
    db_like[sym] = [{"date": f"2026-01-{i+1:02d}", "close": v} for i, v in enumerate(vals)]


def test_predict_returns_insufficient_when_too_few_days():
    closes = {f: [{"date": "2026-01-01", "close": 1.0}] for f in FEATURES}
    r = m_vector.predict(closes, features=FEATURES, min_days=250, k=12, horizon=5)
    assert isinstance(r, PredictionResult)
    assert r.data_ok is False


def test_predict_produces_market_verdict_with_enough_data():
    rng = np.random.default_rng(0)
    closes = {}
    for f in FEATURES:
        prices = (100 + np.cumsum(rng.normal(0, 1, 400))).tolist()
        closes[f] = [{"date": f"d{i}", "close": p} for i, p in enumerate(prices)]
    r = m_vector.predict(closes, features=FEATURES, min_days=250, k=12, horizon=5)
    assert r.system == "vector_analog"
    assert r.scope == "market"
    assert r.data_ok is True
    assert "win_rate" in r.evidence
