# tests/test_prediction_logistic.py
"""시스템6 — numpy 로지스틱 방향 분류 (의존성 0, XGBoost 대체)."""
import numpy as np

from corvin_jarvis.prediction import m_logistic
from corvin_jarvis.prediction.contract import PredictionResult

FEATURES = ["kospi", "nasdaq", "vix"]


def test_sigmoid_bounded():
    z = np.array([-1e3, 0.0, 1e3])
    s = m_logistic.sigmoid(z)
    assert 0.0 <= s[0] < 1e-6
    assert abs(s[1] - 0.5) < 1e-9
    assert 1.0 - 1e-6 < s[2] <= 1.0


def test_fit_learns_separable_pattern():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)  # 선형 분리 가능
    w = m_logistic.fit_logistic(X, y, lr=0.5, epochs=800)
    proba = m_logistic.predict_proba(X, w)
    acc = float(np.mean((proba > 0.5) == (y > 0.5)))
    assert acc > 0.9


def test_fit_stable_on_heavy_tailed_features():
    """두꺼운 꼬리 아웃라이어에도 발산하지 않고 유한 w (matmul inf/NaN 방지)."""
    import warnings
    rng = np.random.default_rng(0)
    X = rng.standard_t(2, (500, 3)) * 4  # 극단 아웃라이어 포함
    y = (X[:, 0] > 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 어떤 RuntimeWarning도 실패로
        w = m_logistic.fit_logistic(X, y)
    assert np.isfinite(w).all()


def test_predict_market_insufficient_when_short():
    closes = {f: [{"date": "2026-01-01", "close": 1.0}] for f in FEATURES}
    r = m_logistic.predict_market(closes, features=FEATURES, min_days=250)
    assert isinstance(r, PredictionResult)
    assert r.data_ok is False


def test_predict_market_probability_in_range():
    rng = np.random.default_rng(1)
    closes = {}
    for f in FEATURES:
        prices = (100 + np.cumsum(rng.normal(0, 1, 400))).tolist()
        closes[f] = [{"date": f"d{i}", "close": p} for i, p in enumerate(prices)]
    r = m_logistic.predict_market(closes, features=FEATURES, min_days=250,
                                  horizon=5)
    assert r.system == "logistic"
    assert r.scope == "market"
    assert r.data_ok is True
    assert 0.0 <= r.evidence["prob_up"] <= 1.0
