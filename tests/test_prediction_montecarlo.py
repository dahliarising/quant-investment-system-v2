# tests/test_prediction_montecarlo.py
"""시스템9 — GBM 몬테카를로 분포 (numpy 전용, 의존성 0)."""
import numpy as np

from corvin_jarvis.prediction import m_montecarlo
from corvin_jarvis.prediction.contract import PredictionResult


def test_insufficient_when_short():
    r = m_montecarlo.run_symbol("AAA", [100, 101, 102], stop=90.0, min_days=60)
    assert isinstance(r, PredictionResult)
    assert r.data_ok is False


def test_distribution_quantiles_ordered():
    rng = np.random.default_rng(0)
    closes = (100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 300)))).tolist()
    r = m_montecarlo.run_symbol("AAA", closes, stop=50.0, min_days=60,
                                n_paths=1000, rng=np.random.default_rng(1))
    assert r.system == "montecarlo"
    assert r.data_ok is True
    e = r.evidence
    assert e["p5"] <= e["p50"] <= e["p95"]
    assert 0.0 <= e["prob_below_stop"] <= 1.0
    assert "e+" not in r.verdict


def test_stop_far_above_price_gives_high_prob_below():
    rng = np.random.default_rng(2)
    closes = (100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300)))).tolist()
    price = closes[-1]
    r = m_montecarlo.run_symbol("AAA", closes, stop=price * 1.5, min_days=60,
                                n_paths=2000, rng=np.random.default_rng(3))
    assert r.evidence["prob_below_stop"] > 0.5


def test_deterministic_with_seeded_rng():
    closes = (100 * np.exp(np.cumsum(
        np.random.default_rng(4).normal(0, 0.01, 200)))).tolist()
    r1 = m_montecarlo.run_symbol("AAA", closes, stop=80.0, min_days=60,
                                 n_paths=500, rng=np.random.default_rng(7))
    r2 = m_montecarlo.run_symbol("AAA", closes, stop=80.0, min_days=60,
                                 n_paths=500, rng=np.random.default_rng(7))
    assert r1.evidence["p50"] == r2.evidence["p50"]
