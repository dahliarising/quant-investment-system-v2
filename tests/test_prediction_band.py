# tests/test_prediction_band.py
"""시스템7 — 경험적 분위수 가격 밴드 (Prophet/LSTM 대체, 의존성 0)."""
import numpy as np

from corvin_jarvis.prediction import m_band
from corvin_jarvis.prediction.contract import PredictionResult


def test_insufficient_when_short():
    r = m_band.run_symbol("AAA", [100, 101, 102], horizon=21, min_days=120)
    assert r.data_ok is False


def test_band_low_le_mid_le_high():
    rng = np.random.default_rng(0)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))).tolist()
    r = m_band.run_symbol("AAA", closes, horizon=21, min_days=120)
    assert r.system == "band"
    assert r.data_ok is True
    e = r.evidence
    assert e["low"] <= e["mid"] <= e["high"]
    assert "e+" not in r.verdict


def test_large_kr_price_no_scientific():
    rng = np.random.default_rng(1)
    closes = (1_000_000 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))).tolist()
    r = m_band.run_symbol("012450", closes, horizon=21, min_days=120)
    assert "e+" not in r.verdict
    assert "," in r.verdict  # 천단위 콤마
