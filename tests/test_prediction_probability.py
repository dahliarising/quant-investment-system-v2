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


def test_large_kr_stop_no_scientific_notation():
    """KR 원화 대형 손절가가 1e+06 과학표기로 나오면 안 됨 (오해 유발)."""
    closes = [1_010_000, 1_005_000, 1_000_500, 1_002_000, 998_000, 1_001_000]
    r = m_probability.run_symbol("012450", closes, stop=1_000_000.0, min_days=5)
    assert r.data_ok is True
    assert "e+" not in r.verdict
    assert "1,000,000" in r.verdict


def test_us_stop_keeps_decimal():
    """US 소형 가격은 천단위 콤마 없이, 의미있는 소수는 보존."""
    closes = [200, 198, 202, 197, 203, 199]
    r = m_probability.run_symbol("NVDA", closes, stop=189.5, min_days=5)
    assert "189.5" in r.verdict
    assert "e+" not in r.verdict
