# tests/test_prediction_vector.py
import numpy as np
from corvin_jarvis.prediction import m_vector


def test_zscore_normalizes_columns():
    m = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    z = m_vector.zscore_columns(m)
    assert abs(z[:, 0].mean()) < 1e-9
    assert abs(z[:, 0].std() - 1.0) < 1e-9


def test_top_k_analogs_finds_most_similar_rows():
    matrix = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [0.0, 1.0]])
    today = np.array([1.0, 0.0])
    idx, sims = m_vector.top_k_analogs(today, matrix, k=2)
    assert list(idx) == [0, 1]          # 가장 닮은 두 행
    assert sims[0] >= sims[1]


def test_forward_distribution_summarizes_returns():
    fwd = [0.02, -0.01, 0.03, 0.00]
    dist = m_vector.forward_distribution(fwd)
    assert abs(dist["mean"] - 0.01) < 1e-9
    assert dist["win_rate"] == 0.5      # >0 비율 (0은 미포함)
    assert dist["n"] == 4
