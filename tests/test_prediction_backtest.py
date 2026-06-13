# tests/test_prediction_backtest.py
"""Phase 2 백테스트 게이트 — walk-forward 검증 + 통과 판정 (안전 척추)."""
import numpy as np

from corvin_jarvis.prediction import backtest


def test_directional_score_perfect_predictor_passes():
    preds = [(0.9, 1), (0.8, 1), (0.1, 0), (0.2, 0)] * 10
    s = backtest.directional_score(preds, margin=0.03)
    assert s["hit_rate"] == 1.0
    assert s["passed"] is True
    assert s["n"] == 40


def test_directional_score_coinflip_fails():
    # 예측이 실현과 무관(절반만 맞음) → baseline 못 넘어 탈락
    preds = [(0.6, 1), (0.6, 0)] * 25  # 항상 up 예측, 실제 50:50 → hit 0.5
    s = backtest.directional_score(preds, margin=0.03)
    assert s["passed"] is False


def test_directional_score_empty_not_passed():
    s = backtest.directional_score([], margin=0.03)
    assert s["n"] == 0 and s["passed"] is False


def test_coverage_score_in_tolerance_passes():
    flags = [True] * 80 + [False] * 20  # coverage 0.8
    s = backtest.coverage_score(flags, target=0.8, tol=0.1)
    assert s["passed"] is True
    assert abs(s["coverage"] - 0.8) < 1e-9


def test_coverage_score_out_of_tolerance_fails():
    flags = [True] * 40 + [False] * 60  # coverage 0.4 vs target 0.8
    s = backtest.coverage_score(flags, target=0.8, tol=0.1)
    assert s["passed"] is False


def test_walk_forward_logistic_runs_on_synthetic():
    # base feature가 다른 feature로 예측가능한 구조 → 동작 + 통과여부 키 존재
    rng = np.random.default_rng(0)
    n = 400
    lead = np.cumsum(rng.normal(0, 1, n))
    closes_by_feature = {
        "base": [{"date": f"d{i}", "close": 100 + lead[i]} for i in range(n)],
        "f2": [{"date": f"d{i}", "close": 100 + lead[i]} for i in range(n)],
    }
    s = backtest.walk_forward_logistic(closes_by_feature, features=["base", "f2"],
                                       horizon=5, min_train=250, step=20)
    assert "hit_rate" in s and "passed" in s and s["n"] >= 1
