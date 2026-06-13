# corvin_jarvis/prediction/m_montecarlo.py
"""시스템9 — GBM 몬테카를로 분포 예측 (numpy 전용, 의존성 0).

일일 로그수익률 μ,σ 추정 → H일 경로 N개 시뮬 → 종가 분포(p5/p50/p95) + 손절이탈 확률.
표본 부족 시 insufficient (가짜 정밀도 금지).
"""
from __future__ import annotations

import math

import numpy as np

from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def log_returns(closes: list[float]) -> np.ndarray:
    """양수 종가쌍의 일일 로그수익률."""
    a = np.asarray(closes, dtype=float)
    pairs = (a[:-1] > 0) & (a[1:] > 0)
    return np.log(a[1:][pairs] / a[:-1][pairs])


def simulate_terminal_returns(mu: float, sigma: float, horizon: int,
                              n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """H일 누적 로그수익률(=일일 N(mu,sigma) 합) → 단순수익률 분포."""
    daily = rng.normal(mu, sigma, size=(n_paths, horizon))
    terminal_log = daily.sum(axis=1)
    return np.exp(terminal_log) - 1.0


def run_symbol(symbol: str, closes: list[float], stop: float | None = None,
               horizon: int = 21, n_paths: int = 2000, min_days: int = 60,
               rng: np.random.Generator | None = None) -> PredictionResult:
    if len(closes) < min_days:
        return insufficient("montecarlo", symbol, f"일봉 {len(closes)} < {min_days}")
    rets = log_returns(closes)
    if len(rets) < 2 or float(np.std(rets)) == 0.0:
        return insufficient("montecarlo", symbol, "수익률 표본/변동 부족")
    rng = rng if rng is not None else np.random.default_rng()
    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1))
    price = float(closes[-1])
    term = simulate_terminal_returns(mu, sigma, horizon, n_paths, rng)
    p5, p50, p95 = (float(np.percentile(term, q) * 100) for q in (5, 50, 95))
    prob_below = None
    if stop is not None and stop > 0:
        prices = price * (1.0 + term)
        prob_below = float(np.mean(prices < stop))
    verdict = (f"몬테카를로 {horizon}일: 중앙 {p50:+.1f}%, "
               f"5~95% [{p5:+.1f}%, {p95:+.1f}%]")
    if prob_below is not None:
        verdict += f", 손절이탈 {prob_below*100:.0f}%"
    conf = round(min(90.0, 55 + abs(p50) * 2.0), 1)
    return PredictionResult("montecarlo", symbol, verdict, conf,
                            {"p5": p5, "p50": p50, "p95": p95, "mu": mu,
                             "sigma": sigma, "horizon": horizon, "n_paths": n_paths,
                             "prob_below_stop": prob_below}, data_ok=True)
