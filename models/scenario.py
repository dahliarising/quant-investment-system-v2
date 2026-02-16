"""
시나리오 분석 모듈
- Geometric Brownian Motion (GBM) 기반 주가 경로 시뮬레이션
- Base / Optimistic / Pessimistic 3-시나리오
- 확률 분포 시각화용 데이터
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScenarioResult:
    ticker: str
    current_price: float
    pred_return: float
    annual_vol: float
    horizon_days: int
    paths: pd.DataFrame       # columns: base, optimistic, pessimistic
    percentiles: pd.DataFrame  # columns: p5, p25, p50, p75, p95
    summary: dict


def estimate_annual_vol(ohlcv: pd.DataFrame, window: int = 60) -> float:
    """OHLCV에서 연간 변동성 추정"""
    cols = [c.lower() for c in ohlcv.columns]
    if "close" in cols:
        close = ohlcv[ohlcv.columns[cols.index("close")]]
    elif "adj close" in cols:
        close = ohlcv[ohlcv.columns[cols.index("adj close")]]
    else:
        close = ohlcv.iloc[:, 3]

    daily_ret = close.pct_change().dropna().tail(window)
    if len(daily_ret) < 20:
        return 0.25  # 기본값
    return float(daily_ret.std() * np.sqrt(252))


def simulate_gbm_paths(
    current_price: float,
    annual_drift: float,
    annual_vol: float,
    horizon_days: int = 252,
    n_paths: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    GBM 시뮬레이션

    Returns:
        ndarray shape (horizon_days+1, n_paths), 각 경로의 가격
    """
    rng = np.random.RandomState(seed)
    dt = 1 / 252
    mu = annual_drift
    sigma = annual_vol

    # 로그 정규 경로
    z = rng.standard_normal((horizon_days, n_paths))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_prices = np.vstack([np.zeros(n_paths), np.cumsum(log_ret, axis=0)])
    prices = current_price * np.exp(log_prices)
    return prices


def make_scenario_paths(
    current_price: float,
    pred_return: float,
    annual_vol: float,
    horizon_days: int = 252,
    n_paths: int = 1000,
    seed: int = 42,
) -> ScenarioResult:
    """
    3-시나리오 분석 실행

    Args:
        current_price: 현재가
        pred_return: 예측 수익률 (horizon 기준)
        annual_vol: 연간 변동성
        horizon_days: 투자 기간 (일)

    Returns:
        ScenarioResult with paths, percentiles, summary
    """
    # drift 추정: pred_return -> annual drift
    annual_drift = pred_return * (252 / horizon_days) if horizon_days != 252 else pred_return

    prices = simulate_gbm_paths(
        current_price, annual_drift, annual_vol,
        horizon_days=horizon_days, n_paths=n_paths, seed=seed,
    )

    # 3-시나리오 경로 (대표 경로)
    base_path = current_price * np.exp(
        np.cumsum(np.full(horizon_days + 1, (annual_drift - 0.5 * annual_vol**2) / 252))
    )
    base_path[0] = current_price

    optimistic_path = current_price * np.exp(
        np.cumsum(np.full(horizon_days + 1, (annual_drift + annual_vol - 0.5 * annual_vol**2) / 252))
    )
    optimistic_path[0] = current_price

    pessimistic_path = current_price * np.exp(
        np.cumsum(np.full(horizon_days + 1, (annual_drift - annual_vol - 0.5 * annual_vol**2) / 252))
    )
    pessimistic_path[0] = current_price

    days = np.arange(horizon_days + 1)
    paths_df = pd.DataFrame({
        "day": days,
        "base": base_path,
        "optimistic": optimistic_path,
        "pessimistic": pessimistic_path,
    })

    # 확률 분포 (각 시점의 percentile)
    pcts = pd.DataFrame({
        "day": days,
        "p5": np.percentile(prices, 5, axis=1),
        "p25": np.percentile(prices, 25, axis=1),
        "p50": np.percentile(prices, 50, axis=1),
        "p75": np.percentile(prices, 75, axis=1),
        "p95": np.percentile(prices, 95, axis=1),
    })

    # 만기 가격 분포
    final_prices = prices[-1, :]
    summary = {
        "current_price": current_price,
        "pred_return": pred_return,
        "base_target": float(base_path[-1]),
        "optimistic_target": float(optimistic_path[-1]),
        "pessimistic_target": float(pessimistic_path[-1]),
        "median_target": float(np.median(final_prices)),
        "mean_target": float(np.mean(final_prices)),
        "prob_profit": float((final_prices > current_price).mean()),
        "prob_loss_10pct": float((final_prices < current_price * 0.9).mean()),
        "var_95": float(np.percentile(final_prices, 5)),
        "expected_return": float(np.mean(final_prices) / current_price - 1),
    }

    return ScenarioResult(
        ticker="",
        current_price=current_price,
        pred_return=pred_return,
        annual_vol=annual_vol,
        horizon_days=horizon_days,
        paths=paths_df,
        percentiles=pcts,
        summary=summary,
    )
