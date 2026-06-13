# corvin_jarvis/prediction/m_vector.py
"""시스템5 — 벡터 analog 예측 순수함수.

오늘 시장상태를 벡터로 인코딩 → 코사인 유사 과거 Top-K → forward 분포로 예측.
numpy만 사용 (외부 의존 0).
"""
from __future__ import annotations

import numpy as np


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """열(feature)별 z-score 정규화. std=0 열은 0으로."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std_safe


def top_k_analogs(today: np.ndarray, matrix: np.ndarray, k: int):
    """today 벡터 vs matrix 각 행 코사인 유사도 → 상위 k 인덱스·유사도."""
    eps = 1e-12
    tn = today / (np.linalg.norm(today) + eps)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + eps)
    sims = mn @ tn
    order = np.argsort(-sims)[:k]
    return order, sims[order]


def forward_distribution(forward_returns: list[float]) -> dict:
    """analog 날들의 forward 수익률 요약."""
    arr = np.array(forward_returns, dtype=float)
    wins = int((arr > 0).sum())
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "win_rate": wins / len(arr) if len(arr) else 0.0, "n": len(arr)}
