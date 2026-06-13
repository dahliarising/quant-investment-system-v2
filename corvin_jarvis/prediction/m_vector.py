# corvin_jarvis/prediction/m_vector.py
"""시스템5 — 벡터 analog 예측 순수함수.

오늘 시장상태를 벡터로 인코딩 → 코사인 유사 과거 Top-K → forward 분포로 예측.
numpy만 사용 (외부 의존 0).
"""
from __future__ import annotations

import numpy as np

from corvin_jarvis.prediction.contract import PredictionResult, insufficient


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
    sims = np.dot(mn, tn)  # @ triggers spurious Accelerate BLAS RuntimeWarning on macOS
    order = np.argsort(-sims)[:k]
    return order, sims[order]


def forward_distribution(forward_returns: list[float]) -> dict:
    """analog 날들의 forward 수익률 요약."""
    arr = np.array(forward_returns, dtype=float)
    wins = int((arr > 0).sum())
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "win_rate": wins / len(arr) if len(arr) else 0.0, "n": len(arr)}


def _returns(closes: list[dict]) -> list[float]:
    out = []
    for prev, cur in zip(closes[:-1], closes[1:]):
        p, c = prev["close"], cur["close"]
        if p and c and p > 0:
            out.append(c / p - 1.0)
        else:
            out.append(0.0)
    return out


def predict(closes_by_feature: dict[str, list[dict]], *, features: list[str],
            min_days: int = 250, k: int = 12, horizon: int = 5,
            min_similarity: float = 0.0) -> PredictionResult:
    """각 feature의 일봉 close → 수익률 행렬 → 오늘 벡터 analog 예측."""
    series = {f: _returns(closes_by_feature.get(f, [])) for f in features}
    n = min((len(s) for s in series.values()), default=0)
    if n < min_days:
        return insufficient("vector_analog", "market",
                            f"공통 과거 {n}일 < 최소 {min_days}일")
    matrix = np.column_stack([np.array(series[f][-n:]) for f in features])
    z = zscore_columns(matrix)
    # 마지막 행 = 오늘, forward horizon 확보 위해 후보는 [0, n-horizon)
    today = z[-1]
    candidates = z[:n - horizon]
    if len(candidates) < k:
        return insufficient("vector_analog", "market", f"analog 후보 {len(candidates)} < k {k}")
    idx, sims = top_k_analogs(today, candidates, k)
    # idx와 sims는 top_k_analogs에서 위치로 짝지어 반환됨 → zip으로 각 후보를
    # 자기 유사도와 직접 페어링 (값 기준 .index() 조회의 오짝 방지).
    idx = [int(i) for i, s in zip(idx, sims) if s >= min_similarity]
    # forward = analog 날(i) 이후 horizon일 누적수익률 (kospi 기준 = features[0])
    base = np.array(series[features[0]][-n:])
    fwd = []
    for i in idx:
        window = base[i + 1:i + 1 + horizon]
        fwd.append(float(np.prod(1 + window) - 1) if len(window) else 0.0)
    dist = forward_distribution(fwd)
    direction = "상승" if dist["mean"] > 0 else "하락"
    conf = round(min(95.0, 50 + abs(dist["win_rate"] - 0.5) * 90), 1)
    verdict = (f"현재 국면과 닮은 과거 {dist['n']}개 → 다음 {horizon}일 "
               f"평균 {dist['mean']*100:+.1f}%, 승률 {dist['win_rate']*100:.0f}% ({direction} 우위)")
    return PredictionResult("vector_analog", "market", verdict, conf,
                            {**dist, "k": k, "horizon": horizon}, data_ok=True)
