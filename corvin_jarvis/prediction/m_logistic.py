# corvin_jarvis/prediction/m_logistic.py
"""시스템6 — numpy 로지스틱 방향 분류 (의존성 0, XGBoost 대체).

feature = 지수·매크로 일일수익률 z-score. label = base feature의 forward H일 상승여부.
오늘 벡터로 '다음 H일 상승확률 P%' 출력. 표본 부족 시 insufficient.
"""
from __future__ import annotations

import numpy as np

from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.3,
                 epochs: int = 600, l2: float = 1e-3) -> np.ndarray:
    """절편 포함 경사하강 로지스틱 회귀. w[0]=bias."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n
        grad[1:] += l2 * w[1:]          # 절편 미규제
        w -= lr * grad
    return w


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return sigmoid(Xb @ w)


def _returns(closes: list[dict]) -> list[float]:
    out = []
    for prev, cur in zip(closes[:-1], closes[1:]):
        p, c = prev["close"], cur["close"]
        out.append(c / p - 1.0 if (p and c and p > 0) else 0.0)
    return out


def _zscore(m: np.ndarray) -> np.ndarray:
    std = m.std(axis=0)
    return (m - m.mean(axis=0)) / np.where(std == 0, 1.0, std)


def predict_market(closes_by_feature: dict[str, list[dict]], *,
                   features: list[str], min_days: int = 250, horizon: int = 5
                   ) -> PredictionResult:
    series = {f: _returns(closes_by_feature.get(f, [])) for f in features}
    n = min((len(s) for s in series.values()), default=0)
    if n < min_days:
        return insufficient("logistic", "market",
                            f"공통 과거 {n}일 < 최소 {min_days}일")
    X = _zscore(np.column_stack([np.array(series[f][-n:]) for f in features]))
    base = np.array(series[features[0]][-n:])
    # forward H일 누적수익률 부호 라벨
    fwd = np.array([float(np.prod(1 + base[i + 1:i + 1 + horizon]) - 1)
                    for i in range(n - horizon)])
    y = (fwd > 0).astype(float)
    X_train = X[:n - horizon]
    if len(np.unique(y)) < 2:
        return insufficient("logistic", "market", "라벨 단일클래스(학습 불가)")
    w = fit_logistic(X_train, y)
    prob_up = float(predict_proba(X[-1:], w)[0])
    direction = "상승" if prob_up >= 0.5 else "하락"
    verdict = f"로지스틱: 다음 {horizon}일 상승확률 {prob_up*100:.0f}% ({direction} 우위)"
    conf = round(min(90.0, 50 + abs(prob_up - 0.5) * 80), 1)
    return PredictionResult("logistic", "market", verdict, conf,
                            {"prob_up": prob_up, "horizon": horizon,
                             "n_train": int(len(y))}, data_ok=True)
