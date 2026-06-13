# corvin_jarvis/prediction/m_band.py
"""시스템7 — 경험적 분위수 가격 밴드 (Prophet/LSTM 대체, 의존성 0).

과거 H일 forward 수익률 경험분포의 분위수를 현재가에 곱해 [low, mid, high] 밴드.
표본 부족 시 insufficient.
"""
from __future__ import annotations

import numpy as np

from corvin_jarvis.prediction.contract import PredictionResult, insufficient
from corvin_jarvis.prediction.fmt import fmt_price

_MIN_FWD = 30


def forward_returns(closes: list[float], horizon: int) -> np.ndarray:
    a = np.asarray(closes, dtype=float)
    out = [a[i + horizon] / a[i] - 1.0
           for i in range(len(a) - horizon) if a[i] > 0]
    return np.asarray(out, dtype=float)


def run_symbol(symbol: str, closes: list[float], horizon: int = 21,
               min_days: int = 120, low_q: float = 10.0,
               high_q: float = 90.0) -> PredictionResult:
    if len(closes) < min_days:
        return insufficient("band", symbol, f"일봉 {len(closes)} < {min_days}")
    fwd = forward_returns(closes, horizon)
    if len(fwd) < _MIN_FWD:
        return insufficient("band", symbol, f"forward 표본 {len(fwd)} < {_MIN_FWD}")
    price = float(closes[-1])
    low = price * (1 + np.percentile(fwd, low_q))
    mid = price * (1 + np.percentile(fwd, 50))
    high = price * (1 + np.percentile(fwd, high_q))
    verdict = (f"{horizon}일 밴드 [{fmt_price(low)} ~ {fmt_price(high)}] "
               f"중앙 {fmt_price(mid)}")
    width = (high - low) / price if price else 0.0
    conf = round(min(85.0, 45 + (1 - min(width, 1.0)) * 40), 1)
    return PredictionResult("band", symbol, verdict, conf,
                            {"low": float(low), "mid": float(mid),
                             "high": float(high), "horizon": horizon,
                             "low_q": low_q, "high_q": high_q}, data_ok=True)
