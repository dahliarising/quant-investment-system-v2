# corvin_jarvis/prediction/m_probability.py
"""시스템2 어댑터 — predict.probability_below 재사용 (log-normal)."""
from __future__ import annotations

import math
import statistics

from corvin_jarvis import predict
from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def _log_returns(closes: list[float]) -> list[float]:
    out = []
    for p, c in zip(closes[:-1], closes[1:]):
        if p > 0 and c > 0:
            out.append(math.log(c / p))
    return out


def run_symbol(symbol: str, closes: list[float], stop: float,
               horizon_days: int = 5, min_days: int = 20) -> PredictionResult:
    if stop is None or stop <= 0.0:
        return insufficient("probability", symbol, "손절가 미설정(stop<=0)")
    if len(closes) < min_days:
        return insufficient("probability", symbol, f"일봉 {len(closes)} < {min_days}")
    rets = _log_returns(closes)
    if len(rets) < 2:
        return insufficient("probability", symbol, "수익률 표본 부족")
    mu = statistics.mean(rets)
    sigma = statistics.stdev(rets)
    price = closes[-1]
    prob = predict.probability_below(price, stop, mu, sigma, horizon_days)
    verdict = f"{horizon_days}일 내 손절가({stop:g}) 이탈 확률 {prob*100:.0f}%"
    conf = round(60 + abs(prob - 0.5) * 60, 1)
    return PredictionResult("probability", symbol, verdict, conf,
                            {"prob_below_stop": prob, "mu": mu, "sigma": sigma,
                             "price": price, "horizon_days": horizon_days}, data_ok=True)
