"""순수 기술지표 — 외부 의존 없음, 리스트 in → 스칼라 out."""
from __future__ import annotations


def sma(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def rsi(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for prev, cur in zip(prices[-(period + 1):-1], prices[-period:]):
        diff = cur - prev
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def recent_high(prices: list[float], window: int = 252) -> float | None:
    if not prices:
        return None
    return max(prices[-window:])
