"""Corvin 선행 인텔리전스 — 시장 거래량 fetch 어댑터.

KR=pykrx OHLCV, US=yfinance history. 실패 시 (None, None).
⚠️ yfinance는 US만 (메모리 feedback_no_yfinance_kr). KR은 pykrx.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

log = logging.getLogger("corvin.market_data")


def _is_kr(symbol: str) -> bool:
    return symbol.isdigit() and len(symbol) == 6


def fetch_volume(symbol: str) -> tuple[float | None, float | None]:
    """(최근 거래량, 20일평균). 실패 시 (None, None)."""
    try:
        if _is_kr(symbol):
            return _fetch_kr_volume(symbol)
        return _fetch_us_volume(symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("거래량 fetch 실패 %s: %s", symbol, e)
        return None, None


def _fetch_kr_volume(code: str) -> tuple[float | None, float | None]:
    from pykrx import stock

    from corvin_jarvis.signals import ensemble
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=40)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv_by_date(start, end, code)
    if df.empty:
        return None, None
    vols = [float(v) for v in df["거래량"].tolist()]
    return ensemble.volume_stats(vols)


def _fetch_us_volume(symbol: str) -> tuple[float | None, float | None]:
    import yfinance as yf

    from corvin_jarvis.signals import ensemble
    hist = yf.Ticker(symbol).history(period="2mo")
    if hist.empty or "Volume" not in hist:
        return None, None
    vols = [float(v) for v in hist["Volume"].tolist()]
    return ensemble.volume_stats(vols)
