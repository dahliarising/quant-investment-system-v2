"""
레이블 생성 모듈
- 미래 수익률 (30일, 1년)
- Alpha vs SPY (벤치마크 대비 초과수익)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    cols = [str(c).strip().lower() for c in df.columns]
    if "close" in cols:
        s = df[df.columns[cols.index("close")]].astype(float)
    elif "adj close" in cols:
        s = df[df.columns[cols.index("adj close")]].astype(float)
    else:
        s = df.iloc[:, 0].astype(float)
    s.name = "close"
    return s


def _future_return(close: pd.Series, horizon_days: int) -> pd.Series:
    r = close.shift(-horizon_days) / close - 1.0
    r.name = f"fwd_ret_{horizon_days}d"
    return r


def make_future_return_30d(ohlcv: pd.DataFrame) -> pd.Series:
    return _future_return(_get_close(ohlcv), 30)


def make_future_return_1y(ohlcv: pd.DataFrame) -> pd.Series:
    return _future_return(_get_close(ohlcv), 252)


def make_future_return_custom(ohlcv: pd.DataFrame, days: int) -> pd.Series:
    return _future_return(_get_close(ohlcv), days)


def make_future_return_alpha(
    ohlcv_ticker: pd.DataFrame,
    ohlcv_benchmark: pd.DataFrame,
    horizon_days: int = 30,
) -> pd.Series:
    """Alpha = ticker 미래수익 - benchmark 미래수익"""
    c_t = _get_close(ohlcv_ticker)
    c_b = _get_close(ohlcv_benchmark)
    rt = _future_return(c_t, horizon_days)
    rb = _future_return(c_b, horizon_days)
    rt2, rb2 = rt.align(rb, join="inner")
    alpha = rt2 - rb2
    alpha.name = f"alpha_{horizon_days}d"
    return alpha


def make_future_return_30d_alpha_vs_spy(
    ohlcv_ticker: pd.DataFrame, ohlcv_spy: pd.DataFrame,
) -> pd.Series:
    alpha = make_future_return_alpha(ohlcv_ticker, ohlcv_spy, 30)
    alpha.name = "alpha_30d_vs_spy"
    return alpha


def make_future_return_1y_alpha_vs_spy(
    ohlcv_ticker: pd.DataFrame, ohlcv_spy: pd.DataFrame,
) -> pd.Series:
    alpha = make_future_return_alpha(ohlcv_ticker, ohlcv_spy, 252)
    alpha.name = "alpha_1y_vs_spy"
    return alpha
