"""
레짐 감지 시스템
- Daily RankIC 기반 시장 상태 판별
- Rolling Z-score로 WARMUP / NORMAL / WARNING / BREAKDOWN 분류
- 레짐 기반 포지션 규모 조절
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

RegimeLabel = Literal["WARMUP", "NORMAL", "WARNING", "BREAKDOWN"]

REGIME_COLORS = {
    "NORMAL": "#22c55e",      # green
    "WARNING": "#f59e0b",     # amber
    "BREAKDOWN": "#ef4444",   # red
    "WARMUP": "#94a3b8",      # gray
}

REGIME_EXPOSURE = {
    "NORMAL": 1.0,
    "WARNING": 0.5,
    "BREAKDOWN": 0.0,
    "WARMUP": 0.5,
}


def daily_rankic(
    panel: pd.DataFrame,
    score_col: str = "score",
    ret_col: str = "future_return",
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    날짜별 cross-sectional RankIC 계산

    Args:
        panel: date, ticker, score, future_return 포함 DataFrame
        score_col: 스코어 컬럼명
        ret_col: 미래수익 컬럼명

    Returns:
        DataFrame[date, rankic]
    """
    df = panel.dropna(subset=[score_col, ret_col]).copy()
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for d, sub in df.groupby("date"):
        if len(sub) < min_stocks:
            continue
        x = sub[score_col].rank(method="average")
        y = sub[ret_col].rank(method="average")
        r = np.corrcoef(x.values, y.values)[0, 1]
        rows.append({"date": d, "rankic": float(r)})

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def assign_regime(
    rankic_df: pd.DataFrame,
    lookback: int = 60,
    z_warning: float = -1.0,
    z_breakdown: float = -2.0,
) -> pd.DataFrame:
    """
    RankIC 시계열에서 레짐 분류

    Returns:
        DataFrame[date, rankic, mu, sd, z, regime, regime_color, exposure]
    """
    df = rankic_df.sort_values("date").copy()
    df["mu"] = df["rankic"].rolling(lookback).mean()
    df["sd"] = df["rankic"].rolling(lookback).std(ddof=0).replace(0, np.nan)
    df["z"] = (df["rankic"] - df["mu"]) / df["sd"]

    def _label(z):
        if pd.isna(z):
            return "WARMUP"
        if z <= z_breakdown:
            return "BREAKDOWN"
        if z <= z_warning:
            return "WARNING"
        return "NORMAL"

    df["regime"] = df["z"].map(_label)
    df["regime_color"] = df["regime"].map(REGIME_COLORS)
    df["exposure"] = df["regime"].map(REGIME_EXPOSURE)

    return df[["date", "rankic", "mu", "sd", "z", "regime", "regime_color", "exposure"]]


def detect_regime_collapse(
    rankic_series: pd.Series,
    window: int = 20,
    z_thresh: float = -2.0,
) -> pd.DataFrame:
    """
    레짐 붕괴 감지 (Rolling Z-score 기반)

    Returns:
        DataFrame[rankic, roll_mean, roll_std, z, collapse]
    """
    s = pd.to_numeric(rankic_series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=["rankic", "roll_mean", "roll_std", "z", "collapse"])

    rm = s.rolling(window).mean()
    rs = s.rolling(window).std(ddof=0).replace(0.0, np.nan)
    z = (s - rm) / rs

    df = pd.DataFrame({
        "rankic": s, "roll_mean": rm, "roll_std": rs, "z": z,
    }).dropna()
    df["collapse"] = df["z"] < z_thresh
    return df


def get_current_regime(rankic_df: pd.DataFrame, lookback: int = 60) -> dict:
    """현재 레짐 상태 반환"""
    regime_df = assign_regime(rankic_df, lookback=lookback)
    if regime_df.empty:
        return {"regime": "WARMUP", "z": np.nan, "exposure": 0.5, "color": REGIME_COLORS["WARMUP"]}

    latest = regime_df.iloc[-1]
    return {
        "regime": latest["regime"],
        "z": float(latest["z"]) if pd.notna(latest["z"]) else np.nan,
        "exposure": REGIME_EXPOSURE[latest["regime"]],
        "color": REGIME_COLORS[latest["regime"]],
        "date": latest["date"],
    }
