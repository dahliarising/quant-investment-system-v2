"""
팩터 스코어링 시스템
- Momentum / Value / Quality / Risk 4대 팩터
- Winsorize + Z-score 정규화
- 섹터 중립화 옵션
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.dropna().empty:
        return s2
    lo, hi = s2.quantile(lower), s2.quantile(upper)
    return s2.clip(lo, hi)


def _zscore(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    mu, sd = s2.mean(skipna=True), s2.std(skipna=True)
    if sd is None or pd.isna(sd) or sd == 0:
        return s2 * 0.0
    return (s2 - mu) / sd


def _safe_inverse(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s2 = s2.where(s2 > 0)
    return 1.0 / s2


def compute_factor_scores(
    latest_features: pd.DataFrame,
    *,
    sector_neutralize: bool = True,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    각 티커의 최신 피처 DataFrame -> 팩터 스코어 계산

    Args:
        latest_features: index=ticker, columns=피처들
        sector_neutralize: Value/Quality 섹터 중립화 여부
        weights: 팩터별 가중치 {'momentum': 0.25, 'value': 0.25, 'quality': 0.25, 'risk': 0.25}

    Returns:
        원본 + factor_momentum, factor_value, factor_quality, factor_risk, factor_total
    """
    w = weights or {"momentum": 0.25, "value": 0.25, "quality": 0.25, "risk": 0.25}
    df = latest_features.copy()

    def col(name: str) -> pd.Series:
        if name in df.columns:
            out = df[name]
            if isinstance(out, pd.Series):
                return out
        return pd.Series(index=df.index, dtype=float)

    def wz(name: str) -> pd.Series:
        return _winsorize(col(name))

    # --- Momentum ---
    mom_12m = wz("ret_12m")
    mom_1m = wz("ret_1m")
    mom_spread = mom_12m - mom_1m  # reversal 보호
    mom_6m = wz("ret_6m")
    mom_trend = wz("ma_gap_200")
    mom_score = (_zscore(mom_spread) + _zscore(mom_6m) + _zscore(mom_trend)) / 3.0

    # --- Value ---
    inv_pe = _winsorize(_safe_inverse(col("trailingPE")))
    inv_pb = _winsorize(_safe_inverse(col("priceToBook")))
    val_raw = (_zscore(inv_pe) + _zscore(inv_pb)) / 2.0

    # --- Quality ---
    roe = wz("returnOnEquity")
    pm = wz("profitMargins")
    om = wz("operatingMargins")
    eg = wz("earningsGrowth")
    qual_raw = (_zscore(roe) + _zscore(pm) + _zscore(om) + _zscore(eg)) / 4.0

    # --- Risk (낮을수록 좋음) ---
    vol = wz("vol_3m")
    dvol = wz("down_vol_3m")
    mdd = wz("max_drawdown_6m")
    risk_score = (-_zscore(vol) + -_zscore(dvol) + -_zscore(mdd)) / 3.0

    # --- 섹터 중립화 ---
    if sector_neutralize and "sector" in df.columns:
        sec = df["sector"].fillna("Unknown").astype(str)
        val_neutral = val_raw - val_raw.groupby(sec).transform("mean")
        qual_neutral = qual_raw - qual_raw.groupby(sec).transform("mean")
        value_score = _zscore(_winsorize(val_neutral))
        quality_score = _zscore(_winsorize(qual_neutral))
    else:
        value_score = _zscore(_winsorize(val_raw))
        quality_score = _zscore(_winsorize(qual_raw))

    momentum_score = _zscore(_winsorize(mom_score))
    risk_score = _zscore(_winsorize(risk_score))

    # --- Total ---
    total = (
        w["momentum"] * momentum_score
        + w["value"] * value_score
        + w["quality"] * quality_score
        + w["risk"] * risk_score
    )

    out = df.copy()
    out["factor_momentum"] = momentum_score
    out["factor_value"] = value_score
    out["factor_quality"] = quality_score
    out["factor_risk"] = risk_score
    out["factor_total"] = total

    # 설명용 중간값
    out["_mom_12m_1m_spread"] = mom_spread
    out["_mom_6m"] = mom_6m
    out["_val_inv_pe"] = inv_pe
    out["_val_inv_pb"] = inv_pb
    out["_qual_roe"] = roe
    out["_qual_profit_margin"] = pm
    out["_risk_vol_3m"] = vol
    out["_risk_mdd_6m"] = mdd

    return out
