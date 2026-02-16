"""
랭킹 파이프라인 (Factor + ML 하이브리드)
- 팩터 스코어 + ML 예측을 결합한 최종 랭킹
- 유니버스별 데이터 자동 수집/가공
- TopN by Group 기능
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

HorizonName = Literal["30d", "1y"]
LabelMode = Literal["absolute", "alpha_vs_spy"]
ModelName = Literal["ridge", "rf", "xgb", "lgbm"]


@dataclass(frozen=True)
class RankConfig:
    period: str = "8y"
    model_name: ModelName = "lgbm"
    min_train_rows: int = 350
    horizon: HorizonName = "1y"
    label_mode: LabelMode = "alpha_vs_spy"
    sector_neutralize: bool = True
    top_n: int = 15


def rank_dataframe(
    features_df: pd.DataFrame,
    factor_scores: pd.DataFrame,
    ml_predictions: Optional[pd.Series] = None,
    *,
    ml_weight: float = 0.5,
    factor_weight: float = 0.5,
    score_col: str = "factor_total",
) -> pd.DataFrame:
    """
    팩터 스코어와 ML 예측을 결합한 최종 랭킹

    Args:
        features_df: 피처 DataFrame (index=ticker)
        factor_scores: compute_factor_scores 결과
        ml_predictions: ML 예측 수익률 Series (index=ticker)
        ml_weight: ML 예측 가중치
        factor_weight: 팩터 스코어 가중치

    Returns:
        종합 랭킹 DataFrame
    """
    out = factor_scores.copy()

    if ml_predictions is not None and not ml_predictions.empty:
        out["pred_return"] = ml_predictions
        # ML 예측의 Z-score
        pr = pd.to_numeric(out["pred_return"], errors="coerce")
        mu, sd = pr.mean(), pr.std()
        if sd and sd > 0:
            ml_z = (pr - mu) / sd
        else:
            ml_z = pr * 0.0
        out["ml_zscore"] = ml_z

        # 종합 스코어
        factor_z = pd.to_numeric(out.get(score_col, 0), errors="coerce").fillna(0)
        out["score_combined"] = factor_weight * factor_z + ml_weight * ml_z
    else:
        out["pred_return"] = np.nan
        out["ml_zscore"] = 0.0
        out["score_combined"] = out.get(score_col, 0)

    out = out.sort_values("score_combined", ascending=False).reset_index(drop=False)
    out["rank"] = range(1, len(out) + 1)
    return out


def topn_by_group(
    ranked: pd.DataFrame,
    group_col: str = "sector",
    top_n: int = 3,
    score_col: str = "score_combined",
) -> pd.DataFrame:
    """그룹(섹터)별 Top N 선택"""
    if ranked.empty or group_col not in ranked.columns:
        return ranked.head(top_n)

    df = ranked.copy()
    df[group_col] = df[group_col].fillna("Unknown")

    parts = []
    for g, sub in df.groupby(group_col, dropna=False):
        parts.append(sub.sort_values(score_col, ascending=False).head(top_n))

    return pd.concat(parts, ignore_index=True).sort_values(score_col, ascending=False)


def apply_sector_cap(
    ranked: pd.DataFrame,
    top_n: int = 15,
    max_sector_pct: float = 0.25,
    score_col: str = "score_combined",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """
    섹터 비중 제한을 적용한 Top-N 선택

    Args:
        ranked: 랭킹 DataFrame
        top_n: 최종 선택 종목 수
        max_sector_pct: 섹터별 최대 비중 (0.25 = 25%)
    """
    if ranked.empty:
        return ranked

    df = ranked.sort_values(score_col, ascending=False).copy()
    df[sector_col] = df[sector_col].fillna("Unknown")
    max_per_sector = max(1, int(top_n * max_sector_pct))

    selected = []
    sector_counts = {}
    for _, row in df.iterrows():
        if len(selected) >= top_n:
            break
        sec = row[sector_col]
        cnt = sector_counts.get(sec, 0)
        if cnt < max_per_sector:
            selected.append(row)
            sector_counts[sec] = cnt + 1

    return pd.DataFrame(selected).reset_index(drop=True)


def compute_portfolio_weights(
    selected: pd.DataFrame,
    method: str = "equal",
    score_col: str = "score_combined",
    max_weight: float = 0.15,
) -> pd.DataFrame:
    """
    선택된 종목에 가중치 부여

    Args:
        selected: 선택된 종목 DataFrame
        method: 'equal', 'score_weighted', 'inverse_vol'
        max_weight: 개별 종목 최대 비중
    """
    df = selected.copy()
    n = len(df)
    if n == 0:
        return df

    if method == "equal":
        df["weight"] = 1.0 / n
    elif method == "score_weighted":
        scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
        scores = scores - scores.min() + 1e-6  # 양수 보장
        df["weight"] = scores / scores.sum()
    else:
        df["weight"] = 1.0 / n

    # max_weight 제한
    df["weight"] = df["weight"].clip(upper=max_weight)
    df["weight"] = df["weight"] / df["weight"].sum()

    return df
