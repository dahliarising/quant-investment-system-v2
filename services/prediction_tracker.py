"""
ML 예측 추적 시스템
- 일일 예측 로깅 (모델, 팩터 스코어, 예측 수익률)
- Horizon 경과 후 실제 수익률 평가
- 정확도 분석 (Hit Rate, Rank IC, 섹터별/월별)
- 포트폴리오 수준 추적 (Traceability)

CSV 기반 영속 저장 — trade_logger.py 패턴 따름
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_PRED_PATH = Path(__file__).resolve().parents[1] / "data_cache" / "prediction_log.csv"

PRED_COLUMNS = [
    "date", "ticker", "pred_return", "actual_return", "hit",
    "model", "horizon",
    "factor_momentum", "factor_value", "factor_quality", "factor_risk",
    "score_total", "entry_price", "exit_price",
    "sector", "evaluated",
]


def _ensure_pred_log(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=PRED_COLUMNS)


# ═══════════════════════════════════════
# 1. 예측 로깅
# ═══════════════════════════════════════
def log_predictions(
    ranked_df: pd.DataFrame,
    model: str = "lgbm",
    horizon: str = "1y",
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    일일 ML 예측을 로그에 기록.

    ranked_df 필수 컬럼: ticker, pred_return, current_price
    선택 컬럼: score_total, factor_momentum/value/quality/risk, sector, score_combined

    날짜+모델 기준 중복 방지.
    """
    path = log_path or DEFAULT_PRED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _ensure_pred_log(path)
    today = datetime.now().strftime("%Y-%m-%d")

    # 중복 방지
    if not existing.empty:
        mask = (existing["date"].astype(str) == today) & (existing["model"] == model)
        if mask.any():
            return pd.DataFrame()

    records = []
    for _, row in ranked_df.iterrows():
        records.append({
            "date": today,
            "ticker": row.get("ticker", ""),
            "pred_return": row.get("pred_return", np.nan),
            "actual_return": np.nan,
            "hit": np.nan,
            "model": model,
            "horizon": horizon,
            "factor_momentum": row.get("factor_momentum", np.nan),
            "factor_value": row.get("factor_value", np.nan),
            "factor_quality": row.get("factor_quality", np.nan),
            "factor_risk": row.get("factor_risk", np.nan),
            "score_total": row.get("score_total", row.get("score_combined", np.nan)),
            "entry_price": row.get("current_price", np.nan),
            "exit_price": np.nan,
            "sector": row.get("sector", ""),
            "evaluated": False,
        })

    if not records:
        return pd.DataFrame()

    new_df = pd.DataFrame(records)
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(path, index=False)
    return new_df


# ═══════════════════════════════════════
# 2. 예측 평가
# ═══════════════════════════════════════
def evaluate_predictions(
    price_fetcher,
    horizon_days: int = 30,
    log_path: Optional[Path] = None,
) -> int:
    """
    Horizon 경과 후 actual_return 계산, hit 판정.

    Args:
        price_fetcher: callable(ticker: str) -> float
        horizon_days: 평가 기간 (30, 252 등)

    Returns:
        업데이트된 행 수
    """
    path = log_path or DEFAULT_PRED_PATH
    if not path.exists():
        return 0

    df = pd.read_csv(path, parse_dates=["date"])
    updated = 0

    for idx, row in df.iterrows():
        if row.get("evaluated") is True or str(row.get("evaluated")).lower() == "true":
            continue

        asof = pd.Timestamp(row["date"])
        days_elapsed = (pd.Timestamp.now() - asof).days

        if days_elapsed < horizon_days:
            continue

        entry_px = row.get("entry_price", np.nan)
        if pd.isna(entry_px) or entry_px <= 0:
            continue

        try:
            exit_px = price_fetcher(str(row["ticker"]))
            if exit_px is None or not np.isfinite(exit_px):
                continue

            actual_ret = exit_px / entry_px - 1.0
            pred_ret = row.get("pred_return", np.nan)
            hit = False
            if np.isfinite(pred_ret):
                hit = bool((pred_ret > 0) == (actual_ret > 0))

            df.at[idx, "exit_price"] = exit_px
            df.at[idx, "actual_return"] = actual_ret
            df.at[idx, "hit"] = hit
            df.at[idx, "evaluated"] = True
            updated += 1
        except Exception:
            continue

    if updated > 0:
        df.to_csv(path, index=False)

    return updated


# ═══════════════════════════════════════
# 3. 이력 조회
# ═══════════════════════════════════════
def get_prediction_history(
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
    evaluated_only: bool = False,
) -> pd.DataFrame:
    """필터 적용된 예측 이력 조회."""
    path = log_path or DEFAULT_PRED_PATH
    df = _ensure_pred_log(path)

    if df.empty:
        return df

    if model:
        df = df[df["model"] == model].copy()

    if evaluated_only:
        df = df[df["evaluated"].astype(str).str.lower() == "true"].copy()

    return df.sort_values("date", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════
# 4. 정확도 분석
# ═══════════════════════════════════════
def get_accuracy_summary(
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> dict:
    """
    전체 정확도 요약.

    Returns:
        dict with: total_predictions, evaluated_count, hit_rate,
        rank_ic_mean, rank_ic_std, mean_pred_return, mean_actual_return,
        mse, by_sector, by_month
    """
    df = get_prediction_history(log_path, model=model)

    result = {
        "total_predictions": len(df),
        "evaluated_count": 0,
        "hit_rate": None,
        "rank_ic_mean": None,
        "rank_ic_std": None,
        "mean_pred_return": None,
        "mean_actual_return": None,
        "mse": None,
        "by_sector": {},
        "by_month": {},
    }

    if df.empty:
        return result

    evaluated = df[df["evaluated"].astype(str).str.lower() == "true"].copy()
    result["evaluated_count"] = len(evaluated)

    if evaluated.empty:
        return result

    actual = pd.to_numeric(evaluated["actual_return"], errors="coerce")
    pred = pd.to_numeric(evaluated["pred_return"], errors="coerce")
    hits = evaluated["hit"].astype(str).str.lower() == "true"

    result["hit_rate"] = float(hits.mean()) if len(hits) > 0 else None
    result["mean_pred_return"] = float(pred.mean()) if pred.notna().any() else None
    result["mean_actual_return"] = float(actual.mean()) if actual.notna().any() else None

    # MSE
    valid = pred.notna() & actual.notna()
    if valid.sum() > 0:
        result["mse"] = float(((pred[valid] - actual[valid]) ** 2).mean())

    # Rank IC 계산 — 날짜별
    rank_ics = compute_daily_rank_ic(log_path, model=model)
    if not rank_ics.empty:
        result["rank_ic_mean"] = float(rank_ics.mean())
        result["rank_ic_std"] = float(rank_ics.std())

    # 섹터별 hit rate
    if "sector" in evaluated.columns:
        for sector, grp in evaluated.groupby("sector"):
            if not sector or pd.isna(sector):
                continue
            grp_hits = grp["hit"].astype(str).str.lower() == "true"
            result["by_sector"][str(sector)] = {
                "count": len(grp),
                "hit_rate": float(grp_hits.mean()),
            }

    # 월별 hit rate
    evaluated["month"] = pd.to_datetime(evaluated["date"]).dt.to_period("M").astype(str)
    for month, grp in evaluated.groupby("month"):
        grp_hits = grp["hit"].astype(str).str.lower() == "true"
        result["by_month"][str(month)] = {
            "count": len(grp),
            "hit_rate": float(grp_hits.mean()),
        }

    return result


def compute_daily_rank_ic(
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> pd.Series:
    """
    날짜별 cross-sectional Rank IC (Spearman)
    pred_return vs actual_return.
    """
    df = get_prediction_history(log_path, model=model, evaluated_only=True)
    if df.empty:
        return pd.Series(dtype=float)

    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    pred = pd.to_numeric(df["pred_return"], errors="coerce")
    actual = pd.to_numeric(df["actual_return"], errors="coerce")

    ics = {}
    for date_val, grp in df.groupby("date_str"):
        p = pred.loc[grp.index].dropna()
        a = actual.loc[grp.index].dropna()
        common = p.index.intersection(a.index)
        if len(common) < 3:
            continue
        ic = p.loc[common].rank().corr(a.loc[common].rank())
        ics[date_val] = ic

    return pd.Series(ics, dtype=float).sort_index()
