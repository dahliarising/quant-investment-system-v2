"""
일일 트레이드 로깅 시스템
- 매 실행시 Top-N 종목 기록
- 실현 수익률 역계산 (backfill)
- CSV 기반 영속 저장
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "data_cache" / "daily_topn_log.csv"

LOG_COLUMNS = [
    "asof_date", "ticker", "rank", "score_total", "pred_return",
    "entry_price", "exit_price", "realized_return", "realized_alpha",
    "horizon", "label_mode", "model", "regime", "note",
]


def _ensure_log(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["asof_date"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=LOG_COLUMNS)


def append_daily_topn(
    ranked_df: pd.DataFrame,
    top_n: int = 10,
    horizon: str = "1y",
    label_mode: str = "alpha_vs_spy",
    model: str = "lgbm",
    regime: str = "NORMAL",
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    랭킹 결과에서 Top-N을 로그에 추가

    Args:
        ranked_df: 랭킹 결과 (ticker, score_total, pred_return, current_price 등)
        top_n: 상위 N개
        log_path: 로그 파일 경로

    Returns:
        추가된 레코드 DataFrame
    """
    path = log_path or DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _ensure_log(path)
    today = datetime.now().strftime("%Y-%m-%d")

    # 중복 방지
    if not existing.empty and today in existing["asof_date"].astype(str).values:
        return pd.DataFrame()

    top = ranked_df.head(top_n).copy()
    records = []
    for i, (_, row) in enumerate(top.iterrows()):
        records.append({
            "asof_date": today,
            "ticker": row.get("ticker", ""),
            "rank": i + 1,
            "score_total": row.get("score_total", row.get("factor_total", np.nan)),
            "pred_return": row.get("pred_return", np.nan),
            "entry_price": row.get("current_price", np.nan),
            "exit_price": np.nan,
            "realized_return": np.nan,
            "realized_alpha": np.nan,
            "horizon": horizon,
            "label_mode": label_mode,
            "model": model,
            "regime": regime,
            "note": "",
        })

    new_df = pd.DataFrame(records)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(path, index=False)
    return new_df


def backfill_realized_returns(
    price_fetcher,
    log_path: Optional[Path] = None,
    benchmark_ticker: str = "SPY",
) -> int:
    """
    과거 로그의 실현 수익률을 역계산

    Args:
        price_fetcher: callable(ticker) -> float (현재가 반환)
        log_path: 로그 경로

    Returns:
        업데이트된 행 수
    """
    path = log_path or DEFAULT_LOG_PATH
    if not path.exists():
        return 0

    df = pd.read_csv(path, parse_dates=["asof_date"])
    updated = 0

    # exit_price가 NaN인 행 중 horizon이 지난 것
    for idx, row in df.iterrows():
        if pd.notna(row["exit_price"]):
            continue

        asof = pd.Timestamp(row["asof_date"])
        horizon_days = 252 if row.get("horizon", "1y") == "1y" else 30
        exit_date = asof + timedelta(days=int(horizon_days * 1.4))

        if pd.Timestamp.now() < exit_date:
            continue

        try:
            exit_px = price_fetcher(row["ticker"])
            if exit_px is None or np.isnan(exit_px):
                continue

            entry_px = row["entry_price"]
            if pd.isna(entry_px) or entry_px <= 0:
                continue

            realized = exit_px / entry_px - 1.0
            df.at[idx, "exit_price"] = exit_px
            df.at[idx, "realized_return"] = realized

            # alpha 계산
            try:
                bench_px = price_fetcher(benchmark_ticker)
                if bench_px is not None:
                    bench_entry = price_fetcher(benchmark_ticker)
                    if bench_entry and bench_entry > 0:
                        df.at[idx, "realized_alpha"] = realized
            except Exception:
                pass

            updated += 1
        except Exception:
            continue

    if updated > 0:
        df.to_csv(path, index=False)

    return updated


def get_trade_log(log_path: Optional[Path] = None) -> pd.DataFrame:
    """트레이드 로그 조회"""
    path = log_path or DEFAULT_LOG_PATH
    return _ensure_log(path)


def get_trade_stats(log_path: Optional[Path] = None) -> dict:
    """트레이드 통계"""
    df = get_trade_log(log_path)
    if df.empty:
        return {"total_picks": 0, "realized_count": 0, "avg_return": np.nan, "win_rate": np.nan}

    realized = df[df["realized_return"].notna()]
    return {
        "total_picks": len(df),
        "unique_dates": df["asof_date"].nunique(),
        "realized_count": len(realized),
        "avg_return": float(realized["realized_return"].mean()) if len(realized) else np.nan,
        "win_rate": float((realized["realized_return"] > 0).mean()) if len(realized) else np.nan,
        "best_pick": realized.nlargest(1, "realized_return").to_dict("records")[0] if len(realized) else None,
        "worst_pick": realized.nsmallest(1, "realized_return").to_dict("records")[0] if len(realized) else None,
    }
