"""
히든챔피언 / 니치 종목 스크리너
모멘텀 + 퀄리티 + 밸류 복합 스코어로 잠재력 높은 종목 발굴
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services.universe import (
    USSource,
    KRUniverseConfig,
    compute_metrics,
    fetch_kr_universe,
    fetch_meta_batch,
    fetch_ohlcv_kr,
    fetch_ohlcv_us,
    fetch_us_universe,
    _zscore,
)


# ═══════════════════════════════════════
# Screener Configs
# ═══════════════════════════════════════
@dataclass
class USScreenerConfig:
    source: USSource = "sp500_github"
    max_pool: int = 500
    min_rows: int = 260
    min_price: float = 2.0
    min_dollar_vol_20d: float = 2e6
    # 스코어 가중치
    w_ret_60d: float = 0.35
    w_ret_1y: float = 0.65
    w_vol_penalty: float = 0.35
    top_n: int = 30
    verbose: bool = False


@dataclass
class KRScreenerConfig:
    market: str = "KOSPI"
    min_mcap_krw: float = 1e12
    min_rows: int = 180
    min_price: float = 500.0
    min_volume_20d: float = 50_000
    w_ret_60d: float = 0.45
    w_ret_1y: float = 0.35
    w_vol_penalty: float = 0.20
    top_n: int = 30
    verbose: bool = False


@dataclass
class HiddenChampionConfig:
    """소형주 + 니치 산업에서 급성장 종목 찾기"""
    source: USSource = "nasdaq_listings_github"
    max_pool: int = 3000
    # 소형주 필터 (시총 $300M~$10B)
    min_mcap: float = 3e8
    max_mcap: float = 1e10
    min_rows: int = 260
    min_price: float = 5.0
    min_dollar_vol_20d: float = 1e6
    # 퀄리티 + 성장 가중치 높임
    w_ret_60d: float = 0.20
    w_ret_1y: float = 0.30
    w_roe: float = 0.20
    w_earnings_growth: float = 0.20
    w_vol_penalty: float = 0.10
    top_n: int = 30
    verbose: bool = False


# ═══════════════════════════════════════
# US Screener
# ═══════════════════════════════════════
def screen_us(cfg: Optional[USScreenerConfig] = None, progress_cb=None) -> pd.DataFrame:
    cfg = cfg or USScreenerConfig()

    universe = fetch_us_universe(cfg.source)
    if universe.empty:
        return pd.DataFrame()

    tickers = universe["ticker"].tolist()[:cfg.max_pool]
    total = len(tickers)
    rows: List[Dict[str, Any]] = []
    fail = {"no_data": 0, "quality": 0, "error": 0}

    for i, t in enumerate(tickers):
        if progress_cb:
            progress_cb(i / total, f"({i+1}/{total}) {t}")
        try:
            df = fetch_ohlcv_us(t, period="5y")
            if df.empty:
                fail["no_data"] += 1
                continue

            met = compute_metrics(df)
            if not met or met["n_rows"] < cfg.min_rows:
                fail["quality"] += 1
                continue
            if met["current_price"] < cfg.min_price:
                fail["quality"] += 1
                continue
            if not np.isfinite(met.get("liq_20d", np.nan)) or met["liq_20d"] < cfg.min_dollar_vol_20d:
                fail["quality"] += 1
                continue

            u_row = universe[universe["ticker"] == t].iloc[0] if t in universe["ticker"].values else {}
            rows.append({
                "ticker": t,
                "name": u_row.get("name", "") if isinstance(u_row, dict) else u_row.get("name", ""),
                "sector": u_row.get("sector", "") if isinstance(u_row, dict) else u_row.get("sector", ""),
                "industry": u_row.get("industry", "") if isinstance(u_row, dict) else u_row.get("industry", ""),
                **met,
            })
        except Exception:
            fail["error"] += 1

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["score"] = (
        cfg.w_ret_60d * _zscore(out["ret_60d"])
        + cfg.w_ret_1y * _zscore(out["ret_1y"])
        - cfg.w_vol_penalty * _zscore(out["vol_1y"])
    )
    out = out.sort_values("score", ascending=False).head(cfg.top_n).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


# ═══════════════════════════════════════
# KR Screener
# ═══════════════════════════════════════
def screen_kr(cfg: Optional[KRScreenerConfig] = None, progress_cb=None) -> pd.DataFrame:
    cfg = cfg or KRScreenerConfig()

    kr_cfg = KRUniverseConfig(market=cfg.market, exclude_preferred=True, min_mcap_krw=cfg.min_mcap_krw)
    universe = fetch_kr_universe(kr_cfg)
    if universe.empty:
        return pd.DataFrame()

    tickers = universe["ticker"].tolist()
    total = len(tickers)

    end = _dt.datetime.now().strftime("%Y%m%d")
    start = (_dt.datetime.now() - _dt.timedelta(days=730)).strftime("%Y%m%d")

    rows: List[Dict[str, Any]] = []

    for i, t in enumerate(tickers):
        if progress_cb:
            progress_cb(i / total, f"({i+1}/{total}) {t}")
        try:
            df = fetch_ohlcv_kr(t, start=start, end=end)
            if df.empty:
                continue
            met = compute_metrics(df)
            if not met or met["n_rows"] < cfg.min_rows:
                continue
            if met["current_price"] < cfg.min_price:
                continue
            if not np.isfinite(met.get("liq_20d", np.nan)) or met["liq_20d"] < cfg.min_volume_20d:
                continue

            u_row = universe[universe["ticker"] == t]
            name = u_row.iloc[0]["name"] if not u_row.empty else ""

            rows.append({"ticker": t, "name": name, "market": cfg.market, **met})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["score"] = (
        cfg.w_ret_60d * _zscore(out["ret_60d"])
        + cfg.w_ret_1y * _zscore(out["ret_1y"])
        - cfg.w_vol_penalty * _zscore(out["vol_1y"])
    )
    out = out.sort_values("score", ascending=False).head(cfg.top_n).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


# ═══════════════════════════════════════
# Hidden Champion Screener
# ═══════════════════════════════════════
def screen_hidden_champions(cfg: Optional[HiddenChampionConfig] = None, progress_cb=None) -> pd.DataFrame:
    """
    소형~중형주에서 고성장 + 고수익성 종목 발굴
    NASDAQ 전체 리스트에서 시총 $300M~$10B 필터 후
    모멘텀 + ROE + 이익성장률 복합 스코어
    """
    cfg = cfg or HiddenChampionConfig()

    universe = fetch_us_universe(cfg.source)
    if universe.empty:
        return pd.DataFrame()

    tickers = universe["ticker"].tolist()[:cfg.max_pool]
    total = len(tickers)

    # Step 1: OHLCV metrics
    price_rows: List[Dict[str, Any]] = []
    passed_tickers: List[str] = []

    for i, t in enumerate(tickers):
        if progress_cb:
            progress_cb(i / total * 0.7, f"가격 데이터 ({i+1}/{total}) {t}")
        try:
            df = fetch_ohlcv_us(t, period="5y")
            if df.empty:
                continue
            met = compute_metrics(df)
            if not met or met["n_rows"] < cfg.min_rows:
                continue
            if met["current_price"] < cfg.min_price:
                continue
            if not np.isfinite(met.get("liq_20d", np.nan)) or met["liq_20d"] < cfg.min_dollar_vol_20d:
                continue

            u_row = universe[universe["ticker"] == t]
            name = u_row.iloc[0]["name"] if not u_row.empty else ""
            sector = u_row.iloc[0].get("sector", "") if not u_row.empty else ""
            industry = u_row.iloc[0].get("industry", "") if not u_row.empty else ""

            price_rows.append({"ticker": t, "name": name, "sector": sector, "industry": industry, **met})
            passed_tickers.append(t)
        except Exception:
            continue

    if not price_rows:
        return pd.DataFrame()

    # Step 2: Fundamental meta (batch)
    if progress_cb:
        progress_cb(0.7, "펀더멘탈 데이터 수집 중...")

    meta = fetch_meta_batch(passed_tickers)

    out = pd.DataFrame(price_rows)
    if not meta.empty:
        meta["ticker"] = meta["ticker"].astype(str).str.upper()
        out = out.merge(meta[["ticker", "marketCap", "returnOnEquity", "earningsGrowth", "trailingPE", "priceToBook"]],
                        on="ticker", how="left")

    # Step 3: Market cap filter for small/mid cap
    if "marketCap" in out.columns:
        mcap = pd.to_numeric(out["marketCap"], errors="coerce")
        mask = (mcap >= cfg.min_mcap) & (mcap <= cfg.max_mcap)
        out = out[mask].copy()

    if out.empty:
        return pd.DataFrame()

    # Step 4: Composite score
    z_ret60 = _zscore(out["ret_60d"])
    z_ret1y = _zscore(out["ret_1y"])
    z_vol = _zscore(out["vol_1y"])

    z_roe = pd.Series(np.zeros(len(out)), index=out.index)
    z_eg = pd.Series(np.zeros(len(out)), index=out.index)

    if "returnOnEquity" in out.columns:
        z_roe = _zscore(pd.to_numeric(out["returnOnEquity"], errors="coerce").fillna(0))
    if "earningsGrowth" in out.columns:
        z_eg = _zscore(pd.to_numeric(out["earningsGrowth"], errors="coerce").fillna(0))

    out["score"] = (
        cfg.w_ret_60d * z_ret60
        + cfg.w_ret_1y * z_ret1y
        + cfg.w_roe * z_roe
        + cfg.w_earnings_growth * z_eg
        - cfg.w_vol_penalty * z_vol
    )

    out = out.sort_values("score", ascending=False).head(cfg.top_n).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    if progress_cb:
        progress_cb(1.0, "완료!")

    return out


# ═══════════════════════════════════════
# Blended KR+US
# ═══════════════════════════════════════
@dataclass
class BlendConfig:
    kr_market: str = "KOSPI"
    final_top_n: int = 30
    w_kr: float = 0.5
    w_us: float = 0.5
    us_per_industry_k: int = 3


def _minmax01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)


def screen_blended(kr_df: pd.DataFrame, us_df: pd.DataFrame, cfg: Optional[BlendConfig] = None) -> pd.DataFrame:
    cfg = cfg or BlendConfig()

    kr = kr_df.copy() if kr_df is not None and not kr_df.empty else pd.DataFrame()
    us = us_df.copy() if us_df is not None and not us_df.empty else pd.DataFrame()

    if kr.empty and us.empty:
        return pd.DataFrame()

    if not kr.empty:
        kr["source"] = "KR"
    if not us.empty:
        us["source"] = "US"
        if "industry" in us.columns and cfg.us_per_industry_k > 0:
            us = (
                us.sort_values("score", ascending=False)
                .groupby(us["industry"].fillna(""), group_keys=False)
                .head(cfg.us_per_industry_k)
                .reset_index(drop=True)
            )

    keep = ["ticker", "name", "score", "source"]
    parts = []
    for part in [kr, us]:
        if part.empty:
            continue
        for c in keep:
            if c not in part.columns:
                part[c] = ""
        parts.append(part[keep].copy())

    if not parts:
        return pd.DataFrame()

    for p in parts:
        p["score_norm"] = _minmax01(p["score"])

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("score_norm", ascending=False).head(cfg.final_top_n).reset_index(drop=True)
    combined.insert(0, "rank", range(1, len(combined) + 1))
    return combined
