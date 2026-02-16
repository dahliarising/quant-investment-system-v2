"""
동적 유니버스 관리 — S&P500, NASDAQ-100, NASDAQ 전체, KOSPI, KOSDAQ
stock_success의 universe/kr_universe 로직을 v2 아키텍처에 통합
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import requests


# ═══════════════════════════════════════
# US Universe
# ═══════════════════════════════════════
USSource = Literal["sp500_github", "sp500_wiki", "nasdaq100_wiki", "nasdaq_listings_github"]

_SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
_NASDAQ_CSV = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
_SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NASDAQ100_WIKI = "https://en.wikipedia.org/wiki/Nasdaq-100"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)


def _normalize_ticker(s: str) -> str:
    return str(s).strip().upper().replace(".", "-")


def _standardize_us(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry"])

    out = df.copy()
    colmap = {
        "Symbol": "ticker", "symbol": "ticker", "Ticker": "ticker",
        "Security": "name", "Name": "name", "Company Name": "name",
        "GICS Sector": "sector", "Sector": "sector",
        "GICS Sub-Industry": "industry", "Industry": "industry",
    }
    rename = {c: colmap[c] for c in out.columns if c in colmap}
    out = out.rename(columns=rename)

    for c in ["ticker", "name", "sector", "industry"]:
        if c not in out.columns:
            out[c] = None

    out["ticker"] = out["ticker"].apply(_normalize_ticker)
    out = out.dropna(subset=["ticker"])
    out = out[out["ticker"].astype(str).str.len() > 0]
    out = out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out[["ticker", "name", "sector", "industry"]]


def fetch_us_universe(source: USSource = "sp500_github", timeout: int = 20) -> pd.DataFrame:
    headers = {"User-Agent": _UA}

    if source == "sp500_github":
        df = pd.read_csv(_SP500_CSV)
        return _standardize_us(df)

    if source == "sp500_wiki":
        from io import StringIO
        html = requests.get(_SP500_WIKI, headers=headers, timeout=timeout).text
        tables = pd.read_html(StringIO(html))
        return _standardize_us(tables[0])

    if source == "nasdaq100_wiki":
        from io import StringIO
        html = requests.get(_NASDAQ100_WIKI, headers=headers, timeout=timeout).text
        tables = pd.read_html(StringIO(html))
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                return _standardize_us(t)
        return _standardize_us(tables[0])

    if source == "nasdaq_listings_github":
        df = pd.read_csv(_NASDAQ_CSV)
        return _standardize_us(df)

    raise ValueError(f"Unknown source: {source}")


# ═══════════════════════════════════════
# KR Universe
# ═══════════════════════════════════════
def _is_common_stock(ticker: str, name: str) -> bool:
    if not isinstance(ticker, str) or not ticker.isdigit():
        return False
    if isinstance(name, str) and "우" in name:
        return False
    return True


@dataclass
class KRUniverseConfig:
    market: str = "KOSPI"
    exclude_preferred: bool = True
    min_mcap_krw: Optional[float] = None
    top_mcap_n: Optional[int] = None


def fetch_kr_universe(cfg: Optional[KRUniverseConfig] = None) -> pd.DataFrame:
    cfg = cfg or KRUniverseConfig()

    from pykrx import stock

    asof = stock.get_nearest_business_day_in_a_week()

    tickers = stock.get_market_ticker_list(asof, market=cfg.market)
    rows = [{"ticker": str(t), "name": stock.get_market_ticker_name(t), "market": cfg.market} for t in tickers]
    u = pd.DataFrame(rows)
    if u.empty:
        return u

    if cfg.exclude_preferred:
        u = u[u.apply(lambda r: _is_common_stock(r["ticker"], r["name"]), axis=1)].copy()
    if u.empty:
        return u

    cap = stock.get_market_cap_by_ticker(asof, market=cfg.market)
    cap = cap.reset_index().rename(columns={"티커": "ticker", "시가총액": "mcap_krw"})
    cap["ticker"] = cap["ticker"].astype(str)
    u = u.merge(cap[["ticker", "mcap_krw"]], on="ticker", how="left")

    if cfg.min_mcap_krw is not None:
        u = u[pd.to_numeric(u["mcap_krw"], errors="coerce") >= float(cfg.min_mcap_krw)].copy()
    if cfg.top_mcap_n is not None:
        u = u.sort_values("mcap_krw", ascending=False).head(int(cfg.top_mcap_n)).copy()

    u = u.sort_values(["mcap_krw", "ticker"], ascending=[False, True]).reset_index(drop=True)
    return u


# ═══════════════════════════════════════
# OHLCV Fetchers
# ═══════════════════════════════════════
def fetch_ohlcv_us(ticker: str, period: str = "5y") -> pd.DataFrame:
    import yfinance as yf
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        t = ticker.strip().upper()
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        if t in lvl0:
            df = df.xs(t, axis=1, level=0, drop_level=True)
        elif t in lvl1:
            df = df.xs(t, axis=1, level=1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].sort_index() if keep else pd.DataFrame()


def fetch_ohlcv_kr(ticker: str, start: str, end: str) -> pd.DataFrame:
    from pykrx import stock
    s = str(start).replace("-", "")
    e = str(end).replace("-", "")
    try:
        df = stock.get_market_ohlcv_by_date(s, e, ticker, adjusted=True)
    except TypeError:
        df = stock.get_market_ohlcv_by_date(s, e, ticker)
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df.index = pd.to_datetime(df.index)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[keep].dropna(subset=["Close"])


# ═══════════════════════════════════════
# Metadata (yfinance .info)
# ═══════════════════════════════════════
_META_FIELDS = [
    "sector", "industry", "trailingPE", "priceToBook",
    "returnOnEquity", "profitMargins", "operatingMargins",
    "earningsGrowth", "marketCap", "shortName",
]


def fetch_meta_batch(tickers: List[str]) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        row = {"ticker": t}
        for k in _META_FIELDS:
            row[k] = info.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════
def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 60:
        return {}

    def _ret(days):
        if len(close) < days + 1:
            return np.nan
        return float(close.iloc[-1] / close.iloc[-days - 1] - 1.0)

    ret_60d = _ret(60)
    ret_1y = _ret(252)

    rets = close.pct_change().dropna()
    vol_1y = float(rets.std(ddof=0) * np.sqrt(252)) if len(rets) >= 60 else np.nan

    liq_20d = np.nan
    if "Volume" in df.columns:
        v = pd.to_numeric(df["Volume"], errors="coerce").reindex(close.index).dropna()
        if len(v) >= 20:
            liq_20d = float((close * v).tail(20).mean())

    return {
        "n_rows": int(len(close)),
        "ret_60d": ret_60d,
        "ret_1y": ret_1y,
        "vol_1y": vol_1y,
        "liq_20d": liq_20d,
        "current_price": float(close.iloc[-1]),
    }


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd
