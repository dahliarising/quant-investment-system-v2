"""한국 종목/지수 데이터 utility — pykrx + FinanceDataReader.

yfinance는 한국 시장 데이터가 stale하므로 한국 종목·지수는 절대 yfinance로 호출하지 않는다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import FinanceDataReader as fdr
from pykrx import stock

KST = ZoneInfo("Asia/Seoul")
LOOKBACK_DAYS = 10


def is_korean_ticker(ticker: str) -> bool:
    upper = ticker.upper()
    if upper.endswith(".KS") or upper.endswith(".KQ"):
        return True
    if ticker.isdigit() and len(ticker) == 6:
        return True
    return False


def normalize_kr_code(ticker: str) -> str:
    upper = ticker.upper()
    if upper.endswith(".KS") or upper.endswith(".KQ"):
        return ticker[:-3]
    return ticker


def _ticker_name(code: str) -> str:
    try:
        return stock.get_market_ticker_name(code)
    except Exception:
        return "N/A"


def _market_cap(code: str, date_str: str) -> int | None:
    try:
        cap_df = stock.get_market_cap_by_date(date_str, date_str, code)
        if not cap_df.empty:
            return int(cap_df.iloc[-1]["시가총액"])
    except Exception:
        return None
    return None


def get_kr_stock_data(ticker: str) -> dict[str, Any]:
    """한국 종목 OHLCV + Fundamental (pykrx)."""
    code = normalize_kr_code(ticker)
    today_str = datetime.now(KST).strftime("%Y%m%d")
    base = datetime.strptime(today_str, "%Y%m%d")

    for delta in range(0, LOOKBACK_DAYS):
        try_date = (base - timedelta(days=delta)).strftime("%Y%m%d")
        ohlcv = stock.get_market_ohlcv_by_date(try_date, try_date, code)
        if ohlcv.empty:
            continue
        fund = stock.get_market_fundamental_by_date(try_date, try_date, code)
        o_row = ohlcv.iloc[-1]
        f_row = fund.iloc[-1] if not fund.empty else None

        result: dict[str, Any] = {
            "종목": ticker,
            "이름": _ticker_name(code),
            "현재가": int(o_row["종가"]),
            "시가": int(o_row["시가"]),
            "고가": int(o_row["고가"]),
            "저가": int(o_row["저가"]),
            "거래량": int(o_row["거래량"]),
            "등락률(%)": round(float(o_row.get("등락률", 0)), 2),
            "시가총액": _market_cap(code, try_date),
            "_date": try_date,
            "_source": "pykrx",
        }
        if f_row is not None:
            per = float(f_row.get("PER", 0) or 0)
            pbr = float(f_row.get("PBR", 0) or 0)
            div = float(f_row.get("DIV", 0) or 0)
            eps = float(f_row.get("EPS", 0) or 0)
            bps = float(f_row.get("BPS", 0) or 0)
            result["PER"] = round(per, 2) if per else None
            result["PBR"] = round(pbr, 2) if pbr else None
            result["배당수익률(%)"] = round(div, 2) if div else None
            result["EPS"] = eps if eps else None
            result["BPS"] = bps if bps else None
        return result

    return {"종목": ticker, "에러": "pykrx 데이터 없음", "_source": "pykrx"}


def get_kr_index_data(index_code: str) -> dict[str, Any]:
    """한국 지수 (KS11=KOSPI, KQ11=KOSDAQ) — FinanceDataReader."""
    today = datetime.now(KST)
    start = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    df = fdr.DataReader(index_code, start, end)
    if df.empty:
        return {"에러": "FDR 데이터 없음", "_source": "FinanceDataReader"}
    row = df.iloc[-1]
    return {
        "현재가": round(float(row["Close"]), 2),
        "전일대비(%)": round(float(row.get("Change", 0)) * 100, 2),
        "시가": round(float(row["Open"]), 2),
        "고가": round(float(row["High"]), 2),
        "저가": round(float(row["Low"]), 2),
        "거래량": int(row.get("Volume", 0)),
        "_date": df.index[-1].strftime("%Y-%m-%d"),
        "_source": "FinanceDataReader",
    }
