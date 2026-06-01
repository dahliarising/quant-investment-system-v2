"""Corvin 선행 인텔리전스 — 재무 metrics 어댑터.

raw API 응답(yfinance.info / pykrx EPS)을 financial_growth_score 입력
metrics dict로 변환. 파서는 순수(테스트 가능), fetch는 thin 어댑터.

⚠️ yfinance는 US 종목만 (메모리 feedback_no_yfinance_kr). KR은 pykrx.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

log = logging.getLogger("corvin.financial_metrics")


def parse_us_financials(info: dict[str, Any]) -> dict[str, float]:
    """yfinance .info → metrics. 가용/유효 필드만. debtToEquity는 %→ratio."""
    m: dict[str, float] = {}
    eg = info.get("earningsGrowth")
    if eg is not None:
        m["eps_growth_yoy"] = float(eg)
    rg = info.get("revenueGrowth")
    if rg is not None:
        m["revenue_growth_yoy"] = float(rg)
    dte = info.get("debtToEquity")
    if dte is not None:
        m["debt_ratio"] = float(dte) / 100.0   # yfinance는 percent 형
    return m


def parse_kr_fundamental(
    eps_now: float | None, eps_year_ago: float | None
) -> dict[str, float]:
    """현재/1년전 EPS → eps_growth_yoy. 둘 중 누락 또는 과거≤0이면 빈 dict.

    과거 EPS ≤ 0(적자)이면 성장률 정의가 왜곡 → 산출 안 함(추측 금지).
    """
    if eps_now is None or eps_year_ago is None:
        return {}
    if eps_year_ago <= 0:
        return {}
    return {"eps_growth_yoy": round((eps_now - eps_year_ago) / eps_year_ago, 6)}


def _is_kr(symbol: str) -> bool:
    return symbol.isdigit() and len(symbol) == 6


def fetch_metrics(symbol: str) -> dict[str, float] | None:
    """종목 재무 metrics fetch. US=yfinance, KR=pykrx 2시점 EPS. 실패시 None."""
    try:
        if _is_kr(symbol):
            return _fetch_kr(symbol)
        return _fetch_us(symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("재무 fetch 실패 %s: %s", symbol, e)
        return None


def _fetch_us(symbol: str) -> dict[str, float] | None:
    import yfinance as yf
    info = yf.Ticker(symbol).info
    m = parse_us_financials(info or {})
    return m or None


def _fetch_kr(symbol: str) -> dict[str, float] | None:
    from datetime import datetime

    from pykrx import stock
    now = datetime.now()
    eps_now = _kr_eps_near(stock, symbol, now)
    eps_ago = _kr_eps_near(stock, symbol, now - timedelta(days=365))
    m = parse_kr_fundamental(eps_now, eps_ago)
    return m or None


def _kr_eps_near(stock_mod: Any, code: str, when: Any) -> float | None:
    """when 근처 영업일의 pykrx EPS. 최대 10일 역탐색."""
    for delta in range(0, 10):
        d = (when - timedelta(days=delta)).strftime("%Y%m%d")
        df = stock_mod.get_market_fundamental_by_date(d, d, code)
        if not df.empty:
            eps = float(df.iloc[-1].get("EPS", 0) or 0)
            return eps if eps else None
    return None
