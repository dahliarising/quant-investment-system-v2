"""scripts → corvin_jarvis.quote_provider 브리지.

각 scripts/*.py는 다음 한 줄로 KIS 실시간 시세 사용 가능:

    from quote_bridge import live_quote, daily_closes

기존 yfinance/pykrx 직접 호출은 quote_provider 내부에서 폴백으로 작동.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corvin_jarvis import quote_provider  # noqa: E402

Quote = quote_provider.Quote


def live_quote(symbol: str) -> Quote:
    """KIS 우선 + yfinance/pykrx 폴백. price + pct_change."""
    return quote_provider.get_stock_quote(symbol)


def live_quote_any(symbol: str) -> Quote:
    """주식·지수·원자재·FX·crypto 자동 라우팅."""
    return quote_provider.get_quote(symbol)


def daily_closes(symbol: str, days: int = 252) -> list[float]:
    """일봉 종가 oldest→newest (KIS 우선)."""
    return quote_provider.get_stock_daily_closes(symbol, days=days)


def fundamentals(symbol: str) -> dict:
    """PER/PBR/시총 — KR=pykrx, US=yfinance (KIS는 펀더 부족)."""
    return quote_provider.get_stock_fundamentals(symbol)
