"""
데이터 소스 선택 공유 컴포넌트
- 3개 페이지(Factor Analysis, Live Monitor, Dashboard)에서 공통 사용
- 데이터 소스 선택기 + 뱃지 + 데이터 라우터
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DataSourceType = Literal["demo", "real_us", "real_kr"]

_SOURCE_OPTIONS = {
    "📊 Demo Data (랜덤 시뮬레이션)": "demo",
    "🇺🇸 Real Data — US Market (yfinance)": "real_us",
    "🇰🇷 Real Data — KR Market (pykrx)": "real_kr",
}


# ═══════════════════════════════════════
# 선택기
# ═══════════════════════════════════════
def render_data_source_selector(key_prefix: str = "default") -> DataSourceType:
    """사이드바에 데이터 소스 선택기 렌더링. 선택된 소스 타입 반환."""
    st.markdown("### 📡 데이터 소스")
    choice = st.selectbox(
        "데이터 소스 선택",
        list(_SOURCE_OPTIONS.keys()),
        key=f"{key_prefix}_data_source_select",
        label_visibility="collapsed",
    )
    source = _SOURCE_OPTIONS[choice]

    if source == "demo":
        st.caption("랜덤으로 생성된 시뮬레이션 데이터입니다.")
    elif source == "real_us":
        st.caption("Yahoo Finance에서 실시간 데이터를 가져옵니다.")
    else:
        st.caption("한국거래소(KRX)에서 실시간 데이터를 가져옵니다.")

    st.markdown("---")
    return source


# ═══════════════════════════════════════
# 뱃지
# ═══════════════════════════════════════
def render_data_source_badge(source: DataSourceType) -> None:
    """현재 데이터 소스를 나타내는 시각적 뱃지."""
    if source == "demo":
        html = """
        <span style="background: rgba(148,163,184,0.15); color: #94a3b8;
                     border: 1px solid rgba(148,163,184,0.3); padding: 0.3rem 0.8rem;
                     border-radius: 20px; font-size: 0.75rem; font-weight: 700;
                     display: inline-block; margin-bottom: 0.5rem;">
            ⚠ DEMO DATA — 랜덤 시뮬레이션
        </span>"""
    elif source == "real_us":
        html = """
        <span style="background: rgba(16,185,129,0.15); color: #10b981;
                     border: 1px solid rgba(16,185,129,0.3); padding: 0.3rem 0.8rem;
                     border-radius: 20px; font-size: 0.75rem; font-weight: 700;
                     display: inline-block; margin-bottom: 0.5rem;">
            ● LIVE — US Market (Yahoo Finance)
        </span>"""
    else:
        html = """
        <span style="background: rgba(239,68,68,0.15); color: #ef4444;
                     border: 1px solid rgba(239,68,68,0.3); padding: 0.3rem 0.8rem;
                     border-radius: 20px; font-size: 0.75rem; font-weight: 700;
                     display: inline-block; margin-bottom: 0.5rem;">
            ● LIVE — KR Market (KRX pykrx)
        </span>"""

    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════
# 데이터 페처
# ═══════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data_for_source(
    source: DataSourceType,
    tickers: tuple,
    period: str = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    소스에 따라 OHLCV 데이터를 가져오는 통합 라우터.
    tickers는 tuple (st.cache_data hashability).
    """
    result = {}

    if source == "demo":
        dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq="B")
        for t in tickers:
            np.random.seed(hash(t) % 2**31)
            base = np.random.uniform(50, 500)
            close = base * np.cumprod(1 + np.random.randn(len(dates)) * 0.015)
            result[t] = pd.DataFrame({
                "Open": close * (1 + np.random.randn(len(dates)) * 0.005),
                "High": close * (1 + abs(np.random.randn(len(dates)) * 0.01)),
                "Low": close * (1 - abs(np.random.randn(len(dates)) * 0.01)),
                "Close": close,
                "Volume": np.random.randint(1_000_000, 20_000_000, len(dates)),
            }, index=dates)
        return result

    if source == "real_us":
        from services.universe import fetch_ohlcv_us
        for t in tickers:
            try:
                df = fetch_ohlcv_us(t, period=period)
                if not df.empty:
                    result[t] = df
            except Exception:
                continue
        return result

    if source == "real_kr":
        from services.universe import fetch_ohlcv_kr
        import datetime as _dt
        end = end_date or _dt.datetime.now().strftime("%Y%m%d")
        period_days = {"5d": 15, "1y": 365, "2y": 730, "5y": 1825}
        days = period_days.get(period, 365)
        start = start_date or (_dt.datetime.now() - _dt.timedelta(days=days)).strftime("%Y%m%d")

        for t in tickers:
            try:
                df = fetch_ohlcv_kr(t, start=start, end=end)
                if not df.empty:
                    result[t] = df
            except Exception:
                continue
        return result

    return result


@st.cache_data(ttl=600, show_spinner=False)
def fetch_universe_for_source(source: DataSourceType) -> pd.DataFrame:
    """소스에 따른 유니버스 목록."""
    if source == "demo":
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                    'JPM', 'V', 'WMT', 'UNH', 'HD', 'PG', 'JNJ', 'CRM']
        return pd.DataFrame({
            "ticker": tickers,
            "name": tickers,
            "sector": ["Technology"] * 7 + ["Finance"] * 2 + ["Consumer"] * 3 + ["Healthcare"] * 2 + ["Technology"],
            "industry": [""] * len(tickers),
        })

    if source == "real_us":
        from services.universe import fetch_us_universe
        return fetch_us_universe("sp500_github")

    if source == "real_kr":
        from services.universe import fetch_kr_universe, KRUniverseConfig
        return fetch_kr_universe(KRUniverseConfig(market="KOSPI", min_mcap_krw=1e12))

    return pd.DataFrame()


def get_default_tickers(source: DataSourceType) -> list:
    """소스에 맞는 기본 종목 목록."""
    if source == "real_kr":
        return ["005930", "000660", "035420", "051910", "006400", "035720", "003550", "105560"]
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]
