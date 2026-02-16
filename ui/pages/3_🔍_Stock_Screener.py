"""
종목 발굴 & 스크리너 페이지
히든챔피언 · 니치 종목 · 유니버스 탐색
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Stock Screener", page_icon="🔍", layout="wide")

# ───────────────────────────────────────
# Dark Theme CSS
# ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #6366f1;
    --accent: #8b5cf6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #334155;
    --glass: rgba(30, 41, 59, 0.8);
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); }

.glass-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
}

.metric-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
}
.metric-label { color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #6366f1, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.guide-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.6;
}
.guide-box strong { color: #a5b4fc; }
.guide-box .guide-title { font-weight: 700; color: #818cf8; margin-bottom: 0.5rem; font-size: 1rem; }

.badge-us { background: #3b82f6; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
.badge-kr { background: #ef4444; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
.badge-hidden { background: #f59e0b; color: #1e293b; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }

.section-header { color: var(--text-primary); font-size: 1.3rem; font-weight: 700; margin: 1.5rem 0 0.5rem; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h3 { color: var(--text-primary); }

.stock-row {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s;
}
.stock-row:hover { border-color: var(--primary); }
.stock-ticker { font-weight: 700; color: var(--text-primary); font-size: 1rem; }
.stock-name { color: var(--text-secondary); font-size: 0.85rem; }
.stock-score { font-weight: 700; font-size: 1.1rem; }
.stock-ret { font-size: 0.85rem; }
.pos { color: #10b981; }
.neg { color: #ef4444; }

.footer-text { text-align: center; color: var(--text-muted); padding: 1.5rem 0; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#94a3b8'),
    xaxis=dict(gridcolor='rgba(51,65,85,0.5)', zerolinecolor='rgba(51,65,85,0.5)'),
    yaxis=dict(gridcolor='rgba(51,65,85,0.5)', zerolinecolor='rgba(51,65,85,0.5)'),
    hoverlabel=dict(bgcolor='#1e293b', font_size=12, font_color='#f1f5f9'),
    margin=dict(l=0, r=0, t=40, b=0),
)
COLORS = ['#818cf8', '#a78bfa', '#c084fc', '#f472b6', '#fb923c', '#34d399', '#38bdf8', '#fbbf24']

# ───────────────────────────────────────
# Header
# ───────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #f59e0b 0%, #ef4444 50%, #8b5cf6 100%);
            padding: 1.8rem 2.5rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
            box-shadow: 0 20px 60px rgba(245,158,11,0.3); position: relative; overflow: hidden;'>
    <div style='position: absolute; top: -50%; right: -10%; width: 300px; height: 300px;
                background: rgba(255,255,255,0.05); border-radius: 50%;'></div>
    <h1 style='margin: 0; font-size: 2rem; position: relative;'>🔍 Stock Screener & Hidden Champions</h1>
    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9; position: relative;'>
        종목 발굴 · 니치 리더 · 히든챔피언 스캐너 · KR+US 블렌딩
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────
# Guide
# ───────────────────────────────────────
st.markdown("""
<div class="guide-box">
    <div class="guide-title">📖 이 페이지 사용법</div>
    이 페이지는 <strong>종목을 직접 발굴</strong>하는 도구입니다. 3가지 모드를 제공합니다:<br><br>
    <strong>🇺🇸 US 스크리너</strong> — S&P 500, NASDAQ-100, 또는 NASDAQ 전체에서 모멘텀+리스크 기반 스코어링<br>
    <strong>🇰🇷 KR 스크리너</strong> — KOSPI/KOSDAQ에서 시총·유동성 필터 후 모멘텀+리스크 스코어링<br>
    <strong>💎 히든챔피언</strong> — 소형~중형주(시총 $300M~$10B)에서 <strong>ROE + 이익성장률 + 모멘텀</strong> 복합 스코어로 잠재력 높은 종목 발굴<br><br>
    왼쪽 사이드바에서 모드와 필터를 선택한 뒤 <strong>"🚀 스크리닝 시작"</strong> 버튼을 누르세요.
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────
# Sidebar
# ───────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 스크리너 설정")

    mode = st.selectbox("모드 선택", [
        "🇺🇸 US 스크리너",
        "🇰🇷 KR 스크리너",
        "💎 히든챔피언 (소형주)",
    ])

    st.markdown("---")

    if "🇺🇸" in mode:
        st.markdown("### 🌐 US 설정")
        us_source = st.selectbox("유니버스", [
            "S&P 500 (GitHub)",
            "S&P 500 (Wikipedia)",
            "NASDAQ-100 (Wikipedia)",
            "NASDAQ 전체 (GitHub)",
        ])
        source_map = {
            "S&P 500 (GitHub)": "sp500_github",
            "S&P 500 (Wikipedia)": "sp500_wiki",
            "NASDAQ-100 (Wikipedia)": "nasdaq100_wiki",
            "NASDAQ 전체 (GitHub)": "nasdaq_listings_github",
        }
        us_source_key = source_map[us_source]
        us_top_n = st.slider("상위 N개 결과", 10, 100, 30)
        us_min_price = st.number_input("최소 주가 ($)", value=2.0, step=1.0)

    elif "🇰🇷" in mode:
        st.markdown("### 🇰🇷 한국 시장 설정")
        kr_market = st.selectbox("시장", ["KOSPI", "KOSDAQ"])
        kr_min_mcap = st.selectbox("최소 시총", [
            "1조원 이상", "5000억 이상", "1000억 이상", "제한 없음",
        ])
        mcap_map = {"1조원 이상": 1e12, "5000억 이상": 5e11, "1000억 이상": 1e11, "제한 없음": 0}
        kr_mcap_val = mcap_map[kr_min_mcap]
        kr_top_n = st.slider("상위 N개 결과", 10, 100, 30)

    else:  # 히든챔피언
        st.markdown("### 💎 히든챔피언 설정")
        hc_min_mcap = st.number_input("최소 시총 ($M)", value=300, step=50)
        hc_max_mcap = st.number_input("최대 시총 ($B)", value=10, step=1)
        hc_top_n = st.slider("상위 N개 결과", 10, 50, 30)

        st.markdown("---")
        st.markdown("### ⚖️ 스코어 가중치")
        hc_w_mom = st.slider("모멘텀", 0.0, 1.0, 0.50, 0.05)
        hc_w_quality = st.slider("퀄리티 (ROE+성장)", 0.0, 1.0, 0.40, 0.05)
        hc_w_risk = st.slider("리스크 패널티", 0.0, 1.0, 0.10, 0.05)

    st.markdown("---")
    run_btn = st.button("🚀 스크리닝 시작", use_container_width=True, type="primary")

# ───────────────────────────────────────
# Main content
# ───────────────────────────────────────
def _pct(v, digits=1):
    if pd.isna(v) or not np.isfinite(v):
        return "N/A"
    return f"{v*100:+.{digits}f}%"


def _fmt_mcap(v):
    if pd.isna(v) or not np.isfinite(v):
        return "N/A"
    if v >= 1e12:
        return f"${v/1e12:.1f}T"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def render_results(df: pd.DataFrame, mode_label: str):
    """결과 테이블 + 차트 렌더링"""
    if df.empty:
        st.warning("필터 조건에 맞는 종목이 없습니다. 조건을 완화해 보세요.")
        return

    # 상단 요약 카드
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">발굴 종목 수</div>
            <div class="metric-value">{len(df)}</div>
        </div>""", unsafe_allow_html=True)

    avg_ret60 = df["ret_60d"].mean() if "ret_60d" in df.columns else np.nan
    avg_ret1y = df["ret_1y"].mean() if "ret_1y" in df.columns else np.nan
    avg_score = df["score"].mean() if "score" in df.columns else np.nan

    with cols[1]:
        c = "pos" if (np.isfinite(avg_ret60) and avg_ret60 > 0) else "neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">평균 60일 수익률</div>
            <div class="metric-value {c}">{_pct(avg_ret60)}</div>
        </div>""", unsafe_allow_html=True)

    with cols[2]:
        c = "pos" if (np.isfinite(avg_ret1y) and avg_ret1y > 0) else "neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">평균 1년 수익률</div>
            <div class="metric-value {c}">{_pct(avg_ret1y)}</div>
        </div>""", unsafe_allow_html=True)

    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">평균 스코어</div>
            <div class="metric-value">{avg_score:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 차트 영역
    chart_col, table_col = st.columns([1, 1])

    with chart_col:
        st.markdown('<div class="section-header">📊 스코어 분포</div>', unsafe_allow_html=True)

        # 바 차트 — 상위 종목 스코어
        top20 = df.head(20)
        colors_bar = ['#10b981' if s > 0 else '#ef4444' for s in top20["score"]]

        fig_bar = go.Figure(data=[go.Bar(
            x=top20["ticker"],
            y=top20["score"],
            marker_color=colors_bar,
            text=top20["score"].apply(lambda x: f"{x:.2f}"),
            textposition='outside',
            textfont=dict(color='#94a3b8', size=10),
        )])
        fig_bar.update_layout(
            height=400,
            xaxis_title="종목",
            yaxis_title="복합 스코어",
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with table_col:
        st.markdown('<div class="section-header">📈 수익률 비교</div>', unsafe_allow_html=True)

        if "ret_60d" in df.columns and "ret_1y" in df.columns:
            top15 = df.head(15)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=top15["ret_60d"] * 100,
                y=top15["ret_1y"] * 100 if "ret_1y" in top15.columns else [0]*len(top15),
                mode='markers+text',
                text=top15["ticker"],
                textposition='top center',
                textfont=dict(color='#94a3b8', size=9),
                marker=dict(
                    size=top15["score"].clip(lower=0) * 8 + 8,
                    color=top15["score"],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=dict(text="스코어", side="right"), tickfont=dict(color='#94a3b8')),
                    line=dict(width=1, color='rgba(51,65,85,0.5)'),
                ),
                hovertemplate='%{text}<br>60일: %{x:.1f}%<br>1년: %{y:.1f}%<extra></extra>',
            ))
            fig_scatter.update_layout(
                height=400,
                xaxis_title="60일 수익률 (%)",
                yaxis_title="1년 수익률 (%)",
                **PLOTLY_DARK,
            )
            # 사분면 가이드선
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.4)")
            fig_scatter.add_vline(x=0, line_dash="dash", line_color="rgba(100,116,139,0.4)")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # 해석 가이드
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">💡 결과 해석법</div>
        • <strong>스코어</strong>: 모멘텀 + 퀄리티 - 리스크의 복합 Z-score. 높을수록 유망<br>
        • <strong>수익률 차트</strong>: 오른쪽 위(↗)에 있을수록 단기+장기 모두 강세. 원 크기 = 스코어<br>
        • <strong>60일 수익률</strong>: 최근 모멘텀. 양수면 최근 상승 추세<br>
        • <strong>1년 수익률</strong>: 장기 추세. 양수면 꾸준히 상승
    </div>
    """, unsafe_allow_html=True)

    # 섹터/산업 분포
    if "sector" in df.columns and df["sector"].notna().any():
        st.markdown('<div class="section-header">🏢 섹터 분포</div>', unsafe_allow_html=True)
        sector_counts = df["sector"].value_counts().head(10)
        if not sector_counts.empty:
            fig_pie = go.Figure(data=[go.Pie(
                labels=sector_counts.index,
                values=sector_counts.values,
                hole=0.4,
                marker=dict(colors=COLORS[:len(sector_counts)]),
                textinfo='label+percent',
                textfont=dict(color='#f1f5f9', size=11),
            )])
            fig_pie.update_layout(
                height=350,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # 전체 결과 테이블
    st.markdown('<div class="section-header">📋 전체 결과</div>', unsafe_allow_html=True)

    display_cols = ["rank", "ticker", "name"]
    if "sector" in df.columns:
        display_cols.append("sector")
    if "industry" in df.columns:
        display_cols.append("industry")
    display_cols.extend(["score", "ret_60d", "ret_1y", "vol_1y", "current_price"])
    if "marketCap" in df.columns:
        display_cols.append("marketCap")
    if "returnOnEquity" in df.columns:
        display_cols.append("returnOnEquity")
    if "earningsGrowth" in df.columns:
        display_cols.append("earningsGrowth")

    display_cols = [c for c in display_cols if c in df.columns]
    display_df = df[display_cols].copy()

    # 컬럼명 한국어화
    col_rename = {
        "rank": "순위", "ticker": "종목코드", "name": "종목명",
        "sector": "섹터", "industry": "산업",
        "score": "스코어", "ret_60d": "60일 수익률", "ret_1y": "1년 수익률",
        "vol_1y": "연변동성", "current_price": "현재가",
        "marketCap": "시가총액", "returnOnEquity": "ROE", "earningsGrowth": "이익성장률",
    }
    display_df = display_df.rename(columns=col_rename)

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)


# ───────────────────────────────────────
# Execution
# ───────────────────────────────────────
if run_btn:
    try:
        if "🇺🇸" in mode:
            from services.stock_screener import screen_us, USScreenerConfig

            with st.spinner(f"🇺🇸 {us_source} 스크리닝 중... (종목 수에 따라 1~5분 소요)"):
                progress = st.progress(0, text="시작 중...")

                def us_progress(pct, text):
                    progress.progress(min(pct, 1.0), text=text)

                cfg = USScreenerConfig(
                    source=us_source_key,
                    top_n=us_top_n,
                    min_price=us_min_price,
                    verbose=True,
                )
                result = screen_us(cfg, progress_cb=us_progress)
                progress.empty()

            render_results(result, "US")

        elif "🇰🇷" in mode:
            from services.stock_screener import screen_kr, KRScreenerConfig

            with st.spinner(f"🇰🇷 {kr_market} 스크리닝 중... (1~3분 소요)"):
                progress = st.progress(0, text="시작 중...")

                def kr_progress(pct, text):
                    progress.progress(min(pct, 1.0), text=text)

                cfg = KRScreenerConfig(
                    market=kr_market,
                    min_mcap_krw=kr_mcap_val,
                    top_n=kr_top_n,
                    verbose=True,
                )
                result = screen_kr(cfg, progress_cb=kr_progress)
                progress.empty()

            render_results(result, "KR")

        else:  # 히든챔피언
            from services.stock_screener import screen_hidden_champions, HiddenChampionConfig

            st.markdown("""
            <div class="guide-box">
                <div class="guide-title">⏳ 히든챔피언 스캐닝 중...</div>
                NASDAQ 전체 리스트에서 소형~중형주를 필터링하고, 각 종목의 가격 데이터와
                펀더멘탈(ROE, 이익성장률)을 수집합니다. <strong>5~15분</strong> 소요될 수 있습니다.
            </div>
            """, unsafe_allow_html=True)

            progress = st.progress(0, text="시작 중...")

            def hc_progress(pct, text):
                progress.progress(min(pct, 1.0), text=text)

            # 가중치 배분
            mom_total = hc_w_mom
            qual_total = hc_w_quality

            cfg = HiddenChampionConfig(
                min_mcap=hc_min_mcap * 1e6,
                max_mcap=hc_max_mcap * 1e9,
                top_n=hc_top_n,
                w_ret_60d=mom_total * 0.4,
                w_ret_1y=mom_total * 0.6,
                w_roe=qual_total * 0.5,
                w_earnings_growth=qual_total * 0.5,
                w_vol_penalty=hc_w_risk,
                verbose=True,
            )
            result = screen_hidden_champions(cfg, progress_cb=hc_progress)
            progress.empty()

            if not result.empty:
                st.markdown("""
                <div class="guide-box">
                    <div class="guide-title">💎 히든챔피언이란?</div>
                    대형주(S&P 500 등)에 가려져 잘 알려지지 않았지만, <strong>높은 수익성(ROE)</strong>과
                    <strong>이익 성장률</strong>을 보이는 소형~중형주입니다.<br>
                    • 시총 $300M~$10B: 대형주보다 성장 여력이 큼<br>
                    • ROE가 높음: 자본 효율성이 좋은 회사<br>
                    • 이익이 성장 중: 미래 가치가 현재보다 클 가능성<br>
                    • 모멘텀 양호: 시장이 이미 가치를 인식하기 시작
                </div>
                """, unsafe_allow_html=True)

            render_results(result, "Hidden Champion")

    except ImportError as e:
        st.error(f"필요한 패키지가 없습니다: {e}\n`pip install yfinance pykrx requests`")
    except Exception as e:
        st.error(f"스크리닝 중 오류 발생: {type(e).__name__}: {e}")

else:
    # 기본 화면 — 모드 설명
    st.markdown('<div class="section-header">📌 스크리닝 모드 비교</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #3b82f6; margin-top: 0;">🇺🇸 US 스크리너</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                <strong style="color: #f1f5f9;">유니버스 선택:</strong><br>
                • S&P 500 (대형 500개)<br>
                • NASDAQ-100 (기술주 100개)<br>
                • NASDAQ 전체 (3000+ 종목)<br><br>
                <strong style="color: #f1f5f9;">스코어링:</strong><br>
                60일 모멘텀 + 1년 모멘텀 - 변동성<br><br>
                <strong style="color: #f1f5f9;">소요 시간:</strong> 1~5분
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #ef4444; margin-top: 0;">🇰🇷 KR 스크리너</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                <strong style="color: #f1f5f9;">시장 선택:</strong><br>
                • KOSPI (대형·우량주)<br>
                • KOSDAQ (성장·기술주)<br><br>
                <strong style="color: #f1f5f9;">필터:</strong><br>
                시총, 최소 가격, 거래량 기준<br><br>
                <strong style="color: #f1f5f9;">소요 시간:</strong> 1~3분
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #f59e0b; margin-top: 0;">💎 히든챔피언</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                <strong style="color: #f1f5f9;">대상:</strong><br>
                NASDAQ 전체에서 소형~중형주<br>
                (시총 $300M ~ $10B)<br><br>
                <strong style="color: #f1f5f9;">스코어링:</strong><br>
                모멘텀 + ROE + 이익성장률<br>
                - 변동성 패널티<br><br>
                <strong style="color: #f1f5f9;">소요 시간:</strong> 5~15분
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">🤔 어떤 모드를 선택해야 할까?</div>
        • <strong>안정적인 대형주</strong>를 원하면 → 🇺🇸 US 스크리너 (S&P 500)<br>
        • <strong>성장 기술주</strong>를 원하면 → 🇺🇸 US 스크리너 (NASDAQ-100)<br>
        • <strong>한국 주식</strong>을 원하면 → 🇰🇷 KR 스크리너<br>
        • <strong>아직 많이 알려지지 않은 성장주</strong>를 원하면 → 💎 히든챔피언<br>
        • <strong>최대한 많은 종목</strong> 중에서 찾고 싶다면 → NASDAQ 전체 또는 히든챔피언
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div class="footer-text">Stock Screener · Hidden Champions · Quant Investment System v2</div>', unsafe_allow_html=True)
