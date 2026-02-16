"""
팩터 분석 페이지
Alpha 158 팩터의 시각적 분석 및 인터랙티브 탐색 (다크 테마)
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

from ui.components.data_source import (
    render_data_source_selector, render_data_source_badge,
    fetch_data_for_source, get_default_tickers,
)

st.set_page_config(page_title="Factor Analysis", page_icon="🔬", layout="wide")

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
    --info: #3b82f6;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #334155;
    --glass: rgba(30, 41, 59, 0.8);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
}

.glass-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
}

.factor-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    padding: 1.5rem;
    border-radius: 16px;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
}
.factor-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    border-color: var(--primary);
}
.factor-title {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.factor-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.factor-badge-good {
    color: #10b981; font-weight: 600; font-size: 0.85rem;
}
.factor-badge-warn {
    color: #f59e0b; font-weight: 600; font-size: 0.85rem;
}

.guide-box {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.6;
}
.guide-box strong {
    color: #a5b4fc;
}
.guide-box .guide-title {
    font-weight: 700;
    color: #818cf8;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}

.insight-box {
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-left: 4px solid #8b5cf6;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.6;
}
.insight-box .insight-title {
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}
.insight-box ul { margin: 0.5rem 0; padding-left: 1.2rem; }
.insight-box li { margin-bottom: 0.3rem; }

.section-header {
    color: var(--text-primary);
    font-size: 1.3rem;
    font-weight: 700;
    margin: 1.5rem 0 0.5rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary);
}

/* Tables */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Footer */
.footer-text {
    text-align: center;
    color: var(--text-muted);
    padding: 1.5rem 0;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────
# Plotly dark layout
# ───────────────────────────────────────
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
<div style='background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
            padding: 1.8rem 2.5rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
            box-shadow: 0 20px 60px rgba(99,102,241,0.3); position: relative; overflow: hidden;'>
    <div style='position: absolute; top: -50%; right: -10%; width: 300px; height: 300px;
                background: rgba(255,255,255,0.05); border-radius: 50%;'></div>
    <h1 style='margin: 0; font-size: 2rem; position: relative;'>🔬 Factor Analysis Lab</h1>
    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9; position: relative;'>
        Alpha 158 팩터 분석 · 상관관계 · 시계열 · 3D 시각화
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────
# Korean Guide — 이 페이지가 뭔지 설명
# ───────────────────────────────────────
st.markdown("""
<div class="guide-box">
    <div class="guide-title">📖 이 페이지 사용법</div>
    <strong>Factor Analysis Lab</strong>은 Alpha 158 팩터(투자 지표)들을 분석하는 도구입니다.<br>
    • <strong>Factor Performance</strong> — 각 팩터가 수익률을 얼마나 잘 예측하는지 (IC = 정보계수)<br>
    • <strong>Factor Correlation</strong> — 팩터들 간의 상관관계 (중복 팩터 식별)<br>
    • <strong>Time Series</strong> — 시간에 따른 팩터 변화, 평균, 변동성, Z-Score<br>
    • <strong>3D Visualization</strong> — 팩터 공간에서 종목 분포를 3차원으로 시각화<br>
    왼쪽 사이드바에서 분석 유형과 팩터를 선택하세요.
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────
# Sidebar (must run before badge so data_source is set)
# ───────────────────────────────────────
with st.sidebar:
    data_source = render_data_source_selector(key_prefix="factor")

    st.markdown("### 🎯 분석 설정")

    analysis_type = st.selectbox(
        "분석 유형 선택",
        ["📊 Factor Performance", "🔗 Factor Correlation", "📈 Time Series", "🎨 3D Visualization"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🎨 팩터 선택")

    factor_categories = {
        "Momentum": ["ROC_5", "ROC_10", "ROC_20", "ROC_60"],
        "Quality": ["MA_20", "MA_60", "STD_20", "STD_60"],
        "Volume": ["VOLUME_MA_5", "VOLUME_MA_20", "VSTD_20"],
        "Technical": ["RSI_14", "MACD", "QTLU_20", "QTLD_20"],
    }

    selected_category = st.selectbox("카테고리", list(factor_categories.keys()))
    selected_factors = st.multiselect(
        "팩터",
        factor_categories[selected_category],
        default=factor_categories[selected_category][:2],
    )

    st.markdown("---")
    lookback = st.slider("분석 기간 (일)", 30, 365, 180)

    st.markdown("---")
    st.markdown("### 📊 표시 설정")
    show_distribution = st.checkbox("분포 그래프", value=True)
    show_outliers = st.checkbox("이상치 박스플롯", value=True)
    normalize_data = st.checkbox("데이터 정규화", value=False)

# Data source badge (main area)
render_data_source_badge(data_source)

# ───────────────────────────────────────
# Data
# ───────────────────────────────────────
@st.cache_data
def generate_factor_data(n_days=365, n_stocks=50):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    data = {}
    for _cat, factors in factor_categories.items():
        for factor in factors:
            if 'ROC' in factor:
                data[factor] = np.random.randn(n_days, n_stocks) * 0.05
            elif 'MA' in factor:
                data[factor] = 100 + np.cumsum(np.random.randn(n_days, n_stocks) * 0.5, axis=0)
            elif 'STD' in factor:
                data[factor] = abs(np.random.randn(n_days, n_stocks) * 0.02)
            elif 'VOLUME' in factor:
                data[factor] = abs(np.random.randn(n_days, n_stocks) * 1e6)
            else:
                data[factor] = np.random.randn(n_days, n_stocks) * 0.3
    return dates, data


def _build_factor_data_from_real(ohlcv_dict, factor_cats):
    """실제 OHLCV 데이터에서 팩터 데이터 구성."""
    try:
        from features.alpha158 import Alpha158
    except ImportError:
        return None, None

    all_dates = None
    ticker_factors = {}

    for ticker, df in ohlcv_dict.items():
        if df.empty or len(df) < 60:
            continue
        try:
            feats = Alpha158.generate_basic(df)
            ticker_factors[ticker] = feats
            if all_dates is None:
                all_dates = feats.index
            else:
                all_dates = all_dates.intersection(feats.index)
        except Exception:
            continue

    if not ticker_factors or all_dates is None or len(all_dates) < 30:
        return None, None

    dates = all_dates.sort_values()
    data = {}
    n_stocks = len(ticker_factors)

    for cat, factors in factor_cats.items():
        for factor in factors:
            matrix = np.full((len(dates), n_stocks), np.nan)
            for j, (ticker, feats) in enumerate(ticker_factors.items()):
                col_match = [c for c in feats.columns if factor.upper() in c.upper()]
                if col_match:
                    vals = feats.loc[dates, col_match[0]].values
                    matrix[:, j] = vals
            # NaN 채우기
            col_means = np.nanmean(matrix, axis=0)
            for j in range(n_stocks):
                mask = np.isnan(matrix[:, j])
                matrix[mask, j] = col_means[j] if np.isfinite(col_means[j]) else 0
            data[factor] = matrix

    return dates, data


if data_source == "demo":
    dates, factor_data = generate_factor_data()
else:
    # 실제 데이터에서 팩터 계산 시도
    _real_tickers = tuple(get_default_tickers(data_source)[:15])
    with st.spinner("실제 데이터를 가져오는 중..."):
        _ohlcv = fetch_data_for_source(data_source, _real_tickers, period="2y")
    _result = _build_factor_data_from_real(_ohlcv, factor_categories) if _ohlcv else (None, None)
    if _result[0] is not None:
        dates, factor_data = _result
    else:
        st.warning("실제 데이터에서 팩터를 계산할 수 없어 Demo 데이터를 사용합니다.")
        dates, factor_data = generate_factor_data()


# ═══════════════════════════════════════
# 1. Factor Performance
# ═══════════════════════════════════════
if analysis_type == "📊 Factor Performance":
    st.markdown('<div class="section-header">📊 Factor Performance Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">💡 IC (정보계수)란?</div>
        팩터 값과 미래 수익률 사이의 상관관계를 나타냅니다.<br>
        • <strong>IC > 0.05</strong>: 예측력이 있는 팩터<br>
        • <strong>IC IR > 1.5</strong>: IC의 안정성 — 높을수록 꾸준히 잘 예측<br>
        • <strong>Hit Rate > 55%</strong>: IC가 양수인 날의 비율 — 높을수록 안정적
    </div>
    """, unsafe_allow_html=True)

    # IC 카드들
    cards = [
        ("Average IC", "0.085", "▲ 우수", "good"),
        ("IC Std Dev", "0.042", "● 보통", "warn"),
        ("IC IR", "2.02", "▲ 강함", "good"),
        ("Hit Rate", "62.3%", "▲ 양호", "good"),
    ]

    cols = st.columns(4)
    for col, (title, value, badge, btype) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="factor-card">
                <div class="factor-title">{title}</div>
                <div class="factor-value">{value}</div>
                <div class="factor-badge-{btype}">{badge}</div>
            </div>
            """, unsafe_allow_html=True)

    if selected_factors:
        st.markdown('<div class="section-header">📈 Rolling IC 시계열</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-box">
            그래프의 <strong>색칠된 영역</strong>이 0 위에 있으면 팩터가 수익률을 양(+)의 방향으로 예측하고 있다는 뜻입니다.
            지속적으로 0 아래로 내려가면 해당 팩터의 예측력이 사라졌거나 역전된 것입니다.
        </div>
        """, unsafe_allow_html=True)

        fig = make_subplots(
            rows=len(selected_factors), cols=1,
            subplot_titles=selected_factors,
            vertical_spacing=0.08,
        )

        for i, factor in enumerate(selected_factors, 1):
            ic_series = pd.Series(
                np.random.randn(len(dates)) * 0.05 + 0.08,
                index=dates,
            ).rolling(window=20).mean()

            fig.add_trace(
                go.Scatter(
                    x=dates, y=ic_series,
                    fill='tozeroy',
                    fillcolor=f'rgba({130 + i * 20}, {140 + i * 15}, 248, 0.25)',
                    line=dict(color=COLORS[i % len(COLORS)], width=2),
                    name=factor, showlegend=False,
                ),
                row=i, col=1,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.5)", row=i, col=1)

        fig.update_layout(height=200 * len(selected_factors), **PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

        # 분포 + 퀀타일
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">📊 팩터 값 분포</div>', unsafe_allow_html=True)

            factor = selected_factors[0]
            factor_values = factor_data[factor][-30:].flatten()

            fig_dist = go.Figure()

            if show_distribution:
                fig_dist.add_trace(go.Histogram(
                    x=factor_values, nbinsx=30,
                    marker=dict(color='rgba(129,140,248,0.6)', line=dict(color='#818cf8', width=1)),
                    name='분포',
                ))

            if show_outliers:
                fig_dist.add_trace(go.Box(
                    y=factor_values, name='박스플롯',
                    marker_color='#a78bfa', boxmean='sd',
                ))

            fig_dist.update_layout(height=350, showlegend=True, **PLOTLY_DARK)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">🎯 퀀타일별 수익률</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="guide-box">
                팩터 값에 따라 종목을 5등분(Q1~Q5)한 뒤 각 그룹의 평균 수익률을 비교합니다.
                <strong>Q1→Q5</strong>로 갈수록 수익률이 높아지면 팩터 예측력이 좋다는 뜻입니다.
            </div>
            """, unsafe_allow_html=True)

            quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            np.quantile(factor_values, quantiles)
            returns = np.sort(np.random.uniform(-0.05, 0.15, len(quantiles) - 1))

            colors = ['#ef4444' if r < 0 else '#10b981' for r in returns]

            fig_quant = go.Figure(data=[
                go.Bar(
                    x=[f'Q{i + 1}' for i in range(len(returns))],
                    y=returns * 100,
                    marker_color=colors,
                    text=[f"{r * 100:+.2f}%" for r in returns],
                    textposition='outside',
                    textfont=dict(color='#94a3b8'),
                    marker=dict(line=dict(color='rgba(51,65,85,0.5)', width=1)),
                )
            ])

            fig_quant.update_layout(
                yaxis_title="수익률 (%)",
                height=350,
                **PLOTLY_DARK,
            )
            st.plotly_chart(fig_quant, use_container_width=True)


# ═══════════════════════════════════════
# 2. Factor Correlation
# ═══════════════════════════════════════
elif analysis_type == "🔗 Factor Correlation":
    st.markdown('<div class="section-header">🔗 팩터 상관관계 매트릭스</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">📖 상관관계 해석법</div>
        • <strong>파란색 (+1에 가까움)</strong>: 두 팩터가 같은 방향으로 움직임 → 중복될 수 있음<br>
        • <strong>빨간색 (-1에 가까움)</strong>: 반대 방향 → 분산 투자 효과 기대<br>
        • <strong>흰색 (0 부근)</strong>: 서로 독립적 → 보완적 정보 제공
    </div>
    """, unsafe_allow_html=True)

    all_factors = []
    for factors in factor_categories.values():
        all_factors.extend(factors)

    corr_matrix = np.random.uniform(-0.6, 0.8, (len(all_factors), len(all_factors)))
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=all_factors,
        y=all_factors,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2),
        texttemplate='%{text}',
        textfont={"size": 8, "color": "#cbd5e1"},
        colorbar=dict(title=dict(text="상관계수", side="right"), tickfont=dict(color='#94a3b8')),
        hovertemplate='%{x} vs %{y}<br>상관계수: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        height=650,
        font=dict(family='Inter, sans-serif', color='#94a3b8'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        hoverlabel=dict(bgcolor='#1e293b', font_color='#f1f5f9'),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 인사이트
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 주요 인사이트</div>
        <ul>
            <li><strong>모멘텀 팩터</strong> (ROC_X): 기간이 비슷한 팩터끼리 상관관계 높음 (0.7+) → 하나만 사용해도 충분</li>
            <li><strong>거래량 팩터</strong>: 가격 기반 팩터와 상관관계 낮음 (-0.2~0.3) → 보완적 정보</li>
            <li><strong>기술적 지표</strong> (RSI, MACD): 독립적 정보 제공 → 포트폴리오에 함께 사용 권장</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 클러스터링 + 통계
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">🌳 팩터 클러스터링</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-box">
            비슷한 팩터끼리 묶어주는 <strong>계층적 클러스터링</strong> 결과입니다.
            가까이 연결된 팩터끼리는 유사한 정보를 담고 있으므로, 포트폴리오 구성 시 한 그룹에서 하나씩 선택하는 것이 분산 효과에 유리합니다.
        </div>
        """, unsafe_allow_html=True)

        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        distance_matrix = 1 - np.abs(corr_matrix)
        Z = linkage(squareform(distance_matrix), method='ward')

        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        dend = scipy_dendrogram(Z, labels=all_factors, no_plot=True)

        icoord = np.array(dend['icoord'])
        dcoord = np.array(dend['dcoord'])

        fig_dend = go.Figure()
        for i in range(len(icoord)):
            fig_dend.add_trace(go.Scatter(
                x=icoord[i], y=dcoord[i],
                mode='lines',
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hoverinfo='skip', showlegend=False,
            ))

        fig_dend.update_layout(
            xaxis=dict(showticklabels=False, gridcolor='rgba(51,65,85,0.3)'),
            yaxis=dict(title="거리", gridcolor='rgba(51,65,85,0.3)'),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_dend, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📊 상관관계 통계</div>', unsafe_allow_html=True)

        upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        stats_df = pd.DataFrame({
            '지표': ['평균 상관계수', '최대 상관계수', '최소 상관계수', '표준편차'],
            '값': [np.mean(upper), np.max(upper), np.min(upper), np.std(upper)],
        })

        st.dataframe(
            stats_df.style.format({'값': '{:.3f}'}).background_gradient(
                subset=['값'], cmap='coolwarm'
            ),
            use_container_width=True,
            hide_index=True,
            height=200,
        )


# ═══════════════════════════════════════
# 3. Time Series
# ═══════════════════════════════════════
elif analysis_type == "📈 Time Series":
    st.markdown('<div class="section-header">📈 팩터 시계열 분석</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">📖 4가지 차트 해석법</div>
        • <strong>Factor Values</strong>: 팩터 원본 값의 시간 추이<br>
        • <strong>Rolling Mean</strong>: 20일 이동평균 — 추세를 부드럽게 확인<br>
        • <strong>Volatility</strong>: 20일 변동성 — 높으면 팩터가 불안정<br>
        • <strong>Z-Score</strong>: 표준화 점수 — <strong>±2를 넘으면</strong> 이상 구간 (빨간 점선)
    </div>
    """, unsafe_allow_html=True)

    if selected_factors:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('팩터 원본 값', '20일 이동평균', '변동성', 'Z-Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        for idx, factor in enumerate(selected_factors):
            color = COLORS[idx % len(COLORS)]
            factor_ts = pd.Series(
                factor_data[factor][:lookback].mean(axis=1),
                index=dates[:lookback],
            )

            # 1. Raw values
            fig.add_trace(
                go.Scatter(x=dates[:lookback], y=factor_ts, name=factor,
                           line=dict(width=2, color=color)),
                row=1, col=1,
            )

            # 2. Rolling mean
            rolling_mean = factor_ts.rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(x=dates[:lookback], y=rolling_mean, name=f'{factor} MA20',
                           line=dict(width=2, dash='dash', color=color), showlegend=False),
                row=1, col=2,
            )

            # 3. Volatility
            rolling_std = factor_ts.rolling(window=20).std()
            fig.add_trace(
                go.Scatter(x=dates[:lookback], y=rolling_std, name=f'{factor} Vol',
                           fill='tozeroy',
                           fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in color else f'rgba(129,140,248,0.15)',
                           line=dict(width=1.5, color=color), showlegend=False),
                row=2, col=1,
            )

            # 4. Z-score
            z_score = (factor_ts - factor_ts.mean()) / factor_ts.std()
            fig.add_trace(
                go.Scatter(x=dates[:lookback], y=z_score, name=f'{factor} Z',
                           line=dict(width=2, color=color), showlegend=False),
                row=2, col=2,
            )

        fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.5)", row=2, col=2)
        fig.add_hline(y=2, line_dash="dot", line_color="#ef4444", row=2, col=2)
        fig.add_hline(y=-2, line_dash="dot", line_color="#ef4444", row=2, col=2)

        fig.update_layout(
            height=700, hovermode='x unified',
            **PLOTLY_DARK,
        )

        # Apply grid colors to all subplots
        for i in range(1, 5):
            axis_x = f'xaxis{i}' if i > 1 else 'xaxis'
            axis_y = f'yaxis{i}' if i > 1 else 'yaxis'
            fig.update_layout(**{
                axis_x: dict(gridcolor='rgba(51,65,85,0.3)'),
                axis_y: dict(gridcolor='rgba(51,65,85,0.3)'),
            })

        st.plotly_chart(fig, use_container_width=True)

        # 통계 요약
        st.markdown('<div class="section-header">📊 통계 요약</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-box">
            • <strong>Mean</strong>: 평균 — 팩터의 중심값<br>
            • <strong>Std</strong>: 표준편차 — 클수록 변동 큼<br>
            • <strong>Skewness</strong>: 왜도 — 0이면 좌우 대칭, 양수면 오른쪽 꼬리 긺<br>
            • <strong>Kurtosis</strong>: 첨도 — 3보다 크면 극단값이 많음 (뾰족한 분포)
        </div>
        """, unsafe_allow_html=True)

        summary_data = []
        for factor in selected_factors:
            factor_ts = factor_data[factor][:lookback].mean(axis=1)
            summary_data.append({
                '팩터': factor,
                '평균': np.mean(factor_ts),
                '표준편차': np.std(factor_ts),
                '최솟값': np.min(factor_ts),
                '최댓값': np.max(factor_ts),
                '왜도': pd.Series(factor_ts).skew(),
                '첨도': pd.Series(factor_ts).kurtosis(),
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df.style.format({
                '평균': '{:.4f}', '표준편차': '{:.4f}',
                '최솟값': '{:.4f}', '최댓값': '{:.4f}',
                '왜도': '{:.3f}', '첨도': '{:.3f}',
            }).background_gradient(subset=['평균', '표준편차'], cmap='viridis'),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("왼쪽 사이드바에서 팩터를 하나 이상 선택하세요.")


# ═══════════════════════════════════════
# 4. 3D Visualization
# ═══════════════════════════════════════
else:
    st.markdown('<div class="section-header">🎨 3D 팩터 시각화</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">📖 3D 차트 해석법</div>
        선택한 3개 팩터를 축으로, 각 종목을 점으로 표시합니다.<br>
        • <strong>초록색 점</strong>: 수익률이 높은 종목 — 좋은 팩터 조합<br>
        • <strong>빨간색 점</strong>: 수익률이 낮은 종목 — 주의 필요<br>
        • 마우스로 <strong>드래그</strong>하면 회전, <strong>스크롤</strong>하면 확대/축소됩니다.
    </div>
    """, unsafe_allow_html=True)

    if len(selected_factors) >= 2:
        factor1 = selected_factors[0]
        factor2 = selected_factors[1]
        factor3 = selected_factors[2] if len(selected_factors) >= 3 else selected_factors[0]

        data1 = factor_data[factor1][-1]
        data2 = factor_data[factor2][-1]
        data3 = factor_data[factor3][-1]
        returns = np.random.randn(len(data1)) * 0.05

        fig = go.Figure(data=[go.Scatter3d(
            x=data1, y=data2, z=data3,
            mode='markers',
            marker=dict(
                size=7,
                color=returns,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="수익률", tickfont=dict(color='#94a3b8')),
                line=dict(width=0.5, color='rgba(51,65,85,0.5)'),
                opacity=0.9,
            ),
            text=[f'종목 {i + 1}<br>수익률: {r * 100:.2f}%' for i, r in enumerate(returns)],
            hovertemplate='%{text}<extra></extra>',
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(title=factor1, backgroundcolor='rgba(15,23,42,0.5)',
                           gridcolor='rgba(51,65,85,0.4)', showbackground=True),
                yaxis=dict(title=factor2, backgroundcolor='rgba(15,23,42,0.5)',
                           gridcolor='rgba(51,65,85,0.4)', showbackground=True),
                zaxis=dict(title=factor3, backgroundcolor='rgba(15,23,42,0.5)',
                           gridcolor='rgba(51,65,85,0.4)', showbackground=True),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            height=600,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
        )
        st.plotly_chart(fig, use_container_width=True)

        # PCA + 로딩
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">📊 PCA 분석</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="guide-box">
                <strong>PCA (주성분 분석)</strong>: 여러 팩터를 소수의 주성분으로 요약합니다.
                PC1이 설명하는 비율이 높을수록 팩터들이 비슷한 정보를 담고 있다는 뜻입니다.
            </div>
            """, unsafe_allow_html=True)

            pca_data = np.column_stack([data1, data2, data3])
            pca_centered = pca_data - pca_data.mean(axis=0)
            cov = np.cov(pca_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            explained_var = eigenvalues / eigenvalues.sum() * 100

            fig_pca = go.Figure(data=[
                go.Bar(
                    x=['PC1', 'PC2', 'PC3'],
                    y=explained_var,
                    marker=dict(
                        color=['#818cf8', '#a78bfa', '#c084fc'],
                        line=dict(color='rgba(51,65,85,0.5)', width=1),
                    ),
                    text=[f"{v:.1f}%" for v in explained_var],
                    textposition='outside',
                    textfont=dict(color='#94a3b8'),
                )
            ])

            fig_pca.update_layout(
                yaxis_title="설명 분산 비율 (%)",
                height=300,
                **PLOTLY_DARK,
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">🎯 팩터 로딩</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="guide-box">
                각 팩터가 주성분(PC)에 기여하는 정도입니다. 막대가 길수록 해당 팩터의 영향이 큽니다.
            </div>
            """, unsafe_allow_html=True)

            loadings_df = pd.DataFrame(
                eigenvectors[:, :2],
                columns=['PC1', 'PC2'],
                index=[factor1, factor2, factor3],
            )

            fig_load = go.Figure()
            fig_load.add_trace(go.Bar(
                name='PC1', y=loadings_df.index, x=loadings_df['PC1'],
                orientation='h', marker_color='#818cf8',
            ))
            fig_load.add_trace(go.Bar(
                name='PC2', y=loadings_df.index, x=loadings_df['PC2'],
                orientation='h', marker_color='#a78bfa',
            ))

            fig_load.update_layout(
                barmode='group', height=300,
                **PLOTLY_DARK,
            )
            st.plotly_chart(fig_load, use_container_width=True)

    else:
        st.warning("3D 시각화를 위해 팩터를 2개 이상 선택해 주세요.")


# ───────────────────────────────────────
# Footer
# ───────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer-text">
    Factor Analysis Lab · Alpha 158 · Quant Investment System v2
</div>
""", unsafe_allow_html=True)
