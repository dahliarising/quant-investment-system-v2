"""
Quant Investment System v2 - 통합 대시보드
모든 기능을 하나의 프로페셔널 UI로 통합
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ───────────────────────────────────────
# Page Config
# ───────────────────────────────────────
st.set_page_config(
    page_title="Quant Investment System v2",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────
# Premium CSS
# ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --accent: #8b5cf6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --info: #3b82f6;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --bg-hover: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #334155;
    --glass: rgba(30, 41, 59, 0.8);
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -1px rgba(0,0,0,0.2);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Dark theme override */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    padding: 1.8rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-header h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    margin: 0.3rem 0 0;
    opacity: 0.85;
    font-size: 0.95rem;
    font-weight: 400;
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--primary);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0.3rem 0;
    line-height: 1;
}
.metric-change {
    font-size: 0.8rem;
    font-weight: 600;
}
.positive { color: var(--success); }
.negative { color: var(--danger); }
.neutral { color: var(--text-secondary); }

/* Section Headers */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
    display: inline-block;
}

/* Regime Badge */
.regime-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.regime-normal { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.regime-warning { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.regime-breakdown { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.regime-warmup { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

/* Glass Panel */
.glass-panel {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
}

/* Factor Bar */
.factor-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
}
.factor-name {
    width: 90px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
}
.factor-fill {
    height: 8px;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--glass);
    padding: 0.4rem;
    border-radius: 12px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    color: var(--text-secondary);
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 700;
    border: none;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: var(--text-secondary) !important;
}

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Hide Streamlit defaults */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#94a3b8'),
    margin=dict(l=0, r=0, t=40, b=0),
    hovermode='x unified',
)

COLOR_PALETTE = {
    "primary": "#6366f1",
    "accent": "#8b5cf6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",
    "cyan": "#06b6d4",
    "pink": "#ec4899",
    "gradient": ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe"],
    "heatmap": ["#ef4444", "#f59e0b", "#fbbf24", "#10b981", "#06b6d4"],
}


def _pct(v, decimals=2):
    if pd.isna(v):
        return "N/A"
    return f"{v * 100:+.{decimals}f}%"


def _num(v, decimals=2):
    if pd.isna(v):
        return "N/A"
    return f"{v:,.{decimals}f}"


def metric_card(label, value, change=None, change_type="positive"):
    change_html = ""
    if change is not None:
        arrow = "" if change_type == "positive" else ""
        change_html = f'<div class="metric-change {change_type}">{arrow} {change}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """


def regime_badge(regime: str) -> str:
    cls = f"regime-{regime.lower()}"
    icons = {"NORMAL": "", "WARNING": "", "BREAKDOWN": "", "WARMUP": ""}
    icon = icons.get(regime, "")
    return f'<span class="regime-badge {cls}">{icon} {regime}</span>'


# ───────────────────────────────────────
# Sample Data Generator (실제 데이터 연결 전 시연용)
# ───────────────────────────────────────
@st.cache_data
def generate_demo_data(start_date, end_date, top_n=10):
    np.random.seed(42)
    dates = pd.date_range(start_date, end_date, freq='B')

    # Portfolio performance
    daily_ret = np.random.randn(len(dates)) * 0.012 + 0.0003
    bench_ret = np.random.randn(len(dates)) * 0.010 + 0.0002
    portfolio_value = 100_000_000 * np.cumprod(1 + daily_ret)
    benchmark_value = 100_000_000 * np.cumprod(1 + bench_ret)

    perf = pd.DataFrame({
        'Date': dates,
        'Portfolio': portfolio_value,
        'Benchmark': benchmark_value,
        'Daily_Return': daily_ret,
    })

    # Top stocks with factor scores — 확장된 유니버스
    symbols = [
        'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'JPM', 'V',
        'WMT', 'UNH', 'MA', 'HD', 'PG', 'JNJ', 'CRM', 'ADBE', 'NFLX', 'AMD',
        'COST', 'LLY', 'ABBV', 'MRK', 'PEP', 'TMO', 'ORCL', 'ACN', 'CSCO', 'MCD',
        'ABT', 'DHR', 'TXN', 'NEE', 'PM', 'UPS', 'MS', 'RTX', 'LOW', 'INTC',
    ]
    sectors = [
        'Technology', 'Technology', 'Technology', 'Communication', 'Consumer',
        'Communication', 'Consumer', 'Technology', 'Finance', 'Finance',
        'Consumer', 'Healthcare', 'Finance', 'Consumer', 'Consumer',
        'Healthcare', 'Technology', 'Technology', 'Communication', 'Technology',
        'Consumer', 'Healthcare', 'Healthcare', 'Healthcare', 'Consumer',
        'Healthcare', 'Technology', 'Technology', 'Technology', 'Consumer',
        'Healthcare', 'Healthcare', 'Technology', 'Utilities', 'Consumer',
        'Industrials', 'Finance', 'Industrials', 'Consumer', 'Technology',
    ]

    n = min(top_n, len(symbols))
    stocks = pd.DataFrame({
        'ticker': symbols[:n],
        'sector': sectors[:n],
        'current_price': np.random.uniform(80, 800, n).round(2),
        'factor_momentum': np.random.uniform(-2, 2, n).round(3),
        'factor_value': np.random.uniform(-2, 2, n).round(3),
        'factor_quality': np.random.uniform(-2, 2, n).round(3),
        'factor_risk': np.random.uniform(-2, 2, n).round(3),
        'factor_total': np.random.uniform(-1.5, 1.5, n).round(3),
        'score_combined': np.random.uniform(0.3, 0.95, n).round(3),
        'pred_return': np.random.uniform(-0.1, 0.35, n).round(4),
        'weight': np.random.dirichlet(np.ones(n)).round(4),
        'ret_1m': np.random.uniform(-0.08, 0.15, n).round(4),
        'ret_3m': np.random.uniform(-0.15, 0.3, n).round(4),
        'ret_12m': np.random.uniform(-0.2, 0.8, n).round(4),
        'vol_3m': np.random.uniform(0.15, 0.45, n).round(4),
        'trailingPE': np.random.uniform(8, 60, n).round(1),
        'returnOnEquity': np.random.uniform(0.05, 0.45, n).round(3),
    }).sort_values('score_combined', ascending=False)
    stocks['rank'] = range(1, len(stocks) + 1)

    # Regime data
    regime_dates = pd.date_range(start_date, end_date, freq='B')
    rankic_values = np.random.randn(len(regime_dates)) * 0.15 + 0.05
    regime_df = pd.DataFrame({
        'date': regime_dates,
        'rankic': rankic_values,
    })
    mu = pd.Series(rankic_values).rolling(60).mean()
    sd = pd.Series(rankic_values).rolling(60).std().replace(0, np.nan)
    z = (pd.Series(rankic_values) - mu) / sd
    regime_df['z'] = z.values
    regime_df['regime'] = z.apply(
        lambda x: 'BREAKDOWN' if x <= -2 else ('WARNING' if x <= -1 else ('WARMUP' if pd.isna(x) else 'NORMAL'))
    ).values

    # Monthly returns
    monthly = perf.set_index('Date').resample('ME')['Daily_Return'].sum()

    # Sector distribution
    sector_dist = stocks.groupby('sector')['weight'].sum().reset_index()

    return perf, stocks, regime_df, monthly, sector_dist


# ───────────────────────────────────────
# Header
# ───────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>Quant Investment System</h1>
    <p>AI-Powered Multi-Factor & ML Hybrid Investment Platform</p>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────
# Sidebar
# ───────────────────────────────────────
with st.sidebar:
    st.markdown("### Market")
    market = st.selectbox(
        "Market", ["US Equities", "KR KOSPI/KOSDAQ"],
        label_visibility="collapsed",
    )

    st.markdown("### Analysis Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End", value=datetime.now())

    st.markdown("---")
    st.markdown("### Strategy")
    model_name = st.selectbox("ML Model", ["lgbm", "xgb", "rf", "ridge"])
    horizon = st.selectbox("Horizon", ["1y", "30d"])
    label_mode = st.selectbox("Label Mode", ["alpha_vs_spy", "absolute"])
    top_n = st.slider("Portfolio Size", 5, 20, 10)
    rebalance = st.select_slider("Rebalance", ["Daily", "Weekly", "Monthly", "Quarterly"], "Monthly")

    st.markdown("---")
    st.markdown("### Factor Weights")
    w_mom = st.slider("Momentum", 0.0, 1.0, 0.25, 0.05)
    w_val = st.slider("Value", 0.0, 1.0, 0.25, 0.05)
    w_qual = st.slider("Quality", 0.0, 1.0, 0.25, 0.05)
    w_risk = st.slider("Risk", 0.0, 1.0, 0.25, 0.05)

    st.markdown("---")
    st.markdown("### Risk Controls")
    max_position_pct = st.slider("Max Position %", 5, 30, 15)
    max_sector_pct = st.slider("Max Sector %", 20, 60, 40)
    sector_neutral = st.checkbox("Sector Neutralize", True)

    st.markdown("---")
    run_btn = st.button("Run Strategy", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
                border-radius: 12px; padding: 0.8rem 1rem; margin-top: 0.5rem;">
        <div style="color: #fbbf24; font-weight: 700; font-size: 0.9rem;">🔍 종목 발굴</div>
        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;">
            더 많은 종목이나 히든챔피언을 찾으려면<br>
            <strong style="color: #f1f5f9;">Stock Screener</strong> 페이지를 사용하세요
        </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────
# Data
# ───────────────────────────────────────
perf_df, stocks_df, regime_df, monthly_ret, sector_dist = generate_demo_data(start_date, end_date, top_n)

# 예측 로깅 (Run Strategy 클릭 시)
if run_btn:
    try:
        from services.prediction_tracker import log_predictions
        logged = log_predictions(stocks_df, model=model_name, horizon=horizon)
        if not logged.empty:
            st.toast(f"✅ {len(logged)}개 종목 예측이 기록되었습니다.", icon="📊")
    except Exception:
        pass

# Current regime
current_regime = regime_df.dropna(subset=['regime']).iloc[-1]['regime'] if not regime_df.empty else "WARMUP"

# ───────────────────────────────────────
# Top Metrics Row
# ───────────────────────────────────────
total_ret = (perf_df['Portfolio'].iloc[-1] / perf_df['Portfolio'].iloc[0] - 1)
bench_ret = (perf_df['Benchmark'].iloc[-1] / perf_df['Benchmark'].iloc[0] - 1)
alpha = total_ret - bench_ret
sharpe = float(perf_df['Daily_Return'].mean() / perf_df['Daily_Return'].std() * np.sqrt(252))
running_max = perf_df['Portfolio'].expanding().max()
drawdown = ((perf_df['Portfolio'] - running_max) / running_max)
max_dd = drawdown.min()
win_rate = (perf_df['Daily_Return'] > 0).mean()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    ct = "positive" if total_ret > 0 else "negative"
    st.markdown(metric_card("Total Return", _pct(total_ret, 1), f"Alpha: {_pct(alpha, 1)}", ct), unsafe_allow_html=True)
with col2:
    ct = "positive" if sharpe > 1.0 else "neutral"
    st.markdown(metric_card("Sharpe Ratio", f"{sharpe:.2f}", "Annualized", ct), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("Max Drawdown", _pct(max_dd, 1), f"Current: {_pct(drawdown.iloc[-1], 1)}", "negative"), unsafe_allow_html=True)
with col4:
    ct = "positive" if win_rate > 0.5 else "negative"
    st.markdown(metric_card("Win Rate", f"{win_rate*100:.1f}%", f"{len(perf_df)} trading days", ct), unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Market Regime</div>
        <div style="margin: 0.5rem 0;">{regime_badge(current_regime)}</div>
        <div class="metric-change neutral">Model: {model_name.upper()}</div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────
# Main Tabs
# ───────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Rankings & Factors",
    "Regime Analysis",
    "Scenario Analysis",
    "Model Explain",
    "Portfolio",
    "Trade Log",
])


# ═══════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════
with tab1:
    # Performance chart
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=perf_df['Date'], y=perf_df['Portfolio'],
        name='Portfolio', line=dict(color=COLOR_PALETTE["primary"], width=2.5),
        fill='tonexty', fillcolor='rgba(99,102,241,0.08)',
    ))
    fig_perf.add_trace(go.Scatter(
        x=perf_df['Date'], y=perf_df['Benchmark'],
        name='Benchmark', line=dict(color=COLOR_PALETTE["danger"], width=1.5, dash='dot'),
    ))
    fig_perf.update_layout(
        **PLOTLY_LAYOUT, height=380,
        title=dict(text="Cumulative Performance", font=dict(size=16, color='#e2e8f0')),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(color='#94a3b8')),
        xaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
        yaxis=dict(gridcolor='rgba(148,163,184,0.1)', title="Value"),
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        # Monthly returns heatmap
        st.markdown('<div class="section-title">Monthly Returns</div>', unsafe_allow_html=True)
        colors = [COLOR_PALETTE["success"] if r > 0 else COLOR_PALETTE["danger"] for r in monthly_ret.values]
        fig_monthly = go.Figure(data=[go.Bar(
            x=monthly_ret.index.strftime('%Y-%m'),
            y=monthly_ret.values * 100,
            marker_color=colors,
            text=[f"{r*100:+.1f}%" for r in monthly_ret.values],
            textposition='outside',
            textfont=dict(size=9, color='#94a3b8'),
        )])
        fig_monthly.update_layout(
            **PLOTLY_LAYOUT, height=280,
            yaxis_title="Return (%)",
            xaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
            yaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        # Sector pie
        st.markdown('<div class="section-title">Sector Allocation</div>', unsafe_allow_html=True)
        fig_sector = go.Figure(data=[go.Pie(
            labels=sector_dist['sector'],
            values=sector_dist['weight'],
            hole=0.55,
            marker=dict(colors=px.colors.qualitative.Pastel),
            textinfo='label+percent',
            textfont=dict(size=10, color='#e2e8f0'),
        )])
        fig_sector.update_layout(
            **PLOTLY_LAYOUT, height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    # Drawdown chart
    st.markdown('<div class="section-title">Drawdown</div>', unsafe_allow_html=True)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=perf_df['Date'], y=drawdown * 100,
        fill='tozeroy', fillcolor='rgba(239,68,68,0.1)',
        line=dict(color=COLOR_PALETTE["danger"], width=1.5),
    ))
    fig_dd.update_layout(
        **PLOTLY_LAYOUT, height=200, showlegend=False,
        yaxis_title="Drawdown (%)",
        xaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
        yaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ═══════════════════════════════════════
# TAB 2: Rankings & Factors
# ═══════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Stock Rankings (Factor + ML)</div>', unsafe_allow_html=True)

    # Factor radar chart for top stock
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        top_stock = stocks_df.iloc[0]
        categories = ['Momentum', 'Value', 'Quality', 'Risk']
        values = [
            top_stock['factor_momentum'], top_stock['factor_value'],
            top_stock['factor_quality'], top_stock['factor_risk'],
        ]
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[max(0, v + 2) for v in values],
            theta=categories,
            fill='toself',
            fillcolor='rgba(99,102,241,0.2)',
            line=dict(color=COLOR_PALETTE["primary"], width=2),
            name=top_stock['ticker'],
        ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT, height=300,
            title=dict(text=f"#{1} {top_stock['ticker']} Factor Profile", font=dict(size=14, color='#e2e8f0')),
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 4], gridcolor='rgba(148,163,184,0.2)', color='#64748b'),
                angularaxis=dict(gridcolor='rgba(148,163,184,0.2)', color='#94a3b8'),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="glass-panel">
            <h3 style="color: #e2e8f0; margin-top:0;">Top Pick: {top_stock['ticker']}</h3>
            <table style="width:100%; color: #94a3b8;">
                <tr><td>Sector</td><td style="text-align:right; color:#e2e8f0;">{top_stock['sector']}</td></tr>
                <tr><td>Price</td><td style="text-align:right; color:#e2e8f0;">${top_stock['current_price']:.2f}</td></tr>
                <tr><td>Pred Return</td><td style="text-align:right; color:{'#10b981' if top_stock['pred_return']>0 else '#ef4444'};">{_pct(top_stock['pred_return'])}</td></tr>
                <tr><td>Combined Score</td><td style="text-align:right; color:#8b5cf6;">{top_stock['score_combined']:.3f}</td></tr>
                <tr><td>1M Return</td><td style="text-align:right; color:{'#10b981' if top_stock['ret_1m']>0 else '#ef4444'};">{_pct(top_stock['ret_1m'])}</td></tr>
                <tr><td>Trailing PE</td><td style="text-align:right; color:#e2e8f0;">{top_stock['trailingPE']:.1f}</td></tr>
                <tr><td>ROE</td><td style="text-align:right; color:#e2e8f0;">{_pct(top_stock['returnOnEquity'])}</td></tr>
                <tr><td>Volatility (3M)</td><td style="text-align:right; color:#f59e0b;">{_pct(top_stock['vol_3m'])}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Ranking table
    display_cols = ['rank', 'ticker', 'sector', 'current_price', 'score_combined',
                    'pred_return', 'factor_momentum', 'factor_value', 'factor_quality',
                    'factor_risk', 'weight']
    available_cols = [c for c in display_cols if c in stocks_df.columns]
    st.dataframe(
        stocks_df[available_cols].style
        .format({
            'current_price': '${:.2f}',
            'score_combined': '{:.3f}',
            'pred_return': '{:+.2%}',
            'factor_momentum': '{:+.3f}',
            'factor_value': '{:+.3f}',
            'factor_quality': '{:+.3f}',
            'factor_risk': '{:+.3f}',
            'weight': '{:.2%}',
        })
        .background_gradient(subset=['score_combined'], cmap='RdYlGn')
        .background_gradient(subset=['pred_return'], cmap='RdYlGn'),
        use_container_width=True, height=450, hide_index=True,
    )

    # Factor heatmap
    st.markdown('<div class="section-title">Factor Heatmap</div>', unsafe_allow_html=True)
    factor_cols = ['factor_momentum', 'factor_value', 'factor_quality', 'factor_risk']
    heatmap_data = stocks_df.set_index('ticker')[factor_cols].head(15)
    fig_hm = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['Momentum', 'Value', 'Quality', 'Risk'],
        y=heatmap_data.index,
        colorscale='RdYlGn',
        zmid=0,
        text=heatmap_data.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorbar=dict(title="Z-Score", tickfont=dict(color='#94a3b8'), titlefont=dict(color='#94a3b8')),
    ))
    fig_hm.update_layout(
        **PLOTLY_LAYOUT, height=max(300, len(heatmap_data) * 28),
        xaxis=dict(side='top'),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════
# TAB 3: Regime Analysis
# ═══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Market Regime Detection</div>', unsafe_allow_html=True)
    st.markdown(f"Current: {regime_badge(current_regime)}", unsafe_allow_html=True)

    # Regime timeline
    regime_colors_map = {'NORMAL': '#10b981', 'WARNING': '#f59e0b', 'BREAKDOWN': '#ef4444', 'WARMUP': '#94a3b8'}
    fig_regime = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

    # RankIC line
    fig_regime.add_trace(go.Scatter(
        x=regime_df['date'], y=regime_df['rankic'],
        name='Daily RankIC',
        line=dict(color=COLOR_PALETTE["primary"], width=1),
    ), row=1, col=1)

    # Rolling mean
    rollmean = regime_df['rankic'].rolling(20).mean()
    fig_regime.add_trace(go.Scatter(
        x=regime_df['date'], y=rollmean,
        name='Rolling Mean (20d)',
        line=dict(color=COLOR_PALETTE["warning"], width=2),
    ), row=1, col=1)

    fig_regime.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.3)", row=1, col=1)

    # Z-score
    fig_regime.add_trace(go.Scatter(
        x=regime_df['date'], y=regime_df['z'],
        name='Z-Score',
        line=dict(color=COLOR_PALETTE["cyan"], width=1.5),
    ), row=2, col=1)
    fig_regime.add_hline(y=-1, line_dash="dash", line_color=COLOR_PALETTE["warning"], row=2, col=1,
                         annotation_text="WARNING", annotation_font_color=COLOR_PALETTE["warning"])
    fig_regime.add_hline(y=-2, line_dash="dash", line_color=COLOR_PALETTE["danger"], row=2, col=1,
                         annotation_text="BREAKDOWN", annotation_font_color=COLOR_PALETTE["danger"])

    fig_regime.update_layout(
        **PLOTLY_LAYOUT, height=500,
        title=dict(text="RankIC & Regime Detection", font=dict(size=16, color='#e2e8f0')),
        legend=dict(orientation="h", y=1.08, font=dict(color='#94a3b8')),
    )
    fig_regime.update_xaxes(gridcolor='rgba(148,163,184,0.1)')
    fig_regime.update_yaxes(gridcolor='rgba(148,163,184,0.1)')
    st.plotly_chart(fig_regime, use_container_width=True)

    # Regime stats
    st.markdown('<div class="section-title">Regime Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col_ui, reg_name in zip([c1, c2, c3, c4], ['NORMAL', 'WARNING', 'BREAKDOWN', 'WARMUP']):
        with col_ui:
            count = (regime_df['regime'] == reg_name).sum()
            pct = count / len(regime_df) * 100 if len(regime_df) else 0
            color = regime_colors_map[reg_name]
            st.markdown(f"""
            <div class="glass-panel" style="text-align:center; border-left: 3px solid {color};">
                <div style="color:{color}; font-weight:700; font-size:1.5rem;">{count}</div>
                <div style="color:#94a3b8; font-size:0.8rem;">{reg_name} ({pct:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════
# TAB 4: Scenario Analysis
# ═══════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Scenario Analysis (GBM Simulation)</div>', unsafe_allow_html=True)

    sc_col1, sc_col2 = st.columns([0.3, 0.7])
    with sc_col1:
        sc_ticker = st.selectbox("Ticker", stocks_df['ticker'].tolist())
        sc_stock = stocks_df[stocks_df['ticker'] == sc_ticker].iloc[0]
        sc_price = sc_stock['current_price']
        sc_pred = sc_stock['pred_return']
        sc_vol = sc_stock['vol_3m']
        sc_days = 252 if horizon == "1y" else 30

        st.markdown(f"""
        <div class="glass-panel">
            <div style="color:#94a3b8; font-size:0.8rem;">Current Price</div>
            <div style="color:#e2e8f0; font-size:1.3rem; font-weight:700;">${sc_price:.2f}</div>
            <br>
            <div style="color:#94a3b8; font-size:0.8rem;">Predicted Return</div>
            <div style="color:{'#10b981' if sc_pred>0 else '#ef4444'}; font-size:1.3rem; font-weight:700;">{_pct(sc_pred)}</div>
            <br>
            <div style="color:#94a3b8; font-size:0.8rem;">Annual Volatility</div>
            <div style="color:#f59e0b; font-size:1.3rem; font-weight:700;">{_pct(sc_vol)}</div>
        </div>
        """, unsafe_allow_html=True)

    with sc_col2:
        try:
            from models.scenario import make_scenario_paths
            result = make_scenario_paths(sc_price, sc_pred, max(sc_vol, 0.1), horizon_days=sc_days)

            fig_sc = go.Figure()
            # Confidence bands
            fig_sc.add_trace(go.Scatter(
                x=result.percentiles['day'], y=result.percentiles['p95'],
                line=dict(width=0), showlegend=False, hoverinfo='skip',
            ))
            fig_sc.add_trace(go.Scatter(
                x=result.percentiles['day'], y=result.percentiles['p5'],
                fill='tonexty', fillcolor='rgba(99,102,241,0.08)',
                line=dict(width=0), name='90% CI',
            ))
            fig_sc.add_trace(go.Scatter(
                x=result.percentiles['day'], y=result.percentiles['p75'],
                line=dict(width=0), showlegend=False, hoverinfo='skip',
            ))
            fig_sc.add_trace(go.Scatter(
                x=result.percentiles['day'], y=result.percentiles['p25'],
                fill='tonexty', fillcolor='rgba(99,102,241,0.15)',
                line=dict(width=0), name='50% CI',
            ))
            # Scenario paths
            fig_sc.add_trace(go.Scatter(
                x=result.paths['day'], y=result.paths['optimistic'],
                name='Optimistic (+1σ)', line=dict(color=COLOR_PALETTE["success"], width=2, dash='dash'),
            ))
            fig_sc.add_trace(go.Scatter(
                x=result.paths['day'], y=result.paths['base'],
                name='Base (Drift)', line=dict(color=COLOR_PALETTE["primary"], width=3),
            ))
            fig_sc.add_trace(go.Scatter(
                x=result.paths['day'], y=result.paths['pessimistic'],
                name='Pessimistic (-1σ)', line=dict(color=COLOR_PALETTE["danger"], width=2, dash='dash'),
            ))
            fig_sc.add_hline(y=sc_price, line_dash="dot", line_color="rgba(148,163,184,0.5)",
                             annotation_text=f"Current: ${sc_price:.0f}")
            fig_sc.update_layout(
                **PLOTLY_LAYOUT, height=400,
                title=dict(text=f"{sc_ticker} Price Scenario ({sc_days}d)", font=dict(size=14, color='#e2e8f0')),
                yaxis_title="Price ($)",
                xaxis_title="Trading Days",
                legend=dict(orientation="h", y=-0.15, font=dict(color='#94a3b8')),
            )
            fig_sc.update_xaxes(gridcolor='rgba(148,163,184,0.1)')
            fig_sc.update_yaxes(gridcolor='rgba(148,163,184,0.1)')
            st.plotly_chart(fig_sc, use_container_width=True)

            # Summary cards
            sm = result.summary
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(metric_card("Base Target", f"${sm['base_target']:.0f}", _pct(sm['base_target']/sc_price-1), "positive"), unsafe_allow_html=True)
            with s2:
                st.markdown(metric_card("Optimistic", f"${sm['optimistic_target']:.0f}", _pct(sm['optimistic_target']/sc_price-1), "positive"), unsafe_allow_html=True)
            with s3:
                st.markdown(metric_card("Pessimistic", f"${sm['pessimistic_target']:.0f}", _pct(sm['pessimistic_target']/sc_price-1), "negative"), unsafe_allow_html=True)
            with s4:
                ct = "positive" if sm['prob_profit'] > 0.5 else "negative"
                st.markdown(metric_card("Profit Probability", f"{sm['prob_profit']*100:.0f}%", f"VaR95: ${sm['var_95']:.0f}", ct), unsafe_allow_html=True)

        except Exception as e:
            st.info(f"Scenario module: {e}")


# ═══════════════════════════════════════
# TAB 5: Model Explain
# ═══════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Model Explainability</div>', unsafe_allow_html=True)

    # Simulated feature importance (SHAP-like)
    features = ['ret_12m', 'ret_6m', 'vol_3m', 'ma_gap_200', 'trailingPE',
                'returnOnEquity', 'profitMargins', 'ret_1m', 'down_vol_3m', 'priceToBook']
    importances = np.sort(np.random.uniform(0.02, 0.18, len(features)))[::-1]
    directions = np.random.choice(['+', '-'], len(features))

    fig_imp = go.Figure()
    colors_imp = [COLOR_PALETTE["success"] if d == '+' else COLOR_PALETTE["danger"] for d in directions]

    fig_imp.add_trace(go.Bar(
        y=features[::-1],
        x=importances[::-1],
        orientation='h',
        marker_color=colors_imp[::-1],
        text=[f"{v:.3f} ({d})" for v, d in zip(importances[::-1], directions[::-1])],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
    ))
    fig_imp.update_layout(
        **PLOTLY_LAYOUT, height=400,
        title=dict(text="Feature Importance (SHAP Values)", font=dict(size=14, color='#e2e8f0')),
        xaxis_title="Absolute SHAP Value",
        xaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
        yaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # CV Performance
    st.markdown('<div class="section-title">Cross-Validation Performance</div>', unsafe_allow_html=True)
    cv_data = pd.DataFrame({
        'Model': ['LightGBM', 'XGBoost', 'Random Forest', 'Ridge'],
        'RMSE': [0.142, 0.155, 0.168, 0.195],
        'MAE': [0.098, 0.108, 0.115, 0.142],
        'RankIC': [0.082, 0.075, 0.058, 0.042],
        'Hit Rate': [0.585, 0.572, 0.548, 0.522],
        'Daily RankIC Mean': [0.065, 0.058, 0.042, 0.032],
        'Status': ['Best', 'Good', 'OK', 'Baseline'],
    })

    st.dataframe(
        cv_data.style
        .format({'RMSE': '{:.3f}', 'MAE': '{:.3f}', 'RankIC': '{:.3f}', 'Hit Rate': '{:.1%}', 'Daily RankIC Mean': '{:.3f}'})
        .background_gradient(subset=['RankIC'], cmap='Greens')
        .background_gradient(subset=['RMSE'], cmap='Reds_r'),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════
# TAB 6: Portfolio
# ═══════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">Portfolio Construction</div>', unsafe_allow_html=True)

    pc1, pc2 = st.columns([0.6, 0.4])
    with pc1:
        # Treemap
        fig_tree = px.treemap(
            stocks_df.head(top_n),
            path=['sector', 'ticker'],
            values='weight',
            color='pred_return',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
        )
        fig_tree.update_layout(
            **PLOTLY_LAYOUT, height=400,
            title=dict(text="Portfolio Treemap", font=dict(size=14, color='#e2e8f0')),
            coloraxis_colorbar=dict(title="Pred Return", tickfont=dict(color='#94a3b8')),
        )
        fig_tree.update_traces(textfont=dict(color='white', size=12))
        st.plotly_chart(fig_tree, use_container_width=True)

    with pc2:
        # Weight bar chart
        port = stocks_df.head(top_n).sort_values('weight', ascending=True)
        fig_w = go.Figure(data=[go.Bar(
            y=port['ticker'],
            x=port['weight'] * 100,
            orientation='h',
            marker=dict(
                color=port['score_combined'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score", tickfont=dict(color='#94a3b8')),
            ),
            text=[f"{w*100:.1f}%" for w in port['weight']],
            textposition='outside',
            textfont=dict(color='#94a3b8'),
        )])
        fig_w.update_layout(
            **PLOTLY_LAYOUT, height=400,
            title=dict(text="Position Weights", font=dict(size=14, color='#e2e8f0')),
            xaxis_title="Weight (%)",
            xaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
            yaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
        )
        st.plotly_chart(fig_w, use_container_width=True)

    # Portfolio stats
    st.markdown('<div class="section-title">Portfolio Summary</div>', unsafe_allow_html=True)
    ps1, ps2, ps3, ps4 = st.columns(4)
    with ps1:
        st.markdown(metric_card("Positions", str(top_n), f"Max weight: {max_position_pct}%", "neutral"), unsafe_allow_html=True)
    with ps2:
        avg_pred = stocks_df.head(top_n)['pred_return'].mean()
        st.markdown(metric_card("Avg Pred Return", _pct(avg_pred), "Weighted", "positive" if avg_pred > 0 else "negative"), unsafe_allow_html=True)
    with ps3:
        n_sectors = stocks_df.head(top_n)['sector'].nunique()
        st.markdown(metric_card("Sectors", str(n_sectors), f"Max: {max_sector_pct}%", "neutral"), unsafe_allow_html=True)
    with ps4:
        avg_score = stocks_df.head(top_n)['score_combined'].mean()
        st.markdown(metric_card("Avg Score", f"{avg_score:.3f}", "Combined", "positive"), unsafe_allow_html=True)


# ═══════════════════════════════════════
# TAB 7: Trade Log
# ═══════════════════════════════════════
with tab7:
    st.markdown('<div class="section-title">Trade History Log</div>', unsafe_allow_html=True)

    try:
        from services.trade_logger import get_trade_log, get_trade_stats
        log_df = get_trade_log()
        stats = get_trade_stats()

        if not log_df.empty:
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                st.markdown(metric_card("Total Picks", str(stats['total_picks']), f"{stats['unique_dates']} dates", "neutral"), unsafe_allow_html=True)
            with tc2:
                st.markdown(metric_card("Realized", str(stats['realized_count']), "Trades completed", "neutral"), unsafe_allow_html=True)
            with tc3:
                ar = stats['avg_return']
                ct = "positive" if (ar and not np.isnan(ar) and ar > 0) else "negative"
                st.markdown(metric_card("Avg Return", _pct(ar) if ar and not np.isnan(ar) else "N/A", "", ct), unsafe_allow_html=True)
            with tc4:
                wr = stats['win_rate']
                ct = "positive" if (wr and not np.isnan(wr) and wr > 0.5) else "negative"
                st.markdown(metric_card("Win Rate", f"{wr*100:.1f}%" if wr and not np.isnan(wr) else "N/A", "", ct), unsafe_allow_html=True)

            st.dataframe(log_df.tail(50), use_container_width=True, hide_index=True)
        else:
            st.info("No trade history yet. Run the strategy to start logging.")
    except Exception:
        st.info("Trade logger ready. Execute a strategy to see trade history.")

    # ── 예측 추적 (Prediction Tracking) ──
    st.markdown('<div class="section-title">📊 Prediction Tracking</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.25);
                border-left: 4px solid #6366f1; border-radius: 12px; padding: 0.8rem 1.2rem;
                margin-bottom: 1rem; color: #cbd5e1; font-size: 0.88rem;">
        <strong style="color: #818cf8;">예측 추적</strong>: "Run Strategy" 클릭 시 예측값이 자동 기록됩니다.
        시간이 지나면 실제 수익률과 비교하여 모델 정확도를 평가할 수 있습니다.
        상세 분석은 <strong style="color: #f1f5f9;">Prediction Eval</strong> 페이지에서 확인하세요.
    </div>
    """, unsafe_allow_html=True)

    try:
        from services.prediction_tracker import get_accuracy_summary, get_prediction_history, evaluate_predictions

        pred_summary = get_accuracy_summary()
        if pred_summary["total_predictions"] > 0:
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                st.markdown(metric_card("예측 기록", str(pred_summary["total_predictions"]),
                            f"평가 완료: {pred_summary['evaluated_count']}", "neutral"), unsafe_allow_html=True)
            with pc2:
                hr = pred_summary["hit_rate"]
                ct = "positive" if hr and hr > 0.5 else "negative"
                st.markdown(metric_card("Hit Rate",
                            f"{hr*100:.1f}%" if hr else "N/A", "방향 예측 정확도", ct), unsafe_allow_html=True)
            with pc3:
                ric = pred_summary["rank_ic_mean"]
                ct = "positive" if ric and ric > 0.03 else "neutral"
                st.markdown(metric_card("Rank IC",
                            f"{ric:.3f}" if ric else "N/A", "순위 상관계수", ct), unsafe_allow_html=True)
            with pc4:
                mse = pred_summary["mse"]
                st.markdown(metric_card("MSE",
                            f"{mse:.4f}" if mse else "N/A", "예측 오차", "neutral"), unsafe_allow_html=True)

            # 평가 버튼
            if st.button("🔄 지금 평가하기 (30일 경과 예측)", key="eval_pred"):
                try:
                    from services.universe import fetch_ohlcv_us
                    def _price_fetcher(ticker):
                        df = fetch_ohlcv_us(ticker, period="5d")
                        if df.empty:
                            return None
                        return float(df["Close"].iloc[-1])
                    n = evaluate_predictions(_price_fetcher, horizon_days=30)
                    st.success(f"✅ {n}개 예측이 평가되었습니다." if n > 0 else "평가할 예측이 아직 없습니다.")
                except Exception as e:
                    st.warning(f"평가 중 오류: {e}")

            pred_hist = get_prediction_history(evaluated_only=False)
            if not pred_hist.empty:
                st.dataframe(pred_hist.head(30), use_container_width=True, hide_index=True)
        else:
            st.info("예측 기록이 없습니다. 'Run Strategy'를 클릭하면 예측이 자동 기록됩니다.")
    except Exception:
        st.info("Prediction tracker ready.")

    st.markdown("---")

    # Quick Log Export
    st.markdown('<div class="section-title">Quick Export</div>', unsafe_allow_html=True)
    if st.button("Export Rankings as CSV"):
        csv = stocks_df.to_csv(index=False)
        st.download_button("Download", csv, "rankings.csv", "text/csv")


# ───────────────────────────────────────
# Footer
# ───────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#64748b; padding:1.5rem 0;'>
    <p style='margin:0; font-size:0.85rem;'>
        <strong style="color:#94a3b8;">Quant Investment System v2.0</strong><br>
        Factor Scoring + ML Prediction + Regime Detection + Scenario Analysis<br>
        Architecture: VN.py + Zipline + Qlib + stock_success
    </p>
</div>
""", unsafe_allow_html=True)
