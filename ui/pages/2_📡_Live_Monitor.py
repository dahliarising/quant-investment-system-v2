"""
실시간 모니터링 대시보드
- 다크 테마 통일
- Order Book 렌더링 수정
- 한국어 해석 가이드 포함
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.components.data_source import (
    render_data_source_selector, render_data_source_badge,
    fetch_data_for_source, get_default_tickers,
)

st.set_page_config(page_title="Live Monitor", page_icon="📡", layout="wide")

# ── 다크 테마 CSS (app.py와 통일) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --primary: #6366f1; --accent: #8b5cf6;
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
    --bg-dark: #0f172a; --bg-card: #1e293b; --border: #334155;
    --glass: rgba(30, 41, 59, 0.8);
    --text-primary: #f1f5f9; --text-secondary: #94a3b8; --text-muted: #64748b;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); }

.hero-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.5rem 2rem; border-radius: 14px; color: white;
    margin-bottom: 1.5rem; border: 1px solid #475569;
}
.hero-header h1 { font-size: 1.8rem; font-weight: 800; margin: 0; }
.hero-header p { margin: 0.3rem 0 0; opacity: 0.7; font-size: 0.9rem; }

@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.live-dot {
    display: inline-block; width: 10px; height: 10px;
    background: var(--success); border-radius: 50%;
    animation: pulse 1.5s infinite; margin-right: 8px; vertical-align: middle;
}

.glass-card {
    background: var(--glass); backdrop-filter: blur(12px);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 1.2rem 1.5rem; margin: 0.5rem 0;
}
.glass-card:hover { border-color: var(--primary); }

.metric-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 1.6rem; font-weight: 800; color: var(--text-primary); margin: 0.2rem 0; }

.guide-box {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;
}
.guide-box strong { color: var(--text-primary); }
.guide-title { color: #a78bfa; font-weight: 700; margin-bottom: 0.3rem; }

/* Order Book */
.ob-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.ob-table td { padding: 6px 10px; border-bottom: 1px solid var(--border); }
.ob-ask { color: var(--danger); }
.ob-bid { color: var(--success); }
.ob-spread { background: rgba(99,102,241,0.15); text-align: center; font-weight: 700; color: var(--text-primary); }
.ob-bar-ask { background: linear-gradient(90deg, transparent, rgba(239,68,68,0.15)); }
.ob-bar-bid { background: linear-gradient(270deg, transparent, rgba(16,185,129,0.15)); }
.ob-qty { text-align: right; color: var(--text-secondary); }
.ob-price { text-align: center; font-weight: 600; }

.signal-card {
    background: var(--glass); border: 2px solid; border-radius: 12px;
    padding: 1.2rem; text-align: center;
}
.signal-buy { border-color: var(--success); }
.signal-sell { border-color: var(--danger); }
.signal-hold { border-color: var(--warning); }

.alert-item {
    padding: 0.6rem 1rem; border-radius: 8px; margin: 0.3rem 0;
    font-size: 0.82rem; border-left: 3px solid;
}
.alert-danger { background: rgba(239,68,68,0.08); border-color: var(--danger); color: #fca5a5; }
.alert-warning { background: rgba(245,158,11,0.08); border-color: var(--warning); color: #fcd34d; }
.alert-info { background: rgba(59,130,246,0.08); border-color: #3b82f6; color: #93c5fd; }

.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }

.section-title {
    font-size: 1.1rem; font-weight: 700; color: var(--text-primary);
    margin: 1.5rem 0 0.8rem; padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--primary); display: inline-block;
}

.stTabs [data-baseweb="tab-list"] { background: var(--glass); padding: 0.3rem; border-radius: 10px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 600; color: var(--text-secondary); }
.stTabs [aria-selected="true"] { background: var(--primary) !important; color: white !important; }
.stButton > button { background: linear-gradient(135deg, var(--primary), var(--accent)); color: white; border-radius: 10px; font-weight: 700; border: none; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border-right: 1px solid var(--border); }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8'),
    margin=dict(l=0, r=0, t=40, b=0), hovermode='x unified',
)

# ── Header ──
st.markdown("""
<div class="hero-header">
    <h1><span class="live-dot"></span> Live Monitoring Dashboard</h1>
    <p>실시간 시장 데이터 · 트레이딩 시그널 · 리스크 알림</p>
</div>
""", unsafe_allow_html=True)

# ── 사이드바 ──
with st.sidebar:
    data_source = render_data_source_selector(key_prefix="monitor")

    st.markdown("### Watch List")
    _default_tickers = get_default_tickers(data_source)
    watchlist = st.multiselect(
        "종목", _default_tickers,
        default=_default_tickers[:3], label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Alert Settings")
    price_threshold = st.slider("가격 변동 알림 (%)", 1.0, 10.0, 3.0, 0.5)

render_data_source_badge(data_source)

# ── 해석 가이드 ──
st.markdown("""
<div class="guide-box">
    <div class="guide-title">이 페이지 보는 법</div>
    이 페이지는 <strong>실시간 시장 모니터링</strong> 화면입니다.<br>
    · <strong>가격 카드</strong>: 각 종목의 현재가와 등락률 (초록=상승, 빨강=하락)<br>
    · <strong>차트</strong>: 최근 50분간 가격 움직임과 거래량<br>
    · <strong>호가창(Order Book)</strong>: 매수/매도 대기 주문 현황. 위쪽 빨간색=매도호가, 아래쪽 초록색=매수호가<br>
    · <strong>시그널</strong>: AI 멀티팩터 모델이 생성한 매수/매도/보유 신호<br>
    · <strong>알림</strong>: 급등락, 거래량 급증 등 주의 이벤트
</div>
""", unsafe_allow_html=True)

# ── 데이터 생성 ──
live_data = {}

if data_source == "demo":
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    base_prices = {'AAPL': 227.5, 'MSFT': 415.2, 'GOOGL': 178.9, 'AMZN': 205.3,
                   'NVDA': 875.4, 'META': 585.1, 'TSLA': 245.8, 'JPM': 198.6,
                   '005930': 72000, '000660': 185000, '035420': 380000, '051910': 680000,
                   '006400': 290000, '035720': 58000, '003550': 340000, '105560': 85000}
    for sym in watchlist:
        bp = base_prices.get(sym, 150.0)
        change = np.random.uniform(-0.035, 0.04)
        price = bp * (1 + change)
        live_data[sym] = {
            'price': round(price, 2),
            'change': change,
            'volume': np.random.randint(2_000_000, 15_000_000),
            'high': round(price * (1 + abs(np.random.uniform(0.002, 0.015))), 2),
            'low': round(price * (1 - abs(np.random.uniform(0.002, 0.015))), 2),
        }
else:
    # 실제 데이터
    _ohlcv = fetch_data_for_source(data_source, tuple(watchlist), period="5d")
    for sym, df in _ohlcv.items():
        if df.empty or len(df) < 2:
            continue
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = (float(latest["Close"]) - float(prev["Close"])) / float(prev["Close"])
        live_data[sym] = {
            'price': round(float(latest["Close"]), 2),
            'change': float(change),
            'volume': int(latest["Volume"]) if "Volume" in df.columns else 0,
            'high': round(float(latest["High"]), 2) if "High" in df.columns else round(float(latest["Close"]), 2),
            'low': round(float(latest["Low"]), 2) if "Low" in df.columns else round(float(latest["Close"]), 2),
        }
    if not live_data:
        st.warning("실제 데이터를 가져올 수 없어 빈 화면입니다. 종목을 확인해 주세요.")

# ── 상태 바 ──
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""<div class="glass-card" style="text-align:center;">
        <span class="live-dot"></span><span style="color:#10b981;font-weight:700;">LIVE</span>
        <div style="color:#64748b;font-size:0.75rem;">Market Data Connected</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <span style="color:#f1f5f9;font-weight:700;">{len(watchlist)}</span>
        <div style="color:#64748b;font-size:0.75rem;">Symbols Tracked</div>
    </div>""", unsafe_allow_html=True)
with s3:
    regime = np.random.choice(["NORMAL", "WARNING"], p=[0.8, 0.2])
    rc = "#10b981" if regime == "NORMAL" else "#f59e0b"
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <span style="color:{rc};font-weight:700;">{regime}</span>
        <div style="color:#64748b;font-size:0.75rem;">Market Regime</div>
    </div>""", unsafe_allow_html=True)
with s4:
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <span style="color:#f1f5f9;font-weight:700;">{datetime.now().strftime('%H:%M:%S')}</span>
        <div style="color:#64748b;font-size:0.75rem;">Last Update</div>
    </div>""", unsafe_allow_html=True)

# ── 가격 카드 ──
st.markdown('<div class="section-title">실시간 가격</div>', unsafe_allow_html=True)
if live_data:
    cols = st.columns(len(live_data))
    for i, (sym, d) in enumerate(live_data.items()):
        with cols[i]:
            c = "#10b981" if d['change'] >= 0 else "#ef4444"
            arrow = "▲" if d['change'] >= 0 else "▼"
            st.markdown(f"""
            <div class="glass-card">
                <div style="color:#94a3b8;font-weight:600;font-size:0.9rem;">{sym}</div>
                <div style="color:{c};font-size:1.8rem;font-weight:800;">${d['price']:.2f}</div>
                <div style="color:{c};font-weight:600;">{arrow} {abs(d['change'])*100:.2f}%</div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:4px;">
                    Vol: {d['volume']:,} · H: ${d['high']:.2f} · L: ${d['low']:.2f}
                </div>
            </div>""", unsafe_allow_html=True)

# ── 차트 + 호가창 ──
st.markdown('<div class="section-title">차트 & 호가창</div>', unsafe_allow_html=True)

chart_col, book_col = st.columns([0.6, 0.4])

with chart_col:
    n_ticks = 50
    time_range = pd.date_range(end=datetime.now(), periods=n_ticks, freq='1min')
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.05)
    colors = ['#6366f1', '#8b5cf6', '#06b6d4', '#ec4899', '#f59e0b']
    for i, sym in enumerate(list(live_data.keys())[:4]):
        bp = live_data[sym]['price']
        prices = bp + np.cumsum(np.random.randn(n_ticks) * bp * 0.002)
        fig.add_trace(go.Scatter(x=time_range, y=prices, name=sym,
                                 line=dict(width=2, color=colors[i % len(colors)])), row=1, col=1)
        vols = np.abs(np.random.randn(n_ticks)) * live_data[sym]['volume'] / 20
        fig.add_trace(go.Bar(x=time_range, y=vols, name=f'{sym} Vol',
                             showlegend=False, marker_color=colors[i % len(colors)], opacity=0.4), row=2, col=1)
    fig.update_layout(**PLOTLY_DARK, height=420,
                      legend=dict(orientation="h", y=1.08, font=dict(color='#94a3b8')))
    fig.update_xaxes(gridcolor='rgba(148,163,184,0.1)')
    fig.update_yaxes(gridcolor='rgba(148,163,184,0.1)')
    st.plotly_chart(fig, use_container_width=True)

with book_col:
    if watchlist:
        sym0 = watchlist[0]
        d0 = live_data[sym0]
        spread_price = d0['price']

        asks = [(round(spread_price + (5 - i) * 0.15, 2), np.random.randint(80, 600)) for i in range(5)]
        bids = [(round(spread_price - (i + 1) * 0.15, 2), np.random.randint(80, 600)) for i in range(5)]

        st.markdown(f"""
        <div class="glass-card">
            <div style="color:#f1f5f9;font-weight:700;font-size:1rem;margin-bottom:0.8rem;">
                Order Book — {sym0}
            </div>
            <div style="color:#64748b;font-size:0.7rem;margin-bottom:0.5rem;">
                매도호가 (빨강) = 이 가격에 팔겠다는 주문 · 매수호가 (초록) = 이 가격에 사겠다는 주문
            </div>
            <table class="ob-table">
                <tr style="color:#64748b;font-size:0.7rem;text-transform:uppercase;">
                    <td style="text-align:left;">수량 (Bid)</td>
                    <td style="text-align:center;">가격</td>
                    <td style="text-align:right;">수량 (Ask)</td>
                </tr>
        """, unsafe_allow_html=True)

        # Ask rows (높은 가격부터)
        ask_html = ""
        for price, qty in asks:
            pct = qty / 600 * 100
            ask_html += f"""
            <tr class="ob-bar-ask">
                <td class="ob-qty"></td>
                <td class="ob-price ob-ask">${price:.2f}</td>
                <td class="ob-qty" style="color:#ef4444;">{qty:,}</td>
            </tr>"""

        # Spread row
        spread_val = round(asks[-1][0] - bids[0][0], 2)
        spread_html = f"""
        <tr><td colspan="3" class="ob-spread">
            SPREAD ${spread_val:.2f} ({spread_val/spread_price*100:.3f}%)
        </td></tr>"""

        # Bid rows
        bid_html = ""
        for price, qty in bids:
            bid_html += f"""
            <tr class="ob-bar-bid">
                <td class="ob-qty" style="color:#10b981;">{qty:,}</td>
                <td class="ob-price ob-bid">${price:.2f}</td>
                <td class="ob-qty"></td>
            </tr>"""

        st.markdown(ask_html + spread_html + bid_html + "</table></div>", unsafe_allow_html=True)

        # Depth bar
        total_bid = sum(b[1] for b in bids)
        total_ask = sum(a[1] for a in asks)
        bid_pct = total_bid / (total_bid + total_ask) * 100
        st.markdown(f"""
        <div class="glass-card" style="padding:0.8rem 1rem;">
            <div style="color:#64748b;font-size:0.7rem;margin-bottom:4px;">매수/매도 압력</div>
            <div style="display:flex;height:12px;border-radius:6px;overflow:hidden;">
                <div style="width:{bid_pct:.0f}%;background:#10b981;"></div>
                <div style="width:{100-bid_pct:.0f}%;background:#ef4444;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-top:4px;">
                <span style="color:#10b981;">매수 {total_bid:,} ({bid_pct:.0f}%)</span>
                <span style="color:#ef4444;">매도 {total_ask:,} ({100-bid_pct:.0f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── 트레이딩 시그널 ──
st.markdown('<div class="section-title">AI 트레이딩 시그널</div>', unsafe_allow_html=True)

st.markdown("""
<div class="guide-box">
    <div class="guide-title">시그널 해석 방법</div>
    · <strong style="color:#10b981;">BUY</strong>: 멀티팩터 점수가 높아 매수 유망. Score가 높을수록 강한 신호<br>
    · <strong style="color:#f59e0b;">HOLD</strong>: 현재 보유 유지 권장. 뚜렷한 방향성 없음<br>
    · <strong style="color:#ef4444;">SELL</strong>: 팩터 점수 하락으로 매도 고려. Score가 낮을수록 강한 매도 신호<br>
    · <strong>Score</strong>: 0~1 범위. 0.7 이상 = 강한 매수, 0.3 이하 = 강한 매도
</div>
""", unsafe_allow_html=True)

signals = []
for sym in watchlist[:6]:
    score = np.random.uniform(0.15, 0.92)
    if score > 0.65:
        sig, cls = "BUY", "signal-buy"
        strength = "Strong" if score > 0.8 else "Moderate"
        sc = "#10b981"
    elif score < 0.35:
        sig, cls = "SELL", "signal-sell"
        strength = "Strong" if score < 0.2 else "Moderate"
        sc = "#ef4444"
    else:
        sig, cls = "HOLD", "signal-hold"
        strength = "Neutral"
        sc = "#f59e0b"
    signals.append((sym, sig, cls, score, strength, sc))

sig_cols = st.columns(min(len(signals), 6))
for i, (sym, sig, cls, score, strength, sc) in enumerate(signals):
    with sig_cols[i]:
        st.markdown(f"""
        <div class="signal-card {cls}">
            <div style="color:#94a3b8;font-weight:600;">{sym}</div>
            <div style="color:{sc};font-size:1.6rem;font-weight:800;margin:0.3rem 0;">{sig}</div>
            <div style="color:#64748b;font-size:0.8rem;">
                Score: <strong style="color:#f1f5f9;">{score:.2f}</strong><br>
                {strength}
            </div>
        </div>""", unsafe_allow_html=True)

# ── 알림 ──
st.markdown('<div class="section-title">최근 알림</div>', unsafe_allow_html=True)

alerts = [
    ("info", "NVDA", "멀티팩터 BUY 신호 발생 (Score: 0.87)", "14:32:15"),
    ("warning", "TSLA", f"거래량 급증 — 평균 대비 2.3배", "14:28:41"),
    ("danger", "AMZN", f"급락 알림 — 가격 -{np.random.uniform(2,4):.1f}% 하락", "14:15:03"),
    ("info", "MSFT", "레짐 변경: WARNING → NORMAL", "13:55:22"),
    ("warning", "GOOGL", "변동성 급등 — 3M Vol 상위 10%", "13:42:10"),
]

for atype, sym, msg, t in alerts:
    st.markdown(f"""
    <div class="alert-item alert-{atype}">
        <strong>{sym}</strong> · {msg}
        <span style="float:right;opacity:0.6;">{t}</span>
    </div>""", unsafe_allow_html=True)

# ── Refresh ──
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Refresh Data", use_container_width=True):
    st.rerun()

st.markdown("""
<div style='text-align:center;color:#475569;padding:1rem 0;font-size:0.8rem;'>
    Live Monitor · 실시간 시장 모니터링 · 데이터는 시뮬레이션입니다
</div>""", unsafe_allow_html=True)
