"""
예측 평가 페이지
ML 예측 vs 실제 결과 추적 · 정확도 분석 · 포트폴리오 Traceability
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

from services.prediction_tracker import (
    get_prediction_history, get_accuracy_summary,
    compute_daily_rank_ic, evaluate_predictions,
)

st.set_page_config(page_title="Prediction Eval", page_icon="📊", layout="wide")

# ───────────────────────────────────────
# Dark Theme CSS
# ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --primary: #6366f1; --accent: #8b5cf6;
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
    --bg-dark: #0f172a; --bg-card: #1e293b;
    --text-primary: #f1f5f9; --text-secondary: #94a3b8; --text-muted: #64748b;
    --border: #334155; --glass: rgba(30, 41, 59, 0.8);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); }

.glass-card {
    background: var(--glass); backdrop-filter: blur(12px);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1rem;
}
.metric-card {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.2rem; text-align: center;
}
.metric-label { color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #6366f1, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-sub { color: var(--text-muted); font-size: 0.8rem; }
.metric-good .metric-value { background: linear-gradient(135deg, #10b981, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-bad .metric-value { background: linear-gradient(135deg, #ef4444, #f87171); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.guide-box {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1; border-radius: 12px;
    padding: 1rem 1.5rem; margin: 1rem 0; color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;
}
.guide-box strong { color: #a5b4fc; }
.guide-box .guide-title { font-weight: 700; color: #818cf8; margin-bottom: 0.5rem; font-size: 1rem; }

.section-header { color: var(--text-primary); font-size: 1.3rem; font-weight: 700; margin: 1.5rem 0 0.5rem; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h3 { color: var(--text-primary); }

.footer-text { text-align: center; color: var(--text-muted); padding: 1.5rem 0; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#94a3b8'),
    xaxis=dict(gridcolor='rgba(51,65,85,0.5)'), yaxis=dict(gridcolor='rgba(51,65,85,0.5)'),
    hoverlabel=dict(bgcolor='#1e293b', font_size=12, font_color='#f1f5f9'),
    margin=dict(l=0, r=0, t=40, b=0),
)
COLORS = ['#818cf8', '#a78bfa', '#c084fc', '#f472b6', '#fb923c', '#34d399', '#38bdf8', '#fbbf24']


def _mc(label, value, sub="", style=""):
    cls = f"metric-card {style}"
    return f"""<div class="{cls}"><div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>"""


# ───────────────────────────────────────
# Header
# ───────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #10b981 0%, #6366f1 50%, #8b5cf6 100%);
            padding: 1.8rem 2.5rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
            box-shadow: 0 20px 60px rgba(16,185,129,0.3); position: relative; overflow: hidden;'>
    <div style='position: absolute; top: -50%; right: -10%; width: 300px; height: 300px;
                background: rgba(255,255,255,0.05); border-radius: 50%;'></div>
    <h1 style='margin: 0; font-size: 2rem; position: relative;'>📊 Prediction Evaluation</h1>
    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9; position: relative;'>
        ML 예측 vs 실제 결과 · 정확도 분석 · 포트폴리오 Traceability
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="guide-box">
    <div class="guide-title">📖 이 페이지 사용법</div>
    메인 대시보드에서 <strong>"Run Strategy"</strong>를 클릭하면 ML 예측값이 자동 기록됩니다.<br>
    시간이 지나면 실제 수익률을 가져와 예측 정확도를 평가합니다.<br><br>
    • <strong>예측 vs 실제</strong> — 산점도로 예측 정확도 시각화<br>
    • <strong>일일 Hit Rate</strong> — 날짜별 방향 예측 성공률<br>
    • <strong>섹터/시기별</strong> — 어떤 섹터/시기에 모델이 잘 맞는지<br>
    • <strong>포트폴리오 추적</strong> — 예측 기반 포트폴리오의 실제 성과<br>
    • <strong>전체 이력</strong> — 모든 예측 기록 + CSV 다운로드
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────
# Sidebar
# ───────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 필터 설정")
    model_filter = st.selectbox("모델", ["전체", "lgbm", "xgb", "rf", "ridge"])
    model_arg = None if model_filter == "전체" else model_filter

    st.markdown("---")
    st.markdown("### ⚡ 평가 실행")
    eval_horizon = st.selectbox("평가 기간", [30, 60, 90, 252], format_func=lambda x: f"{x}일")

    if st.button("🔄 지금 평가하기", use_container_width=True, type="primary"):
        try:
            from services.universe import fetch_ohlcv_us

            def _price_fetch(ticker):
                df = fetch_ohlcv_us(ticker, period="5d")
                if df.empty:
                    return None
                return float(df["Close"].iloc[-1])

            n = evaluate_predictions(_price_fetch, horizon_days=eval_horizon)
            st.success(f"✅ {n}개 평가 완료" if n > 0 else "평가할 예측이 없습니다.")
        except Exception as e:
            st.error(f"오류: {e}")

# ───────────────────────────────────────
# Data
# ───────────────────────────────────────
summary = get_accuracy_summary(model=model_arg)
all_preds = get_prediction_history(model=model_arg)
eval_preds = get_prediction_history(model=model_arg, evaluated_only=True)
daily_rics = compute_daily_rank_ic(model=model_arg)

# ───────────────────────────────────────
# Summary Metrics
# ───────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(_mc("총 예측", str(summary["total_predictions"]), f"평가 완료: {summary['evaluated_count']}"), unsafe_allow_html=True)
with c2:
    hr = summary["hit_rate"]
    s = "metric-good" if hr and hr > 0.5 else "metric-bad" if hr else ""
    st.markdown(_mc("Hit Rate", f"{hr*100:.1f}%" if hr else "N/A", "방향 예측 정확도", s), unsafe_allow_html=True)
with c3:
    ric = summary["rank_ic_mean"]
    s = "metric-good" if ric and ric > 0.03 else ""
    st.markdown(_mc("Rank IC", f"{ric:.3f}" if ric else "N/A", f"± {summary['rank_ic_std']:.3f}" if summary.get("rank_ic_std") else "", s), unsafe_allow_html=True)
with c4:
    mse = summary["mse"]
    st.markdown(_mc("MSE", f"{mse:.4f}" if mse else "N/A", "예측 오차"), unsafe_allow_html=True)
with c5:
    mp = summary["mean_pred_return"]
    ma = summary["mean_actual_return"]
    st.markdown(_mc("평균 수익률", f"예측 {mp*100:.1f}%" if mp else "N/A", f"실제 {ma*100:.1f}%" if ma else ""), unsafe_allow_html=True)

# ───────────────────────────────────────
# Tabs
# ───────────────────────────────────────
if summary["total_predictions"] == 0:
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">📝 아직 예측 기록이 없습니다</div>
        메인 대시보드(app.py)에서 <strong>"Run Strategy"</strong> 버튼을 클릭하면<br>
        ML 모델의 예측값이 자동으로 여기에 기록됩니다.
    </div>
    """, unsafe_allow_html=True)
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 예측 vs 실제",
        "📉 일일 Hit Rate",
        "🏢 섹터/시기별",
        "💼 포트폴리오 추적",
        "📋 전체 이력",
    ])

    # ── Tab 1: Prediction vs Actual ──
    with tab1:
        st.markdown('<div class="section-header">예측 수익률 vs 실제 수익률</div>', unsafe_allow_html=True)

        if not eval_preds.empty:
            st.markdown("""
            <div class="guide-box">
                점이 <strong>대각선(점선)</strong>에 가까울수록 예측이 정확합니다.<br>
                • 대각선 위: 실제가 예측보다 좋음 (보수적 예측)<br>
                • 대각선 아래: 실제가 예측보다 나쁨 (낙관적 예측)
            </div>
            """, unsafe_allow_html=True)

            pred_vals = pd.to_numeric(eval_preds["pred_return"], errors="coerce") * 100
            actual_vals = pd.to_numeric(eval_preds["actual_return"], errors="coerce") * 100

            fig = go.Figure()
            # 섹터별 색상
            sectors = eval_preds["sector"].fillna("Unknown")
            unique_sectors = sectors.unique()
            for i, sec in enumerate(unique_sectors):
                mask = sectors == sec
                fig.add_trace(go.Scatter(
                    x=pred_vals[mask], y=actual_vals[mask],
                    mode='markers', name=str(sec),
                    marker=dict(size=8, color=COLORS[i % len(COLORS)], opacity=0.8),
                    text=eval_preds.loc[mask, "ticker"],
                    hovertemplate='%{text}<br>예측: %{x:.1f}%<br>실제: %{y:.1f}%<extra>%{fullData.name}</extra>',
                ))

            # 대각선
            mn = min(pred_vals.min(), actual_vals.min()) if pred_vals.notna().any() else -10
            mx = max(pred_vals.max(), actual_vals.max()) if pred_vals.notna().any() else 30
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                     line=dict(dash='dash', color='rgba(100,116,139,0.5)', width=1),
                                     showlegend=False))

            fig.update_layout(height=500, xaxis_title="예측 수익률 (%)", yaxis_title="실제 수익률 (%)", **PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("아직 평가된 예측이 없습니다. 사이드바에서 '지금 평가하기'를 시도하세요.")

    # ── Tab 2: Daily Hit Rate ──
    with tab2:
        st.markdown('<div class="section-header">일일 Hit Rate & Rank IC</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-box">
            • <strong>Hit Rate</strong>: 예측 방향이 맞은 비율. <strong>50% 이상</strong>이면 동전 던지기보다 나음<br>
            • <strong>Rank IC</strong>: 예측 순위와 실제 순위의 상관관계. <strong>0.05 이상</strong>이면 양호
        </div>
        """, unsafe_allow_html=True)

        if not eval_preds.empty:
            eval_preds["date_dt"] = pd.to_datetime(eval_preds["date"])
            eval_preds["hit_bool"] = eval_preds["hit"].astype(str).str.lower() == "true"
            daily_hr = eval_preds.groupby("date_dt")["hit_bool"].mean()

            fig = make_subplots(rows=2, cols=1, subplot_titles=("일일 Hit Rate", "일일 Rank IC"),
                                vertical_spacing=0.12)

            # Hit rate
            rolling_hr = daily_hr.rolling(20, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=daily_hr.index, y=daily_hr.values * 100, name="일일",
                                     line=dict(color='rgba(129,140,248,0.3)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=rolling_hr.index, y=rolling_hr.values * 100, name="20일 평균",
                                     line=dict(color='#818cf8', width=2.5)), row=1, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b", row=1, col=1)

            # Rank IC
            if not daily_rics.empty:
                colors_ic = ['#10b981' if v > 0 else '#ef4444' for v in daily_rics.values]
                fig.add_trace(go.Bar(x=daily_rics.index, y=daily_rics.values, name="Rank IC",
                                     marker_color=colors_ic), row=2, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.5)", row=2, col=1)

            fig.update_layout(height=600, **PLOTLY_DARK)
            for i in range(1, 3):
                fig.update_layout(**{
                    f'xaxis{i}': dict(gridcolor='rgba(51,65,85,0.3)'),
                    f'yaxis{i}': dict(gridcolor='rgba(51,65,85,0.3)'),
                })
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("평가된 예측이 필요합니다.")

    # ── Tab 3: By Sector / Period ──
    with tab3:
        st.markdown('<div class="section-header">섹터별 / 월별 정확도</div>', unsafe_allow_html=True)

        col_s, col_m = st.columns(2)

        with col_s:
            st.markdown("#### 🏢 섹터별 Hit Rate")
            by_sector = summary.get("by_sector", {})
            if by_sector:
                sec_df = pd.DataFrame([
                    {"섹터": k, "Hit Rate (%)": v["hit_rate"] * 100, "예측 수": v["count"]}
                    for k, v in by_sector.items()
                ]).sort_values("Hit Rate (%)", ascending=True)

                fig_sec = go.Figure(data=[go.Bar(
                    y=sec_df["섹터"], x=sec_df["Hit Rate (%)"],
                    orientation='h',
                    marker_color=['#10b981' if h > 50 else '#ef4444' for h in sec_df["Hit Rate (%)"]],
                    text=sec_df["Hit Rate (%)"].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside', textfont=dict(color='#94a3b8'),
                )])
                fig_sec.add_vline(x=50, line_dash="dash", line_color="#f59e0b")
                fig_sec.update_layout(height=max(250, len(sec_df) * 40), xaxis_title="Hit Rate (%)", **PLOTLY_DARK)
                st.plotly_chart(fig_sec, use_container_width=True)
            else:
                st.info("섹터 데이터 없음")

        with col_m:
            st.markdown("#### 📅 월별 Hit Rate")
            by_month = summary.get("by_month", {})
            if by_month:
                month_df = pd.DataFrame([
                    {"월": k, "Hit Rate (%)": v["hit_rate"] * 100, "예측 수": v["count"]}
                    for k, v in by_month.items()
                ]).sort_values("월")

                fig_month = go.Figure(data=[go.Bar(
                    x=month_df["월"], y=month_df["Hit Rate (%)"],
                    marker_color=['#10b981' if h > 50 else '#ef4444' for h in month_df["Hit Rate (%)"]],
                    text=month_df["Hit Rate (%)"].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside', textfont=dict(color='#94a3b8'),
                )])
                fig_month.add_hline(y=50, line_dash="dash", line_color="#f59e0b")
                fig_month.update_layout(height=350, xaxis_title="월", yaxis_title="Hit Rate (%)", **PLOTLY_DARK)
                st.plotly_chart(fig_month, use_container_width=True)
            else:
                st.info("월별 데이터 없음")

    # ── Tab 4: Portfolio Tracking ──
    with tab4:
        st.markdown('<div class="section-header">포트폴리오 수준 추적</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-box">
            <div class="guide-title">💼 포트폴리오 Traceability</div>
            각 날짜에 모델이 추천한 종목들의 <strong>예측 평균 수익률</strong>과 <strong>실제 평균 수익률</strong>을 비교합니다.<br>
            예측선과 실제선이 비슷하게 움직이면 모델의 포트폴리오 수준 예측이 정확하다는 뜻입니다.
        </div>
        """, unsafe_allow_html=True)

        if not eval_preds.empty:
            eval_preds["date_dt"] = pd.to_datetime(eval_preds["date"])
            port = eval_preds.groupby("date_dt").agg(
                pred_avg=("pred_return", lambda x: pd.to_numeric(x, errors="coerce").mean()),
                actual_avg=("actual_return", lambda x: pd.to_numeric(x, errors="coerce").mean()),
                n_stocks=("ticker", "count"),
            ).dropna()

            if not port.empty:
                port["cum_pred"] = (1 + port["pred_avg"]).cumprod() * 100
                port["cum_actual"] = (1 + port["actual_avg"]).cumprod() * 100

                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(x=port.index, y=port["cum_pred"], name="예측 포트폴리오",
                                              line=dict(color='#818cf8', width=2.5)))
                fig_port.add_trace(go.Scatter(x=port.index, y=port["cum_actual"], name="실제 포트폴리오",
                                              line=dict(color='#10b981', width=2.5)))
                fig_port.add_hline(y=100, line_dash="dash", line_color="rgba(100,116,139,0.4)")
                fig_port.update_layout(height=450, yaxis_title="누적 수익률 (시작=100)", **PLOTLY_DARK)
                st.plotly_chart(fig_port, use_container_width=True)

                # 요약 카드
                pc1, pc2, pc3 = st.columns(3)
                pred_total = (port["cum_pred"].iloc[-1] / 100 - 1) if len(port) > 0 else 0
                actual_total = (port["cum_actual"].iloc[-1] / 100 - 1) if len(port) > 0 else 0
                with pc1:
                    st.markdown(_mc("예측 누적 수익률", f"{pred_total*100:+.1f}%", ""), unsafe_allow_html=True)
                with pc2:
                    s = "metric-good" if actual_total > 0 else "metric-bad"
                    st.markdown(_mc("실제 누적 수익률", f"{actual_total*100:+.1f}%", "", s), unsafe_allow_html=True)
                with pc3:
                    gap = actual_total - pred_total
                    st.markdown(_mc("예측 오차", f"{gap*100:+.1f}%p", "양수=보수적 예측"), unsafe_allow_html=True)
            else:
                st.info("포트폴리오 데이터 부족")
        else:
            st.info("평가된 예측이 필요합니다.")

    # ── Tab 5: Full History ──
    with tab5:
        st.markdown('<div class="section-header">전체 예측 이력</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-box">
            모든 예측 기록의 전체 이력입니다. <strong>evaluated=True</strong>인 행은 실제 수익률이 계산된 것입니다.
        </div>
        """, unsafe_allow_html=True)

        if not all_preds.empty:
            # 컬럼명 한국어화
            display = all_preds.copy()
            col_map = {
                "date": "날짜", "ticker": "종목", "pred_return": "예측수익률",
                "actual_return": "실제수익률", "hit": "적중", "model": "모델",
                "horizon": "기간", "factor_momentum": "모멘텀", "factor_value": "밸류",
                "factor_quality": "퀄리티", "factor_risk": "리스크",
                "score_total": "총점", "entry_price": "진입가",
                "exit_price": "청산가", "sector": "섹터", "evaluated": "평가완료",
            }
            display = display.rename(columns={k: v for k, v in col_map.items() if k in display.columns})

            st.dataframe(display, use_container_width=True, hide_index=True, height=500)

            # CSV 다운로드
            csv = all_preds.to_csv(index=False)
            st.download_button(
                "📥 CSV 다운로드",
                csv,
                "prediction_history.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.info("예측 기록이 없습니다.")

# Footer
st.markdown("---")
st.markdown('<div class="footer-text">Prediction Evaluation · ML Traceability · Quant Investment System v2</div>',
            unsafe_allow_html=True)
