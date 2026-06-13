"""실시간 토론 툴 — Phase A: 검증 컨텍스트 빌더 테스트.

모든 수치를 data_verify로 교차검증 + 신호 신뢰도 태깅. 네트워크 없음(DI).
"""
from corvin_jarvis.live_debate import context as ctxmod


def _fetchers_for(sym):
    return {
        "MSFT": {"yf": lambda: 416.67, "kis": lambda: 417.00},   # 일치
        "XYZ": {"yf": lambda: 100.0, "kis": lambda: 110.0},      # 불일치
    }[sym]


def test_build_context_verifies_prices_and_pnl():
    holdings = [
        {"symbol": "MSFT", "shares": 6, "avg_price": 383.25, "stored_pnl_pct": 12.95},
        {"symbol": "XYZ", "shares": 1, "avg_price": 100.0, "stored_pnl_pct": 0.0},
    ]
    ctx = ctxmod.build_context(holdings, _fetchers_for, tol_pct=1.5)
    h = {x["symbol"]: x for x in ctx["holdings"]}

    # 가격 2소스 일치 → high
    assert h["MSFT"]["price"]["confidence"] == "high"
    # 라이브 손익 = (416.835/383.25-1)*100 ≈ 8.76%
    assert abs(h["MSFT"]["live_pnl_pct"] - 8.76) < 0.1
    # 저장 12.95% vs 라이브 8.76% 괴리 → stale flag (TSLA형 버그 가드)
    assert h["MSFT"]["pnl"]["stale"] is True

    # 소스 불일치 → 저신뢰
    assert h["XYZ"]["confidence"] == "low"
    assert "XYZ" in ctx["low_confidence"]


def test_attach_signal_reliability_tags_backtest_edge():
    ctx = {"signals": {"semis": "amber", "vix_term": "amber"}}
    report = {
        "semis_red": {"precision": 0.18, "n_events_rate": 0.16},
        "vix_term_red": {"precision": 0.33, "n_events_rate": 0.16},
    }
    out = ctxmod.attach_signal_reliability(ctx, report)
    # semis: 정밀도 ≈ 기저율 → low-edge
    assert out["signal_reliability"]["semis"]["edge"] == "low"
    # vix_term: 정밀도 ~2배 → has-edge
    assert out["signal_reliability"]["vix_term"]["edge"] == "has-edge"


def test_context_facts_extracts_allowed_numbers():
    ctx = {"holdings": [
        {"symbol": "MSFT", "live_pnl_pct": 8.76, "price": {"value": 416.84}},
    ]}
    facts = ctxmod.context_facts(ctx)
    assert ("MSFT", "live_pnl_pct", 8.76) in facts
    assert ("MSFT", "price", 416.84) in facts
