"""Tests for corvin_jarvis.auto_strategy — 종목선정 + 가드 (행동·리스크).

오늘 진단한 실패패턴(추격·FOMO·집중)을 코드 룰로 검증.
"""
from __future__ import annotations

import pytest

from corvin_jarvis import auto_strategy as st
from corvin_jarvis import paper_portfolio as pp


# ============================================================
# 행동 가드 — no-chase (추격 금지)
# ============================================================

@pytest.mark.unit
def test_chasing_when_extended_above_ma():
    m = st.ChaseMetrics(pct_above_ma20=20.0, rsi=55, pct_from_52w_high=-10)
    chasing, why = st.is_chasing(m)
    assert chasing is True
    assert "MA" in why


@pytest.mark.unit
def test_chasing_when_overbought_rsi():
    m = st.ChaseMetrics(pct_above_ma20=5, rsi=75.0, pct_from_52w_high=-10)
    assert st.is_chasing(m)[0] is True


@pytest.mark.unit
def test_chasing_when_near_52w_high():
    """52주 고점 -2% 이내 = 거의 고점 진입 = 추격 (SK하이닉스 케이스)."""
    m = st.ChaseMetrics(pct_above_ma20=5, rsi=60, pct_from_52w_high=-2.0)
    assert st.is_chasing(m)[0] is True


@pytest.mark.unit
def test_not_chasing_when_pulled_back():
    """눌림(MA 근처·RSI 중립·고점서 멀리) = 진입 허용."""
    m = st.ChaseMetrics(pct_above_ma20=2, rsi=45, pct_from_52w_high=-20)
    assert st.is_chasing(m)[0] is False


@pytest.mark.unit
def test_missing_metrics_not_chasing():
    """지표 없으면 추격 판정 안 함 (보수적으로 통과, 단 None은 차단 안 함)."""
    m = st.ChaseMetrics(pct_above_ma20=None, rsi=None, pct_from_52w_high=None)
    assert st.is_chasing(m)[0] is False


# ============================================================
# 리스크 가드 — 비중 상한
# ============================================================

@pytest.mark.unit
def test_cap_qty_limits_to_max_weight():
    """총자산 10M, 상한 20% → 종목당 최대 2M. 가격 500k면 4주 캡."""
    pf = pp.PaperPortfolio(cash_krw=10_000_000, initial_krw=10_000_000)
    qty = st.cap_qty_to_weight(pf, "AAA", 100, 500_000, {}, max_pct=20.0)
    assert qty == 4  # 2M / 500k


@pytest.mark.unit
def test_cap_qty_accounts_existing_position():
    """이미 15% 보유 중이면 상한 20%까지 5%만 추가 가능."""
    pf = pp.PaperPortfolio(cash_krw=10_000_000, initial_krw=10_000_000)
    pf = pp.buy(pf, "AAA", 3, 500_000)  # 1.5M = 15% of 10M
    qty = st.cap_qty_to_weight(pf, "AAA", 100, 500_000, {"AAA": 500_000}, max_pct=20.0)
    # 여유 = 20%*10M - 1.5M = 0.5M → 1주
    assert qty == 1


@pytest.mark.unit
def test_cap_qty_zero_when_full():
    pf = pp.PaperPortfolio(cash_krw=10_000_000, initial_krw=10_000_000)
    pf = pp.buy(pf, "AAA", 4, 500_000)  # 2M = 20%
    qty = st.cap_qty_to_weight(pf, "AAA", 100, 500_000, {"AAA": 500_000}, max_pct=20.0)
    assert qty == 0


# ============================================================
# plan_cycle — 통합 (매도 + 가드된 매수)
# ============================================================

@pytest.mark.unit
def test_plan_buys_guarded_and_sells():
    pf = pp.PaperPortfolio(cash_krw=10_000_000, initial_krw=10_000_000)
    pf = pp.buy(pf, "012450", 4, 1_000_000)  # 보유 (매도 대상)
    plans = st.plan_cycle(
        pf,
        buy_signals=[{"symbol": "005930", "action": "매수"}],
        sell_signals=[{"symbol": "012450", "qty": 4, "reason": "손절"}],
        prices_krw={"005930": 300_000, "012450": 900_000},
        chase_metrics={"005930": st.ChaseMetrics(2, 45, -20)},  # 눌림 → 통과
        buy_krw=500_000, max_position_pct=20.0,
    )
    sides = {(p["symbol"], p["side"], p["status"]) for p in plans}
    assert ("012450", "sell", "planned") in sides
    assert ("005930", "buy", "planned") in sides


@pytest.mark.unit
def test_plan_skips_chasing_buy():
    pf = pp.PaperPortfolio(cash_krw=10_000_000, initial_krw=10_000_000)
    plans = st.plan_cycle(
        pf,
        buy_signals=[{"symbol": "000660", "action": "매수"}],
        sell_signals=[],
        prices_krw={"000660": 2_300_000},
        chase_metrics={"000660": st.ChaseMetrics(50, 65, -1)},  # 고점 추격
        buy_krw=5_000_000, max_position_pct=50.0,
    )
    p = [x for x in plans if x["symbol"] == "000660"][0]
    assert p["status"] == "skipped"
    assert "고점" in p["skip_reason"] or "MA" in p["skip_reason"]
