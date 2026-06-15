"""Tests for run_auto_trader 순수 로직 — 세션 게이트 + verdict→신호 추출."""
from __future__ import annotations

from datetime import datetime

import pytest

from corvin_jarvis import run_auto_trader as rat
from corvin_jarvis import paper_portfolio as pp


# ============================================================
# 시장 세션 게이트 (KST)
# ============================================================

@pytest.mark.unit
@pytest.mark.parametrize("hh,mm,expected", [
    (9, 30, "KR"), (15, 0, "KR"), (8, 0, None),
    (23, 0, "US"), (2, 0, "US"), (16, 0, None), (22, 0, None),
])
def test_current_session(hh, mm, expected):
    assert rat.current_session(datetime(2026, 6, 16, hh, mm)) == expected


# ============================================================
# verdict → 매수/매도 신호
# ============================================================

@pytest.mark.unit
def test_buy_signals_from_verdicts():
    verdicts = {
        "005930": {"action": "매수"}, "AMD": {"action": "분할매수"},
        "012450": {"action": "비중축소"}, "META": {"action": "홀딩"},
    }
    pf = pp.PaperPortfolio(cash_krw=10_000_000, initial_krw=10_000_000)
    buys, _ = rat.signals_from_verdicts(verdicts, pf)
    syms = {b["symbol"] for b in buys}
    assert syms == {"005930", "AMD"}


@pytest.mark.unit
def test_sell_signals_only_for_held():
    """매도/비중축소는 *페이퍼 보유분*에만. 미보유는 무시."""
    verdicts = {"012450": {"action": "매도"}, "207940": {"action": "비중축소"}}
    pf = pp.PaperPortfolio(cash_krw=5_000_000, initial_krw=10_000_000)
    pf = pp.buy(pf, "012450", 6, 800_000)  # 보유
    # 207940 미보유
    _, sells = rat.signals_from_verdicts(verdicts, pf)
    by = {s["symbol"]: s for s in sells}
    assert "012450" in by and by["012450"]["qty"] == 6   # 매도=전량
    assert "207940" not in by  # 미보유 → 신호 없음


@pytest.mark.unit
def test_reduce_sells_one_third():
    verdicts = {"012450": {"action": "비중축소"}}
    pf = pp.PaperPortfolio(
        cash_krw=0, initial_krw=10_000_000,
        holdings={"012450": pp.Holding(9, 800_000)})
    _, sells = rat.signals_from_verdicts(verdicts, pf)
    assert sells[0]["qty"] == 3  # 9 * 1/3
