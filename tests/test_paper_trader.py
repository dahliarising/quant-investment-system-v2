"""STAGE② 페이퍼 트레이딩 — 모의 청산 + 가상 원장 + 실현손익 (TDD). 실주문 0."""
import json

import pytest

from corvin_jarvis import paper_trader as pt
from corvin_jarvis.signal_engine import EngineSignal


def _sig(symbol, kind, price, action="—"):
    return EngineSignal(symbol=symbol, kind=kind, action=action, urgency=50,
                        reason="test", price=price, pnl_pct=None, stop=None,
                        stop_distance_pct=None)


def _pos(symbol, shares, avg, market="US"):
    return {"symbol": symbol, "shares": shares, "avgPrice": avg, "market": market}


@pytest.mark.unit
def test_stop_signal_simulates_full_exit():
    t = pt.simulate_from_signal(_sig("TSLA", "STOP", 370.0), _pos("TSLA", 2, 437.0),
                                ts="2026-06-09T15:30")
    assert t is not None
    assert t.side == "SELL"
    assert t.qty == 2          # 손절 = 전량 청산
    assert t.price == 370.0
    assert t.kind == "STOP"


@pytest.mark.unit
def test_trim_signal_simulates_partial_exit():
    t = pt.simulate_from_signal(_sig("MSFT", "TRIM", 500.0), _pos("MSFT", 6, 383.25),
                                ts="2026-06-09T05:00", fraction=0.5)
    assert t.side == "SELL"
    assert t.qty == 3          # 50% 트림 (6 → 3)
    assert t.kind == "TRIM"


@pytest.mark.unit
def test_hold_and_watch_produce_no_trade():
    assert pt.simulate_from_signal(_sig("NVDA", "HOLD", 208.0), _pos("NVDA", 12, 197.0)) is None
    assert pt.simulate_from_signal(_sig("X", "WATCH", 100.0), _pos("X", 1, 100.0)) is None
    assert pt.simulate_from_signal(_sig("Y", "UNKNOWN", None), _pos("Y", 1, 100.0)) is None


@pytest.mark.unit
def test_trim_rounds_down_and_skips_when_zero():
    # 1주 보유 50% 트림 → 0주 → 거래 없음 (소수주 매도 안 함)
    assert pt.simulate_from_signal(_sig("Z", "TRIM", 10.0), _pos("Z", 1, 8.0), fraction=0.5) is None


@pytest.mark.unit
def test_realized_pnl_native_per_symbol():
    trades = [
        pt.PaperTrade(ts="t1", symbol="TSLA", side="SELL", qty=2, price=370.0,
                      reason="stop", kind="STOP"),
        pt.PaperTrade(ts="t2", symbol="MSFT", side="SELL", qty=3, price=500.0,
                      reason="trim", kind="TRIM"),
    ]
    out = pt.realized_pnl(trades, {"TSLA": 437.0, "MSFT": 383.25})
    assert out["TSLA"]["realized"] == pytest.approx(2 * (370.0 - 437.0))   # -134.0
    assert out["MSFT"]["realized"] == pytest.approx(3 * (500.0 - 383.25))  # +350.25
    assert out["TSLA"]["realized_pct"] == pytest.approx((370.0 / 437.0 - 1) * 100, rel=1e-3)


@pytest.mark.unit
def test_propose_exits_triggers_only_on_stop():
    holdings = [
        {"symbol": "TSLA", "market": "US", "price": 370.0, "pnl_pct": -15.0, "shares": 2},
        {"symbol": "NVDA", "market": "US", "price": 208.0, "pnl_pct": 5.8, "shares": 12},
    ]
    props = pt.propose_exits(holdings, stops={"TSLA": 377.0, "NVDA": 189.0})
    # TSLA 종가 < 손절선 377 → 청산 제안 1건, NVDA HOLD → 없음
    assert len(props) == 1
    assert props[0].symbol == "TSLA"
    assert props[0].side == "SELL"
    assert props[0].qty == 2


@pytest.mark.unit
def test_propose_exits_empty_when_all_hold():
    holdings = [{"symbol": "NVDA", "market": "US", "price": 208.0, "pnl_pct": 5.8, "shares": 12}]
    assert pt.propose_exits(holdings, stops={"NVDA": 189.0}) == []


@pytest.mark.unit
def test_ledger_roundtrip(tmp_path):
    p = tmp_path / "paper_trades.json"
    led = pt.PaperLedger.load(p)
    assert led.trades == []
    led.add(pt.PaperTrade(ts="t1", symbol="TSLA", side="SELL", qty=2, price=370.0,
                          reason="stop", kind="STOP"))
    led.save()
    # reload
    led2 = pt.PaperLedger.load(p)
    assert len(led2.trades) == 1
    assert led2.trades[0].symbol == "TSLA"
    # file is valid json with list
    assert isinstance(json.loads(p.read_text()), list)


@pytest.mark.unit
def test_ledger_add_is_immutable_append(tmp_path):
    led = pt.PaperLedger.load(tmp_path / "x.json")
    before = led.trades
    led.add(pt.PaperTrade(ts="t", symbol="A", side="SELL", qty=1, price=1.0,
                          reason="r", kind="STOP"))
    assert before == []          # 원본 리스트 불변
    assert len(led.trades) == 1
