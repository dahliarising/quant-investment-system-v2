"""STAGE① 시그널 엔진 — 명시적 가격 손절선 + 근접 시그널 + 통합 랭킹 (TDD)."""
import pytest

from corvin_jarvis import signal_engine as se


def _pos(symbol, price, pnl_pct, market="US"):
    return {"symbol": symbol, "market": market, "price": price, "pnl_pct": pnl_pct}


@pytest.mark.unit
def test_close_below_explicit_stop_emits_stop_sell():
    # 종가가 명시 손절선 아래 → STOP / 손절검토, 최고 긴급도
    sig = se.evaluate([_pos("TSLA", 370.0, -15.0)], stops={"TSLA": 377.0})
    assert len(sig) == 1
    s = sig[0]
    assert s.kind == "STOP"
    assert s.action == "손절검토"
    assert s.urgency >= 90
    assert s.stop == 377.0
    assert s.stop_distance_pct is not None and s.stop_distance_pct < 0


@pytest.mark.unit
def test_near_stop_emits_watch():
    # 손절선 위 근접(기본 3% 이내) → WATCH / 관찰
    sig = se.evaluate([_pos("012450", 1010000.0, -17.8, market="KR")],
                      stops={"012450": 1000000.0})
    s = sig[0]
    assert s.kind == "WATCH"
    assert "손절선" in s.reason
    assert 0 <= s.stop_distance_pct <= 3.0
    assert 60 <= s.urgency < 90


@pytest.mark.unit
def test_take_profit_emits_trim():
    # 명시 손절선 없고 익절선(+25%) 도달 → TRIM / 비중축소
    sig = se.evaluate([_pos("MSFT", 500.0, 30.0)])
    s = sig[0]
    assert s.kind == "TRIM"
    assert s.action == "비중축소"


@pytest.mark.unit
def test_pnl_stop_without_explicit_line():
    # 명시 손절선 없고 PnL이 -8% 이하 → STOP (verdict 임계 재사용)
    sig = se.evaluate([_pos("XYZ", 50.0, -9.0)])
    assert sig[0].kind == "STOP"


@pytest.mark.unit
def test_healthy_hold_low_urgency():
    sig = se.evaluate([_pos("NVDA", 208.0, 5.8)], stops={"NVDA": 189.0})
    s = sig[0]
    assert s.kind == "HOLD"
    assert s.urgency < 40


@pytest.mark.unit
def test_ranking_urgency_desc():
    # 여러 종목 → 긴급도 내림차순 정렬
    sig = se.evaluate(
        [_pos("NVDA", 208.0, 5.8), _pos("TSLA", 370.0, -15.0), _pos("MSFT", 500.0, 30.0)],
        stops={"TSLA": 377.0, "NVDA": 189.0},
    )
    urg = [s.urgency for s in sig]
    assert urg == sorted(urg, reverse=True)
    assert sig[0].symbol == "TSLA"   # 손절 위반이 1순위


@pytest.mark.unit
def test_missing_price_is_safe():
    # 가격 없으면 평가 불가 → kind UNKNOWN, 예외 없음
    sig = se.evaluate([{"symbol": "ZZZ", "market": "US", "price": None, "pnl_pct": None}])
    assert sig[0].kind == "UNKNOWN"


@pytest.mark.unit
def test_load_stops_from_json(tmp_path):
    p = tmp_path / "stops.json"
    p.write_text('{"TSLA": 377, "012450": 1000000}', encoding="utf-8")
    stops = se.load_stops(p)
    assert stops["TSLA"] == 377.0
    assert stops["012450"] == 1000000.0


@pytest.mark.unit
def test_empty_holdings():
    assert se.evaluate([]) == []
