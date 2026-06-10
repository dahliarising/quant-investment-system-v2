"""Phase 4 — data_gate 순수 게이트 코어 테스트."""
import pytest

from corvin_jarvis.signals import data_gate as dg


def _sig(symbol="TSLA", kind="STOP", **kw):
    base = {"symbol": symbol, "kind": kind, "urgency": 70, "confidence": 65.0,
            "message": "현재가 $310 — 손절 임박"}
    base.update(kw)
    return base


@pytest.mark.unit
def test_gate_passes_clean_signals_unchanged():
    """검증 통과 — 신호 내용 그대로 (장중이므로 라벨 미교정)."""
    sigs = [_sig()]
    res = dg.gate_signals(sigs, market_open=True)
    assert res["passed"] == sigs
    assert res["blocked"] == []
    assert res["warnings"] == []


@pytest.mark.unit
def test_gate_drops_nan_inf_signals():
    """NaN/inf 수치 신호만 drop — 나머지는 통과 (스펙 §7 row 4)."""
    bad_nan = _sig(symbol="AAA", confidence=float("nan"))
    bad_inf = _sig(symbol="BBB", urgency=float("inf"))
    good = _sig(symbol="CCC")
    res = dg.gate_signals([bad_nan, bad_inf, good], market_open=True)
    assert [s["symbol"] for s in res["passed"]] == ["CCC"]
    assert {b["reason"] for b in res["blocked"]} == {"invalid_numeric"}


@pytest.mark.unit
def test_gate_holds_signal_on_price_discrepancy():
    """가격 소스 불일치 → 해당 종목 신호 보류 + 경고 (스펙 §7 row 1)."""
    checks = {"TSLA": {"value": 310.0, "flag": "discrepancy", "spread_pct": 2.4},
              "NVDA": {"value": 180.0, "flag": None, "spread_pct": 0.1}}
    res = dg.gate_signals([_sig("TSLA"), _sig("NVDA")],
                          price_checks=checks, market_open=True)
    assert [s["symbol"] for s in res["passed"]] == ["NVDA"]
    assert res["blocked"][0]["reason"] == "price_discrepancy"
    assert any("TSLA" in w for w in res["warnings"])


@pytest.mark.unit
def test_gate_corrects_price_label_when_market_closed():
    """장마감 — '현재가' → '전일종가' 자동 교정, 원본 dict 비변이 (스펙 §7 row 2)."""
    orig = _sig()
    res = dg.gate_signals([orig], market_open=False)
    assert res["passed"][0]["message"] == "전일종가 $310 — 손절 임박"
    assert orig["message"] == "현재가 $310 — 손절 임박"  # 불변성


@pytest.mark.unit
def test_gate_keeps_label_when_market_open():
    res = dg.gate_signals([_sig()], market_open=True)
    assert "현재가" in res["passed"][0]["message"]


@pytest.mark.unit
def test_gate_blocks_all_on_stale_data():
    """데이터 한도 초과 — 전체 차단 + 갱신 요청 경고 (스펙 §7 row 3)."""
    res = dg.gate_signals([_sig("TSLA"), _sig("NVDA")],
                          market_open=True, data_age_days=8)
    assert res["passed"] == []
    assert all(b["reason"] == "stale_data" for b in res["blocked"])
    assert any("갱신" in w for w in res["warnings"])


@pytest.mark.unit
def test_gate_allows_within_stale_limit():
    """기본 한도(3일) 이내 — 통과."""
    res = dg.gate_signals([_sig()], market_open=True, data_age_days=2)
    assert len(res["passed"]) == 1


@pytest.mark.unit
def test_gate_custom_stale_limit():
    """max_age_days 커스텀 — 통합부의 7일 정책 지원."""
    res = dg.gate_signals([_sig()], market_open=True,
                          data_age_days=5, max_age_days=6)
    assert len(res["passed"]) == 1
    res2 = dg.gate_signals([_sig()], market_open=True,
                           data_age_days=7, max_age_days=6)
    assert res2["passed"] == []
