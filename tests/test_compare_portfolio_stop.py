"""compare.check_portfolio — ATR 변동성 조정 손절 + A/B 버킷 (2026-06-11)."""
import pytest

from corvin_jarvis import compare


def _latest(positions):
    return {"portfolio": positions}


_TH = {"stop_loss_pct": -8.0, "take_profit_pct": 25.0, "pnl_alert_pct": 10.0}


def _stop_alerts(alerts):
    return [a for a in alerts if a.metric.startswith("stop_loss_")]


@pytest.mark.unit
def test_trade_position_flat_stop_fires_without_atr():
    """버킷 미지정(=trade) + ATR 없음 — 기존 평면 -8% 동작 보존."""
    alerts = compare.check_portfolio(
        _latest([{"symbol": "TSLA", "pnl_pct": -9.0}]), _TH)
    assert len(_stop_alerts(alerts)) == 1


@pytest.mark.unit
def test_dca_position_skips_price_stop():
    """B(dca) 버킷 — 가격 손절 미발동. -20%여도 STOP 알림 없음."""
    alerts = compare.check_portfolio(
        _latest([{"symbol": "012450", "pnl_pct": -20.0, "bucket": "dca"}]), _TH)
    assert _stop_alerts(alerts) == []


@pytest.mark.unit
def test_atr_position_uses_wider_stop():
    """고변동 ATR 손절(-13%)이 평면(-8)보다 넓음 — -10%는 발동 안 함."""
    alerts = compare.check_portfolio(
        _latest([{"symbol": "TSLA", "pnl_pct": -10.0, "atr_pct": 2.6}]), _TH)
    assert _stop_alerts(alerts) == []


@pytest.mark.unit
def test_atr_position_stop_fires_when_breached():
    """ATR 손절선(-13%) 돌파 — STOP 발동, 임계가 ATR값."""
    alerts = compare.check_portfolio(
        _latest([{"symbol": "TSLA", "pnl_pct": -14.0, "atr_pct": 2.6}]), _TH)
    sa = _stop_alerts(alerts)
    assert len(sa) == 1
    assert abs(sa[0].threshold - (-13.0)) < 0.01


@pytest.mark.unit
def test_per_position_stop_override():
    """포지션별 stop_loss_pct 명시 override — ATR·평면보다 우선."""
    pos = {"symbol": "X", "pnl_pct": -12.0, "stop_loss_pct": -15.0}
    assert _stop_alerts(compare.check_portfolio(_latest([pos]), _TH)) == []
    pos2 = {"symbol": "X", "pnl_pct": -16.0, "stop_loss_pct": -15.0}
    assert len(_stop_alerts(compare.check_portfolio(_latest([pos2]), _TH))) == 1


@pytest.mark.unit
def test_dca_position_take_profit_still_fires():
    """B 버킷도 익절은 발동 (가격손절만 면제)."""
    alerts = compare.check_portfolio(
        _latest([{"symbol": "Y", "pnl_pct": 30.0, "bucket": "dca"}]), _TH)
    assert any(a.metric.startswith("take_profit_") for a in alerts)


@pytest.mark.unit
def test_bucket_survives_positionquote_serialization(monkeypatch):
    """회귀 가드: pulse.PositionQuote→asdict 경로에서 bucket 보존 (compare 입력)."""
    from dataclasses import asdict
    from corvin_jarvis import pulse

    class _Q:
        price, pct_change, source, error = 100.0, 0.0, "test", None
    monkeypatch.setattr(pulse.quote_provider, "get_stock_quote", lambda s: _Q())
    pq = pulse._fetch_position_quote(
        {"symbol": "012450", "shares": 4, "currency": "KRW",
         "avgPriceKRW": 125.0, "bucket": "dca"})   # -20% 손실 유발
    pos = asdict(pq)
    assert pos["bucket"] == "dca"
    # 직렬화 dict가 compare로 들어가도 dca 면제 작동
    alerts = compare.check_portfolio(_latest([pos]), _TH)
    assert _stop_alerts(alerts) == []
