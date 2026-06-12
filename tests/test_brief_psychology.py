from corvin_jarvis.brief.psychology import build_psych_guard


def test_drawdown_trigger():
    positions = [{"symbol": "012450", "pnl_pct": -17.4}]
    g = build_psych_guard(positions, has_new_buy_candidate=False)
    assert g.triggered is True
    assert g.pattern == "드로다운"
    assert "룰 손절선" in g.question


def test_fomo_trigger_when_no_drawdown():
    positions = [{"symbol": "MSFT", "pnl_pct": 12.0}]
    g = build_psych_guard(positions, has_new_buy_candidate=True)
    assert g.triggered is True
    assert g.pattern == "FOMO"


def test_no_trigger():
    positions = [{"symbol": "MSFT", "pnl_pct": 2.0}]
    g = build_psych_guard(positions, has_new_buy_candidate=False)
    assert g.triggered is False
    assert g.question is None
