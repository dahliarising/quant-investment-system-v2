from corvin_jarvis.brief.position_overlay import build_position_lines


def test_winner_held():
    pos = [{"symbol": "MSFT", "pnl_pct": 12.9, "bucket": "trade",
            "currency": "USD", "current_price": 390.3, "error": None}]
    lines = build_position_lines(pos, fresh_label="종가")
    assert lines[0].action_icon == "✅"
    assert lines[0].symbol == "MSFT"
    assert "유지" in lines[0].label


def test_dca_stop_zone():
    pos = [{"symbol": "012450", "pnl_pct": -17.4, "bucket": "dca",
            "currency": "KRW", "current_price": 1014000, "error": None}]
    lines = build_position_lines(pos, fresh_label="종가")
    assert lines[0].action_icon == "✂️"
    assert "손절존" in lines[0].label


def test_quote_failure_is_watch():
    pos = [{"symbol": "X", "pnl_pct": None, "bucket": "trade",
            "currency": "USD", "current_price": None, "error": "no data"}]
    lines = build_position_lines(pos, fresh_label="종가")
    assert lines[0].action_icon == "👀"
    assert "시세없음" in lines[0].label
