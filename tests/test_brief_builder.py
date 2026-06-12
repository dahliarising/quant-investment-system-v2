from corvin_jarvis.brief.builder import build_brief


def test_build_brief_end_to_end():
    positions = [
        {"symbol": "012450", "pnl_pct": -17.4, "bucket": "dca",
         "currency": "KRW", "current_price": 1014000, "error": None},
        {"symbol": "MSFT", "pnl_pct": 12.9, "bucket": "trade",
         "currency": "USD", "current_price": 390.3, "error": None},
    ]
    actions = [
        {"symbol": "TSLA", "action": "매도검토", "urgency": 95,
         "rationale": "추세이탈", "sources": ["jarvis"]},
        {"symbol": "BWXT", "action": "매수후보", "urgency": 55,
         "rationale": "원자력", "sources": ["leading"]},
    ]
    calibration = {"leading": {"매수후보": {"n": 12, "hit_rate": 0.6}}}
    brief = build_brief(positions=positions, actions=actions,
                        calibration=calibration, held={"012450", "MSFT"},
                        market_state="장마감", fresh_label="종가",
                        as_of="2026-06-12 05:45 KST")
    syms = {p.symbol for p in brief.positions}
    assert syms == {"012450", "MSFT"}
    # 012450 깊은 손실 → 드로다운 심리 가드
    assert brief.psych.triggered and brief.psych.pattern == "드로다운"
    assert brief.framing is not None
    assert brief.headline.startswith("🦅")
