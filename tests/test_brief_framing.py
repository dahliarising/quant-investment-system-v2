from corvin_jarvis.brief.framing import build_framing


def test_framing_counts_directions():
    actions = [
        {"symbol": "TSLA", "action": "매도검토", "urgency": 95, "rationale": "추세이탈"},
        {"symbol": "NVDA", "action": "유지", "urgency": 40, "rationale": "모멘텀양호"},
        {"symbol": "BWXT", "action": "매수후보", "urgency": 55, "rationale": "원자력테제"},
    ]
    f = build_framing(actions)
    assert "추세이탈" in f.bear
    assert "모멘텀양호" in f.bull or "매수후보" in f.bull or "원자력테제" in f.bull
    assert "방어" in f.base
    # 최고 urgency=TSLA 매도검토 기준 counterfactual
    assert "TSLA" in f.counterfactual


def test_framing_empty_is_safe():
    f = build_framing([])
    assert f.base
    assert f.counterfactual
