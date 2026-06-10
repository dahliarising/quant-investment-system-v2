"""signals/arbiter.py — 종목별 상충 중재 (안전 우선 → 적중률 가중 → 보수 기본)."""
from corvin_jarvis.signals import arbiter


def _sig(**over):
    base = {"engine": "signal_engine", "symbol": "NVDA", "kind": "STOP",
            "intent": "defensive", "urgency": 95, "confidence": None, "note": "손절선 이탈"}
    base.update(over)
    return base


def test_defensive_overrides_buy_with_conflict_flag():
    """규칙① 안전 우선 — STOP이 BUY_NOW를 이긴다."""
    sigs = [_sig(),
            _sig(engine="playbook", kind="BUY_NOW", intent="buy", urgency=50, note="딥밸류존")]
    acts = arbiter.arbitrate(sigs)
    assert len(acts) == 1
    a = acts[0]
    assert a.symbol == "NVDA"
    assert a.action == "매도검토"
    assert a.conflict is True
    assert "playbook" in a.sources and "signal_engine" in a.sources


def test_buy_vs_warn_both_unproven_holds():
    """규칙② 미검증 상충 → 보수적 보류."""
    sigs = [_sig(engine="playbook", kind="BUY_NOW", intent="buy", urgency=50),
            _sig(engine="predictive", kind="RS_WEAK", intent="warn", urgency=48)]
    acts = arbiter.arbitrate(sigs)  # calibration 없음 → 둘 다 미검증
    assert acts[0].action == "보류"
    assert acts[0].conflict is True


def test_buy_vs_warn_hit_rate_winner_buy():
    """규칙② 적중률 가중 — buy 쪽 엔진이 검증 우세(n>=10)면 매수후보."""
    sigs = [_sig(engine="playbook", kind="BUY_NOW", intent="buy", urgency=50),
            _sig(engine="predictive", kind="RS_WEAK", intent="warn", urgency=48)]
    cal = {"playbook": {"BUY_NOW": {"n": 20, "hit_rate": 0.8}},
           "predictive": {"RS_WEAK": {"n": 15, "hit_rate": 0.3}}}
    acts = arbiter.arbitrate(sigs, calibration=cal)
    assert acts[0].action == "매수후보"
    assert acts[0].conflict is True
    assert "적중률" in acts[0].rationale


def test_buy_vs_warn_hit_rate_winner_warn():
    """규칙② 적중률 가중 — warn 쪽 우세면 관찰."""
    sigs = [_sig(engine="playbook", kind="BUY_NOW", intent="buy", urgency=50),
            _sig(engine="predictive", kind="RS_WEAK", intent="warn", urgency=48)]
    cal = {"playbook": {"BUY_NOW": {"n": 20, "hit_rate": 0.3}},
           "predictive": {"RS_WEAK": {"n": 15, "hit_rate": 0.8}}}
    acts = arbiter.arbitrate(sigs, calibration=cal)
    assert acts[0].action == "관찰"


def test_small_sample_calibration_treated_unproven():
    """n<10 적중률은 미검증 취급 → 보류."""
    sigs = [_sig(engine="playbook", kind="BUY_NOW", intent="buy", urgency=50),
            _sig(engine="predictive", kind="RS_WEAK", intent="warn", urgency=48)]
    cal = {"playbook": {"BUY_NOW": {"n": 3, "hit_rate": 1.0}},
           "predictive": {"RS_WEAK": {"n": 2, "hit_rate": 0.0}}}
    acts = arbiter.arbitrate(sigs, calibration=cal)
    assert acts[0].action == "보류"


def test_trim_without_conflict():
    sigs = [_sig(kind="TRIM", intent="trim", urgency=55, note="익절선 도달")]
    acts = arbiter.arbitrate(sigs)
    assert acts[0].action == "비중축소"
    assert acts[0].conflict is False


def test_buy_only_is_buy_candidate():
    sigs = [_sig(engine="playbook", kind="BUY_NOW", intent="buy", urgency=50)]
    assert arbiter.arbitrate(sigs)[0].action == "매수후보"


def test_warn_only_is_watch():
    sigs = [_sig(engine="predictive", kind="RS_WEAK", intent="warn", urgency=48)]
    assert arbiter.arbitrate(sigs)[0].action == "관찰"


def test_hold_only_is_hold():
    sigs = [_sig(kind="HOLD", intent="hold", urgency=20, note="보유 논리 유효")]
    assert arbiter.arbitrate(sigs)[0].action == "홀딩"


def test_macro_empty_symbol_excluded():
    sigs = [_sig(symbol="", engine="predictive", kind="EVENT", intent="warn", urgency=80)]
    assert arbiter.arbitrate(sigs) == []


def test_multiple_symbols_sorted_by_priority():
    sigs = [_sig(symbol="GOOGL", engine="playbook", kind="BUY_NOW", intent="buy", urgency=50),
            _sig(symbol="012450")]  # defensive
    acts = arbiter.arbitrate(sigs)
    assert [a.symbol for a in acts] == ["012450", "GOOGL"]  # 매도검토(95) > 매수후보(50)


def test_to_dict_roundtrip():
    acts = arbiter.arbitrate([_sig()])
    d = acts[0].to_dict()
    assert d["symbol"] == "NVDA" and d["action"] == "매도검토" and isinstance(d["sources"], list)
