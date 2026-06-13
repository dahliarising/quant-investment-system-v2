from corvin_jarvis import early_warning as ew

CFG = {
    "vix_term": {"green_max": 0.90, "amber_max": 1.00},
    "breadth": {"green_min": 60.0, "amber_min": 40.0},
    "hy": {"green_max": 3.5, "amber_max": 5.0, "rise_amber_5d": 0.3, "rise_red_5d": 0.5},
    "curve": {"green_min": 0.5, "amber_min": 0.0, "fast_move_5d": 0.15},
    "semis": {"divergence_high_dist_pct": 3.0},
}


# ── Task 2: band classifiers ─────────────────────────────────
def test_classify_vix_term():
    assert ew.classify_vix_term(0.85, CFG) == ew.GREEN
    assert ew.classify_vix_term(0.95, CFG) == ew.AMBER
    assert ew.classify_vix_term(1.05, CFG) == ew.RED


def test_classify_breadth():
    assert ew.classify_breadth(70, CFG) == ew.GREEN
    assert ew.classify_breadth(50, CFG) == ew.AMBER
    assert ew.classify_breadth(30, CFG) == ew.RED


def test_classify_hy_level_and_rise():
    assert ew.classify_hy(2.74, 0.0, CFG) == ew.GREEN
    assert ew.classify_hy(4.0, 0.0, CFG) == ew.AMBER       # level band
    assert ew.classify_hy(3.0, 0.35, CFG) == ew.AMBER      # rising fast
    assert ew.classify_hy(6.0, 0.0, CFG) == ew.RED         # level
    assert ew.classify_hy(3.0, 0.6, CFG) == ew.RED         # rise red


def test_classify_curve_inversion_and_fastmove():
    assert ew.classify_curve(0.8, 0.0, CFG) == ew.GREEN
    assert ew.classify_curve(0.3, 0.0, CFG) == ew.AMBER
    assert ew.classify_curve(-0.1, 0.0, CFG) == ew.RED     # inverted
    assert ew.classify_curve(0.8, 0.2, CFG) == ew.RED      # fast move


# ── Task 3: semis divergence ─────────────────────────────────
def test_classify_semis_divergence():
    cfg = {"semis": {"divergence_high_dist_pct": 3.0}}
    assert ew.classify_semis(1.0, 1.05, -0.02, -1.0, cfg) == ew.RED      # divergence
    assert ew.classify_semis(1.06, 1.05, -0.01, -1.0, cfg) == ew.AMBER   # rollover above MA
    assert ew.classify_semis(1.10, 1.05, 0.02, -1.0, cfg) == ew.GREEN    # rising
    assert ew.classify_semis(1.0, 1.05, -0.02, -8.0, cfg) == ew.AMBER    # spx already down → not divergence


# ── Task 4: gauge + action ───────────────────────────────────
def test_composite_gauge():
    g, a, r = ew.GREEN, ew.AMBER, ew.RED
    assert ew.composite_gauge({"semis": g, "vix_term": g, "breadth": g, "hy": g, "curve": g})[0] == ew.BUY
    assert ew.composite_gauge({"semis": r, "vix_term": g, "breadth": g, "hy": g, "curve": g})[0] == ew.REDUCE  # semis red escalates
    assert ew.composite_gauge({"vix_term": r, "breadth": g, "hy": g, "curve": g})[0] == ew.HOLD               # 1 red, not semis
    assert ew.composite_gauge({"vix_term": r, "curve": r, "breadth": g, "hy": g})[0] == ew.REDUCE             # 2 red
    assert ew.composite_gauge({"semis": r, "vix_term": r, "breadth": r, "hy": g, "curve": g})[0] == ew.SELL   # 3 red
    assert ew.composite_gauge({"semis": g, "vix_term": g, "breadth": r, "hy": r, "curve": g})[0] == ew.SELL   # combo


def test_action_label_held_vs_unheld():
    assert "파세요" in ew.action_label(ew.SELL, held=True)
    assert "진입 미루" in ew.action_label(ew.SELL, held=False)
    assert "사세요" in ew.action_label(ew.BUY, held=True)


# ── Task 5: transitions + messages ───────────────────────────
def test_detect_transitions_only_worsening():
    prev = {"semis": ew.GREEN, "vix_term": ew.AMBER}
    cur = {"semis": ew.RED, "vix_term": ew.AMBER}
    t = ew.detect_transitions(prev, cur)
    assert [x["key"] for x in t] == ["semis"]
    assert t[0]["to"] == ew.RED


def test_no_transition_when_stable_or_improving():
    prev = {"semis": ew.RED}
    assert ew.detect_transitions(prev, {"semis": ew.RED}) == []
    assert ew.detect_transitions(prev, {"semis": ew.GREEN}) == []


def test_indicator_message():
    assert "반도체 줄이세요" in ew.indicator_message("semis", ew.RED)


# ── Task 6: hard stop ────────────────────────────────────────
def test_hard_stop_only_held_breaches():
    positions = [
        {"sym": "012450", "pnl_pct": -14.6},
        {"sym": "TSLA", "pnl_pct": -10.5},
        {"sym": "MSFT", "pnl_pct": 8.7},
        {"sym": "BWXT", "pnl_pct": None},
    ]
    hits = ew.hard_stop(positions, threshold=-8.0)
    assert [h["sym"] for h in hits] == ["012450", "TSLA"]
    assert "파세요" in hits[0]["message"]
