"""signals/scorer.py — 채점 규칙 (순수 함수, 가격 주입)."""
from corvin_jarvis.signals import scorer


def _row(**over):
    base = {"id": 1, "kind": "VELOCITY", "symbol": "NVDA", "direction": None,
            "horizon_days": 5, "age_days": 6,
            "evidence": {"stop": 180.0, "current_price": 190.0}}
    base.update(over)
    return base


# ── VELOCITY ──────────────────────────────────────────
def test_velocity_hit_when_stop_reached_within_horizon():
    closes = [188, 185, 179, 182, 184]  # 3일째 179 ≤ 180
    status, outcome = scorer.score_row(_row(), closes_after=closes, bench_after=[])
    assert status == "hit"


def test_velocity_late_hit_within_1_5x_horizon():
    # horizon=5 내 미도달, 7일째(≤7.5) 도달
    closes = [188, 186, 185, 184, 183, 182, 179]
    row = _row(age_days=8)
    status, _ = scorer.score_row(row, closes_after=closes, bench_after=[])
    assert status == "late_hit"


def test_velocity_pending_during_grace_window():
    # horizon=5, age=6 (<7.5) 인데 아직 미도달 → 판정 보류 (None)
    closes = [188, 186, 185, 184, 183, 182]
    assert scorer.score_row(_row(age_days=6), closes_after=closes, bench_after=[]) is None


def test_velocity_miss_after_grace_window():
    closes = [188, 186, 185, 184, 183, 182, 181, 185]
    status, _ = scorer.score_row(_row(age_days=9), closes_after=closes, bench_after=[])
    assert status == "miss"


def test_velocity_unscorable_without_closes():
    status, _ = scorer.score_row(_row(), closes_after=[], bench_after=[])
    assert status == "unscorable"


# ── RS_WEAK ───────────────────────────────────────────
def test_rs_weak_hit_when_underperformance_continues():
    row = _row(kind="RS_WEAK", horizon_days=10, age_days=10, evidence={})
    sym = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]    # -9%
    bench = [100, 100, 101, 101, 102, 102, 103, 103, 104, 104]  # +4%
    status, outcome = scorer.score_row(row, closes_after=sym, bench_after=bench)
    assert status == "hit"
    assert outcome["rs_pct"] < 0


def test_rs_weak_miss_when_recovers():
    row = _row(kind="RS_WEAK", horizon_days=10, age_days=10, evidence={})
    sym = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
    bench = [100, 100, 101, 101, 102, 102, 103, 103, 104, 104]
    status, _ = scorer.score_row(row, closes_after=sym, bench_after=bench)
    assert status == "miss"


# ── EVENT ─────────────────────────────────────────────
def test_event_hit_on_vol_expansion():
    row = _row(kind="EVENT", symbol="", horizon_days=2, age_days=3, evidence={})
    bench_before = [100, 100.2, 100.1, 100.3, 100.2, 100.4]  # 일변동 ~0.15%
    bench_after = [99.0]  # -1.4% → 평소의 1.3배 초과
    status, _ = scorer.score_row(row, closes_after=[], bench_after=bench_after,
                                 bench_before=bench_before)
    assert status == "hit"


def test_event_miss_on_calm_day():
    row = _row(kind="EVENT", symbol="", horizon_days=2, age_days=3, evidence={})
    bench_before = [100, 101, 99, 101, 99, 101]  # 일변동 ~1.5%
    bench_after = [100.1]
    status, _ = scorer.score_row(row, closes_after=[], bench_after=bench_after,
                                 bench_before=bench_before)
    assert status == "miss"


# ── STOP/WATCH ────────────────────────────────────────
def test_stop_hit_when_further_decline():
    row = _row(kind="STOP", horizon_days=5, age_days=5,
               evidence={"price": 100.0})
    status, _ = scorer.score_row(row, closes_after=[99, 98, 96, 97, 95], bench_after=[])
    assert status == "hit"  # 방어 신호 유효 — 신호 후 추가 하락


def test_watch_miss_when_recovered():
    row = _row(kind="WATCH", horizon_days=5, age_days=5,
               evidence={"price": 100.0})
    status, _ = scorer.score_row(row, closes_after=[101, 102, 103, 104, 105], bench_after=[])
    assert status == "miss"


# ── 방향성 (leading/EW) ───────────────────────────────
def test_directional_bear_hit_on_decline():
    row = _row(kind="RS_PILLAR", direction="bear", horizon_days=5, age_days=8,
               evidence={})
    status, _ = scorer.score_row(row, closes_after=[99, 98, 97, 96, 95], bench_after=[])
    assert status == "hit"


def test_unknown_kind_without_direction_unscorable():
    row = _row(kind="JARVIS_ALERT", direction=None, evidence={})
    status, _ = scorer.score_row(row, closes_after=[100], bench_after=[100])
    assert status == "unscorable"
