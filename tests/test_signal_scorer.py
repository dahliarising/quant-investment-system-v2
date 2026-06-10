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


# ── 경계 조건 (리뷰 반영) ─────────────────────────────
def test_velocity_delayed_scoring_beyond_grace_is_miss():
    """grace(7.5일) 밖 8일째 도달 — 지연 채점이어도 miss (리뷰 재현 케이스)."""
    closes = [188, 186, 185, 184, 183, 182, 181, 180, 175]
    status, _ = scorer.score_row(_row(age_days=9), closes_after=closes, bench_after=[])
    assert status == "miss"


def test_velocity_exactly_at_stop_is_hit():
    closes = [188, 185, 180.0, 184, 186]  # close == stop → 도달
    status, _ = scorer.score_row(_row(), closes_after=closes, bench_after=[])
    assert status == "hit"


def test_stop_exactly_at_ref_is_miss():
    row = _row(kind="STOP", horizon_days=5, age_days=5, evidence={"price": 100.0})
    status, _ = scorer.score_row(row, closes_after=[100.0, 101, 102, 103, 104], bench_after=[])
    assert status == "miss"  # 추가 하락 없음 (low == ref)


def test_rs_weak_zero_rs_is_miss():
    row = _row(kind="RS_WEAK", horizon_days=10, age_days=10, evidence={})
    same = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    status, _ = scorer.score_row(row, closes_after=same, bench_after=list(same))
    assert status == "miss"  # rs == 0 → 약세 지속 아님


def test_event_zero_bench_price_unscorable():
    row = _row(kind="EVENT", symbol="", horizon_days=2, age_days=3, evidence={})
    status, _ = scorer.score_row(row, closes_after=[], bench_after=[99.0],
                                 bench_before=[100, 0.0, 100.2])
    assert status == "unscorable"


# ── 러너 ──────────────────────────────────────────────
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from corvin_jarvis.signals import ledger

KST = ZoneInfo("Asia/Seoul")
NOW = datetime(2026, 6, 10, 16, 30, tzinfo=KST)


def test_run_scores_due_signals_and_updates_ledger(tmp_path):
    db = tmp_path / "ledger.db"
    fired = NOW - timedelta(days=9)  # horizon 5 → 만기 + 유예(7.5) 경과
    ledger.record_batch("predictive", [{
        "symbol": "NVDA", "kind": "VELOCITY", "urgency": 70, "confidence": 65.0,
        "horizon_days": 5, "stop": 180.0, "current_price": 190.0,
    }], db_path=db, now=fired)

    def fake_fetch(symbol, days):
        return [188, 185, 179, 182, 184, 183, 182]  # 3일째 hit

    result = scorer.run(db_path=db, now=NOW, fetch_closes=fake_fetch)
    assert result["scored"] == 1
    assert result["by_status"]["hit"] == 1
    assert ledger.fetch_due(db_path=db, now=NOW) == []


def test_run_keeps_pending_signal_open(tmp_path):
    db = tmp_path / "ledger.db"
    fired = NOW - timedelta(days=6)  # 만기(5) 도달, 유예(7.5) 이내
    ledger.record_batch("predictive", [{
        "symbol": "NVDA", "kind": "VELOCITY", "urgency": 70, "confidence": 65.0,
        "horizon_days": 5, "stop": 180.0,
    }], db_path=db, now=fired)

    def fake_fetch(symbol, days):
        return [188, 186, 185, 184, 183, 182]  # 미도달

    result = scorer.run(db_path=db, now=NOW, fetch_closes=fake_fetch)
    assert result["scored"] == 0
    assert result["pending"] == 1
    assert len(ledger.fetch_due(db_path=db, now=NOW)) == 1  # 여전히 open


def test_run_skips_row_on_fetch_error(tmp_path):
    """fetch 예외 — 해당 행만 skip, open 유지 (cron 배치 생존)."""
    db = tmp_path / "ledger.db"
    fired = NOW - timedelta(days=9)
    ledger.record_batch("predictive", [{
        "symbol": "NVDA", "kind": "VELOCITY", "urgency": 70, "confidence": 65.0,
        "horizon_days": 5, "stop": 180.0,
    }], db_path=db, now=fired)

    def failing_fetch(symbol, days):
        raise ConnectionError("network down")

    result = scorer.run(db_path=db, now=NOW, fetch_closes=failing_fetch)
    assert result["scored"] == 0
    assert result["by_status"]["fetch_error"] == 1
    assert len(ledger.fetch_due(db_path=db, now=NOW)) == 1  # 여전히 open


def test_run_leaves_open_on_empty_fetch(tmp_path):
    """fetch 빈 응답 — 영구 unscorable 금지, open 유지."""
    db = tmp_path / "ledger.db"
    fired = NOW - timedelta(days=9)
    ledger.record_batch("predictive", [{
        "symbol": "NVDA", "kind": "VELOCITY", "urgency": 70, "confidence": 65.0,
        "horizon_days": 5, "stop": 180.0,
    }], db_path=db, now=fired)

    result = scorer.run(db_path=db, now=NOW, fetch_closes=lambda s, d: [])
    assert result["scored"] == 0
    assert result["pending"] == 1
    assert len(ledger.fetch_due(db_path=db, now=NOW)) == 1
