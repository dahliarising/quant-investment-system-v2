"""signals/ledger.py — 신호 원장 기록·중복방지·만기조회·채점마킹."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from corvin_jarvis.signals import ledger

KST = ZoneInfo("Asia/Seoul")
NOW = datetime(2026, 6, 10, 9, 0, tzinfo=KST)


def _sig(**over):
    base = {"symbol": "NVDA", "kind": "VELOCITY", "urgency": 70,
            "confidence": 65.0, "horizon_days": 5,
            "stop": 180.0, "current_price": 190.0}
    base.update(over)
    return base


def test_record_batch_inserts_and_returns_count(tmp_path):
    db = tmp_path / "ledger.db"
    n = ledger.record_batch("predictive", [_sig()], db_path=db, now=NOW)
    assert n == 1


def test_record_batch_dedups_same_open_key(tmp_path):
    db = tmp_path / "ledger.db"
    ledger.record_batch("predictive", [_sig()], db_path=db, now=NOW)
    n2 = ledger.record_batch("predictive", [_sig()], db_path=db, now=NOW)
    assert n2 == 0  # 같은 (engine,symbol,kind) open → skip


def test_record_batch_skips_hold_unknown(tmp_path):
    db = tmp_path / "ledger.db"
    n = ledger.record_batch("signal_engine",
                            [_sig(kind="HOLD"), _sig(kind="UNKNOWN")],
                            db_path=db, now=NOW)
    assert n == 0


def test_default_horizon_stop_watch_5_else_10(tmp_path):
    db = tmp_path / "ledger.db"
    ledger.record_batch("signal_engine",
                        [_sig(kind="STOP", horizon_days=None),
                         _sig(symbol="META", kind="RS_WEAK", horizon_days=None)],
                        db_path=db, now=NOW)
    due_at_6d = ledger.fetch_due(db_path=db, now=NOW + timedelta(days=6))
    kinds = {r["kind"] for r in due_at_6d}
    assert kinds == {"STOP"}  # STOP=5일 만기, RS_WEAK=10일이라 아직


def test_fetch_due_returns_evidence_dict_and_age(tmp_path):
    db = tmp_path / "ledger.db"
    ledger.record_batch("predictive", [_sig()], db_path=db, now=NOW)
    due = ledger.fetch_due(db_path=db, now=NOW + timedelta(days=6))
    assert len(due) == 1
    assert due[0]["evidence"]["stop"] == 180.0
    assert due[0]["age_days"] == 6


def test_mark_scored_closes_signal(tmp_path):
    db = tmp_path / "ledger.db"
    ledger.record_batch("predictive", [_sig()], db_path=db, now=NOW)
    due = ledger.fetch_due(db_path=db, now=NOW + timedelta(days=6))
    ledger.mark_scored(due[0]["id"], "hit", {"min_close": 175.0}, db_path=db, now=NOW)
    assert ledger.fetch_due(db_path=db, now=NOW + timedelta(days=6)) == []
    # 채점 후 같은 키 재기록 가능 (open 아님)
    n = ledger.record_batch("predictive", [_sig()], db_path=db, now=NOW)
    assert n == 1


def test_fetch_due_accepts_naive_datetime(tmp_path):
    db = tmp_path / "ledger.db"
    ledger.record_batch("predictive", [_sig()], db_path=db, now=NOW)
    # naive datetime → KST 가정, TypeError 없이 due 반환
    due = ledger.fetch_due(db_path=db, now=datetime(2026, 6, 16, 9, 0))
    assert len(due) == 1


def test_horizon_zero_clamped_to_one(tmp_path):
    db = tmp_path / "ledger.db"
    ledger.record_batch("predictive", [_sig(horizon_days=0)], db_path=db, now=NOW)
    due = ledger.fetch_due(db_path=db, now=NOW + timedelta(days=1))
    assert len(due) == 1  # horizon 0 → 1로 클램프 저장


def test_leading_horizon_str_mapping(tmp_path):
    """LeadingSignal horizon 문자열 → 일수 매핑."""
    db = tmp_path / "ledger.db"
    n = ledger.record_batch("leading", [
        {"symbol": "NVDA", "kind": "RS_PILLAR", "direction": "bear",
         "confidence": 70.0, "horizon_days": ledger.horizon_str_to_days("days")},
    ], db_path=db, now=NOW)
    assert n == 1
    due = ledger.fetch_due(db_path=db, now=NOW + timedelta(days=5))
    assert due[0]["horizon_days"] == 5


def test_horizon_str_to_days_mapping():
    assert ledger.horizon_str_to_days("intraday") == 1
    assert ledger.horizon_str_to_days("days") == 5
    assert ledger.horizon_str_to_days("weeks") == 20
    assert ledger.horizon_str_to_days("unknown") == 10  # fallback


def test_record_batch_literal_kind_wins_over_evidence_kind(tmp_path):
    """evidence에 'kind' 키가 있어도 canonical kind 유지 (스프레드 순서 회귀 방지)."""
    db = tmp_path / "ledger.db"
    sig = {"kind": "earnings", "days_to": 3}  # event evidence 모사
    ledger.record_batch("leading", [{
        **sig,
        "symbol": "NVDA", "kind": "event", "direction": "neutral",
        "confidence": 60.0, "horizon_days": 5,
    }], db_path=db, now=NOW)
    due = ledger.fetch_due(db_path=db, now=NOW + timedelta(days=5))
    assert due[0]["kind"] == "event"
    assert due[0]["evidence"]["days_to"] == 3
