"""Phase 1 — 신호 적중률 원장. 모든 엔진 신호를 발화 시점에 SQLite 기록.

기존 엔진 무수정(additive). 기록 실패는 호출측 _safe 격리 — 신호 흐름 무영향.
중복 방지: 같은 (engine, symbol, kind) open 신호 존재 시 재기록 생략.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger("corvin.signals.ledger")

KST = ZoneInfo("Asia/Seoul")
DB_PATH = Path(__file__).resolve().parent.parent / "state" / "signal_ledger.db"

_DEFAULT_HORIZON = {"STOP": 5, "WATCH": 5}     # 스펙 §4.1: NULL이면 kind별 기본
_FALLBACK_HORIZON = 10
_HORIZON_STR = {"intraday": 1, "days": 5, "weeks": 20}
_SKIP_KINDS = {"HOLD", "UNKNOWN"}               # 비액션 신호는 기록 제외
_CORE_FIELDS = ("engine", "symbol", "kind", "direction", "urgency", "confidence",
                "horizon_days")

SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  engine TEXT NOT NULL,
  symbol TEXT NOT NULL,
  kind TEXT NOT NULL,
  direction TEXT,
  urgency INTEGER,
  confidence REAL,
  horizon_days INTEGER,
  evidence TEXT,
  status TEXT NOT NULL DEFAULT 'open',
  scored_at TEXT,
  outcome TEXT
);
CREATE INDEX IF NOT EXISTS idx_sl_status ON signal_ledger (status);
CREATE INDEX IF NOT EXISTS idx_sl_key ON signal_ledger (engine, symbol, kind, status);
"""


def init_db(db_path: Path | None = None) -> Path:
    """DB 파일 + 스키마 생성 (idempotent). timeseries.py 패턴."""
    p = db_path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as conn:
        conn.executescript(SCHEMA)
    return p


def horizon_str_to_days(horizon: str) -> int:
    """LeadingSignal.horizon 문자열 → 평가 일수."""
    return _HORIZON_STR.get(horizon, _FALLBACK_HORIZON)


def record_batch(engine: str, signals: list[dict[str, Any]],
                 db_path: Path | None = None,
                 now: datetime | None = None) -> int:
    """신호 dict 리스트 기록. 반환 = 신규 insert 수 (dedup·skip 제외)."""
    p = init_db(db_path)
    t = now or datetime.now(KST)
    if t.tzinfo is None:
        t = t.replace(tzinfo=KST)  # 방어: naive datetime → KST 가정
    ts = t.isoformat(timespec="seconds")
    inserted = 0
    with sqlite3.connect(p) as conn:
        for s in signals:
            kind = str(s.get("kind", ""))
            if not kind or kind in _SKIP_KINDS:
                continue
            sym = str(s.get("symbol", ""))
            dup = conn.execute(
                "SELECT 1 FROM signal_ledger WHERE engine=? AND symbol=? AND kind=?"
                " AND status='open' LIMIT 1", (engine, sym, kind)).fetchone()
            if dup:
                continue
            horizon = s.get("horizon_days")
            if horizon is None:
                horizon = _DEFAULT_HORIZON.get(kind, _FALLBACK_HORIZON)
            horizon = max(1, int(horizon))  # 0 이하 → 만기 불능 방지
            evidence = {k: v for k, v in s.items() if k not in _CORE_FIELDS}
            conn.execute(
                """INSERT INTO signal_ledger
                   (ts, engine, symbol, kind, direction, urgency, confidence,
                    horizon_days, evidence)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ts, engine, sym, kind, s.get("direction"), s.get("urgency"),
                 s.get("confidence"), horizon,
                 json.dumps(evidence, ensure_ascii=False, default=str)))
            inserted += 1
    return inserted


def fetch_due(db_path: Path | None = None,
              now: datetime | None = None) -> list[dict[str, Any]]:
    """만기(발화 후 horizon_days 경과) open 신호. evidence는 dict로 파싱, age_days 포함."""
    p = init_db(db_path)
    t = now or datetime.now(KST)
    if t.tzinfo is None:
        t = t.replace(tzinfo=KST)  # 방어: naive datetime → KST 가정 (aware-naive 빼기 방지)
    out: list[dict[str, Any]] = []
    with sqlite3.connect(p) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute("SELECT * FROM signal_ledger WHERE status='open'"):
            d = dict(r)
            try:
                fired = datetime.fromisoformat(d["ts"])
            except ValueError:
                continue
            age = (t - fired).days
            if age >= (d["horizon_days"] or _FALLBACK_HORIZON):
                d["evidence"] = json.loads(d["evidence"] or "{}")
                d["age_days"] = age
                out.append(d)
    return out


def fetch_open(db_path: Path | None = None) -> list[dict[str, Any]]:
    """open 신호 전체 (만기 무관) — arbiter 입력용. evidence는 dict로 파싱."""
    p = init_db(db_path)
    out: list[dict[str, Any]] = []
    with sqlite3.connect(p) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute("SELECT * FROM signal_ledger WHERE status='open'"):
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"] or "{}")
            out.append(d)
    return out


def mark_scored(row_id: int, status: str, outcome: dict[str, Any],
                db_path: Path | None = None, now: datetime | None = None) -> None:
    """채점 결과 기록 — status: hit | late_hit | miss | unscorable."""
    p = init_db(db_path)
    t = now or datetime.now(KST)
    if t.tzinfo is None:
        t = t.replace(tzinfo=KST)  # 방어: naive datetime → KST 가정
    ts = t.isoformat(timespec="seconds")
    with sqlite3.connect(p) as conn:
        cur = conn.execute(
            "UPDATE signal_ledger SET status=?, scored_at=?, outcome=? WHERE id=?",
            (status, ts, json.dumps(outcome, ensure_ascii=False, default=str), row_id))
        if cur.rowcount == 0:
            log.warning("mark_scored: id=%s not found", row_id)
