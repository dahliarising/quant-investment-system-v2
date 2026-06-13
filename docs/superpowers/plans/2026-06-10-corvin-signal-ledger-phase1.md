# Corvin 신호 적중률 원장 (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모든 시그널 엔진의 발화 신호를 SQLite 원장에 기록하고, 만기 시 채점(HIT/LATE_HIT/MISS/UNSCORABLE)하여 엔진×종류별 적중률 기반 confidence 보정값을 산출한다.

**Architecture:** 기존 엔진 무수정(additive). `signals/ledger.py`(기록) → `signals/scorer.py`(채점, 순수 함수 + 러너) → `signals/calibration.py`(보정) 3모듈. snapshot.py·run_leading.py·jarvis.py가 발화 직후 `_safe` 격리로 record 호출. 채점은 16:30 KST cron.

**Tech Stack:** Python 3 표준 라이브러리(sqlite3, json, datetime) + 기존 quote_provider. 신규 외부 의존성 0.

**Spec:** `docs/superpowers/specs/2026-06-10-corvin-signal-feedback-loop-design.md`

---

## File Structure

```
corvin_jarvis/signals/ledger.py        # 신규 — SQLite 원장 (기록·만기조회·채점마킹)
corvin_jarvis/signals/scorer.py        # 신규 — 채점 규칙(순수) + 러너(가격 fetch) + __main__
corvin_jarvis/signals/calibration.py   # 신규 — 적중률 집계 → calibration.json + scoreboard
corvin_jarvis/run_scorer.sh            # 신규 — cron 스크립트 (run_digest.sh 패턴)
corvin_jarvis/dashboard/snapshot.py    # 수정 — record 훅 + signal_scoreboard 섹션
corvin_jarvis/run_leading.py           # 수정 — leading 신호 record 훅
corvin_jarvis/jarvis.py                # 수정 — alert record 훅
corvin_jarvis/dashboard/static/command_center.html  # 수정 — 적중률 미니패널
tests/test_signal_ledger.py            # 신규
tests/test_signal_scorer.py            # 신규
tests/test_signal_calibration.py       # 신규
tests/test_dashboard_snapshot.py       # 수정 — scoreboard 섹션 + record 훅 테스트 추가
```

**핵심 제약 (구현 전 숙지):**
- `qp.get_stock_daily_closes(symbol, days=N)` → `list[float]` oldest→newest, **날짜 없음**.
  발화일 이후 종가 정렬은 거래일 근사 `est = max(1, round(age_days * 5/7))` 로 `closes[-est:]` 슬라이스.
  근사 오차는 ±1~2 거래일 — HIT 판정엔 충분 (스펙 §4.2 합의).
- 테스트는 절대 실제 전송/실제 네트워크 금지 — scorer 러너는 fetch 함수 주입으로 테스트.
- DB는 모든 public 함수에서 `db_path` 인자 주입 가능 (테스트 tmp_path).

---

### Task 1: `signals/ledger.py` — 원장 기록·중복방지

**Files:**
- Create: `corvin_jarvis/signals/ledger.py`
- Test: `tests/test_signal_ledger.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signal_ledger.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_signal_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError` 또는 `AttributeError` (ledger 모듈 없음)

- [ ] **Step 3: Write the implementation**

```python
# corvin_jarvis/signals/ledger.py
"""Phase 1 — 신호 적중률 원장. 모든 엔진 신호를 발화 시점에 SQLite 기록.

기존 엔진 무수정(additive). 기록 실패는 호출측 _safe 격리 — 신호 흐름 무영향.
중복 방지: 같은 (engine, symbol, kind) open 신호 존재 시 재기록 생략.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
DB_PATH = Path(__file__).resolve().parent.parent / "state" / "signal_ledger.db"

_DEFAULT_HORIZON = {"STOP": 5, "WATCH": 5}     # 스펙 §4.1: NULL이면 kind별 기본
_FALLBACK_HORIZON = 10
_SKIP_KINDS = {"HOLD", "UNKNOWN"}               # 비액션 신호는 기록 제외
_CORE_FIELDS = ("symbol", "kind", "direction", "urgency", "confidence", "horizon_days")

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


def record_batch(engine: str, signals: list[dict[str, Any]],
                 db_path: Path | None = None,
                 now: datetime | None = None) -> int:
    """신호 dict 리스트 기록. 반환 = 신규 insert 수 (dedup·skip 제외)."""
    p = init_db(db_path)
    ts = (now or datetime.now(KST)).isoformat(timespec="seconds")
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
            evidence = {k: v for k, v in s.items() if k not in _CORE_FIELDS}
            conn.execute(
                """INSERT INTO signal_ledger
                   (ts, engine, symbol, kind, direction, urgency, confidence,
                    horizon_days, evidence)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ts, engine, sym, kind, s.get("direction"), s.get("urgency"),
                 s.get("confidence"), int(horizon),
                 json.dumps(evidence, ensure_ascii=False, default=str)))
            inserted += 1
    return inserted


def fetch_due(db_path: Path | None = None,
              now: datetime | None = None) -> list[dict[str, Any]]:
    """만기(발화 후 horizon_days 경과) open 신호. evidence는 dict로 파싱, age_days 포함."""
    p = init_db(db_path)
    t = now or datetime.now(KST)
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


def mark_scored(row_id: int, status: str, outcome: dict[str, Any],
                db_path: Path | None = None, now: datetime | None = None) -> None:
    """채점 결과 기록 — status: hit | late_hit | miss | unscorable."""
    p = init_db(db_path)
    ts = (now or datetime.now(KST)).isoformat(timespec="seconds")
    with sqlite3.connect(p) as conn:
        conn.execute(
            "UPDATE signal_ledger SET status=?, scored_at=?, outcome=? WHERE id=?",
            (status, ts, json.dumps(outcome, ensure_ascii=False, default=str), row_id))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_signal_ledger.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/ledger.py tests/test_signal_ledger.py
git commit -m "feat(signals): signal ledger — 발화 신호 SQLite 기록 + 중복방지 + 만기조회"
```

---

### Task 2: `signals/scorer.py` — 채점 규칙 (순수 함수)

**Files:**
- Create: `corvin_jarvis/signals/scorer.py`
- Test: `tests/test_signal_scorer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signal_scorer.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_signal_scorer.py -v`
Expected: FAIL — scorer 모듈 없음

- [ ] **Step 3: Write the implementation**

```python
# corvin_jarvis/signals/scorer.py
"""만기 신호 채점 — HIT / LATE_HIT / MISS / UNSCORABLE (스펙 §4.2).

score_row는 순수 함수 (가격 주입). run()이 fetch + 채점 + 원장 갱신.
LATE 유예: VELOCITY·방향성 신호는 만기×1.5까지 보류(None 반환) 후 최종 판정.
"""
from __future__ import annotations

from typing import Any, Callable

_EVENT_VOL_MULT = 1.3   # 이벤트일 변동 > 직전 평균 ×1.3 → HIT
_GRACE_MULT = 1.5       # late_hit 유예 배수

Verdict = tuple[str, dict[str, Any]]


def _ret_pct(closes: list[float]) -> float | None:
    if len(closes) < 2 or closes[0] <= 0:
        return None
    return (closes[-1] / closes[0] - 1) * 100.0


def score_row(row: dict[str, Any], *,
              closes_after: list[float],
              bench_after: list[float],
              bench_before: list[float] | None = None) -> Verdict | None:
    """단일 만기 신호 채점. None = late 유예 중 (다음 채점까지 open 유지).

    closes_after/bench_after: 발화일 이후 종가 oldest→newest.
    bench_before: EVENT 채점용 — 발화 직전 벤치마크 종가.
    """
    kind = str(row["kind"])
    horizon = int(row.get("horizon_days") or 10)
    age = int(row.get("age_days") or horizon)
    ev = row.get("evidence") or {}
    grace = horizon * _GRACE_MULT

    if kind == "VELOCITY":
        stop = ev.get("stop")
        if stop is None or not closes_after:
            return "unscorable", {"reason": "no stop or closes"}
        within = closes_after[:horizon]
        if any(c <= stop for c in within):
            return "hit", {"min_close": min(within), "stop": stop}
        late = closes_after[:int(grace) + 1]
        if any(c <= stop for c in late):
            return "late_hit", {"min_close": min(late), "stop": stop}
        if age < grace:
            return None  # 유예 중
        return "miss", {"min_close": min(closes_after), "stop": stop}

    if kind == "RS_WEAK":
        h_ret = _ret_pct(closes_after[:horizon])
        b_ret = _ret_pct(bench_after[:horizon])
        if h_ret is None or b_ret is None:
            return "unscorable", {"reason": "insufficient closes"}
        rs = h_ret - b_ret
        verdict = "hit" if rs < 0 else "miss"
        return verdict, {"rs_pct": round(rs, 2)}

    if kind == "EVENT":
        if not bench_after or not bench_before or len(bench_before) < 3:
            return "unscorable", {"reason": "no bench data"}
        prior_moves = [abs(bench_before[i] / bench_before[i - 1] - 1)
                       for i in range(1, len(bench_before))]
        avg_move = sum(prior_moves) / len(prior_moves)
        event_move = abs(bench_after[0] / bench_before[-1] - 1)
        verdict = "hit" if (avg_move > 0 and event_move > avg_move * _EVENT_VOL_MULT) else "miss"
        return verdict, {"event_move_pct": round(event_move * 100, 2),
                         "avg_move_pct": round(avg_move * 100, 2)}

    if kind in ("STOP", "WATCH"):
        ref = ev.get("price")
        if ref is None or not closes_after:
            return "unscorable", {"reason": "no ref price or closes"}
        low = min(closes_after[:horizon])
        verdict = "hit" if low < ref else "miss"
        return verdict, {"ref_price": ref, "min_close": low}

    direction = row.get("direction")
    if direction in ("bull", "bear"):
        series = closes_after if row.get("symbol") else bench_after
        r = _ret_pct(series[:horizon])
        if r is None:
            return "unscorable", {"reason": "insufficient closes"}
        moved = r > 0 if direction == "bull" else r < 0
        if moved:
            return "hit", {"ret_pct": round(r, 2)}
        if age < grace:
            return None  # 방향성 신호도 유예
        r_grace = _ret_pct(series[:int(grace) + 1])
        late = (r_grace or 0) > 0 if direction == "bull" else (r_grace or 0) < 0
        if late:
            return "late_hit", {"ret_pct": round(r_grace, 2)}
        return "miss", {"ret_pct": round(r, 2)}

    return "unscorable", {"reason": f"no scoring rule for kind={kind}"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_signal_scorer.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/scorer.py tests/test_signal_scorer.py
git commit -m "feat(signals): scorer 채점 규칙 — HIT/LATE_HIT/MISS/UNSCORABLE 순수 함수"
```

---

### Task 3: scorer 러너 — fetch + 채점 + 원장 갱신

**Files:**
- Modify: `corvin_jarvis/signals/scorer.py` (run 함수 + `__main__` 추가)
- Test: `tests/test_signal_scorer.py` (러너 테스트 추가)

- [ ] **Step 1: Write the failing tests** (test_signal_scorer.py에 append)

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_signal_scorer.py -k run -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'run'`

- [ ] **Step 3: Write the implementation** (scorer.py에 append)

```python
# scorer.py 하단에 추가

def _trading_days(age_days: int) -> int:
    """달력일 → 거래일 근사 (주 5일). ±1~2일 오차 허용 — 스펙 §4.2 합의."""
    return max(1, round(age_days * 5 / 7))


def _default_fetch(symbol: str, days: int) -> list[float]:
    from corvin_jarvis import quote_provider as qp
    return qp.get_stock_daily_closes(symbol, days=days, completed_only=True)


def run(db_path=None, now=None,
        fetch_closes: Callable[[str, int], list[float]] | None = None) -> dict[str, Any]:
    """만기 신호 일괄 채점. 반환: {"scored": n, "pending": n, "by_status": {...}}."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from corvin_jarvis.signals import ledger

    kst = ZoneInfo("Asia/Seoul")
    t = now or datetime.now(kst)
    fetch = fetch_closes or _default_fetch
    due = ledger.fetch_due(db_path=db_path, now=t)

    by_status: dict[str, int] = {}
    scored = pending = 0
    bench_cache: dict[str, list[float]] = {}

    def bench_for(symbol: str) -> str:
        return "069500" if symbol.endswith(".KS") or (symbol.isdigit() and len(symbol) == 6) else "SPY"
        # 069500 = KODEX200 ETF (KOSPI 프록시 — pykrx/KIS 모두 조회 가능)

    for row in due:
        sym = row["symbol"]
        age = row["age_days"]
        n_days = _trading_days(age)
        closes_after = []
        if sym:
            full = fetch(sym, n_days + 30)
            closes_after = full[-n_days:] if full else []
        bkey = bench_for(sym or "SPY")
        if bkey not in bench_cache:
            bench_cache[bkey] = fetch(bkey, n_days + 30) or []
        bench_full = bench_cache[bkey]
        bench_after = bench_full[-n_days:] if bench_full else []
        bench_before = bench_full[:-n_days][-21:] if len(bench_full) > n_days else []

        verdict = score_row(row, closes_after=closes_after,
                            bench_after=bench_after, bench_before=bench_before)
        if verdict is None:
            pending += 1
            continue
        status, outcome = verdict
        ledger.mark_scored(row["id"], status, outcome, db_path=db_path, now=t)
        by_status[status] = by_status.get(status, 0) + 1
        scored += 1

    return {"scored": scored, "pending": pending, "by_status": by_status}


if __name__ == "__main__":
    import json as _json
    res = run()
    from corvin_jarvis.signals import calibration
    calibration.write_state()
    print(_json.dumps(res, ensure_ascii=False))
```

주의: `__main__` 블록의 `calibration.write_state()`는 Task 4 완료 전까지 ImportError —
Task 3 커밋 시점에는 해당 2줄을 제외하고 커밋하고, Task 4에서 추가한다.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_signal_scorer.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/scorer.py tests/test_signal_scorer.py
git commit -m "feat(signals): scorer 러너 — 만기 신호 fetch·채점·원장 갱신"
```

---

### Task 4: `signals/calibration.py` — 적중률 → confidence 보정

**Files:**
- Create: `corvin_jarvis/signals/calibration.py`
- Modify: `corvin_jarvis/signals/scorer.py` (`__main__`에 write_state 2줄 추가)
- Test: `tests/test_signal_calibration.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signal_calibration.py
"""signals/calibration.py — 적중률 집계 + 베이지안 수축 보정."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from corvin_jarvis.signals import calibration, ledger

KST = ZoneInfo("Asia/Seoul")
NOW = datetime(2026, 6, 10, 16, 30, tzinfo=KST)


def _seed(db, n_hit, n_miss, n_late=0, engine="predictive", kind="VELOCITY"):
    """채점 완료된 신호 n건 시드."""
    for i in range(n_hit + n_miss + n_late):
        ledger.record_batch(engine, [{
            "symbol": f"SYM{i}", "kind": kind, "confidence": 65.0,
            "horizon_days": 5, "stop": 100.0,
        }], db_path=db, now=NOW)
    rows = []
    import sqlite3
    with sqlite3.connect(db) as conn:
        rows = [r[0] for r in conn.execute("SELECT id FROM signal_ledger").fetchall()]
    statuses = ["hit"] * n_hit + ["late_hit"] * n_late + ["miss"] * n_miss
    for rid, st in zip(rows, statuses):
        ledger.mark_scored(rid, st, {}, db_path=db, now=NOW)


def test_compute_hit_rate_with_late_half_weight(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=6, n_miss=2, n_late=2)  # (6 + 0.5*2)/10 = 0.70
    stats = calibration.compute(db_path=db)
    entry = stats["predictive"]["VELOCITY"]
    assert entry["n"] == 10
    assert abs(entry["hit_rate"] - 0.70) < 1e-9


def test_small_sample_below_10_not_calibrated(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=3, n_miss=2)  # n=5 < 10
    stats = calibration.compute(db_path=db)
    entry = stats["predictive"]["VELOCITY"]
    assert entry["calibrated_confidence"] is None  # 보정 보류


def test_calibrated_confidence_shrinks_toward_hit_rate(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=4, n_miss=6)  # hit_rate 0.40, n=10 → shrink 0.7
    stats = calibration.compute(db_path=db)
    entry = stats["predictive"]["VELOCITY"]
    # 65*0.7 + 40*0.3 = 57.5
    assert abs(entry["calibrated_confidence"] - 57.5) < 0.1


def test_unscorable_excluded_from_denominator(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=10, n_miss=0)
    ledger.record_batch("predictive", [{
        "symbol": "ZZZ", "kind": "VELOCITY", "confidence": 65.0,
        "horizon_days": 5,
    }], db_path=db, now=NOW)
    import sqlite3
    with sqlite3.connect(db) as conn:
        rid = conn.execute("SELECT MAX(id) FROM signal_ledger").fetchone()[0]
    ledger.mark_scored(rid, "unscorable", {}, db_path=db, now=NOW)
    stats = calibration.compute(db_path=db)
    assert stats["predictive"]["VELOCITY"]["n"] == 10  # unscorable 제외


def test_write_state_creates_json(tmp_path):
    db = tmp_path / "ledger.db"
    out = tmp_path / "calibration.json"
    _seed(db, n_hit=8, n_miss=2)
    calibration.write_state(db_path=db, out_path=out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "predictive" in data
    assert data["predictive"]["VELOCITY"]["n"] == 10


def test_scoreboard_rows_for_dashboard(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=7, n_miss=3)
    rows = calibration.scoreboard(db_path=db)
    assert rows == [{"engine": "predictive", "kind": "VELOCITY",
                     "n": 10, "hit_rate": 0.70, "open": 0,
                     "calibrated_confidence": rows[0]["calibrated_confidence"]}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_signal_calibration.py -v`
Expected: FAIL — calibration 모듈 없음

- [ ] **Step 3: Write the implementation**

```python
# corvin_jarvis/signals/calibration.py
"""적중률 집계 → confidence 보정 (스펙 §4.3).

hit_rate = (hit + 0.5*late_hit) / (hit+late_hit+miss). unscorable 분모 제외.
표본 n<10 → 보정 보류 (calibrated_confidence=None, 원본 유지).
베이지안 수축: calibrated = avg_conf*shrink + hit_rate*100*(1-shrink).
shrink = n=10→0.7에서 n≥50→0.2로 선형 감소 — 소표본 과보정 방지.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from corvin_jarvis.signals import ledger

STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "calibration.json"
_MIN_SAMPLES = 10


def _shrink(n: int) -> float:
    if n >= 50:
        return 0.2
    return 0.7 - 0.5 * (n - _MIN_SAMPLES) / 40  # n=10→0.7, n=50→0.2 선형


def compute(db_path: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """엔진×kind별 {n, hit_rate, avg_confidence, calibrated_confidence}."""
    p = ledger.init_db(db_path)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    with sqlite3.connect(p) as conn:
        rows = conn.execute(
            """SELECT engine, kind,
                      SUM(status='hit'), SUM(status='late_hit'), SUM(status='miss'),
                      AVG(confidence)
               FROM signal_ledger
               WHERE status IN ('hit','late_hit','miss')
               GROUP BY engine, kind""").fetchall()
    for engine, kind, hits, lates, misses, avg_conf in rows:
        n = (hits or 0) + (lates or 0) + (misses or 0)
        if n == 0:
            continue
        hit_rate = ((hits or 0) + 0.5 * (lates or 0)) / n
        calibrated = None
        if n >= _MIN_SAMPLES and avg_conf is not None:
            s = _shrink(n)
            calibrated = round(avg_conf * s + hit_rate * 100 * (1 - s), 1)
        out.setdefault(engine, {})[kind] = {
            "n": n, "hit_rate": round(hit_rate, 4),
            "avg_confidence": round(avg_conf, 1) if avg_conf is not None else None,
            "calibrated_confidence": calibrated,
        }
    return out


def write_state(db_path: Path | None = None, out_path: Path | None = None) -> Path:
    """compute 결과를 state/calibration.json에 저장 (Phase 3 주입용)."""
    target = out_path or STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(compute(db_path), ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def scoreboard(db_path: Path | None = None) -> list[dict[str, Any]]:
    """대시보드용 행 — open 신호 수 포함, n 내림차순."""
    p = ledger.init_db(db_path)
    stats = compute(db_path)
    with sqlite3.connect(p) as conn:
        open_counts = dict(conn.execute(
            "SELECT engine || '|' || kind, COUNT(*) FROM signal_ledger"
            " WHERE status='open' GROUP BY engine, kind").fetchall())
    rows = []
    for engine, kinds in stats.items():
        for kind, e in kinds.items():
            rows.append({"engine": engine, "kind": kind, "n": e["n"],
                         "hit_rate": round(e["hit_rate"], 2),
                         "calibrated_confidence": e["calibrated_confidence"],
                         "open": open_counts.get(f"{engine}|{kind}", 0)})
    return sorted(rows, key=lambda r: -r["n"])
```

그리고 scorer.py `__main__`에 Task 3에서 보류한 2줄 추가:

```python
if __name__ == "__main__":
    import json as _json
    res = run()
    from corvin_jarvis.signals import calibration
    calibration.write_state()
    print(_json.dumps(res, ensure_ascii=False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_signal_calibration.py tests/test_signal_scorer.py -v`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/calibration.py corvin_jarvis/signals/scorer.py tests/test_signal_calibration.py
git commit -m "feat(signals): calibration — 적중률 집계 + 베이지안 수축 confidence 보정"
```

---

### Task 5: snapshot.py 통합 — record 훅 + scoreboard 섹션

**Files:**
- Modify: `corvin_jarvis/dashboard/snapshot.py` (build_snapshot 부분)
- Test: `tests/test_dashboard_snapshot.py` (테스트 추가)

- [ ] **Step 1: Write the failing tests** (test_dashboard_snapshot.py에 append — 기존 fixture 패턴 확인 후 동일 스타일로)

```python
# tests/test_dashboard_snapshot.py에 추가

def test_snapshot_has_signal_scoreboard_section(monkeypatch):
    """signal_scoreboard 섹션 존재 — 실패해도 빈 리스트 (panel 격리)."""
    from corvin_jarvis.dashboard import snapshot as snap
    snap._CACHE["data"] = None  # 캐시 무효화
    s = snap.build_snapshot()
    assert "signal_scoreboard" in s
    assert isinstance(s["signal_scoreboard"], list)


def test_snapshot_records_signals_to_ledger(monkeypatch, tmp_path):
    """build_snapshot이 engine/predictive 신호를 원장에 기록."""
    from corvin_jarvis.dashboard import snapshot as snap
    from corvin_jarvis.signals import ledger as led

    recorded = []
    monkeypatch.setattr(led, "record_batch",
                        lambda engine, sigs, **kw: recorded.append((engine, len(sigs))) or 0)
    snap._CACHE["data"] = None
    snap.build_snapshot()
    engines = {e for e, _ in recorded}
    assert "signal_engine" in engines
    assert "predictive" in engines
```

(주의: 기존 test_dashboard_snapshot.py가 quote_provider를 어떻게 mock하는지 먼저 읽고,
동일한 conftest/fixture 패턴을 따를 것. 네트워크 호출이 발생하면 기존 패턴으로 monkeypatch.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard_snapshot.py -k "scoreboard or records" -v`
Expected: FAIL — KeyError 'signal_scoreboard' / record_batch 미호출

- [ ] **Step 3: Modify build_snapshot** (snapshot.py)

기존:

```python
        "engine_signals": _safe(lambda: _engine_signals(held), []),
        "predictive_signals": _safe(lambda: _predictive_signals(held), []),
```

변경 — build_snapshot 본문에서 신호를 지역변수로 계산 후 기록:

```python
def _record_to_ledger(engine_sigs: list[dict], pred_sigs: list[dict]) -> None:
    """Phase 1 원장 기록 — 실패해도 신호 흐름 무영향 (_safe로 호출)."""
    from corvin_jarvis.signals import ledger
    ledger.record_batch("signal_engine", engine_sigs)
    ledger.record_batch("predictive", pred_sigs)


def _scoreboard() -> list[dict]:
    from corvin_jarvis.signals import calibration
    return calibration.scoreboard()
```

build_snapshot 내부:

```python
    engine_sigs = _safe(lambda: _engine_signals(held), [])
    pred_sigs = _safe(lambda: _predictive_signals(held), [])
    _safe(lambda: _record_to_ledger(engine_sigs, pred_sigs), None)
    return {
        ...
        "engine_signals": engine_sigs,
        "predictive_signals": pred_sigs,
        "signal_scoreboard": _safe(_scoreboard, []),
        ...
    }
```

- [ ] **Step 4: Run tests — 신규 + 기존 회귀**

Run: `python3 -m pytest tests/test_dashboard_snapshot.py -v`
Expected: 전체 PASS (기존 테스트 포함 회귀 0)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/dashboard/snapshot.py tests/test_dashboard_snapshot.py
git commit -m "feat(dashboard): snapshot에 원장 기록 훅 + signal_scoreboard 섹션"
```

---

### Task 6: run_leading.py + jarvis.py record 훅

**Files:**
- Modify: `corvin_jarvis/run_leading.py` (collect 결과 기록)
- Modify: `corvin_jarvis/jarvis.py` (alert 기록)
- Test: `tests/test_signal_ledger.py` (horizon 매핑 테스트 추가)

- [ ] **Step 1: Write the failing test** (test_signal_ledger.py에 append)

```python
def test_leading_horizon_str_mapping(tmp_path):
    """LeadingSignal horizon 문자열 → 일수 매핑."""
    from corvin_jarvis.signals import ledger
    db = tmp_path / "ledger.db"
    n = ledger.record_batch("leading", [
        {"symbol": "NVDA", "kind": "RS_PILLAR", "direction": "bear",
         "confidence": 70.0, "horizon_days": ledger.horizon_str_to_days("days")},
    ], db_path=db, now=NOW)
    assert n == 1
    due = ledger.fetch_due(db_path=db, now=NOW + timedelta(days=5))
    assert due[0]["horizon_days"] == 5


def test_horizon_str_to_days_mapping():
    from corvin_jarvis.signals import ledger
    assert ledger.horizon_str_to_days("intraday") == 1
    assert ledger.horizon_str_to_days("days") == 5
    assert ledger.horizon_str_to_days("weeks") == 20
    assert ledger.horizon_str_to_days("unknown") == 10  # fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_signal_ledger.py -k horizon_str -v`
Expected: FAIL — horizon_str_to_days 없음

- [ ] **Step 3: Implement**

ledger.py에 추가:

```python
_HORIZON_STR = {"intraday": 1, "days": 5, "weeks": 20}


def horizon_str_to_days(horizon: str) -> int:
    """LeadingSignal.horizon 문자열 → 평가 일수."""
    return _HORIZON_STR.get(horizon, _FALLBACK_HORIZON)
```

run_leading.py — collect 직후 (기존 코드를 읽고 collect 반환 지점에 추가, `_safe` 동등 패턴 try/except):

```python
# collect() 호출 직후
try:
    from corvin_jarvis.signals import ledger
    ledger.record_batch("leading", [{
        "symbol": s.symbol, "kind": s.pillar, "direction": s.direction,
        "confidence": s.confidence,
        "horizon_days": ledger.horizon_str_to_days(s.horizon),
        "message": s.message, **s.evidence,
    } for s in signals])
except Exception as e:  # noqa: BLE001 — 원장 실패는 신호 흐름 무영향
    log.warning("ledger record failed: %s", e)
```

jarvis.py — alert 생성 직후 (compare 결과 alerts 루프 뒤, 동일 try/except 패턴):

```python
try:
    from corvin_jarvis.signals import ledger
    ledger.record_batch("jarvis", [{
        "symbol": a.get("symbol", ""), "kind": a.get("type", "ALERT"),
        "urgency": {"CRITICAL": 90, "HIGH": 70}.get(a.get("severity"), 40),
        "message": a.get("message", ""),
    } for a in alerts])
except Exception as e:  # noqa: BLE001
    log.warning("ledger record failed: %s", e)
```

(jarvis alert는 방향성 없음 → 만기 시 unscorable로 채점됨. 의도된 동작 —
Phase 2 arbiter가 jarvis alert 소비 시 방향 태깅 추가 예정.)

**주의**: run_leading.py와 jarvis.py의 실제 변수명(signals/alerts)과 로거 이름은
파일을 읽고 기존 구조에 맞출 것. 위 코드는 형태 기준.

- [ ] **Step 4: Run tests + 회귀**

Run: `python3 -m pytest tests/test_signal_ledger.py tests/test_leading*.py tests/test_jarvis*.py -v 2>/dev/null || python3 -m pytest tests/test_signal_ledger.py -v`
Expected: PASS (leading/jarvis 테스트 파일 존재 시 함께 회귀)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/ledger.py corvin_jarvis/run_leading.py corvin_jarvis/jarvis.py tests/test_signal_ledger.py
git commit -m "feat(signals): leading·jarvis 발화 신호 원장 기록 훅 (additive)"
```

---

### Task 7: run_scorer.sh + cron 등록

**Files:**
- Create: `corvin_jarvis/run_scorer.sh`

- [ ] **Step 1: Write the script** (run_digest.sh 패턴 그대로)

```bash
#!/usr/bin/env bash
# Corvin 신호 채점 — 매일 16:30 KST 장마감 후. 만기 신호 채점 + calibration.json 갱신.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
LOG="$SCRIPT_DIR/state/cron.log"
mkdir -p "$SCRIPT_DIR/state"
PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then PY=/opt/homebrew/bin/python3; fi
if [ -f "$SCRIPT_DIR/.env" ]; then set -a; . "$SCRIPT_DIR/.env"; set +a; fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scorer 시작" >> "$LOG"
"$PY" -m corvin_jarvis.signals.scorer >> "$LOG" 2>&1 || echo "[ERROR] scorer 실패" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scorer 완료" >> "$LOG"
```

- [ ] **Step 2: Verify executable + manual run**

```bash
chmod +x corvin_jarvis/run_scorer.sh
bash corvin_jarvis/run_scorer.sh
tail -3 corvin_jarvis/state/cron.log
```

Expected: `Scorer 시작` → JSON 출력 (`{"scored": 0, ...}` — 첫 실행은 만기 신호 0) → `Scorer 완료`

- [ ] **Step 3: cron 등록** (기존 crontab 보존 — additive)

```bash
(crontab -l; echo '# Corvin 신호 채점 — 매일 16:30 KST (Phase 1, 등록: 2026-06-10)'; echo '30 16 * * * /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_scorer.sh') | crontab -
crontab -l | grep scorer
```

Expected: scorer 라인 출력

- [ ] **Step 4: Commit**

```bash
git add corvin_jarvis/run_scorer.sh
git commit -m "feat(signals): run_scorer.sh — 일일 채점 cron (16:30 KST)"
```

---

### Task 8: command_center.html 적중률 미니패널

**Files:**
- Modify: `corvin_jarvis/dashboard/static/command_center.html`

- [ ] **Step 1: HTML 패널 추가** — `#log` 패널 (224행 부근 `<div class="panel glow">` ... `LOG`) 다음 형제로:

```html
  <div class="panel glow">
    <div class="ph"><span class="dot"></span><span class="t">SIGNAL SCOREBOARD</span>
      <span class="lbl" style="margin-left:auto" id="sbMeta">—</span></div>
    <table class="hm"><thead><tr>
      <th style="text-align:left">ENGINE·KIND</th><th>HIT%</th><th>N</th><th>OPEN</th>
    </tr></thead><tbody id="sbBody"></tbody></table>
    <div class="lbl" style="margin-top:4px">※ n<10 = 보정 보류 (회색)</div>
  </div>
```

(기존 패널들의 `ph`/`dot`/`t` 클래스 구조를 그대로 따른다 — 파일 내 다른 패널 헤더 마크업 확인 후 동일하게.)

- [ ] **Step 2: applyLive에 렌더 추가** — `applyLive(live)` 함수(498행 부근) 내부에:

```javascript
  // SIGNAL SCOREBOARD (Phase 1)
  if(live.signal_scoreboard){
    const sb=document.getElementById("sbBody");
    const rows=live.signal_scoreboard;
    document.getElementById("sbMeta").textContent=rows.length?`${rows.length} kinds`:"NO DATA";
    sb.innerHTML=rows.map(r=>{
      const pct=Math.round(r.hit_rate*100);
      const cls=r.n<10?"neu":(pct>=60?"up":"dn");
      return `<tr><td style="text-align:left">${r.engine}·${r.kind}</td>`+
             `<td class="${cls}">${pct}%</td><td>${r.n}</td><td>${r.open}</td></tr>`;
    }).join("");
  }
```

(`up`/`dn`/`neu` 클래스는 파일 내 기존 색상 클래스 — grep으로 실제 클래스명 확인 후 맞출 것.)

- [ ] **Step 3: Visual verify**

```bash
lsof -ti :8765 >/dev/null || (cd /Users/thethethe/Claude/quant_investment_system_v2 && python3 -m corvin_jarvis.dashboard &)
open "http://127.0.0.1:8765/command_center"
```

playwright로 스크린샷 캡처 → SCOREBOARD 패널 표시 확인 (데이터 없으면 "NO DATA").

- [ ] **Step 4: Commit**

```bash
git add corvin_jarvis/dashboard/static/command_center.html
git commit -m "feat(dashboard): 커맨드센터 SIGNAL SCOREBOARD 미니패널"
```

---

### Task 9: 전체 회귀 + E2E 검증 + 보고

- [ ] **Step 1: 전체 테스트 회귀**

Run: `python3 -m pytest tests/ -x -q`
Expected: 전체 PASS (기존 + 신규 ~23건)

- [ ] **Step 2: E2E — 실데이터 파이프라인 1회전**

```bash
# 1. snapshot 1회 → 원장에 현재 신호 기록 확인
python3 -c "
from corvin_jarvis.dashboard import snapshot
s = snapshot.build_snapshot()
print('scoreboard:', s['signal_scoreboard'])
import sqlite3
from corvin_jarvis.signals.ledger import DB_PATH
with sqlite3.connect(DB_PATH) as c:
    print('ledger rows:', c.execute('SELECT engine, symbol, kind, status FROM signal_ledger').fetchall())
"
# 2. scorer 수동 1회 (만기 신호 0이어도 정상 종료 확인)
bash corvin_jarvis/run_scorer.sh && tail -2 corvin_jarvis/state/cron.log
```

Expected: ledger rows에 open 신호 ≥1 (현재 VELOCITY/EVENT 등 활성), scorer 정상 종료

- [ ] **Step 3: 대시보드 스크린샷 → Discord 보고 (폐하 승인 게이트)**

스크린샷 + 원장 기록 수 + 테스트 결과를 Discord로 보고.
**폐하 승인 후에만** Phase 2 진행 (조건부 승인 게이트 규칙 — 자가통과 금지).

---

## Self-Review 결과

- **스펙 커버리지**: §4.1 원장(Task 1)·§4.2 채점(Task 2-3)·§4.3 보정(Task 4)·§4.4 연결점 3곳(Task 5-6)·§4.5 대시보드(Task 8)·cron(Task 7) — 전부 매핑됨.
- **타입 일관성**: record_batch(engine, signals, db_path, now) / fetch_due(db_path, now) / mark_scored(id, status, outcome, db_path, now) 시그니처 Task 1~6 전체 일치 확인.
- **알려진 근사**: 날짜 없는 closes 슬라이스(±1~2 거래일 오차) — 스펙 §4.2 합의 사항, 코드 주석에 명시.
- **순서 의존**: Task 3 `__main__`의 calibration 호출은 Task 4로 이연 — 각 커밋이 독립적으로 green.
