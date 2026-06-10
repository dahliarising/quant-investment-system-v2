# Corvin 통합 중재자 arbiter (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 5개 엔진(signal_engine·predictive·playbook·leading·jarvis)의 상충 신호를 종목별 단일 최종 액션으로 중재한다 — 안전 우선 → 적중률 가중 → 보수 기본값.

**Architecture:** 순수 중재 코어 `signals/arbiter.py`(규칙만, I/O 없음) + 입력 어댑터/CLI `signals/arbiter_inputs.py`(snapshot 헬퍼·ledger open 재사용 → `state/final_actions.json`). 소비자 3곳: snapshot 섹션(`final_actions`) → digest 블록 → 커맨드센터 패널. Phase 1 calibration.json의 hit_rate가 중재 가중치.

**Tech Stack:** Python 표준 라이브러리만. 신규 외부 의존성 0.

**Spec:** `docs/superpowers/specs/2026-06-10-corvin-signal-feedback-loop-design.md` §5

---

## File Structure

```
corvin_jarvis/signals/arbiter.py         # 신규 — FinalAction + arbitrate() 순수 규칙
corvin_jarvis/signals/arbiter_inputs.py  # 신규 — 엔진별 신호 normalize + collect + CLI(__main__)
corvin_jarvis/signals/ledger.py          # 수정 — fetch_open() 추가 (open 신호 전체 조회)
corvin_jarvis/dashboard/snapshot.py      # 수정 — final_actions 섹션
corvin_jarvis/notify.py                  # 수정 — digest에 중재 최종 액션 블록
corvin_jarvis/run_digest.sh              # 수정 — notify 전 arbiter CLI 1줄
corvin_jarvis/dashboard/static/command_center.html  # 수정 — FINAL ACTIONS 패널
tests/test_signal_arbiter.py             # 신규
tests/test_signal_arbiter_inputs.py      # 신규
tests/test_dashboard_snapshot.py         # 수정 — final_actions 섹션 테스트
tests/test_notify_digest_actions.py      # 신규 — digest 블록 테스트
```

**중재 어휘 (행동 지시어 — EW 선례 준수, 추상 라벨 금지):**

| action | 의미 | 아이콘 | 우선순위 |
|--------|------|--------|---------|
| 매도검토 | 방어 신호 발동 (STOP/hardstop 등) | 🛑 | 95 |
| 비중축소 | 익절/TRIM | ✂️ | 70 |
| 보류 | 강세·약세 상충 + 양쪽 모두 미검증 | ⏸️ | 60 |
| 매수후보 | 매수 신호 우세 | ➕ | 50 |
| 관찰 | 약세 경고만 존재 | 👀 | 40 |
| 홀딩 | 신호 없음/정상 | ✅ | 20 |

**normalize 규약 (모든 엔진 → 공통 dict):**
`{"engine": str, "symbol": str, "kind": str, "intent": "buy"|"defensive"|"trim"|"warn"|"hold", "urgency": int, "confidence": float|None, "note": str}`

| 엔진 | kind → intent 매핑 |
|------|--------------------|
| signal_engine | STOP→defensive · WATCH→warn · TRIM→trim · HOLD→hold |
| predictive | VELOCITY→warn(단 urgency≥70이면 defensive) · RS_WEAK→warn · EVENT→제외(symbol="" 매크로) |
| playbook | BUY_NOW→buy · TRIM_NOW→trim |
| leading (ledger open) | direction bull→buy · bear→warn |
| jarvis (ledger open) | stop_loss·hardstop→defensive · 그 외→제외(정보성, Phase 2 방향태깅 전) |

---

### Task 1: `signals/arbiter.py` — FinalAction + 중재 규칙 (순수)

**Files:**
- Create: `corvin_jarvis/signals/arbiter.py`
- Test: `tests/test_signal_arbiter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signal_arbiter.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_signal_arbiter.py -v`
Expected: FAIL — arbiter 모듈 없음

- [ ] **Step 3: Write the implementation**

```python
# corvin_jarvis/signals/arbiter.py
"""Phase 2 — 통합 중재자. 5엔진 상충 신호 → 종목별 단일 최종 액션 (스펙 §5).

중재 규칙 (우선순위 순):
  ① 안전 우선 — defensive(STOP/hardstop류)는 어떤 매수 신호보다 우선.
  ② 적중률 가중 — buy vs warn 상충 시 calibration hit_rate(n>=10) 우세 쪽.
     양쪽 모두 미검증이면 보수적 '보류'.
  ③ 그 외 — trim > buy > warn > hold.

순수 함수 · 부작용 없음. 입력은 arbiter_inputs가 normalize한 공통 dict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

_MIN_SAMPLES = 10          # calibration 신뢰 최소 표본 (calibration._MIN_SAMPLES와 동일)
_VELOCITY_DEFENSIVE_URGENCY = 70

_ACTION_PRIORITY = {"매도검토": 95, "비중축소": 70, "보류": 60,
                    "매수후보": 50, "관찰": 40, "홀딩": 20}


@dataclass(frozen=True)
class FinalAction:
    symbol: str
    action: str                 # 매도검토|비중축소|보류|매수후보|관찰|홀딩
    urgency: int                # _ACTION_PRIORITY 기반
    rationale: str              # 근거 + 상충 내역
    sources: list[str] = field(default_factory=list)   # 관여 엔진들
    conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hit_rate(calibration: dict | None, engine: str, kind: str) -> float | None:
    """n>=10 검증된 적중률만. 미달/부재 → None(미검증)."""
    if not calibration:
        return None
    entry = (calibration.get(engine) or {}).get(kind) or {}
    n = entry.get("n") or 0
    if n < _MIN_SAMPLES:
        return None
    return entry.get("hit_rate")


def _best_rate(group: list[dict], calibration: dict | None) -> float | None:
    rates = [r for r in (_hit_rate(calibration, s["engine"], s["kind"]) for s in group)
             if r is not None]
    return max(rates) if rates else None


def _mk(symbol: str, action: str, rationale: str,
        sources: list[str], conflict: bool) -> FinalAction:
    return FinalAction(symbol=symbol, action=action,
                       urgency=_ACTION_PRIORITY[action], rationale=rationale,
                       sources=sorted(set(sources)), conflict=conflict)


def _note_of(group: list[dict]) -> str:
    notes = [s.get("note", "") for s in group if s.get("note")]
    return notes[0] if notes else ""


def arbitrate(signals: list[dict[str, Any]],
              calibration: dict | None = None) -> list[FinalAction]:
    """normalize된 신호 → 종목당 단일 FinalAction. 우선순위(긴급도) 내림차순."""
    by_sym: dict[str, list[dict]] = {}
    for s in signals:
        sym = str(s.get("symbol", ""))
        if not sym:
            continue  # 매크로(EVENT 등)는 종목 중재 대상 아님
        by_sym.setdefault(sym, []).append(s)

    out: list[FinalAction] = []
    for sym, group in by_sym.items():
        engines = [s["engine"] for s in group]
        defensive = [s for s in group if s["intent"] == "defensive"
                     or (s["kind"] == "VELOCITY" and (s.get("urgency") or 0) >= _VELOCITY_DEFENSIVE_URGENCY)]
        buys = [s for s in group if s["intent"] == "buy"]
        trims = [s for s in group if s["intent"] == "trim"]
        warns = [s for s in group if s["intent"] == "warn" and s not in defensive]

        if defensive:
            why = _note_of(defensive) or "방어 신호 발동"
            conflict = bool(buys)
            if conflict:
                why += f" — 매수 신호({buys[0]['engine']}) 상충, 안전 우선"
            out.append(_mk(sym, "매도검토", why, engines, conflict))
            continue
        if trims:
            out.append(_mk(sym, "비중축소", _note_of(trims) or "익절/축소 신호", engines, False))
            continue
        if buys and warns:
            buy_rate = _best_rate(buys, calibration)
            warn_rate = _best_rate(warns, calibration)
            if buy_rate is not None and (warn_rate is None or buy_rate > warn_rate):
                why = (f"매수·약세 상충 — 적중률 우세({buys[0]['engine']} "
                       f"{buy_rate:.0%} vs {'미검증' if warn_rate is None else f'{warn_rate:.0%}'})")
                out.append(_mk(sym, "매수후보", why, engines, True))
            elif warn_rate is not None and (buy_rate is None or warn_rate >= buy_rate):
                why = (f"매수·약세 상충 — 적중률 우세({warns[0]['engine']} "
                       f"{warn_rate:.0%}) 약세 측")
                out.append(_mk(sym, "관찰", why, engines, True))
            else:
                out.append(_mk(sym, "보류",
                               "매수·약세 상충 — 양쪽 모두 적중률 미검증(n<10), 보수 유지",
                               engines, True))
            continue
        if buys:
            out.append(_mk(sym, "매수후보", _note_of(buys) or "매수 신호", engines, False))
            continue
        if warns:
            out.append(_mk(sym, "관찰", _note_of(warns) or "약세 경고", engines, False))
            continue
        out.append(_mk(sym, "홀딩", _note_of(group) or "신호 정상", engines, False))

    return sorted(out, key=lambda a: -a.urgency)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_signal_arbiter.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/arbiter.py tests/test_signal_arbiter.py
git commit -m "feat(signals): arbiter 중재 코어 — 안전우선·적중률가중·보수기본 (Phase 2)"
```

---

### Task 2: `ledger.fetch_open()` — open 신호 전체 조회

**Files:**
- Modify: `corvin_jarvis/signals/ledger.py` (함수 1개 추가)
- Test: `tests/test_signal_ledger.py` (테스트 추가)

- [ ] **Step 1: Append failing test** (파일에 이미 NOW·timedelta·ledger 임포트 존재)

```python
def test_fetch_open_returns_all_open_regardless_of_maturity(tmp_path):
    """fetch_open — 만기 무관 전체 open (arbiter 입력용)."""
    db = tmp_path / "ledger.db"
    ledger.record_batch("leading", [
        {"symbol": "NVDA", "kind": "ensemble", "direction": "bull",
         "confidence": 70.0, "horizon_days": 20},
    ], db_path=db, now=NOW)
    rows = ledger.fetch_open(db_path=db)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "NVDA"
    assert rows[0]["direction"] == "bull"
    assert rows[0]["evidence"] == {}
    # 채점되면 안 나옴
    ledger.mark_scored(rows[0]["id"], "hit", {}, db_path=db, now=NOW)
    assert ledger.fetch_open(db_path=db) == []
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_signal_ledger.py -k fetch_open -v` — FAIL (AttributeError)

- [ ] **Step 3: Add to ledger.py** (fetch_due 아래):

```python
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
```

- [ ] **Step 4: Run** `python3 -m pytest tests/test_signal_ledger.py -v` — 12 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/ledger.py tests/test_signal_ledger.py
git commit -m "feat(signals): ledger.fetch_open — arbiter 입력용 open 신호 조회"
```

---

### Task 3: `signals/arbiter_inputs.py` — normalize + collect + CLI

**Files:**
- Create: `corvin_jarvis/signals/arbiter_inputs.py`
- Test: `tests/test_signal_arbiter_inputs.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signal_arbiter_inputs.py
"""arbiter_inputs — 엔진별 신호 → 공통 normalize dict."""
from corvin_jarvis.signals import arbiter_inputs as ai


def test_normalize_engine_signals():
    rows = ai.normalize_engine_signals([
        {"symbol": "012450", "kind": "STOP", "urgency": 95, "reason": "손절선 이탈"},
        {"symbol": "NVDA", "kind": "WATCH", "urgency": 70, "reason": "근접"},
        {"symbol": "META", "kind": "TRIM", "urgency": 55, "reason": "익절"},
        {"symbol": "MSFT", "kind": "HOLD", "urgency": 20, "reason": "정상"},
        {"symbol": "GOOGL", "kind": "UNKNOWN", "urgency": 0, "reason": "가격없음"},
    ])
    by = {r["symbol"]: r for r in rows}
    assert by["012450"]["intent"] == "defensive"
    assert by["NVDA"]["intent"] == "warn"
    assert by["META"]["intent"] == "trim"
    assert by["MSFT"]["intent"] == "hold"
    assert "GOOGL" not in by  # UNKNOWN 제외
    assert all(r["engine"] == "signal_engine" for r in rows)


def test_normalize_predictive_signals():
    rows = ai.normalize_predictive_signals([
        {"symbol": "012450", "kind": "VELOCITY", "urgency": 82, "message": "D-1 도달"},
        {"symbol": "BWXT", "kind": "RS_WEAK", "urgency": 45, "message": "상대약세"},
        {"symbol": "", "kind": "EVENT", "urgency": 80, "message": "FOMC D-1"},
    ])
    by = {r["symbol"]: r for r in rows}
    assert by["012450"]["intent"] == "warn"  # defensive 승격은 arbiter 규칙(urgency>=70)이 담당
    assert by["BWXT"]["intent"] == "warn"
    assert "" not in by  # 매크로 제외
    assert all(r["engine"] == "predictive" for r in rows)


def test_normalize_playbook_signals():
    rows = ai.normalize_playbook_signals([
        {"sym": "GOOGL", "zone": "딥밸류", "stance": "ENTER", "color": "green"},
        {"sym": "META", "zone": "고점", "stance": "HARVEST", "color": "amber"},
    ])
    by = {r["symbol"]: r for r in rows}
    assert by["GOOGL"]["intent"] == "buy" and by["GOOGL"]["kind"] == "BUY_NOW"
    assert by["META"]["intent"] == "trim" and by["META"]["kind"] == "TRIM_NOW"
    assert all(r["engine"] == "playbook" for r in rows)


def test_normalize_ledger_open_rows():
    rows = ai.normalize_ledger_open([
        {"engine": "leading", "symbol": "NVDA", "kind": "ensemble",
         "direction": "bull", "urgency": None, "confidence": 70.0,
         "evidence": {"message": "추세+RS"}},
        {"engine": "leading", "symbol": "012450", "kind": "ensemble",
         "direction": "bear", "urgency": None, "confidence": 40.0, "evidence": {}},
        {"engine": "jarvis", "symbol": "TSLA", "kind": "stop_loss",
         "direction": None, "urgency": 90, "confidence": None,
         "evidence": {"message": "STOP LOSS 도달"}},
        {"engine": "jarvis", "symbol": "", "kind": "kospi",
         "direction": None, "urgency": 90, "confidence": None, "evidence": {}},
        {"engine": "jarvis", "symbol": "NVDA", "kind": "rs_confirmed",
         "direction": None, "urgency": 55, "confidence": None, "evidence": {}},
        {"engine": "signal_engine", "symbol": "012450", "kind": "WATCH",
         "direction": None, "urgency": 70, "confidence": None, "evidence": {}},
    ])
    by = {(r["engine"], r["symbol"]): r for r in rows}
    assert by[("leading", "NVDA")]["intent"] == "buy"
    assert by[("leading", "012450")]["intent"] == "warn"
    assert by[("jarvis", "TSLA")]["intent"] == "defensive"
    assert ("jarvis", "") not in by                  # 매크로 제외
    assert ("jarvis", "NVDA") not in by              # 방향성 없는 정보성 제외
    assert ("signal_engine", "012450") not in by     # 라이브 엔진과 중복 — ledger의 engine/predictive/playbook행 제외


def test_build_final_actions_writes_state(tmp_path, monkeypatch):
    """collect→arbitrate→state 파일 쓰기 E2E (입력 전부 주입)."""
    out_path = tmp_path / "final_actions.json"
    result = ai.build_final_actions(
        engine_sigs=[{"symbol": "012450", "kind": "STOP", "urgency": 95, "reason": "이탈"}],
        pred_sigs=[],
        playbook_sigs=[{"sym": "012450", "zone": "딥밸류", "stance": "ENTER", "color": "green"}],
        ledger_open=[],
        calibration={},
        out_path=out_path,
    )
    assert result["actions"][0]["symbol"] == "012450"
    assert result["actions"][0]["action"] == "매도검토"
    assert result["actions"][0]["conflict"] is True
    import json
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["actions"] == result["actions"]
    assert "ts" in saved
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_signal_arbiter_inputs.py -v` — FAIL (모듈 없음)

- [ ] **Step 3: Write the implementation**

```python
# corvin_jarvis/signals/arbiter_inputs.py
"""arbiter 입력 어댑터 — 엔진별 신호를 공통 normalize dict로 + CLI.

CLI(`python3 -m corvin_jarvis.signals.arbiter_inputs`)는 라이브 신호를 수집해
arbitrate 후 state/final_actions.json에 기록 (digest가 읽음).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from corvin_jarvis.signals import arbiter

KST = ZoneInfo("Asia/Seoul")
STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "final_actions.json"

_ENGINE_INTENT = {"STOP": "defensive", "WATCH": "warn", "TRIM": "trim", "HOLD": "hold"}
_JARVIS_DEFENSIVE_KINDS = {"stop_loss", "hardstop"}
# ledger open에서 중재에 쓰는 엔진 — 라이브 평가가 없는 엔진만 (이중 계상 방지)
_LEDGER_ENGINES = {"leading", "jarvis"}


def _row(engine: str, symbol: str, kind: str, intent: str,
         urgency: int, confidence: float | None, note: str) -> dict[str, Any]:
    return {"engine": engine, "symbol": symbol, "kind": kind, "intent": intent,
            "urgency": urgency, "confidence": confidence, "note": note}


def normalize_engine_signals(sigs: list[dict]) -> list[dict]:
    """signal_engine EngineSignal.to_dict() 리스트 → 공통 dict. UNKNOWN 제외."""
    out = []
    for s in sigs:
        intent = _ENGINE_INTENT.get(str(s.get("kind", "")))
        if intent is None:
            continue
        out.append(_row("signal_engine", str(s.get("symbol", "")), s["kind"], intent,
                        int(s.get("urgency") or 0), None, str(s.get("reason", ""))))
    return out


def normalize_predictive_signals(sigs: list[dict]) -> list[dict]:
    """predictive PredictiveSignal.to_dict() → 공통 dict. EVENT(매크로) 제외."""
    out = []
    for s in sigs:
        sym = str(s.get("symbol", ""))
        if not sym:
            continue
        out.append(_row("predictive", sym, str(s.get("kind", "")), "warn",
                        int(s.get("urgency") or 0), s.get("confidence"),
                        str(s.get("message", ""))))
    return out


def normalize_playbook_signals(sigs: list[dict]) -> list[dict]:
    """snapshot._build_signals 출력({sym,zone,stance,color}) → 공통 dict.

    color green=BUY_NOW, amber=TRIM_NOW (snapshot._build_signals의 매핑 역변환).
    """
    out = []
    for s in sigs:
        buy = s.get("color") == "green"
        out.append(_row("playbook", str(s.get("sym", "")),
                        "BUY_NOW" if buy else "TRIM_NOW",
                        "buy" if buy else "trim",
                        50 if buy else 55, None, str(s.get("zone", ""))))
    return out


def normalize_ledger_open(rows: list[dict]) -> list[dict]:
    """ledger open 행 → 공통 dict. leading/jarvis만 (라이브 엔진 이중 계상 방지).

    leading: direction bull→buy / bear→warn.
    jarvis: stop_loss·hardstop→defensive, 그 외 정보성 제외 (방향 없음).
    매크로(symbol="") 제외.
    """
    out = []
    for r in rows:
        engine = str(r.get("engine", ""))
        sym = str(r.get("symbol", ""))
        if engine not in _LEDGER_ENGINES or not sym:
            continue
        kind = str(r.get("kind", ""))
        note = str((r.get("evidence") or {}).get("message", ""))
        conf = r.get("confidence")
        if engine == "leading":
            d = r.get("direction")
            if d == "bull":
                out.append(_row(engine, sym, kind, "buy",
                                int(r.get("urgency") or 50), conf, note))
            elif d == "bear":
                out.append(_row(engine, sym, kind, "warn",
                                int(r.get("urgency") or 50), conf, note))
            continue
        if kind in _JARVIS_DEFENSIVE_KINDS:
            out.append(_row(engine, sym, kind, "defensive",
                            int(r.get("urgency") or 90), conf, note))
    return out


def build_final_actions(*, engine_sigs: list[dict], pred_sigs: list[dict],
                        playbook_sigs: list[dict], ledger_open: list[dict],
                        calibration: dict | None,
                        out_path: Path | None = None,
                        now: datetime | None = None) -> dict[str, Any]:
    """normalize → arbitrate → {ts, actions} 반환 + state 파일 기록."""
    signals = (normalize_engine_signals(engine_sigs)
               + normalize_predictive_signals(pred_sigs)
               + normalize_playbook_signals(playbook_sigs)
               + normalize_ledger_open(ledger_open))
    actions = [a.to_dict() for a in arbiter.arbitrate(signals, calibration=calibration)]
    t = now or datetime.now(KST)
    if t.tzinfo is None:
        t = t.replace(tzinfo=KST)
    result = {"ts": t.isoformat(timespec="seconds"), "actions": actions}
    target = out_path or STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def collect_live() -> dict[str, Any]:
    """라이브 수집 (snapshot 헬퍼 재사용) → build_final_actions. CLI 전용."""
    from corvin_jarvis.dashboard import snapshot as snap
    from corvin_jarvis.playbook import builder
    from corvin_jarvis.signals import calibration as cal_mod
    from corvin_jarvis.signals import ledger

    pf = snap._load_portfolio()
    positions = snap._positions(pf)
    held = snap._held_for_engine(pf, positions)
    engine_sigs = snap._engine_signals(held)
    pred_sigs = snap._predictive_signals(held)
    holdings = builder.load_holdings()
    playbook_sigs = snap._build_signals(holdings)
    ledger_open = ledger.fetch_open()
    calibration = cal_mod.compute()
    return build_final_actions(engine_sigs=engine_sigs, pred_sigs=pred_sigs,
                               playbook_sigs=playbook_sigs, ledger_open=ledger_open,
                               calibration=calibration)


if __name__ == "__main__":
    res = collect_live()
    print(json.dumps({"actions": len(res["actions"]), "ts": res["ts"]}, ensure_ascii=False))
```

- [ ] **Step 4: Run** `python3 -m pytest tests/test_signal_arbiter_inputs.py tests/test_signal_arbiter.py -v` — 18 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/arbiter_inputs.py tests/test_signal_arbiter_inputs.py
git commit -m "feat(signals): arbiter 입력 어댑터 — 5엔진 normalize + state CLI"
```

---

### Task 4: snapshot에 `final_actions` 섹션

**Files:**
- Modify: `corvin_jarvis/dashboard/snapshot.py`
- Test: `tests/test_dashboard_snapshot.py` (append)

- [ ] **Step 1: Append failing test** (기존 autouse `_isolate_signal_ledger` 픽스처와 `_mock_quotes` 헬퍼 활용 — 파일 패턴 따를 것)

```python
def test_snapshot_has_final_actions_section(monkeypatch):
    """final_actions 섹션 — arbiter 중재 결과. 실패해도 빈 리스트."""
    from corvin_jarvis.dashboard import snapshot as snap
    snap._CACHE["data"] = None
    s = snap.build_snapshot()
    assert "final_actions" in s
    assert isinstance(s["final_actions"], list)
```

(기존 record_batch no-op·quote mock 패턴을 이 테스트에도 동일 적용.)

- [ ] **Step 2: Run** — FAIL (KeyError)

- [ ] **Step 3: Modify snapshot.py.** `_scoreboard()` 아래 헬퍼 추가:

```python
def _final_actions(engine_sigs: list[dict], pred_sigs: list[dict],
                   playbook_sigs: list[dict]) -> list[dict]:
    """Phase 2 중재 — 라이브 신호 + ledger open(leading/jarvis) → 종목별 최종 액션."""
    from corvin_jarvis.signals import arbiter_inputs as ai
    from corvin_jarvis.signals import calibration as cal_mod
    from corvin_jarvis.signals import ledger
    res = ai.build_final_actions(
        engine_sigs=engine_sigs, pred_sigs=pred_sigs, playbook_sigs=playbook_sigs,
        ledger_open=ledger.fetch_open(), calibration=cal_mod.compute())
    return res["actions"]
```

build_snapshot 내부 — playbook 신호도 지역변수로 승격 후 섹션 추가:

```python
    playbook_sigs = _safe(lambda: _build_signals(holdings), [])
    ...
        "signals": playbook_sigs,
        ...
        "final_actions": _safe(lambda: _final_actions(engine_sigs, pred_sigs, playbook_sigs), []),
```

(주의: `_final_actions`는 `build_final_actions`를 부르므로 state/final_actions.json도 갱신된다 —
의도된 동작. 테스트에서는 ledger DB가 tmp로 격리되어 있으므로 state 파일 쓰기를
막으려면 테스트에서 `ai.STATE_PATH`를 tmp로 monkeypatch하는 autouse 픽스처를
`_isolate_signal_ledger`에 1줄 추가: `monkeypatch.setattr(ai_mod, "STATE_PATH", tmp_path / "fa.json")`.)

- [ ] **Step 4: Run** `python3 -m pytest tests/test_dashboard_snapshot.py -v` — 전체 PASS (기존 11 + 신규 1)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/dashboard/snapshot.py tests/test_dashboard_snapshot.py
git commit -m "feat(dashboard): snapshot final_actions 섹션 — arbiter 중재 결과"
```

---

### Task 5: digest 통합 — 중재 최종 액션 블록

**Files:**
- Modify: `corvin_jarvis/notify.py`
- Modify: `corvin_jarvis/run_digest.sh`
- Test: `tests/test_notify_digest_actions.py` (신규)

- [ ] **Step 1: Write the failing tests** (notify.py의 기존 테스트 파일들 패턴 확인 — conftest의 실송 차단 안전망 위에서 _format 함수만 단위 테스트)

```python
# tests/test_notify_digest_actions.py
"""notify digest — 중재 최종 액션 블록 (state/final_actions.json 기반)."""
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from corvin_jarvis import notify

KST = ZoneInfo("Asia/Seoul")


def _write_state(tmp_path, ts, actions):
    p = tmp_path / "final_actions.json"
    p.write_text(json.dumps({"ts": ts.isoformat(timespec="seconds"),
                             "actions": actions}, ensure_ascii=False), encoding="utf-8")
    return p


def test_actions_block_renders_with_icons(tmp_path):
    p = _write_state(tmp_path, datetime.now(KST), [
        {"symbol": "012450", "action": "매도검토", "urgency": 95,
         "rationale": "손절선 이탈 — 매수 신호(playbook) 상충, 안전 우선",
         "sources": ["playbook", "signal_engine"], "conflict": True},
        {"symbol": "GOOGL", "action": "관찰", "urgency": 40,
         "rationale": "RS 약세", "sources": ["predictive"], "conflict": False},
    ])
    block = notify._final_actions_block(state_path=p)
    assert "🛑" in block and "012450" in block and "매도검토" in block
    assert "👀" in block and "GOOGL" in block
    assert "⚔️" in block  # conflict 표시


def test_actions_block_empty_when_stale(tmp_path):
    p = _write_state(tmp_path, datetime.now(KST) - timedelta(hours=25), [
        {"symbol": "X", "action": "홀딩", "urgency": 20, "rationale": "", "sources": [], "conflict": False},
    ])
    assert notify._final_actions_block(state_path=p) == ""


def test_actions_block_empty_when_missing(tmp_path):
    assert notify._final_actions_block(state_path=tmp_path / "nope.json") == ""


def test_actions_block_skips_hold_only(tmp_path):
    """홀딩만 있으면 노이즈 — 블록 생략 (alert noise aversion)."""
    p = _write_state(tmp_path, datetime.now(KST), [
        {"symbol": "MSFT", "action": "홀딩", "urgency": 20, "rationale": "정상",
         "sources": ["signal_engine"], "conflict": False},
    ])
    assert notify._final_actions_block(state_path=p) == ""
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_notify_digest_actions.py -v` — FAIL (AttributeError)

- [ ] **Step 3: Implement.** notify.py에 추가 (모듈 상수부에 아이콘 맵, `_format_message` 위쪽):

```python
_ACTION_ICONS = {"매도검토": "🛑", "비중축소": "✂️", "보류": "⏸️",
                 "매수후보": "➕", "관찰": "👀", "홀딩": "✅"}
_FINAL_ACTIONS_PATH = Path(__file__).resolve().parent / "state" / "final_actions.json"
_FINAL_ACTIONS_MAX_AGE_H = 24


def _final_actions_block(state_path: Path | None = None) -> str:
    """digest용 중재 최종 액션 블록. 낡았거나(>24h) 없거나 홀딩뿐이면 ''."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    p = state_path or _FINAL_ACTIONS_PATH
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(data["ts"])
    except (OSError, ValueError, KeyError):
        return ""
    kst = ZoneInfo("Asia/Seoul")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=kst)
    if datetime.now(kst) - ts > timedelta(hours=_FINAL_ACTIONS_MAX_AGE_H):
        return ""
    rows = [a for a in data.get("actions", []) if a.get("action") != "홀딩"]
    if not rows:
        return ""
    lines = ["", "🎯 중재 최종 액션 (5엔진 통합)"]
    for a in rows:
        icon = _ACTION_ICONS.get(a.get("action", ""), "·")
        flag = " ⚔️" if a.get("conflict") else ""
        lines.append(f"{icon} {a.get('symbol')} {a.get('action')}{flag} — {a.get('rationale', '')}")
    return "\n".join(lines)
```

`notify()`의 digest 경로에서 메시지 조립 후 블록 append — `_format_message(...)` 호출 결과를 받는 지점을 찾아(READ the file) digest 모드일 때:

```python
    if mode == "digest":
        actions_block = _final_actions_block()
        if actions_block:
            content += "\n" + actions_block
```

run_digest.sh — jarvis.py 실행 다음, notify digest 전에 1줄 추가:

```bash
"$PY" -m corvin_jarvis.signals.arbiter_inputs >> "$LOG" 2>&1 || echo "[ERROR] arbiter 실패" >> "$LOG"
```

- [ ] **Step 4: Run** `python3 -m pytest tests/test_notify_digest_actions.py tests/test_channels.py -v` 및 notify 관련 기존 테스트 — 전체 PASS. `bash -n corvin_jarvis/run_digest.sh` 문법 OK.

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/notify.py corvin_jarvis/run_digest.sh tests/test_notify_digest_actions.py
git commit -m "feat(notify): digest에 중재 최종 액션 블록 + run_digest arbiter 선실행"
```

---

### Task 6: 커맨드센터 FINAL ACTIONS 패널

**Files:**
- Modify: `corvin_jarvis/dashboard/static/command_center.html`

- [ ] **Step 1: HTML 패널** — SIGNAL SCOREBOARD 패널 바로 다음 형제로 (같은 마크업 컨벤션: `.ph`/`.lbl`, 일반 `table`, `td.sym`):

```html
  <div class="panel glow">
    <div class="ph"><span class="dot"></span><span class="lbl">Final Actions · 중재</span>
      <span class="lbl" style="margin-left:auto" id="faMeta">—</span></div>
    <table><thead><tr><th>SYM</th><th>ACTION</th><th>RATIONALE</th></tr></thead>
      <tbody id="faBody"></tbody></table>
  </div>
```

- [ ] **Step 2: applyLive 렌더러** — scoreboard 렌더 블록 다음에 (esc 헬퍼는 scoreboard 블록의 것을 함수 밖으로 끌어올려 공유하거나 동일 정의 재사용):

```javascript
  // FINAL ACTIONS (Phase 2)
  if(live.final_actions){
    const icons={"매도검토":"🛑","비중축소":"✂️","보류":"⏸️","매수후보":"➕","관찰":"👀","홀딩":"✅"};
    const fa=live.final_actions.filter(a=>a.action!=="홀딩");
    $("#faMeta").textContent=fa.length?`${fa.length} actions`:"ALL HOLD";
    $("#faBody").innerHTML=(fa.slice(0,8).map(a=>{
      const cls=a.action==="매수후보"?"up":(a.action==="매도검토"?"down":"neu");
      return `<tr><td class="sym">${esc(a.symbol)}</td>`+
        `<td class="${cls}">${icons[a.action]||""} ${esc(a.action)}${a.conflict?" ⚔️":""}</td>`+
        `<td class="neu">${esc((a.rationale||"").slice(0,40))}</td></tr>`;
    }).join(""))||`<tr><td colspan="3" class="neu">중재 데이터 없음</td></tr>`;
  }
```

(esc를 두 블록이 공유하도록 applyLive 상단으로 이동해도 됨 — 파일 스타일 유지.)

- [ ] **Step 3: Verify**

```bash
python3 -c "html=open('corvin_jarvis/dashboard/static/command_center.html').read(); assert 'faBody' in html and 'Final Actions' in html; print('HTML OK')"
python3 -m pytest tests/test_dashboard_server.py -q
```

- [ ] **Step 4: Commit**

```bash
git add corvin_jarvis/dashboard/static/command_center.html
git commit -m "feat(dashboard): 커맨드센터 FINAL ACTIONS 중재 패널"
```

---

### Task 7: 전체 회귀 + E2E + 보고

- [ ] **Step 1:** `python3 -m pytest tests/ -q` — 전체 PASS (예상 660+)

- [ ] **Step 2: E2E 실데이터 1회전**

```bash
# arbiter CLI 실행 → state 파일 + 액션 수 확인
python3 -m corvin_jarvis.signals.arbiter_inputs
python3 -c "import json; d=json.load(open('corvin_jarvis/state/final_actions.json')); [print(a['action'], a['symbol'], '⚔️' if a['conflict'] else '', '-', a['rationale'][:60]) for a in d['actions']]"
# digest 블록 미리보기 (실송 없음)
python3 -c "from corvin_jarvis import notify; print(notify._final_actions_block() or '(빈 블록)')"
```

기대: 012450 매도검토(STOP+VELOCITY), 상충 시 ⚔️ 표시 등 실데이터 중재 결과.

- [ ] **Step 3:** 대시보드 서버 재시작 + 패널 스크린샷 → Discord 보고 (폐하 승인 게이트). main 머지는 승인 후.

---

## Self-Review 결과

- **스펙 §5 커버리지**: 중재 규칙 3단(T1) · 종목당 단일 FinalAction+rationale 상충명시(T1) · ledger open 입력(T2-3) · 소비자 snapshot(T4)→digest(T5) 전환 + 대시보드(T6) — 전부 매핑. debate 소비 전환은 스펙상 "단계적" — Phase 2 범위에서 제외(YAGNI), Phase 3+에서.
- **이중 계상 방지**: ledger open에는 snapshot이 막 기록한 engine/predictive 행도 있음 → normalize_ledger_open이 leading/jarvis만 통과시켜 라이브 평가와 중복 계상 차단 (테스트 `test_normalize_ledger_open_rows`가 고정).
- **타입 일관성**: FinalAction.to_dict 키(symbol/action/urgency/rationale/sources/conflict)가 T4 snapshot 섹션·T5 digest 블록·T6 JS 렌더러와 일치 확인.
- **노이즈 통제**: digest 블록은 홀딩 생략·24h 신선도 게이트, 패널은 홀딩 필터+top8 (alert noise aversion 준수).
