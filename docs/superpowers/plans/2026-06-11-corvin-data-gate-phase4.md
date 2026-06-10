# Phase 4 — 데이터 검증 게이트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 신호가 ledger에 기록·발화되기 전 자동 데이터 검증 훅 — 가격 교차검증·시점 라벨 교정·staleness 차단·NaN/inf drop (스펙 §7).

**Architecture:** `data_verify.py` 패턴 그대로 — 순수 판정(`gate_signals`)과 라이브 다소스 바인딩(`collect_price_checks`)을 분리. 게이트는 `snapshot._record_to_ledger` 직전 훅으로 통합되며, 실패해도 신호 흐름이 깨지지 않게 기존 `_safe` 격리를 유지한다. 모든 IO는 DI fetcher로 주입해 순수 함수 테스트.

**Tech Stack:** Python 3.9 호환(기존 코드베이스), pytest, sqlite ledger(기존), data_verify.verified 재사용.

**근거 스펙:** `docs/superpowers/specs/2026-06-10-corvin-signal-feedback-loop-design.md` §7

| 체크 | 규칙 | 실패 시 |
|------|------|---------|
| 가격 교차 | KIS vs yahoo ±1% 초과 불일치 (KR은 KIS vs pykrx) | 신호 보류 + ⚠️ 경고 |
| 시점 라벨 | 장마감 상태면 '전일종가' 라벨 강제 | 라벨 자동 교정 |
| staleness | 데이터 한도 초과 경과 | 신호 차단 + 갱신 요청 |
| NaN/inf | 비정상 수치 | 해당 신호만 drop |

**설계 결정 (staleness 임계):** 모듈 기본값은 스펙 §7의 3일. 단, snapshot 통합부는 portfolio.json 기존 정책(FRESH≤3 진행 / WARN 4-6 진행 / STALE≥7 차단 — CLAUDE.md Portfolio 정합성 규칙)과 충돌하지 않도록 `staleness.check().block_strategy` (≥7일)일 때만 차단한다. WARN 구간(4-6일)에서 신호를 전부 차단하면 기존 정책보다 과격해져 대시보드가 비어버림 — 기존 정책 우선.

---

### Task 1: `signals/data_gate.py` 순수 게이트 코어

**Files:**
- Create: `corvin_jarvis/signals/data_gate.py`
- Test: `tests/test_data_gate.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
"""Phase 4 — data_gate 순수 게이트 코어 테스트."""
import pytest

from corvin_jarvis.signals import data_gate as dg


def _sig(symbol="TSLA", kind="STOP", **kw):
    base = {"symbol": symbol, "kind": kind, "urgency": 70, "confidence": 65.0,
            "message": "현재가 $310 — 손절 임박"}
    base.update(kw)
    return base


@pytest.mark.unit
def test_gate_passes_clean_signals_unchanged():
    """검증 통과 — 신호 내용 그대로 (장중이므로 라벨 미교정)."""
    sigs = [_sig()]
    res = dg.gate_signals(sigs, market_open=True)
    assert res["passed"] == sigs
    assert res["blocked"] == []
    assert res["warnings"] == []


@pytest.mark.unit
def test_gate_drops_nan_inf_signals():
    """NaN/inf 수치 신호만 drop — 나머지는 통과 (스펙 §7 row 4)."""
    bad_nan = _sig(symbol="AAA", confidence=float("nan"))
    bad_inf = _sig(symbol="BBB", urgency=float("inf"))
    good = _sig(symbol="CCC")
    res = dg.gate_signals([bad_nan, bad_inf, good], market_open=True)
    assert [s["symbol"] for s in res["passed"]] == ["CCC"]
    assert {b["reason"] for b in res["blocked"]} == {"invalid_numeric"}


@pytest.mark.unit
def test_gate_holds_signal_on_price_discrepancy():
    """가격 소스 불일치 → 해당 종목 신호 보류 + 경고 (스펙 §7 row 1)."""
    checks = {"TSLA": {"value": 310.0, "flag": "discrepancy", "spread_pct": 2.4},
              "NVDA": {"value": 180.0, "flag": None, "spread_pct": 0.1}}
    res = dg.gate_signals([_sig("TSLA"), _sig("NVDA")],
                          price_checks=checks, market_open=True)
    assert [s["symbol"] for s in res["passed"]] == ["NVDA"]
    assert res["blocked"][0]["reason"] == "price_discrepancy"
    assert any("TSLA" in w for w in res["warnings"])


@pytest.mark.unit
def test_gate_corrects_price_label_when_market_closed():
    """장마감 — '현재가' → '전일종가' 자동 교정, 원본 dict 비변이 (스펙 §7 row 2)."""
    orig = _sig()
    res = dg.gate_signals([orig], market_open=False)
    assert res["passed"][0]["message"] == "전일종가 $310 — 손절 임박"
    assert orig["message"] == "현재가 $310 — 손절 임박"  # 불변성


@pytest.mark.unit
def test_gate_keeps_label_when_market_open():
    res = dg.gate_signals([_sig()], market_open=True)
    assert "현재가" in res["passed"][0]["message"]


@pytest.mark.unit
def test_gate_blocks_all_on_stale_data():
    """데이터 한도 초과 — 전체 차단 + 갱신 요청 경고 (스펙 §7 row 3)."""
    res = dg.gate_signals([_sig("TSLA"), _sig("NVDA")],
                          market_open=True, data_age_days=8)
    assert res["passed"] == []
    assert all(b["reason"] == "stale_data" for b in res["blocked"])
    assert any("갱신" in w for w in res["warnings"])


@pytest.mark.unit
def test_gate_allows_within_stale_limit():
    """기본 한도(3일) 이내 — 통과."""
    res = dg.gate_signals([_sig()], market_open=True, data_age_days=2)
    assert len(res["passed"]) == 1


@pytest.mark.unit
def test_gate_custom_stale_limit():
    """max_age_days 커스텀 — 통합부의 7일 정책 지원."""
    res = dg.gate_signals([_sig()], market_open=True,
                          data_age_days=5, max_age_days=6)
    assert len(res["passed"]) == 1
    res2 = dg.gate_signals([_sig()], market_open=True,
                           data_age_days=7, max_age_days=6)
    assert res2["passed"] == []
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_data_gate.py -v` — 전부 FAIL (`No module named ...data_gate`)

- [ ] **Step 3: Write the implementation**

```python
"""Phase 4 — 데이터 검증 게이트. 신호 발화·기록 전 자동 검증 (스펙 §7).

ledger.record 직전 훅: 통과 신호만 기록·발화. 순수 판정(gate_signals) +
라이브 다소스 바인딩(collect_price_checks) 분리 — data_verify.py 패턴.
"""
from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("corvin.signals.data_gate")

_PRICE_TOL_PCT = 1.0     # KIS vs yahoo/pykrx 허용 괴리 (스펙 §7: ±1%)
_MAX_STALE_DAYS = 3      # 데이터 경과 기본 한도 — 초과 시 신호 차단
_NUMERIC_FIELDS = ("urgency", "confidence", "horizon_days")


def _numbers_ok(sig: dict[str, Any]) -> bool:
    """핵심 수치 필드 NaN/inf/비수치 거부. None은 허용(옵셔널 필드)."""
    for f in _NUMERIC_FIELDS:
        v = sig.get(f)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if math.isnan(fv) or math.isinf(fv):
            return False
    return True


def _fix_label(sig: dict[str, Any], market_open: bool) -> dict[str, Any]:
    """장마감이면 '현재가' → '전일종가' 교정. 새 dict 반환 (비변이)."""
    msg = str(sig.get("message", ""))
    if market_open or "현재가" not in msg:
        return sig
    return {**sig, "message": msg.replace("현재가", "전일종가")}


def gate_signals(signals: list[dict[str, Any]], *,
                 price_checks: dict[str, dict] | None = None,
                 market_open: bool = True,
                 data_age_days: int | None = None,
                 max_age_days: int = _MAX_STALE_DAYS) -> dict[str, Any]:
    """신호 리스트 검증 — {passed, blocked, warnings} 반환.

    - data_age_days > max_age_days → 전체 차단 (stale_data) + 갱신 요청
    - NaN/inf 수치 → 해당 신호 drop (invalid_numeric)
    - price_checks[symbol].flag == "discrepancy" → 보류 (price_discrepancy) + 경고
    - market_open=False → message의 '현재가' → '전일종가' 교정
    """
    checks = price_checks or {}
    if data_age_days is not None and data_age_days > max_age_days:
        return {
            "passed": [],
            "blocked": [{"signal": s, "reason": "stale_data"} for s in signals],
            "warnings": [f"⚠️ 데이터 {data_age_days}일 경과 — "
                         f"신호 {len(signals)}건 차단, 갱신 필요"],
        }
    passed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    warnings: list[str] = []
    for s in signals:
        if not _numbers_ok(s):
            blocked.append({"signal": s, "reason": "invalid_numeric"})
            continue
        chk = checks.get(str(s.get("symbol", "")))
        if chk and chk.get("flag") == "discrepancy":
            blocked.append({"signal": s, "reason": "price_discrepancy"})
            warnings.append(
                f"⚠️ {s.get('symbol')} 가격 소스 불일치 "
                f"{chk.get('spread_pct', 0):.1f}% — 신호 보류")
            continue
        passed.append(_fix_label(s, market_open))
    return {"passed": passed, "blocked": blocked, "warnings": warnings}
```

- [ ] **Step 4: Run** `python3 -m pytest tests/test_data_gate.py -v` — 8 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/data_gate.py tests/test_data_gate.py
git commit -m "feat(signals): data_gate 순수 게이트 코어 — NaN drop·가격불일치 보류·라벨 교정·stale 차단"
```

---

### Task 2: `collect_price_checks` 라이브 다소스 바인딩

**Files:**
- Modify: `corvin_jarvis/signals/data_gate.py` (append)
- Test: `tests/test_data_gate.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# ── collect_price_checks — DI fetcher 바인딩 ─────────────────

@pytest.mark.unit
def test_collect_price_checks_uses_injected_fetchers():
    """주입된 fetcher 페어로 cross_check — 일치 시 flag None."""
    def fetchers_for(sym):
        return {"kis": lambda: 100.0, "alt": lambda: 100.5}
    out = dg.collect_price_checks(["TSLA"], fetchers_for=fetchers_for)
    assert out["TSLA"]["flag"] is None
    assert out["TSLA"]["confidence"] == "high"


@pytest.mark.unit
def test_collect_price_checks_flags_discrepancy():
    """±1% 초과 괴리 — discrepancy flag (스펙 §7 row 1)."""
    def fetchers_for(sym):
        return {"kis": lambda: 100.0, "alt": lambda: 103.0}
    out = dg.collect_price_checks(["TSLA"], fetchers_for=fetchers_for)
    assert out["TSLA"]["flag"] == "discrepancy"


@pytest.mark.unit
def test_collect_price_checks_single_source_passes():
    """한 소스 죽음 — single_source, 게이트는 차단하지 않음 (소스 격리)."""
    def fetchers_for(sym):
        return {"kis": lambda: (_ for _ in ()).throw(RuntimeError("down")),
                "alt": lambda: 100.0}
    out = dg.collect_price_checks(["TSLA"], fetchers_for=fetchers_for)
    assert out["TSLA"]["flag"] == "single_source"
    # single_source는 gate_signals에서 discrepancy가 아니므로 통과되어야 함
    res = dg.gate_signals([{"symbol": "TSLA", "kind": "STOP",
                            "urgency": 50, "confidence": 60.0, "message": "x"}],
                          price_checks=out, market_open=True)
    assert len(res["passed"]) == 1


@pytest.mark.unit
def test_collect_price_checks_dedups_symbols():
    calls = []
    def fetchers_for(sym):
        calls.append(sym)
        return {"kis": lambda: 100.0, "alt": lambda: 100.0}
    dg.collect_price_checks(["TSLA", "TSLA", "NVDA"], fetchers_for=fetchers_for)
    assert calls == ["TSLA", "NVDA"]
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_data_gate.py -k collect -v` — 4 FAIL (`no attribute 'collect_price_checks'`)

- [ ] **Step 3: Append implementation** (data_gate.py 끝에)

```python
# ── 라이브 다소스 바인딩 ──────────────────────────────────

def _kis_price(symbol: str) -> float | None:
    """KIS 시세 (KR/US 자동 분기). env 없으면 None."""
    from corvin_jarvis import kis_quote
    from corvin_jarvis import quote_provider as qp
    env = qp._get_kis_env()
    if env is None:
        return None
    if qp.is_kr_stock(symbol):
        price, _ = kis_quote.get_kr_quote(symbol, env=env)
    else:
        price, _ = kis_quote.get_us_quote(
            symbol, exchange=qp._exchange_for(symbol), env=env)
    return price


def _alt_price(symbol: str) -> float | None:
    """교차 소스 — KR: pykrx (yfinance 금지 규칙), US: yfinance."""
    from corvin_jarvis import quote_provider as qp
    if qp.is_kr_stock(symbol):
        from kr_data import get_kr_stock_data  # noqa: PLC0415
        data = get_kr_stock_data(symbol)
        return None if "에러" in data else float(data["현재가"])
    return qp._yfinance_quote(symbol).price


def collect_price_checks(symbols: list[str],
                         fetchers_for=None,
                         tol_pct: float = _PRICE_TOL_PCT) -> dict[str, dict]:
    """종목별 KIS vs 교차소스 검증 결과 — gate_signals price_checks 입력.

    fetchers_for(sym) -> {name: fetch} 주입 가능 (테스트/커스텀).
    data_verify.verified가 예외·NaN을 소스 격리 처리.
    """
    from corvin_jarvis import data_verify
    out: dict[str, dict] = {}
    for sym in dict.fromkeys(symbols):       # 순서 보존 dedup
        if fetchers_for is not None:
            fetchers = fetchers_for(sym)
        else:
            fetchers = {"kis": lambda s=sym: _kis_price(s),
                        "alt": lambda s=sym: _alt_price(s)}
        out[sym] = data_verify.verified(sym, fetchers, tol_pct=tol_pct,
                                        positive=True)
    return out
```

- [ ] **Step 4: Run** `python3 -m pytest tests/test_data_gate.py -v` — 12 passed

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/data_gate.py tests/test_data_gate.py
git commit -m "feat(signals): data_gate 라이브 가격 교차검증 — KIS vs yahoo/pykrx, DI fetcher"
```

---

### Task 3: snapshot 통합 — record 직전 훅

**Files:**
- Modify: `corvin_jarvis/dashboard/snapshot.py` (`_record_to_ledger`, `build_snapshot`)
- Test: `tests/test_dashboard_snapshot.py` (append)

- [ ] **Step 1: Append failing tests** (tests/test_dashboard_snapshot.py 끝에)

```python
# ── Phase 4: data gate 훅 ─────────────────────────────────

@pytest.mark.unit
def test_record_to_ledger_gates_signals(monkeypatch, tmp_path):
    """게이트 차단 신호는 ledger에 기록되지 않음 (스펙 §7: 통과 신호만 기록)."""
    from corvin_jarvis.dashboard import snapshot as snap
    from corvin_jarvis.signals import ledger

    db = tmp_path / "ledger.db"
    monkeypatch.setattr(ledger, "DB_PATH", db)
    # 가격 교차검증은 라이브 IO — 테스트에서는 항상 빈 dict
    monkeypatch.setattr(
        "corvin_jarvis.signals.data_gate.collect_price_checks",
        lambda syms, **kw: {})
    bad = {"symbol": "BAD", "kind": "STOP", "urgency": 50,
           "confidence": float("nan"), "message": "x"}
    good = {"symbol": "OK", "kind": "STOP", "urgency": 50,
            "confidence": 60.0, "message": "x"}
    info = snap._record_to_ledger([bad, good], [], market_open=True)
    rows = ledger.fetch_open(db_path=db)
    assert [r["symbol"] for r in rows] == ["OK"]
    assert info["blocked"] == 1


@pytest.mark.unit
def test_snapshot_exposes_data_gate_summary(monkeypatch):
    """snapshot에 data_gate 요약 키 — blocked 수 + warnings."""
    from corvin_jarvis.dashboard import snapshot as snap
    monkeypatch.setattr(snap, "_record_to_ledger",
                        lambda *a, **kw: {"blocked": 2, "warnings": ["⚠️ w"]})
    monkeypatch.setattr(snap, "_load_portfolio", lambda: {"positions": []})
    s = snap.build_snapshot()
    assert s["data_gate"]["blocked"] == 2
    assert s["data_gate"]["warnings"] == ["⚠️ w"]
```

(참고: 두 번째 테스트에서 `build_snapshot`의 나머지 fetch는 모두 `_safe` 격리라 네트워크 없이도 빈 값으로 진행됨 — 기존 snapshot 테스트와 동일 패턴. 기존 테스트 파일의 픽스처/mock 헬퍼가 있으면 그 패턴을 따를 것.)

- [ ] **Step 2: Run** `python3 -m pytest tests/test_dashboard_snapshot.py -k "gate" -v` — 2 FAIL

- [ ] **Step 3: Modify `_record_to_ledger`** (snapshot.py)

기존:

```python
def _record_to_ledger(engine_sigs: list[dict], pred_sigs: list[dict]) -> None:
    """Phase 1 원장 기록 — 실패해도 신호 흐름 무영향 (_safe로 호출)."""
    from corvin_jarvis.signals import ledger
    ledger.record_batch("signal_engine", engine_sigs)
    ledger.record_batch("predictive", pred_sigs)
```

변경:

```python
def _record_to_ledger(engine_sigs: list[dict], pred_sigs: list[dict],
                      market_open: bool = True) -> dict:
    """Phase 1 원장 기록 + Phase 4 게이트 — 통과 신호만 기록 (스펙 §7).

    staleness는 portfolio 기존 정책(STALE≥7 차단)을 따름 — WARN(4-6일)
    구간 전체 차단은 기존 정책보다 과격 (plan 설계 결정 참조).
    """
    from corvin_jarvis import staleness
    from corvin_jarvis.signals import data_gate, ledger
    held_syms = [str(s.get("symbol", "")) for s in engine_sigs + pred_sigs
                 if s.get("symbol")]
    checks = _safe(lambda: data_gate.collect_price_checks(held_syms), {})
    rep = _safe(staleness.check, None)
    age = rep.days_since_update if (rep and rep.block_strategy) else None
    g_eng = data_gate.gate_signals(engine_sigs, price_checks=checks,
                                   market_open=market_open, data_age_days=age,
                                   max_age_days=6)
    g_pred = data_gate.gate_signals(pred_sigs, price_checks=checks,
                                    market_open=market_open, data_age_days=age,
                                    max_age_days=6)
    ledger.record_batch("signal_engine", g_eng["passed"])
    ledger.record_batch("predictive", g_pred["passed"])
    for w in g_eng["warnings"] + g_pred["warnings"]:
        log.warning("data_gate: %s", w)
    return {"blocked": len(g_eng["blocked"]) + len(g_pred["blocked"]),
            "warnings": g_eng["warnings"] + g_pred["warnings"]}
```

`build_snapshot` 변경 — 호출부와 반환 dict:

```python
    market_open = market_hours.is_kr_open(now) or market_hours.is_us_open(now)
    gate_info = _safe(lambda: _record_to_ledger(engine_sigs, pred_sigs,
                                                market_open=market_open),
                      {"blocked": 0, "warnings": []})
```

(기존 `_safe(lambda: _record_to_ledger(engine_sigs, pred_sigs), None)` 줄 대체.
기존 `"market_state": "open" if (...)` 줄은 `"market_state": "open" if market_open else "closed"`로 단순화 — 같은 식 재사용.)

반환 dict에 키 추가:

```python
        "data_gate": gate_info,
```

- [ ] **Step 4: Run** `python3 -m pytest tests/test_dashboard_snapshot.py tests/test_data_gate.py -v` — 전부 PASS

- [ ] **Step 5: Run 전체 회귀** `python3 -m pytest tests/ -q` — 전부 PASS (695+ 예상)

- [ ] **Step 6: Commit**

```bash
git add corvin_jarvis/dashboard/snapshot.py tests/test_dashboard_snapshot.py
git commit -m "feat(dashboard): record 직전 data_gate 훅 — 통과 신호만 ledger 기록, gate 요약 노출"
```

---

### Task 4: 전체 회귀 + E2E + 보고

- [ ] **Step 1:** `python3 -m pytest tests/ -q` — 전체 PASS

- [ ] **Step 2: E2E 실데이터**

```bash
python3 -c "
from corvin_jarvis.dashboard import snapshot
snapshot._CACHE['data'] = None
s = snapshot.build_snapshot()
print('market_state:', s['market_state'])
print('data_gate:', s['data_gate'])
for sig in s['predictive_signals']:
    print(sig['kind'], sig['symbol'], '|', sig['message'][:70])
"
```

기대: `data_gate` 키에 blocked/warnings. 장마감 시간대 실행이면 predictive 메시지에 '현재가' 없음(교정 작동). 가격 교차는 KIS·yahoo 정상 시 flag 없음 → blocked=0이 정상.

- [ ] **Step 3:** 대시보드 재시작 + 검증

```bash
kill <기존 PID> && nohup python3 -m corvin_jarvis.dashboard > /tmp/corvin_dashboard.log 2>&1 &
sleep 3 && curl -s http://127.0.0.1:8765/api/snapshot | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data_gate'))"
```

- [ ] **Step 4:** main 머지 (폐하 "모두 진행해" 사전 승인 — 2026-06-11 Discord) + Discord 보고

```bash
git checkout main && git merge --no-ff feat/data-gate-phase4 -m "merge: Phase 4 데이터 검증 게이트 — 신호 발화 전 교차검증·라벨교정·stale차단"
python3 -m pytest tests/ -q   # 머지 후 회귀
```

---

## Self-Review 결과

- **스펙 §7 커버리지**: 가격 교차 ±1%(T1 게이트 + T2 바인딩, KR=pykrx 규칙 준수) · 라벨 교정(T1) · staleness 차단+갱신요청(T1, 통합부는 기존 portfolio 정책과 정합 — 설계 결정 문서화) · NaN/inf drop(T1) · record 직전 훅 + 통과만 기록(T3) — 전부 매핑.
- **Placeholder 스캔**: 없음 — 전 Step 실코드.
- **타입 일관성**: `gate_signals` 반환 {passed, blocked, warnings} — T1 정의·T3 소비 일치. `collect_price_checks(symbols, fetchers_for, tol_pct)` — T2 정의·T3 소비 일치.
- **비고**: jarvis.py·run_leading.py의 record_batch 콜사이트는 이번 범위 외 (스펙 §7은 발화 경로 = snapshot 게이트가 핵심; cron 엔진 게이트는 후속 — 과설계 방지).
