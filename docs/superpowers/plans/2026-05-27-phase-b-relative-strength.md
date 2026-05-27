# Phase B (RS) — Relative Strength Leading Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 감시 universe 종목이 자기 시장 지수보다 얼마나 강한지(상대강도 RS)를 계산해, 임계 초과 시 alert을 낸다. 추가 데이터 조회 없이 기존 스냅샷(종목 pct_change vs 지수 pct_change)만 사용.

**Architecture:** 신규 `corvin_jarvis/signals/leading.py`에 `check_relative_strength()`를 두고, `jarvis.merge_signal_alerts()`가 market별 벤치마크 지수(KR→kospi, US→sp500)의 pct_change를 넘겨 호출. alert dict는 기존 universe/sector와 동일 스키마 + phase 라벨.

**Tech Stack:** Python 3.11+, pytest. severity는 `compare._classify_severity` 재사용.

**Scope:** RS(B1a)만. 모멘텀(B1b)·거래량·52주신고가(B2)는 묶음 조회(bulk fetch) 데이터 레이어 작업과 함께 별도 plan.

**Spec:** `docs/superpowers/specs/2026-05-27-proactive-signal-detection-design.md` (§3.1 B1, §5.2 leading)

---

## File Structure
| 파일 | 책임 | 신규/수정 |
|---|---|---|
| `corvin_jarvis/signals/leading.py` | RS 계산 → alert dict | Create |
| `corvin_jarvis/config.json` | `leading.rs_min_pct` 추가 | Modify |
| `corvin_jarvis/jarvis.py` | merge_signal_alerts에 RS 호출(market별 지수 매핑) | Modify |
| `tests/test_leading.py` | RS 단위테스트 | Create |
| `tests/test_universe_monitor.py` | merge RS 통합테스트 추가 | Modify |

**Alert dict 스키마:** category="leading_rs", metric=f"rs_{symbol}_{phase}", severity, message, value(=rs, %p), threshold, phase.

---

### Task B1: leading.py — relative strength

**Files:**
- Create: `corvin_jarvis/signals/leading.py`
- Test: `tests/test_leading.py`

- [ ] **Step 1: Write the failing test** — `tests/test_leading.py`:
```python
from corvin_jarvis.signals import leading


def _u(pct, sym="005930", name="삼성전자"):
    return [{"symbol": sym, "market": "KR", "sector": "semiconductor", "name": name, "pct_change": pct}]


def test_outperform_index_creates_strong_alert():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    alerts = leading.check_relative_strength(_u(7.0), index_pct=2.5, config=cfg, phase="confirmed", index_name="kospi")
    assert len(alerts) == 1
    a = alerts[0]
    assert a["category"] == "leading_rs"
    assert a["metric"] == "rs_005930_confirmed"
    assert abs(a["value"] - 4.5) < 0.01      # 7.0 - 2.5
    assert "강세" in a["message"]


def test_underperform_index_creates_weak_alert():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    alerts = leading.check_relative_strength(_u(-1.0), index_pct=2.5, config=cfg, phase="confirmed")
    assert len(alerts) == 1
    assert alerts[0]["value"] < 0
    assert "약세" in alerts[0]["message"]


def test_within_threshold_no_alert():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    assert leading.check_relative_strength(_u(3.0), index_pct=2.5, config=cfg, phase="confirmed") == []


def test_none_index_returns_empty():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    assert leading.check_relative_strength(_u(7.0), index_pct=None, config=cfg, phase="confirmed") == []


def test_missing_pct_skipped():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    entries = [{"symbol": "X", "market": "KR", "sector": "auto", "name": "X", "pct_change": None}]
    assert leading.check_relative_strength(entries, index_pct=2.5, config=cfg, phase="confirmed") == []


def test_provisional_label():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    a = leading.check_relative_strength(_u(7.0), index_pct=2.5, config=cfg, phase="provisional")[0]
    assert a["metric"] == "rs_005930_provisional"
    assert "🟡" in a["message"] and "잠정" in a["message"]
```

- [ ] **Step 2: Run — expect FAIL** `python3 -m pytest tests/test_leading.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `corvin_jarvis/signals/leading.py`:
```python
"""선행 신호 (Phase B) — 상대강도(RS): 종목이 시장 지수보다 얼마나 센가/약한가."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from compare import Severity, _classify_severity  # noqa: E402

_PHASE_EMOJI = {"provisional": "🟡", "confirmed": "✅"}
_PHASE_LABEL = {"provisional": "잠정", "confirmed": "확정"}


def _name_tag(entry: dict[str, Any]) -> str:
    name = entry.get("name")
    sym = entry.get("symbol")
    return f"{name}({sym})" if name else str(sym)


def check_relative_strength(
    universe_entries: list[dict[str, Any]],
    index_pct: float | None,
    config: dict[str, Any],
    phase: str,
    index_name: str = "",
) -> list[dict[str, Any]]:
    """종목 pct_change 와 벤치마크 지수 pct_change 의 차이(RS, %p)가 임계 초과 시 alert."""
    if index_pct is None:
        return []
    cfg = config.get("leading", {})
    rs_min = float(cfg.get("rs_min_pct", 2.0))
    emoji, label = _PHASE_EMOJI.get(phase, ""), _PHASE_LABEL.get(phase, phase)
    bench = index_name or "지수"

    alerts: list[dict[str, Any]] = []
    for entry in universe_entries:
        pct = entry.get("pct_change")
        if pct is None:
            continue
        rs = pct - index_pct
        if abs(rs) < rs_min:
            continue
        sev = _classify_severity(rs, rs_min)
        direction = "강세" if rs > 0 else "약세"
        sym = entry.get("symbol")
        alerts.append({
            "category": "leading_rs",
            "metric": f"rs_{sym}_{phase}",
            "severity": sev.value if isinstance(sev, Severity) else str(sev),
            "message": (f"{emoji} {_name_tag(entry)} 상대강도 {direction} {rs:+.2f}%p "
                        f"(종목 {pct:+.2f}% vs {bench} {index_pct:+.2f}%, 임계 ±{rs_min}%p, {label})"),
            "value": round(rs, 4),
            "threshold": rs_min,
            "phase": phase,
        })
    return alerts
```

- [ ] **Step 4: Run — expect PASS** `python3 -m pytest tests/test_leading.py -v` → 6 passed.

- [ ] **Step 5: Commit**
```bash
git add corvin_jarvis/signals/leading.py tests/test_leading.py
git commit -m "feat(corvin): relative strength leading signal (Phase B)"
```

---

### Task B2: config.json — leading threshold

**Files:** Modify `corvin_jarvis/config.json`

- [ ] **Step 1:** In `alert_thresholds`, after the `sector_basket_pct` block, add (comma after the prior `}`):
```json
    "leading": {
      "rs_min_pct": 2.0
    }
```

- [ ] **Step 2: Validate** `python3 -c "import json; print(json.load(open('corvin_jarvis/config.json'))['alert_thresholds']['leading'])"` → `{'rs_min_pct': 2.0}`

- [ ] **Step 3: Commit**
```bash
git add corvin_jarvis/config.json
git commit -m "feat(corvin): add leading RS threshold config (Phase B)"
```

---

### Task B3: wire RS into jarvis.merge_signal_alerts

**Files:**
- Modify: `corvin_jarvis/jarvis.py`
- Test: `tests/test_universe_monitor.py` (append)

- [ ] **Step 1: Append the failing test** to `tests/test_universe_monitor.py`:
```python
def test_merge_signal_alerts_includes_relative_strength(tmp_path, monkeypatch):
    import json
    from corvin_jarvis import jarvis
    from corvin_jarvis.signals import market_phase

    latest = {
        "timestamp_kst": "2026-05-27T16:00:00+09:00",
        "indices": {"kospi": {"price": 8000, "pct_change": 2.5}},
        "universe": [
            {"symbol": "005930", "market": "KR", "sector": "semiconductor", "name": "삼성전자", "price": 320000, "pct_change": 7.02},
        ],
    }
    latest_file = tmp_path / "latest.json"
    alerts_file = tmp_path / "alerts.json"
    latest_file.write_text(json.dumps(latest))
    monkeypatch.setattr(jarvis, "LATEST_FILE", latest_file)
    monkeypatch.setattr(jarvis, "ALERTS_FILE", alerts_file)
    monkeypatch.setattr(market_phase, "phase_for", lambda m, now=None: "confirmed")

    jarvis.merge_signal_alerts()
    data = json.loads(alerts_file.read_text())
    metrics = {a["metric"] for a in data["alerts"]}
    assert "rs_005930_confirmed" in metrics      # 7.02 vs kospi 2.5 → RS +4.52%p
```

- [ ] **Step 2: Run — expect FAIL** `python3 -m pytest tests/test_universe_monitor.py::test_merge_signal_alerts_includes_relative_strength -v` → no rs_ metric.

- [ ] **Step 3: Modify `merge_signal_alerts` in `corvin_jarvis/jarvis.py`.**

Change the import line inside the function from:
```python
    from corvin_jarvis.signals import market_phase, universe_monitor
```
to:
```python
    from corvin_jarvis.signals import market_phase, universe_monitor, leading
```

Add this module-level constant right before `def merge_signal_alerts`:
```python
_INDEX_FOR_MARKET = {"KR": "kospi", "US": "sp500"}
```

In the per-market loop, after the `check_sectors` line, add the RS call:
```python
        idx_name = _INDEX_FOR_MARKET.get(market)
        idx_q = latest.get("indices", {}).get(idx_name, {}) if idx_name else {}
        idx_pct = idx_q.get("pct_change") if isinstance(idx_q, dict) else None
        s_alerts.extend(leading.check_relative_strength(entries, idx_pct, th, phase, index_name=idx_name or ""))
```
(So the loop body runs check_tickers, check_sectors, then check_relative_strength.)

- [ ] **Step 4: Run — expect PASS** `python3 -m pytest tests/test_universe_monitor.py -v` → all pass (incl new RS test).

- [ ] **Step 5: Full suite** `python3 -m pytest tests/ -q` → no regressions.

- [ ] **Step 6: Commit**
```bash
git add corvin_jarvis/jarvis.py tests/test_universe_monitor.py
git commit -m "feat(corvin): wire relative strength into signal pipeline (Phase B)"
```

---

### Task B4: verification

- [ ] **Step 1:** `python3 corvin_jarvis/jarvis.py 2>&1 | grep -iE "Signal alerts merged|ERROR"` → completes, signal alerts merged.
- [ ] **Step 2:** Inspect `alerts.json` for any `rs_*` (leading_rs) entries: `python3 -c "import json; d=json.load(open('corvin_jarvis/state/alerts.json')); print([a['metric'] for a in d['alerts'] if a['category']=='leading_rs'])"` (may be empty if no stock diverges from index beyond threshold — that's valid).
- [ ] **Step 3:** Coverage `python3 -m pytest tests/test_leading.py --cov=corvin_jarvis/signals/leading --cov-report=term-missing -p no:warnings` → ≥80%.

---

## Self-Review
- Spec coverage: §3.1 B1 RS ✅ (Task B1, wired B3). Momentum/B2 explicitly deferred (scope, ties to bulk-fetch).
- Placeholders: none — all steps have complete code/commands.
- Type consistency: `check_relative_strength(universe_entries, index_pct, config, phase, index_name)` signature consistent between leading.py, the test, and the jarvis call site. Alert dict keys match universe/sector schema + phase.

## Notes
- RS threshold (rs_min_pct 2.0%p) is tunable in config; raise it if alerts are noisy on broad up/down days.
- Index benchmark map KR→kospi, US→sp500 (configurable extension later).
