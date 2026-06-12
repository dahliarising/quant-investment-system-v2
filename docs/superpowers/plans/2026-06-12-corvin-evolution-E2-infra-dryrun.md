# Corvin Evolution Loop — Phase E2: 진화 인프라 골격 + 드라이런 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 자가진화 루프의 "심판"(머지 게이트 + pinned 백테스트 평가자 + 세이프티 검사)을 **레포 밖** 별도 repo로 구축하고, **드라이런 모드**(머지 안 함, 후보 평가·로그만)로 게이트가 좋은/나쁜/위험한 후보를 올바르게 판정함을 증명한다. 아직 자가수정 에이전트(선수)는 만들지 않는다 — 심판부터.

**Architecture:** `~/corvin_evolution/`를 **독립 git repo**(supervisor)로 만든다. 피수정 대상인 `quant_investment_system_v2`(subject)와 물리적으로 분리 → 루프가 자기 심판에 쓰기 권한 없음. 게이트는 결정론적(LLM 호출 없음). pinned 평가자는 *자기 소유 백테스트 엔진 + 번들 fixture 데이터 + 자기 채점*을 갖고, 후보로부터는 **개선 대상인 시그널 분류기만** import해 동일 잣대로 측정한다("선수는 바꾸되 심판의 자·초시계·시험지는 못 건드림").

**Tech Stack:** Python 3.11, pytest, PyYAML, git, subprocess. 후보 레포의 `corvin_jarvis.ew_runner._classify_all`/`early_warning`(분류기)만 import.

**선행:** [Evolution Loop 설계서](../specs/2026-06-12-corvin-evolution-loop-design.md) §1·§2·§4 · [E1 플랜](./2026-06-12-corvin-evolution-E1-persuasion-brief.md)(완료·머지).

---

## 핵심 설계 결정 (구현 전 폐하 리뷰 포인트)

1. **edge 메트릭 v1** = pinned 평가자가 번들 fixture 시계열에 후보 분류기를 흘려 산출한 **선행신호 F1**(precision·recall 조화평균, EW 백테스트 채점 방식 차용하되 *평가자 소유 코드*로 재구현). 외부 고정이라 루프가 정의를 못 바꾼다. E4에서 수익·Sharpe로 정교화.
2. **walk-forward**: fixture를 train/oos 구간으로 분할. IS edge(train)·OOS edge(oos) 둘 다 산출 → 과적합 갭 = IS−OOS.
3. **번들 데이터**: 백테스트용 과거 시계열은 `~/corvin_evolution/fixtures/`에 둔다(후보 레포가 아님) → 루프가 데이터로 게임 못 함.
4. **드라이런**: merge_gate는 `dry_run=True`에서 `git merge`를 절대 호출하지 않고 verdict만 로그. E4에서 실머지 on.

---

## File Structure (모두 `~/corvin_evolution/` = 신규 독립 repo)

| 파일 | 책임 |
|------|------|
| `config.yaml` | 경로·ε(최소 엣지 개선)·overfit_gap_threshold·캡·fixture 분할 |
| `fixtures/ew_history.json` | 백테스트용 고정 과거 시계열 (번들, 루프 불가침) |
| `evolution/__init__.py` | 패키지 |
| `evolution/types.py` | frozen dataclass: EdgeResult·Violation·Verdict |
| `evolution/pinned_backtest.py` | 후보 분류기 import + 자기 채점 → EdgeResult |
| `evolution/safety.py` | INV-1/2/3 정적 검사 → list[Violation] |
| `evolution/merge_gate.py` | pytest+백테스트+safety 조합 → Verdict (dry_run) |
| `evolution/snapshot_baseline.py` | main 평가 → baseline.json |
| `runner.sh` | 킬스위치 체크 → 후보 평가 → evolution.log (E2: 에이전트 dispatch 없음) |
| `tests/` | supervisor 자체 pytest |
| `README.md`·`.gitignore` | |

---

## Task 1: supervisor repo 골격 + config + types

**Files (in `~/corvin_evolution/`):**
- Create: `~/corvin_evolution/.gitignore`, `config.yaml`, `evolution/__init__.py`, `evolution/types.py`
- Test: `~/corvin_evolution/tests/test_types.py`

- [ ] **Step 1: repo 초기화**

Run:
```bash
mkdir -p ~/corvin_evolution/evolution ~/corvin_evolution/tests ~/corvin_evolution/fixtures
cd ~/corvin_evolution && git init -q && python3 -m venv .venv 2>/dev/null; true
printf '.venv/\n__pycache__/\nevolution.log\nbaseline.json\n*.pyc\n' > .gitignore
```

- [ ] **Step 2: Write the failing test**

```python
# ~/corvin_evolution/tests/test_types.py
import dataclasses
from evolution.types import EdgeResult, Violation, Verdict


def test_edge_result_overfit_gap():
    e = EdgeResult(oos_edge=0.55, is_edge=0.60, window_id="w1")
    assert round(e.overfit_gap, 4) == 0.05
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        e.oos_edge = 1.0  # type: ignore[misc]


def test_verdict_holds_reasons():
    v = Verdict(passed=False, reasons=["edge below baseline"], dry_run=True, merged=False)
    assert v.passed is False and not v.merged
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/corvin_evolution && python3 -m pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evolution'`

- [ ] **Step 4: Implement**

```python
# ~/corvin_evolution/evolution/__init__.py
"""Corvin Evolution supervisor — 레포 밖 결정론 머지 게이트 (Phase E2)."""
```

```python
# ~/corvin_evolution/evolution/types.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EdgeResult:
    oos_edge: float
    is_edge: float
    window_id: str

    @property
    def overfit_gap(self) -> float:
        return self.is_edge - self.oos_edge


@dataclass(frozen=True)
class Violation:
    code: str          # INV-1 / INV-2 / INV-3
    detail: str


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    dry_run: bool = True
    merged: bool = False
```

```yaml
# ~/corvin_evolution/config.yaml
subject_repo: /Users/thethethe/Claude/quant_investment_system_v2
killswitch: /Users/thethethe/.corvin_killswitch
epsilon: 0.02                 # OOS edge가 baseline보다 최소 이만큼 높아야 통과
overfit_gap_threshold: 0.15   # IS-OOS 갭이 이보다 크면 과적합으로 거부
fixture: fixtures/ew_history.json
train_ratio: 0.6              # walk-forward train/oos 분할
# INV-3 동결 경로 (후보 diff가 건드리면 거부) — 세이프티/게이트 관련
frozen_paths:
  - corvin_jarvis/channels.py
caps:
  max_iterations: 5
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/corvin_evolution && python3 -m pytest tests/test_types.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
cd ~/corvin_evolution && git add -A && git commit -q -m "feat(evolution): supervisor 골격 + config + types (E2)"
```

---

## Task 2: pinned 백테스트 fixture + EdgeResult 산출

**Files:**
- Create: `~/corvin_evolution/fixtures/ew_history.json`, `~/corvin_evolution/evolution/pinned_backtest.py`
- Test: `~/corvin_evolution/tests/test_pinned_backtest.py`

**원리**: 평가자가 *자기 소유* replay+채점을 수행하고, 후보 레포에서는 분류기(`_classify_all`)만 import. fixture는 번들. `evaluate(subject_repo)`가 `sys.path`에 subject_repo를 넣어 분류기를 로드 → fixture를 train/oos로 나눠 각 구간 F1 산출.

fixture 스키마: `[{"date","readings":{...EW 입력...},"spx":float}]`. 최소 30일 합성/과거 데이터(드로다운 이벤트 포함).

- [ ] **Step 1: Write the failing test**

```python
# ~/corvin_evolution/tests/test_pinned_backtest.py
from evolution.pinned_backtest import evaluate, _f1
from evolution.types import EdgeResult


def test_f1_basic():
    # signal_dates가 events를 정확히 맞추면 F1=1
    assert _f1(signal_dates={"d1", "d2"}, event_dates={"d1", "d2"}) == 1.0
    assert _f1(signal_dates=set(), event_dates={"d1"}) == 0.0


def test_evaluate_returns_edgeresult(tmp_path):
    import os
    repo = os.environ.get("SUBJECT_REPO", "/Users/thethethe/Claude/quant_investment_system_v2")
    res = evaluate(subject_repo=repo)
    assert isinstance(res, EdgeResult)
    assert 0.0 <= res.oos_edge <= 1.0
    assert 0.0 <= res.is_edge <= 1.0
```

- [ ] **Step 2: Create fixture**

`~/corvin_evolution/fixtures/ew_history.json` — 40일 시계열. EW readings 키는 subject의 `early_warning` 분류기 입력과 일치해야 함(구현 시 `grep -n 'def _classify\|readings\[' <subject>/corvin_jarvis/early_warning.py`로 키 확인). 드로다운 구간(SPX 하락) 2~3개 포함. (실데이터 추출 권장: subject의 `pulse`/지표 로그에서 과거 40일을 덤프하거나, 합성으로 vix↑·breadth↓·spx↓ 구간을 심는다.)

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/corvin_evolution && python3 -m pytest tests/test_pinned_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: evolution.pinned_backtest`

- [ ] **Step 4: Implement (pinned 채점은 평가자 소유)**

```python
# ~/corvin_evolution/evolution/pinned_backtest.py
"""Pinned 백테스트 — 평가자 소유 채점. 후보의 분류기만 import (게임내성)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evolution.types import EdgeResult

_HORIZON = 5
_DRAWDOWN_PCT = -3.0
_HERE = Path(__file__).resolve().parent.parent


def _load_fixture(name: str) -> list[dict]:
    return json.loads((_HERE / name).read_text(encoding="utf-8"))


def _event_dates(history: list[dict]) -> set[str]:
    # 평가자 소유 라벨링 — 미래 horizon일 내 SPX 드로다운 ≥ |thresh|
    out: set[str] = set()
    for i, rec in enumerate(history):
        base = rec["spx"]
        window = history[i + 1: i + 1 + _HORIZON]
        if window and (min(w["spx"] for w in window) / base - 1) * 100 <= _DRAWDOWN_PCT:
            out.add(rec["date"])
    return out


def _signal_dates(history: list[dict], classify) -> set[str]:
    # 후보의 분류기로 RED/REDUCE 게이지가 뜬 날
    out: set[str] = set()
    for rec in history:
        states = classify(rec["readings"])
        if any(v == "red" for v in states.values()):
            out.add(rec["date"])
    return out


def _f1(signal_dates: set[str], event_dates: set[str]) -> float:
    if not signal_dates and not event_dates:
        return 1.0
    tp = len(signal_dates & event_dates)
    if tp == 0:
        return 0.0
    prec = tp / len(signal_dates)
    rec = tp / len(event_dates)
    return 2 * prec * rec / (prec + rec)


def _make_classifier(subject_repo: str):
    if subject_repo not in sys.path:
        sys.path.insert(0, subject_repo)
    from corvin_jarvis import early_warning as ew  # 후보 분류기 (개선 대상)
    cfg = ew.default_config() if hasattr(ew, "default_config") else {}
    from corvin_jarvis.ew_runner import _classify_all
    return lambda readings: _classify_all(readings, cfg)


def evaluate(subject_repo: str, *, fixture: str = "fixtures/ew_history.json",
             train_ratio: float = 0.6) -> EdgeResult:
    history = _load_fixture(fixture)
    classify = _make_classifier(subject_repo)
    split = int(len(history) * train_ratio)
    train, oos = history[:split], history[split:]

    def edge(seg: list[dict]) -> float:
        return _f1(_signal_dates(seg, classify), _event_dates(seg))

    return EdgeResult(oos_edge=edge(oos), is_edge=edge(train), window_id=f"{train_ratio}")
```

- [ ] **Step 5: Run tests**

Run: `cd ~/corvin_evolution && python3 -m pytest tests/test_pinned_backtest.py -v`
Expected: PASS (2 passed). If `_classify_all`/`default_config` signature differs, fix `_make_classifier` to match the real subject API (verify with grep first) — this is the one integration seam.

- [ ] **Step 6: Commit**

```bash
cd ~/corvin_evolution && git add -A && git commit -q -m "feat(evolution): pinned 백테스트 — 자기채점+번들fixture, 후보 분류기만 import"
```

---

## Task 3: 세이프티 검사 INV-1 (no-trade denylist)

**Files:**
- Create: `~/corvin_evolution/evolution/safety.py`
- Test: `~/corvin_evolution/tests/test_safety_inv1.py`

INV-1: 후보 diff(텍스트)가 **실매매/주문 능력**을 추가하면 거부. denylist 정규식: `place_order`, `submit_order`, `send_order`, `create_order`, `order/cash/buy/sell` 류 KIS 주문 엔드포인트(`TTTC0802U`,`TTTC0801U` 등 주문 TR), `portfolio.json` 쓰기(`open(...portfolio.json..., "w")`, `write_text` to portfolio).

- [ ] **Step 1: Write the failing test**

```python
# ~/corvin_evolution/tests/test_safety_inv1.py
from evolution.safety import check_no_trade


def test_flags_order_call():
    diff = '+    broker.place_order(symbol, qty)\n'
    v = check_no_trade(diff)
    assert v and v[0].code == "INV-1"


def test_flags_portfolio_write():
    diff = '+    open("portfolio.json", "w").write(data)\n'
    assert any(x.code == "INV-1" for x in check_no_trade(diff))


def test_clean_diff_passes():
    diff = '+    return f1_score(signals, events)\n'
    assert check_no_trade(diff) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/corvin_evolution && python3 -m pytest tests/test_safety_inv1.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# ~/corvin_evolution/evolution/safety.py
"""세이프티 불변식 정적 검사 (INV-1/2/3). 후보 diff 텍스트 기반."""
from __future__ import annotations

import re

from evolution.types import Violation

# 추가된 라인(+)만 검사
_ORDER_PATTERNS = [
    r"\b(place|submit|send|create)_order\b",
    r"\bTTTC080[12]U\b",            # KIS 국내주문 TR
    r"\bTTTT1002U\b|\bJTTT1002U\b",  # KIS 해외주문 TR (현금매수)
]
_PORTFOLIO_WRITE = re.compile(
    r"""portfolio\.json["']?\s*,\s*["']w|"""
    r"""(write_text|\.write)\([^)]*portfolio""", re.IGNORECASE)


def _added_lines(diff: str) -> list[str]:
    return [ln[1:] for ln in diff.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")]


def check_no_trade(diff: str) -> list[Violation]:
    out: list[Violation] = []
    body = "\n".join(_added_lines(diff))
    for pat in _ORDER_PATTERNS:
        m = re.search(pat, body)
        if m:
            out.append(Violation("INV-1", f"주문 실행 패턴 추가: {m.group(0)}"))
    if _PORTFOLIO_WRITE.search(body):
        out.append(Violation("INV-1", "portfolio.json 쓰기 시도"))
    return out
```

- [ ] **Step 4: Run tests** — `cd ~/corvin_evolution && python3 -m pytest tests/test_safety_inv1.py -v` → PASS (3)

- [ ] **Step 5: Commit**

```bash
cd ~/corvin_evolution && git add -A && git commit -q -m "feat(evolution): safety INV-1 — no-trade denylist (주문/portfolio쓰기 차단)"
```

---

## Task 4: 세이프티 INV-2(자기브레이크 불가침) + INV-3(scope sanity)

**Files:**
- Modify: `~/corvin_evolution/evolution/safety.py`
- Test: `~/corvin_evolution/tests/test_safety_inv23.py`

INV-2: diff가 킬스위치 경로(`~/.corvin_killswitch`)나 supervisor 경로(`corvin_evolution`)를 참조/수정하려 하면 거부. INV-3: 변경 파일 목록(`changed_files`)이 subject 레포 밖이거나 `frozen_paths`(config)를 건드리면 거부.

- [ ] **Step 1: Write the failing test**

```python
# ~/corvin_evolution/tests/test_safety_inv23.py
from evolution.safety import check_no_self_disable, check_scope


def test_inv2_killswitch_reference():
    diff = '+    os.remove("/Users/x/.corvin_killswitch")\n'
    assert any(v.code == "INV-2" for v in check_no_self_disable(diff))


def test_inv2_supervisor_reference():
    diff = '+    shutil.rmtree("~/corvin_evolution")\n'
    assert any(v.code == "INV-2" for v in check_no_self_disable(diff))


def test_inv3_frozen_path():
    v = check_scope(changed_files=["corvin_jarvis/channels.py"],
                    frozen_paths=["corvin_jarvis/channels.py"])
    assert any(x.code == "INV-3" for x in v)


def test_inv3_out_of_repo():
    v = check_scope(changed_files=["../other_project/x.py"], frozen_paths=[])
    assert any(x.code == "INV-3" for x in v)


def test_inv3_clean():
    assert check_scope(changed_files=["corvin_jarvis/predictive_engine.py"],
                       frozen_paths=["corvin_jarvis/channels.py"]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/corvin_evolution && python3 -m pytest tests/test_safety_inv23.py -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement (append to safety.py)**

```python
# append to ~/corvin_evolution/evolution/safety.py
_KILLSWITCH = re.compile(r"\.corvin_killswitch")
_SUPERVISOR = re.compile(r"corvin_evolution")


def check_no_self_disable(diff: str) -> list[Violation]:
    out: list[Violation] = []
    body = "\n".join(_added_lines(diff))
    if _KILLSWITCH.search(body):
        out.append(Violation("INV-2", "킬스위치 경로 참조/수정 시도"))
    if _SUPERVISOR.search(body):
        out.append(Violation("INV-2", "supervisor(corvin_evolution) 참조 시도"))
    return out


def check_scope(changed_files: list[str], frozen_paths: list[str]) -> list[Violation]:
    out: list[Violation] = []
    frozen = set(frozen_paths)
    for f in changed_files:
        if f.startswith("..") or f.startswith("/"):
            out.append(Violation("INV-3", f"레포 밖 파일 변경: {f}"))
        elif f in frozen:
            out.append(Violation("INV-3", f"동결 경로 변경: {f}"))
    return out


def check_all(diff: str, changed_files: list[str],
              frozen_paths: list[str]) -> list[Violation]:
    return (check_no_trade(diff) + check_no_self_disable(diff)
            + check_scope(changed_files, frozen_paths))
```

- [ ] **Step 4: Run tests** — `cd ~/corvin_evolution && python3 -m pytest tests/test_safety_inv23.py -v` → PASS (5)

- [ ] **Step 5: Commit**

```bash
cd ~/corvin_evolution && git add -A && git commit -q -m "feat(evolution): safety INV-2(자기브레이크)·INV-3(scope) + check_all 조합"
```

---

## Task 5: merge_gate 조합 (pytest + 백테스트 + safety, dry_run)

**Files:**
- Create: `~/corvin_evolution/evolution/merge_gate.py`
- Test: `~/corvin_evolution/tests/test_merge_gate.py`

`evaluate_candidate(subject_repo, candidate_ref, baseline_edge, cfg, *, dry_run=True) -> Verdict`:
1. `git -C subject_repo diff baseline..candidate_ref` → diff 텍스트 + `--name-only` → changed_files.
2. `safety.check_all(diff, changed_files, frozen_paths)` — 위반 있으면 즉시 fail.
3. pytest subprocess: `python3 -m pytest -q`(subject_repo의 candidate 체크아웃 상태에서). 비0 종료면 fail.
4. `pinned_backtest.evaluate(subject_repo)` → cand_edge. `cand.oos_edge >= baseline_edge + epsilon` AND `cand.overfit_gap <= overfit_gap_threshold` 아니면 fail.
5. 통과 + `dry_run` → `Verdict(passed=True, merged=False, dry_run=True)` (머지 안 함). `not dry_run`이면 여기서 머지(E4). 실패면 사유 리스트.

테스트는 safety/backtest/pytest 러너를 **주입**해 IO 격리.

- [ ] **Step 1: Write the failing test**

```python
# ~/corvin_evolution/tests/test_merge_gate.py
from evolution.merge_gate import decide
from evolution.types import EdgeResult, Verdict


_BASE = 0.50
_CFG = {"epsilon": 0.02, "overfit_gap_threshold": 0.15, "frozen_paths": []}


def _good_edge():
    return EdgeResult(oos_edge=0.55, is_edge=0.58, window_id="w")


def test_safety_violation_blocks():
    v = decide(diff="+ place_order(x)", changed_files=["a.py"],
               baseline_edge=_BASE, cand_edge=_good_edge(),
               tests_passed=True, cfg=_CFG, dry_run=True)
    assert v.passed is False and any("INV-1" in r for r in v.reasons)


def test_failing_tests_block():
    v = decide(diff="+ ok", changed_files=["a.py"], baseline_edge=_BASE,
               cand_edge=_good_edge(), tests_passed=False, cfg=_CFG, dry_run=True)
    assert v.passed is False and any("테스트" in r for r in v.reasons)


def test_weak_edge_blocks():
    weak = EdgeResult(oos_edge=0.505, is_edge=0.52, window_id="w")
    v = decide(diff="+ ok", changed_files=["a.py"], baseline_edge=_BASE,
               cand_edge=weak, tests_passed=True, cfg=_CFG, dry_run=True)
    assert v.passed is False and any("엣지" in r for r in v.reasons)


def test_overfit_blocks():
    of = EdgeResult(oos_edge=0.55, is_edge=0.85, window_id="w")
    v = decide(diff="+ ok", changed_files=["a.py"], baseline_edge=_BASE,
               cand_edge=of, tests_passed=True, cfg=_CFG, dry_run=True)
    assert v.passed is False and any("과적합" in r for r in v.reasons)


def test_good_candidate_passes_but_not_merged_in_dryrun():
    v = decide(diff="+ ok", changed_files=["corvin_jarvis/predictive_engine.py"],
               baseline_edge=_BASE, cand_edge=_good_edge(),
               tests_passed=True, cfg=_CFG, dry_run=True)
    assert v.passed is True and v.merged is False and v.dry_run is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/corvin_evolution && python3 -m pytest tests/test_merge_gate.py -v`
Expected: FAIL — `decide` not defined

- [ ] **Step 3: Implement (pure `decide` + IO wrapper)**

```python
# ~/corvin_evolution/evolution/merge_gate.py
"""결정론 머지 게이트. decide()는 순수(테스트용), evaluate_candidate()는 IO 래퍼."""
from __future__ import annotations

import subprocess
from typing import Any

from evolution import safety
from evolution.types import EdgeResult, Verdict


def decide(*, diff: str, changed_files: list[str], baseline_edge: float,
           cand_edge: EdgeResult, tests_passed: bool, cfg: dict[str, Any],
           dry_run: bool = True) -> Verdict:
    reasons: list[str] = []
    viols = safety.check_all(diff, changed_files, cfg.get("frozen_paths", []))
    reasons += [f"{v.code}: {v.detail}" for v in viols]
    if not tests_passed:
        reasons.append("테스트 실패")
    if cand_edge.oos_edge < baseline_edge + cfg["epsilon"]:
        reasons.append(f"엣지 부족: OOS {cand_edge.oos_edge:.3f} < "
                       f"baseline {baseline_edge:.3f}+ε {cfg['epsilon']}")
    if cand_edge.overfit_gap > cfg["overfit_gap_threshold"]:
        reasons.append(f"과적합: 갭 {cand_edge.overfit_gap:.3f} > "
                       f"{cfg['overfit_gap_threshold']}")
    passed = not reasons
    merged = passed and not dry_run
    return Verdict(passed=passed, reasons=reasons, dry_run=dry_run, merged=merged)


def _run_pytest(repo: str) -> bool:
    r = subprocess.run(["python3", "-m", "pytest", "-q"], cwd=repo,
                       capture_output=True, text=True)
    return r.returncode == 0


def evaluate_candidate(subject_repo: str, candidate_ref: str, baseline_edge: float,
                       cfg: dict[str, Any], *, dry_run: bool = True) -> Verdict:
    from evolution import pinned_backtest
    base = "main"
    diff = subprocess.run(["git", "-C", subject_repo, "diff", f"{base}..{candidate_ref}"],
                          capture_output=True, text=True).stdout
    names = subprocess.run(["git", "-C", subject_repo, "diff", "--name-only",
                            f"{base}..{candidate_ref}"], capture_output=True, text=True)
    changed = [f for f in names.stdout.splitlines() if f]
    tests_ok = _run_pytest(subject_repo)
    cand_edge = pinned_backtest.evaluate(subject_repo)
    return decide(diff=diff, changed_files=changed, baseline_edge=baseline_edge,
                  cand_edge=cand_edge, tests_passed=tests_ok, cfg=cfg, dry_run=dry_run)
```

- [ ] **Step 4: Run tests** — `cd ~/corvin_evolution && python3 -m pytest tests/test_merge_gate.py -v` → PASS (5)

- [ ] **Step 5: Commit**

```bash
cd ~/corvin_evolution && git add -A && git commit -q -m "feat(evolution): merge_gate — pytest+백테스트+safety 결정론 조합 (dry_run)"
```

---

## Task 6: baseline 스냅샷 + runner + 킬스위치 (드라이런 E2E)

**Files:**
- Create: `~/corvin_evolution/evolution/snapshot_baseline.py`, `~/corvin_evolution/runner.sh`
- Test: `~/corvin_evolution/tests/test_runner_smoke.py`

`snapshot_baseline.py`: pinned_backtest.evaluate(main) → `baseline.json {oos_edge, is_edge, ts_window}`.
`runner.sh`: 킬스위치 있으면 exit 0(skip 로그) → 없으면 baseline 로드 → `evaluate_candidate(subject, candidate_ref=$1 or "HEAD", ...)` → verdict를 evolution.log에 append. **E2: 에이전트 dispatch·머지 없음(dry_run 고정).**

- [ ] **Step 1: Write snapshot + failing smoke test**

```python
# ~/corvin_evolution/evolution/snapshot_baseline.py
from __future__ import annotations

import json
from pathlib import Path

import yaml

from evolution import pinned_backtest

_HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = yaml.safe_load((_HERE / "config.yaml").read_text())
    res = pinned_backtest.evaluate(cfg["subject_repo"],
                                   fixture=cfg["fixture"],
                                   train_ratio=cfg["train_ratio"])
    (_HERE / "baseline.json").write_text(
        json.dumps({"oos_edge": res.oos_edge, "is_edge": res.is_edge,
                    "window_id": res.window_id}, indent=2))
    print(f"baseline: OOS={res.oos_edge:.3f} IS={res.is_edge:.3f}")


if __name__ == "__main__":
    main()
```

```python
# ~/corvin_evolution/tests/test_runner_smoke.py
import subprocess
from pathlib import Path

HERE = Path.home() / "corvin_evolution"


def test_killswitch_blocks(tmp_path, monkeypatch):
    ks = tmp_path / ".corvin_killswitch"
    ks.write_text("stop")
    r = subprocess.run(["bash", str(HERE / "runner.sh")],
                       env={"CORVIN_KILLSWITCH": str(ks), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "killswitch" in (r.stdout + r.stderr).lower()
```

- [ ] **Step 2: Write runner.sh**

```bash
# ~/corvin_evolution/runner.sh
#!/usr/bin/env bash
# Corvin Evolution runner (E2: dry-run only — 평가·로그, 머지/에이전트 없음)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KS="${CORVIN_KILLSWITCH:-$HOME/.corvin_killswitch}"
if [ -e "$KS" ]; then
  echo "[runner] killswitch present ($KS) — skip"; exit 0
fi
CANDIDATE_REF="${1:-HEAD}"
cd "$HERE"
python3 - "$CANDIDATE_REF" <<'PY'
import json, sys
from pathlib import Path
import yaml
from evolution import merge_gate
HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
cfg = yaml.safe_load(Path("config.yaml").read_text())
base = json.loads(Path("baseline.json").read_text())["oos_edge"] if Path("baseline.json").exists() else 0.0
v = merge_gate.evaluate_candidate(cfg["subject_repo"], sys.argv[1], base, cfg, dry_run=True)
line = f"verdict passed={v.passed} merged={v.merged} reasons={v.reasons}"
print("[runner]", line)
with open("evolution.log", "a") as f:
    f.write(line + "\n")
PY
```

- [ ] **Step 3: chmod + run smoke test**

Run:
```bash
chmod +x ~/corvin_evolution/runner.sh
cd ~/corvin_evolution && python3 -m pytest tests/test_runner_smoke.py -v
```
Expected: PASS (killswitch smoke). runner.sh가 `CORVIN_KILLSWITCH` env를 존중하도록 위 스크립트의 `KS=` 라인이 처리함.

- [ ] **Step 4: baseline 생성 + dry-run 1회**

Run:
```bash
cd ~/corvin_evolution && python3 -m evolution.snapshot_baseline && bash runner.sh HEAD
cat evolution.log
```
Expected: baseline.json 생성, evolution.log에 `verdict passed=... merged=False` 1줄. (HEAD==main이라 엣지 동일 → 엣지부족으로 passed=False 예상 — 정상. 게이트가 "개선 없음"을 거부하는 것 확인.)

- [ ] **Step 5: Commit**

```bash
cd ~/corvin_evolution && git add -A && git commit -q -m "feat(evolution): baseline 스냅샷 + runner(킬스위치·dry-run) — E2E 골격"
```

---

## Task 7: 드라이런 판정 증명 (좋은/나쁜/위험 후보)

**Files:**
- Test: `~/corvin_evolution/tests/test_dryrun_proof.py`

게이트가 실제로 **올바르게 판정**함을 통합 증명. subject 레포에 임시 후보 브랜치를 만들어(테스트 내 git 조작, tmp) 4종을 검증. git 비용이 크면 `decide()` 레벨 통합으로 대체 가능(아래는 decide 통합 — 결정론·고속).

- [ ] **Step 1: Write the proof test**

```python
# ~/corvin_evolution/tests/test_dryrun_proof.py
from evolution.merge_gate import decide
from evolution.types import EdgeResult

CFG = {"epsilon": 0.02, "overfit_gap_threshold": 0.15,
       "frozen_paths": ["corvin_jarvis/channels.py"]}
BASE = 0.50
GOOD = EdgeResult(oos_edge=0.56, is_edge=0.59, window_id="w")


def test_good_candidate_would_pass():
    v = decide(diff="+ better_threshold = 0.7", changed_files=["corvin_jarvis/predictive_engine.py"],
               baseline_edge=BASE, cand_edge=GOOD, tests_passed=True, cfg=CFG, dry_run=True)
    assert v.passed and not v.merged   # 통과하나 드라이런이라 머지 안 함


def test_unsafe_order_rejected_even_if_edge_great():
    v = decide(diff="+ broker.submit_order(sym, qty)", changed_files=["corvin_jarvis/x.py"],
               baseline_edge=BASE, cand_edge=GOOD, tests_passed=True, cfg=CFG, dry_run=True)
    assert not v.passed and any("INV-1" in r for r in v.reasons)


def test_killswitch_tamper_rejected():
    v = decide(diff='+ os.remove(".corvin_killswitch")', changed_files=["corvin_jarvis/x.py"],
               baseline_edge=BASE, cand_edge=GOOD, tests_passed=True, cfg=CFG, dry_run=True)
    assert not v.passed and any("INV-2" in r for r in v.reasons)


def test_frozen_path_rejected():
    v = decide(diff="+ noop", changed_files=["corvin_jarvis/channels.py"],
               baseline_edge=BASE, cand_edge=GOOD, tests_passed=True, cfg=CFG, dry_run=True)
    assert not v.passed and any("INV-3" in r for r in v.reasons)
```

- [ ] **Step 2: Run** — `cd ~/corvin_evolution && python3 -m pytest tests/test_dryrun_proof.py -v` → PASS (4). 이로써 게이트가 *좋은 후보 통과 / 위험·약한 후보 거부*를 결정론적으로 함을 증명.

- [ ] **Step 3: 전체 supervisor 스위트**

Run: `cd ~/corvin_evolution && python3 -m pytest -q`
Expected: 전 테스트 PASS (types·pinned_backtest·safety×2·merge_gate·runner_smoke·dryrun_proof).

- [ ] **Step 4: Commit**

```bash
cd ~/corvin_evolution && git add -A && git commit -q -m "test(evolution): 드라이런 판정 증명 — 좋은통과/위험·약한거부"
```

---

## Task 8: README + 운영 문서

**Files:**
- Create: `~/corvin_evolution/README.md`

내용: 아키텍처(supervisor vs subject 분리), 드라이런 운영법, 킬스위치(`touch ~/.corvin_killswitch`), baseline 갱신, E3/E4 로드맵, **E4 전까지 dry_run 고정** 경고.

- [ ] **Step 1: Write README** (아키텍처·명령·안전 불변식·E3/E4 예고 포함. spec §1·§4 요약 인용.)

- [ ] **Step 2: Commit**

```bash
cd ~/corvin_evolution && git add -A && git commit -q -m "docs(evolution): supervisor README — 운영·킬스위치·E3/E4 로드맵"
```

---

## Self-Review

**Spec 커버리지 (§2 Loop B · §1 안전 척추):**
- 외부 결정론 머지하니스 → Task 5 ✓ / pinned 백테스트(자기채점·번들·후보 분류기만 import) → Task 2 ✓
- baseline → Task 6 ✓ / 킬스위치(레포 밖) → Task 6 ✓
- INV-1/2/3 정적검사 → Task 3·4 ✓ / dry-run(머지 안 함) → Task 5·6·7 ✓
- supervisor=별도 repo(루프 불가침) → Task 1 ✓
- **에이전트 dispatch·자동머지는 의도적으로 E3/E4** (본 plan 범위 밖, README 명시).

**Placeholder scan:** 없음. 단 Task 2 Step 2(fixture 데이터 작성)·Step 5(분류기 시그니처 정합)는 실 subject API 확인이 필요한 통합 작업 — E1의 통합 task와 동일 성격(의도적 수동 검증).

**Type consistency:** `EdgeResult`(oos_edge·is_edge·window_id·overfit_gap)·`Violation`(code·detail)·`Verdict`(passed·reasons·dry_run·merged)가 Task 1 정의와 Task 2·3·4·5·6·7 사용에서 일치. `check_no_trade`·`check_no_self_disable`·`check_scope`·`check_all`·`decide`·`evaluate_candidate`·`evaluate` 시그니처 전 task 일관.

**알려진 통합 리스크 (구현 중 확인):**
- `early_warning.default_config`/`ew_runner._classify_all` 시그니처 — Task 2 Step 5에서 grep 후 정합. fixture readings 키도 분류기 입력과 일치시켜야 함.
- runner.sh의 `CORVIN_KILLSWITCH` env 처리 — smoke test가 강제. 실 cron에선 기본 `~/.corvin_killswitch`.
- pytest 전체 실행(~40s)이 게이트마다 돌아 느림 — E4에서 영향 테스트만 선별 실행으로 최적화 가능(현재는 안전 우선 전체).

---

**다음 Phase**: E3(in-repo 진화 대상 `corvin_jarvis/evolution/` objective·harness·ledger + 자가수정 에이전트 dispatch) → E4(드라이런 N회 안전 확인 후 `dry_run=False` 실머지 활성화). E2는 *심판*만 완성하며 아무것도 자동 머지하지 않는다.
