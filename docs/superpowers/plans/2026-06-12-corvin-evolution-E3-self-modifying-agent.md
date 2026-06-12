# Corvin Evolution Loop — Phase E3: 자가수정 에이전트 (드라이런) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 격리된 git worktree에서 `claude -p` 헤드리스 에이전트가 subject의 시그널 코드를 자율 수정→테스트→실험하고, E2의 외부 결정론 심판이 그 후보를 **드라이런으로 평가**해 "통과할 뻔한 후보"를 폐하 리뷰용으로 모은다. **여전히 머지는 안 한다** (실머지는 E4).

**Architecture:** subject(`quant_investment_system_v2`)에 진화 대상 `corvin_jarvis/evolution/`(objective·실험 harness·세대 ledger)를 두고, supervisor(`~/corvin_evolution/`)에 dispatch 루프를 추가한다. 루프: 킬스위치 체크 → subject worktree 생성 → `claude -p`로 자가수정 에이전트 dispatch(objective + INV 제약을 시스템 프롬프트로 주입) → E2 `merge_gate.evaluate_candidate(worktree, dry_run=True)` → verdict 로그 + would-pass 후보를 `candidates/`에 저장 → worktree 정리 → cap까지 반복.

**Tech Stack:** Python 3.11, pytest, `claude -p` (Claude Code 헤드리스), git worktree, 기존 E2 supervisor·`ew_backtest`.

**선행:** [Evolution spec](../specs/2026-06-12-corvin-evolution-loop-design.md) §6 · [E2 plan](./2026-06-12-corvin-evolution-E2-infra-dryrun.md)(완료). E2 supervisor가 `~/corvin_evolution/`에 존재.

---

## 핵심 설계 결정 (구현 전 폐하 리뷰)

1. **에이전트 dispatch = `claude -p` 헤드리스** — worktree에서 비대화식 실행. INV 제약·objective는 `--append-system-prompt`로 주입. 에이전트는 **subject worktree만** 보고 `~/corvin_evolution/`(심판)엔 접근 불가.
2. **E3도 드라이런 고정** — gate는 평가·로그만, **절대 머지 안 함**. would-pass 후보는 `~/corvin_evolution/candidates/<ts>/`에 patch로 저장 → 폐하가 사람 눈으로 검토. (E2 심판이 발견한 "단일 fixture 암기" 취약점 때문에, 자동 반영은 홀드아웃 데이터 갖춘 E4 전까지 금지.)
3. **보수적 cap** — 1회 야간 실행당 후보 N개(기본 2)·에이전트 토큰 budget·worktree 타임아웃. 비용 폭주·무한루프 방지.
4. **자가수정 범위** — objective가 시그널 품질(early_warning 임계·predictive_engine 로직 등)로 유도. ⓒ(레포 전체)는 허용하나 INV-3 frozen_paths·scope 검사가 가드. 에이전트는 *심판·킬스위치 못 건드림*(레포 밖이라 worktree에 없음).

---

## File Structure

| 파일 | 위치 | 책임 |
|------|------|------|
| `corvin_jarvis/evolution/__init__.py` | subject | 패키지 |
| `corvin_jarvis/evolution/objective.md` | subject | 자가수정 에이전트에 주는 목표·제약·금지(INV) 프롬프트 |
| `corvin_jarvis/evolution/harness.py` | subject | 에이전트의 실험용 인-레포 백테스트(`ew_backtest` 래핑) + JSON 출력 |
| `corvin_jarvis/evolution/ledger.py` | subject | 세대 ledger (`state/evolution_ledger.db`): 변경요약·edge·verdict |
| `evolution/agent_dispatch.py` | supervisor | `claude -p` worktree dispatch (injectable, 캡·타임아웃) |
| `evolution/evolve_loop.py` | supervisor | 전체 루프 오케스트레이션 (killswitch→worktree→dispatch→gate→save→cleanup) |
| `evolve.sh` | supervisor | cron 진입점 (evolve_loop 호출) |
| `candidates/` | supervisor | would-pass 후보 patch 저장 (폐하 리뷰) |

---

## Task 1: in-repo 진화 대상 — objective + 패키지 (subject repo)

**Files (subject `quant_investment_system_v2`, on a feature branch):**
- Create: `corvin_jarvis/evolution/__init__.py`, `corvin_jarvis/evolution/objective.md`
- Test: `tests/test_evolution_objective.py`

`objective.md`는 에이전트가 받는 프롬프트. 반드시 포함: 목표(walk-forward OOS 엣지 개선), 측정법(supervisor가 외부에서 채점하니 fixture 암기/하드코딩 금지 명시), 절대금지(INV-1 주문/portfolio 쓰기, INV-2 킬스위치·supervisor, INV-3 frozen/scope), 작업법(TDD·기존 773 테스트 유지).

- [ ] **Step 1: 브랜치 + 실패 테스트**

Run: `cd /Users/thethethe/Claude/quant_investment_system_v2 && git checkout main && git checkout -b feat/corvin-evolution-e3`

```python
# tests/test_evolution_objective.py
from pathlib import Path

OBJ = Path("corvin_jarvis/evolution/objective.md")


def test_objective_states_constraints():
    txt = OBJ.read_text(encoding="utf-8")
    # 핵심 제약이 프롬프트에 명시돼야 함 (에이전트가 반드시 보게)
    for must in ["OOS", "INV-1", "INV-2", "INV-3", "fixture", "773", "포트폴리오"]:
        assert must in txt, f"objective.md missing: {must}"
```

- [ ] **Step 2: Run → FAIL** (`FileNotFoundError`). Run: `python3 -m pytest tests/test_evolution_objective.py -v`

- [ ] **Step 3: Implement**

```python
# corvin_jarvis/evolution/__init__.py
"""Corvin 자가진화 대상 — 에이전트가 수정하는 시그널 코드 (Phase E3)."""
```

```markdown
<!-- corvin_jarvis/evolution/objective.md -->
# 목표
이 레포(quant_investment_system_v2)의 주식 시그널 예측 정확도를 개선하라.
구체적으로: walk-forward 백테스트의 **OOS(out-of-sample) 엣지**를 baseline 대비 높여라.

# 측정 방법 (중요)
너의 변경은 레포 **밖의 독립 심판**이 채점한다. 심판은 자기 소유의 고정 데이터와
채점 코드를 쓰며, 너의 분류기/시그널 함수만 호출한다. 따라서:
- 특정 날짜에 정답을 **하드코딩하거나 fixture를 암기**하면 진짜 개선이 아니다 (E4의
  홀드아웃 데이터에서 즉시 들통난다). 일반화되는 진짜 신호 개선만 가치 있다.
- `corvin_jarvis/evolution/harness.py`로 네 변경을 미리 실험해볼 수 있다.

# 절대 금지 (위반 시 자동 거부)
- INV-1: 실매매/주문 실행 코드(`*_order`, KIS 주문 TR), `portfolio.json` 쓰기 추가 금지.
- INV-2: 킬스위치(`~/.corvin_killswitch`)·supervisor(`corvin_evolution`) 참조/수정 금지.
- INV-3: 이 레포 밖 파일·동결 경로(`corvin_jarvis/channels.py`) 수정 금지.

# 작업 방식
- TDD. 기존 773개 테스트를 전부 통과 유지하라.
- 작게, 한 가설씩. 변경 이유를 커밋 메시지에 남겨라.
- 대상 예시: `early_warning` 임계값, `predictive_engine` 로직, `regime` 게이팅, 시그널 가중치.
```

- [ ] **Step 4: Run → PASS.** `python3 -m pytest tests/test_evolution_objective.py -v`

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/evolution/__init__.py corvin_jarvis/evolution/objective.md tests/test_evolution_objective.py
git commit -m "feat(evolution): E3 자가수정 에이전트 objective 프롬프트 + 제약"
```

---

## Task 2: in-repo 실험 harness (subject)

**Files:**
- Create: `corvin_jarvis/evolution/harness.py`
- Test: `tests/test_evolution_harness.py`

에이전트가 자기 변경을 실험하는 인-레포 백테스트. `ew_backtest.run_backtest(history, cfg, horizon, thresh_pct)` 래핑 + JSON 출력. (심판의 pinned 평가자와 별개 — 이건 에이전트 도구일 뿐, 머지 판정엔 안 쓰임.)

- [ ] **Step 1: 실패 테스트**

```python
# tests/test_evolution_harness.py
from corvin_jarvis.evolution.harness import run_experiment


def test_run_experiment_returns_metrics():
    history = [
        {"date": f"2025-01-{i:02d}",
         "readings": {"semis": {"ratio": 1.0, "ratio_ma50": 1.0, "slope_5d": 0.0,
                                "spx_dist_from_high_pct": 0.0},
                      "vix_term": {"ratio": 1.0}, "breadth": {"pct_above_ma200": 60.0},
                      "hy": {"value": 4.0, "chg_5d": 0.0}, "curve": {"value": 1.0, "chg_5d": 0.0}},
         "spx": 5000.0 - i * 5}
        for i in range(1, 21)
    ]
    out = run_experiment(history)
    assert "metrics" in out and isinstance(out["metrics"], dict)
```

- [ ] **Step 2: Run → FAIL**. `python3 -m pytest tests/test_evolution_harness.py -v`

- [ ] **Step 3: Implement**

```python
# corvin_jarvis/evolution/harness.py
"""에이전트의 실험용 인-레포 백테스트. ew_backtest 래핑 (심판 아님 — 도구)."""
from __future__ import annotations

import json
from typing import Any

from corvin_jarvis import ew_backtest


def run_experiment(history: list[dict[str, Any]], *, cfg: dict[str, Any] | None = None,
                   horizon: int = 5, thresh_pct: float = -3.0) -> dict[str, Any]:
    metrics = ew_backtest.run_backtest(history, cfg or {}, horizon, thresh_pct)
    return {"metrics": metrics, "n_days": len(history)}


def main() -> None:  # CLI: python -m corvin_jarvis.evolution.harness <history.json>
    import sys
    history = json.loads(open(sys.argv[1]).read()) if len(sys.argv) > 1 else []
    print(json.dumps(run_experiment(history), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS** (run_backtest cfg 빈dict로 KeyError 나면 테스트 history의 readings/cfg를 실제 `_classify_all` 기대키에 맞춰 보정 — `ew_backtest` 사용처 참조).

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/evolution/harness.py tests/test_evolution_harness.py
git commit -m "feat(evolution): in-repo 실험 harness (ew_backtest 래핑)"
```

---

## Task 3: 세대 ledger (subject)

**Files:**
- Create: `corvin_jarvis/evolution/ledger.py`
- Test: `tests/test_evolution_ledger.py`

세대별 시도 기록(`state/evolution_ledger.db` SQLite): ts·branch·summary·oos_edge·is_edge·verdict_passed·reasons. 기존 `signals/ledger.py` 패턴 따름.

- [ ] **Step 1: 실패 테스트**

```python
# tests/test_evolution_ledger.py
from corvin_jarvis.evolution import ledger


def test_record_and_fetch(tmp_path):
    db = tmp_path / "evo.db"
    ledger.record(db, branch="cand-1", summary="lower vix thresh",
                  oos_edge=0.56, is_edge=0.59, passed=True, reasons=[])
    rows = ledger.recent(db, limit=5)
    assert rows[0]["branch"] == "cand-1" and rows[0]["passed"] == 1
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
# corvin_jarvis/evolution/ledger.py
"""자가진화 세대 ledger — 시도·엣지·verdict 기록 (SQLite)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """CREATE TABLE IF NOT EXISTS evolution_ledger (
  id INTEGER PRIMARY KEY, ts TEXT, branch TEXT, summary TEXT,
  oos_edge REAL, is_edge REAL, passed INTEGER, reasons TEXT)"""


def _conn(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    c.execute(_SCHEMA)
    return c


def record(db: Path, *, branch: str, summary: str, oos_edge: float,
           is_edge: float, passed: bool, reasons: list[str], ts: str = "") -> None:
    with _conn(db) as c:
        c.execute("INSERT INTO evolution_ledger"
                  "(ts,branch,summary,oos_edge,is_edge,passed,reasons)"
                  " VALUES(?,?,?,?,?,?,?)",
                  (ts, branch, summary, oos_edge, is_edge, int(passed),
                   json.dumps(reasons, ensure_ascii=False)))


def recent(db: Path, limit: int = 20) -> list[dict[str, Any]]:
    with _conn(db) as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM evolution_ledger ORDER BY id DESC LIMIT ?", (limit,))]
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/evolution/ledger.py tests/test_evolution_ledger.py
git commit -m "feat(evolution): 세대 ledger (SQLite) — 시도·엣지·verdict 기록"
```

- [ ] **Step 6: subject 브랜치 머지 게이트** — 전체 스위트 `python3 -m pytest -q`(기존 797 + 신규, 사전실패 1건 제외 그린) 확인 후 **폐하 승인 받아** main 머지(자가통과 금지). 이 in-repo 부분은 정상 코드라 평소 워크플로로 머지.

---

## Task 4: 에이전트 dispatch (supervisor) — `claude -p` worktree

**Files (supervisor `~/corvin_evolution/`):**
- Create: `evolution/agent_dispatch.py`
- Test: `~/corvin_evolution/tests/test_agent_dispatch.py`

`dispatch_agent(subject_repo, worktree_path, branch, objective_text, *, runner=subprocess.run, token_cap, timeout_s) -> DispatchResult`. `runner`를 주입해 테스트는 실제 claude 호출 없이 검증. 실제 호출은:
```
claude -p "<objective>" --append-system-prompt "<INV 제약>" --allow-dangerously-skip-permissions
```
worktree 디렉토리를 cwd로. **에이전트는 worktree만 봄** (supervisor 경로 미포함).

- [ ] **Step 1: 실패 테스트 (runner 주입)**

```python
# ~/corvin_evolution/tests/test_agent_dispatch.py
from evolution.agent_dispatch import dispatch_agent, DispatchResult


def test_dispatch_builds_claude_command_and_isolates(tmp_path):
    calls = {}
    def fake_runner(cmd, **kw):
        calls["cmd"] = cmd; calls["cwd"] = kw.get("cwd")
        class R: returncode = 0; stdout = "done"; stderr = ""
        return R()
    res = dispatch_agent("/subj", str(tmp_path), "cand-1", "improve OOS edge",
                         runner=fake_runner, token_cap=50000, timeout_s=600)
    assert isinstance(res, DispatchResult) and res.ok
    assert calls["cmd"][0] == "claude" and "-p" in calls["cmd"]
    assert calls["cwd"] == str(tmp_path)         # worktree 격리
    assert "/subj" not in " ".join(calls["cmd"])  # supervisor 경로 노출 안 함
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
# ~/corvin_evolution/evolution/agent_dispatch.py
"""자가수정 에이전트 dispatch — claude -p 헤드리스를 worktree에서 실행."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

_INV = ("제약(위반=자동거부): INV-1 주문/portfolio.json 쓰기 금지; "
        "INV-2 킬스위치·corvin_evolution 참조 금지; "
        "INV-3 이 레포 밖·channels.py 수정 금지. TDD로 기존 테스트 유지.")


@dataclass(frozen=True)
class DispatchResult:
    ok: bool
    stdout: str
    stderr: str


def dispatch_agent(subject_repo: str, worktree_path: str, branch: str,
                   objective_text: str, *, runner: Callable = subprocess.run,
                   token_cap: int = 80000, timeout_s: int = 900) -> DispatchResult:
    cmd = ["claude", "-p", objective_text,
           "--append-system-prompt", _INV,
           "--allow-dangerously-skip-permissions"]
    try:
        r = runner(cmd, cwd=worktree_path, capture_output=True, text=True,
                   timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return DispatchResult(ok=False, stdout="", stderr="timeout")
    return DispatchResult(ok=(r.returncode == 0), stdout=r.stdout, stderr=r.stderr)
```

- [ ] **Step 4: Run → PASS**. `cd ~/corvin_evolution && python3 -m pytest tests/test_agent_dispatch.py -v`

- [ ] **Step 5: Commit** (`cd ~/corvin_evolution && git add -A && git commit -m "feat(evolution): 자가수정 에이전트 dispatch (claude -p worktree, injectable)"`)

---

## Task 5: 진화 루프 오케스트레이션 (supervisor)

**Files:**
- Create: `~/corvin_evolution/evolution/evolve_loop.py`
- Test: `~/corvin_evolution/tests/test_evolve_loop.py`

`run_once(cfg, *, dispatch=..., gate=..., git=...) -> LoopResult`: 의존성 주입으로 테스트. 실 흐름:
1. worktree 생성: `git -C subject worktree add <wt> -b cand-<ts> main`
2. `dispatch_agent(subject, wt, branch, objective)` — 에이전트가 wt에서 코드 수정·커밋
3. `merge_gate.evaluate_candidate(subject, branch, baseline, cfg, dry_run=True)` (※ E3 dry_run **고정**)
4. ledger 기록 + verdict 로그. passed면 `git -C wt diff main > candidates/<ts>.patch` 저장(폐하 리뷰).
5. worktree 제거: `git -C subject worktree remove <wt> --force`. **머지 절대 안 함.**

- [ ] **Step 1: 실패 테스트 (전부 주입)**

```python
# ~/corvin_evolution/tests/test_evolve_loop.py
from evolution.evolve_loop import run_once
from evolution.types import Verdict, EdgeResult


def test_loop_dryrun_never_merges_and_saves_passing_candidate(tmp_path):
    events = []
    fake_git = lambda *a, **k: events.append(("git", a))
    fake_dispatch = lambda *a, **k: events.append(("dispatch", a)) or type("R", (), {"ok": True})()
    fake_gate = lambda *a, **k: Verdict(passed=True, reasons=[], dry_run=True, merged=False)
    res = run_once({"subject_repo": "/subj", "candidates_dir": str(tmp_path)},
                   dispatch=fake_dispatch, gate=fake_gate, git=fake_git, ts="T1")
    assert res.merged is False                       # 절대 머지 안 함
    assert res.passed is True
    # git 호출에 'merge'가 절대 없어야
    assert not any("merge" in str(a) for _, a in events)
    # worktree add/remove는 있어야
    joined = " ".join(str(a) for _, a in events)
    assert "worktree" in joined
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement** (dry_run 고정, git 'merge' 호출 없음)

```python
# ~/corvin_evolution/evolution/evolve_loop.py
"""진화 루프 (E3 드라이런) — worktree→dispatch→gate→save. 절대 머지 안 함."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class LoopResult:
    branch: str
    passed: bool
    merged: bool
    reasons: list[str]


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True, text=True)


def run_once(cfg: dict[str, Any], *, dispatch: Callable, gate: Callable,
             git: Callable = _git, ts: str = "") -> LoopResult:
    subject = cfg["subject_repo"]
    branch = f"cand-{ts}"
    wt = str(Path(cfg.get("worktree_root", "/tmp")) / branch)
    git("-C", subject, "worktree", "add", wt, "-b", branch, "main")
    try:
        dispatch(subject, wt, branch, cfg.get("objective", "improve OOS edge"))
        v = gate(subject, branch, cfg.get("baseline_edge", 0.0), cfg, dry_run=True)
        if v.passed:
            cand_dir = Path(cfg["candidates_dir"]); cand_dir.mkdir(exist_ok=True)
            diff = subprocess.run(["git", "-C", wt, "diff", "main"],
                                  capture_output=True, text=True).stdout
            (cand_dir / f"{ts}.patch").write_text(diff, encoding="utf-8")
        return LoopResult(branch=branch, passed=v.passed, merged=False,
                          reasons=list(v.reasons))
    finally:
        git("-C", subject, "worktree", "remove", wt, "--force")
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit** (`feat(evolution): 진화 루프 오케스트레이션 (dry-run, worktree, 후보 저장)`)

---

## Task 6: evolve.sh 진입점 + 실 dispatch 스모크 (supervisor)

**Files:**
- Create: `~/corvin_evolution/evolve.sh`
- Test: 수동 스모크 (실제 claude 1회)

`evolve.sh`: 킬스위치 → cap만큼 `run_once` 반복(실 dispatch/gate/git) → evolution.log + 텔레그램 요약(있으면). **머지 없음.**

- [ ] **Step 1: evolve.sh 작성**

```bash
# ~/corvin_evolution/evolve.sh
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
KS="${CORVIN_KILLSWITCH:-$HOME/.corvin_killswitch}"
[ -e "$KS" ] && { echo "[evolve] killswitch — skip"; exit 0; }
python3 -m evolution.evolve_run    # evolve_run.py: cfg 로드 + cap만큼 run_once 실머지없이 반복
```

- [ ] **Step 2: `evolution/evolve_run.py`** — config.yaml 로드, baseline.json 읽기, `caps.max_iterations`만큼 `run_once`(실 dispatch=agent_dispatch.dispatch_agent, gate=merge_gate.evaluate_candidate) 호출, 결과 evolution.log append. ts는 호출 시각 문자열(스크립트가 `date`로 주입 — `run_once`는 Date 안 씀).

- [ ] **Step 3: 실 스모크 (폐하 승인 후, 비용 발생)** — `chmod +x evolve.sh && CORVIN_MAX_ITER=1 bash evolve.sh` 1회. 확인: worktree 생겼다 사라짐, 에이전트가 실제로 뭔가 수정 시도, gate가 dry-run verdict 로그, **main 무변경**(`git -C <subject> log --oneline -1` 그대로), candidates/에 patch(통과 시).

- [ ] **Step 4: Commit** (`feat(evolution): evolve.sh 진입점 + evolve_run (dry-run 루프)`)

---

## Task 7: 후보 리뷰 요약 + 안전 검증 (supervisor)

**Files:**
- Create: `~/corvin_evolution/evolution/review_digest.py`
- Test: `~/corvin_evolution/tests/test_review_digest.py`

`build_digest(ledger_rows, candidates_dir) -> str`: 최근 세대 요약(통과/거부 수, would-pass 후보 목록·patch 경로·엣지). 폐하가 사람 눈으로 검토 후 수동 적용 판단. (텔레그램 발송은 기존 채널 재사용 가능하나 선택.)

- [ ] **Step 1: 실패 테스트**

```python
# ~/corvin_evolution/tests/test_review_digest.py
from evolution.review_digest import build_digest


def test_digest_lists_passing_candidates():
    rows = [{"ts": "T1", "branch": "cand-T1", "summary": "lower vix",
             "oos_edge": 0.57, "passed": 1, "reasons": "[]"},
            {"ts": "T2", "branch": "cand-T2", "summary": "bad", "oos_edge": 0.40,
             "passed": 0, "reasons": '["엣지 부족"]'}]
    out = build_digest(rows, candidates_dir="/c")
    assert "cand-T1" in out and "0.57" in out
    assert "통과" in out or "would-pass" in out.lower()
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
# ~/corvin_evolution/evolution/review_digest.py
"""would-pass 후보 사람-리뷰 요약 (E3: 자동적용 없음, 폐하 수동 판단)."""
from __future__ import annotations

from typing import Any


def build_digest(ledger_rows: list[dict[str, Any]], candidates_dir: str) -> str:
    passing = [r for r in ledger_rows if r.get("passed")]
    lines = [f"🧬 진화 세대 요약 — 통과(would-pass) {len(passing)} / 총 {len(ledger_rows)}",
             "※ E3 드라이런: 자동 반영 안 됨. patch 검토 후 수동 적용 판단.", ""]
    for r in passing:
        lines.append(f"✅ {r['branch']} — {r.get('summary','')} "
                     f"(OOS {r.get('oos_edge')}) → {candidates_dir}/{r['ts']}.patch")
    if not passing:
        lines.append("이번 회차 would-pass 후보 없음 (심판이 전부 거부 — 정상).")
    return "\n".join(lines)
```

- [ ] **Step 4: Run → PASS** · **Step 5: Commit** (`feat(evolution): 후보 리뷰 요약 digest`)

- [ ] **Step 6: supervisor 전체 스위트** `cd ~/corvin_evolution && python3 -m pytest -q` → 전 그린.

---

## Self-Review

**Spec §6 E3 커버리지:** in-repo objective·harness·ledger → Task 1·2·3 ✓ / 자가수정 에이전트 dispatch → Task 4 ✓ / 루프(worktree·dry-run) → Task 5·6 ✓ / **여전히 dry_run, 머지 없음** → Task 5(merged=False 고정)·6(스모크 main무변경) ✓ / would-pass 후보 사람 리뷰 → Task 5(patch 저장)·7(digest) ✓.

**Placeholder scan:** 없음. Task 6 Step 2(evolve_run.py)는 조립 코드라 단계 내 명세대로, Step 3은 실 claude 호출이라 의도적 수동 스모크(비용·승인). Task 2 Step 4·harness cfg 키 정합은 통합 확인.

**Type consistency:** `DispatchResult`(ok·stdout·stderr)·`LoopResult`(branch·passed·merged·reasons)·E2의 `Verdict`(passed·reasons·dry_run·merged) 일관. `dispatch_agent`·`run_once`·`build_digest`·`record`/`recent` 시그니처 task 간 일치.

**안전 불변식 재확인:** ① E3도 `dry_run=True` 고정 — 어떤 후보도 자동 머지 안 됨 (Task 5 test가 'merge' 호출 부재 강제). ② 에이전트는 worktree만 봄, supervisor·킬스위치 미접근 (Task 4 격리 test). ③ would-pass도 **patch로 저장만**, 적용은 폐하 수동. ④ E2 심판이 발견한 "fixture 암기" 취약점 → E3는 자동적용 안 하므로 무해, E4 전 홀드아웃 데이터 필수(objective.md·README 명시).

**알려진 리스크 (구현/운영 중):**
- `claude -p --allow-dangerously-skip-permissions`를 cron에서 자율 실행 = 비용·예측불가 행동. cap(max_iterations·timeout)·killswitch가 1차 방어. **첫 가동은 폐하 입회 수동 1회**(Task 6 Step 3) 권장, cron 등록은 그 후.
- 에이전트가 fixture를 못 보게: worktree에 supervisor fixture 없음(레포 분리)으로 자연 차단. 단 에이전트가 in-repo harness로 자기 실험은 가능(정상).
- E4 진입 조건: 드라이런 N회에서 would-pass 후보가 *사람 검토 시 실제로 타당*함이 반복 확인 + 홀드아웃 데이터 구축 후.

---

**다음 Phase E4**: 홀드아웃/회전 데이터로 fixture-암기 취약점 차단 → 드라이런 신뢰 누적 후 `dry_run=False` 실머지 활성화(budget cap 보수적). E3는 *에이전트가 돌되 아무것도 자동 반영 안 함* — 폐하가 patch를 사람 눈으로 보는 단계.
