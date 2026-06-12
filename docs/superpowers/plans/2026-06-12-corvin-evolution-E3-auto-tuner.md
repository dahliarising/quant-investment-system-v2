# Corvin Evolution Loop — Phase E3 (무료): 자동 파라미터 튜너 (드라이런) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** **API 0원**으로 시그널 설정값(early_warning 임계값 등) 공간을 자동 탐색해, walk-forward 백테스트 점수가 baseline보다 높고 과적합 안 한 설정 후보를 찾아 폐하 리뷰용으로 저장한다. LLM·worktree·`claude` 호출 없음 — 순수 로컬 계산. **자동 반영은 안 함** (실머지는 E4).

**Architecture:** supervisor(`~/corvin_evolution/`)에 결정론적 **옵티마이저**를 추가. 흐름: 킬스위치 체크 → 파라미터 공간에서 후보 설정 N개 샘플(random search + coordinate descent) → 각 후보를 E2 pinned 백테스트로 **train(최적화)·OOS(검증)** 채점 → baseline+ε·과적합 게이트 통과한 최고 후보를 `candidates/<ts>.json`에 저장 + ledger 기록. 비용=전기세만.

**Tech Stack:** Python 3.11, pytest, 기존 E2 supervisor(pinned_backtest·merge_gate.decide). `claude`/worktree/네트워크 안 씀. `random.Random(seed)`로 재현성.

**선행:** [Evolution spec](../specs/2026-06-12-corvin-evolution-loop-design.md) · [E2 plan](./2026-06-12-corvin-evolution-E2-infra-dryrun.md)(완료, supervisor 존재). 이 문서가 [E3 LLM-agent](./2026-06-12-corvin-evolution-E3-self-modifying-agent.md)를 **대체**(API 비용 회피).

---

## 핵심 설계 결정

1. **API 0원** — 자가수정 주체가 LLM이 아니라 **결정론적 옵티마이저**. 후보는 *코드*가 아니라 *설정 딕셔너리*(임계값). 따라서 worktree·claude·코드 diff 불필요.
2. **튜닝 대상** = `corvin_jarvis/config.json["early_warning"]`의 숫자 임계값 (vix_term·breadth·hy·curve·semis). 화이트리스트 키만 — 임의 키 추가 금지(안전).
3. **과적합 방어** — 옵티마이저는 **train 구간 edge로만 후보 선택**, 심판은 **OOS 구간으로 검증**(옵티마이저가 직접 최대화 안 한 데이터). 과적합 갭(is−oos) 게이트 추가. ⚠️단일 fixture라 완벽치 않음 → E4에서 **홀드아웃/회전 데이터** 필수(이 한계 명시).
4. **E3도 드라이런** — 통과 후보는 `candidates/`에 설정 json으로 저장만. 폐하가 사람 눈으로 보고 수동 적용 판단. 자동 반영 없음.

---

## File Structure (전부 supervisor `~/corvin_evolution/`)

| 파일 | 책임 |
|------|------|
| `evolution/pinned_backtest.py` (수정) | `evaluate(subject_repo, config_override=None)` — 임의 후보 설정 채점 |
| `evolution/search_space.py` (신규) | 튜너블 파라미터 + 탐색 범위 정의 (화이트리스트) |
| `evolution/optimizer.py` (신규) | random search + coordinate descent, walk-forward 채점 → 최고 (config, EdgeResult) |
| `evolution/tuner.py` (신규) | 후보 게이트(decide)·저장·ledger 기록 오케스트레이션 |
| `evolution/ledger.py` (신규) | 세대 ledger (SQLite, supervisor 쪽) |
| `tune.sh` + `evolution/tune_run.py` (신규) | cron 진입점 — 킬스위치→옵티마이저→저장→digest (API 0) |
| `evolution/review_digest.py` (신규) | would-apply 후보 사람-리뷰 요약 |
| `candidates/` | 통과 후보 설정 json (폐하 리뷰·수동 적용) |

---

## Task 1: pinned_backtest 설정 오버라이드 (게이밍 없이 임의 후보 채점)

**Files:**
- Modify: `~/corvin_evolution/evolution/pinned_backtest.py`
- Test: `~/corvin_evolution/tests/test_pinned_config_override.py`

현재 `_make_classifier(subject_repo)`는 subject의 `config.json["early_warning"]`만 로드. 옵티마이저가 **후보 설정**을 채점하려면 override가 필요. `evaluate(subject_repo, *, config_override=None, fixture=..., train_ratio=...)` 추가 — override 있으면 그 설정으로 분류기 생성. 라벨링·채점·fixture는 그대로 supervisor 소유(게임내성 유지).

- [ ] **Step 1: 실패 테스트**

```python
# ~/corvin_evolution/tests/test_pinned_config_override.py
from evolution.pinned_backtest import evaluate

SUBJ = "/Users/thethethe/Claude/quant_investment_system_v2"


def test_override_changes_edge():
    base = evaluate(SUBJ)
    # 극단 설정(전부 즉시 red) override → edge가 baseline과 달라야 함(채점이 설정을 반영)
    override = {"semis": {"divergence_high_dist_pct": 0.0},
                "vix_term": {"green_max": 0.0, "amber_max": 0.0},
                "breadth": {"green_min": 100.0, "amber_min": 100.0},
                "hy": {"green_max": 0.0, "amber_max": 0.0, "rise_amber_5d": 0.0, "rise_red_5d": 0.0},
                "curve": {"green_min": 100.0, "amber_min": 100.0, "fast_move_5d": 0.0},
                "hard_stop_pct": -8.0, "enabled": True}
    alt = evaluate(SUBJ, config_override=override)
    assert alt.oos_edge != base.oos_edge or alt.is_edge != base.is_edge
```

- [ ] **Step 2: Run → FAIL** (`evaluate() got unexpected keyword 'config_override'`).

- [ ] **Step 3: Implement** — `_make_classifier(subject_repo, config_override=None)`에서 override 있으면 `cfg = config_override`, 없으면 기존대로 `config.json["early_warning"]`. `evaluate`에 `config_override` 인자 추가해 전달.

```python
# pinned_backtest.py — _make_classifier 수정 + evaluate 시그니처 확장 (핵심만)
def _make_classifier(subject_repo: str, config_override=None):
    if subject_repo not in sys.path:
        sys.path.insert(0, subject_repo)
    from corvin_jarvis.ew_runner import _classify_all
    if config_override is not None:
        cfg = config_override
    else:
        import json as _j
        cfg = _j.loads((Path(subject_repo) / "corvin_jarvis" / "config.json"
                        ).read_text())["early_warning"]
    return lambda readings: _classify_all(readings, cfg)


def evaluate(subject_repo: str, *, config_override=None,
             fixture: str = "fixtures/ew_history.json", train_ratio: float = 0.6) -> EdgeResult:
    history = _load_fixture(fixture)
    classify = _make_classifier(subject_repo, config_override)
    split = int(len(history) * train_ratio)
    train, oos = history[:split], history[split:]
    def edge(seg): return _f1(_signal_dates(seg, classify), _event_dates(seg))
    return EdgeResult(oos_edge=edge(oos), is_edge=edge(train), window_id=f"{train_ratio}")
```

- [ ] **Step 4: Run → PASS.** 기존 `tests/test_pinned_backtest.py`도 회귀 없는지 `cd ~/corvin_evolution && python3 -m pytest tests/test_pinned_backtest.py tests/test_pinned_config_override.py -q`.

- [ ] **Step 5: Commit** (`feat(evolution): pinned_backtest config override — 임의 후보 설정 채점`)

---

## Task 2: 탐색 공간 정의 (화이트리스트)

**Files:**
- Create: `~/corvin_evolution/evolution/search_space.py`
- Test: `~/corvin_evolution/tests/test_search_space.py`

튜너블 키 + 범위. 현재값 중심. `sample(rng) -> config`(전체 설정 dict)와 `neighbors(config) -> list[config]`(coordinate descent용 ±1스텝) 제공. 화이트리스트 외 키는 절대 생성 안 함(안전).

- [ ] **Step 1: 실패 테스트**

```python
# ~/corvin_evolution/tests/test_search_space.py
import random
from evolution.search_space import sample, neighbors, WHITELIST


def test_sample_only_whitelist_keys():
    cfg = sample(random.Random(0))
    for section, params in cfg.items():
        if section in ("hard_stop_pct", "enabled"):
            continue
        assert section in WHITELIST
        for key in params:
            assert (section, key) in WHITELIST[section] or key in ("rise_amber_5d", "rise_red_5d")


def test_sample_in_range_and_deterministic():
    a = sample(random.Random(42))
    b = sample(random.Random(42))
    assert a == b                              # 시드 고정 → 재현
    assert 0.0 <= a["vix_term"]["green_max"] <= 1.5


def test_neighbors_nonempty():
    cfg = sample(random.Random(1))
    assert len(neighbors(cfg)) > 0
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement** — `WHITELIST`: 섹션별 (key, lo, hi, step). 예:

```python
# search_space.py (핵심)
from __future__ import annotations
import random
from typing import Any

# (lo, hi, step) — 현재값 중심 범위
WHITELIST: dict[str, dict[str, tuple[float, float, float]]] = {
    "semis":    {"divergence_high_dist_pct": (1.0, 6.0, 0.5)},
    "vix_term": {"green_max": (0.7, 1.1, 0.05), "amber_max": (0.9, 1.3, 0.05)},
    "breadth":  {"green_min": (45.0, 75.0, 2.5), "amber_min": (25.0, 55.0, 2.5)},
    "hy":       {"green_max": (2.5, 4.5, 0.25), "amber_max": (4.0, 6.0, 0.25),
                 "rise_amber_5d": (0.1, 0.5, 0.05), "rise_red_5d": (0.3, 0.8, 0.05)},
    "curve":    {"green_min": (0.0, 1.0, 0.1), "amber_min": (-0.5, 0.5, 0.1),
                 "fast_move_5d": (0.05, 0.3, 0.025)},
}
_FIXED = {"hard_stop_pct": -8.0, "enabled": True}


def _snap(lo: float, hi: float, step: float, v: float) -> float:
    v = max(lo, min(hi, v))
    return round(lo + round((v - lo) / step) * step, 4)


def sample(rng: random.Random) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for sec, params in WHITELIST.items():
        cfg[sec] = {k: _snap(lo, hi, st, rng.uniform(lo, hi))
                    for k, (lo, hi, st) in params.items()}
    cfg.update(_FIXED)
    return cfg


def neighbors(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sec, params in WHITELIST.items():
        for k, (lo, hi, st) in params.items():
            for d in (-st, st):
                nb = {s: dict(p) for s, p in cfg.items() if isinstance(p, dict)}
                nb.update(_FIXED)
                nb[sec][k] = _snap(lo, hi, st, cfg[sec][k] + d)
                if nb[sec][k] != cfg[sec][k]:
                    out.append(nb)
    return out
```

- [ ] **Step 4: Run → PASS** · **Step 5: Commit** (`feat(evolution): 파라미터 탐색공간 (화이트리스트·range·neighbors)`)

---

## Task 3: 옵티마이저 (random search + coordinate descent, walk-forward)

**Files:**
- Create: `~/corvin_evolution/evolution/optimizer.py`
- Test: `~/corvin_evolution/tests/test_optimizer.py`

`optimize(score_fn, *, n_random, seed, refine_steps) -> OptResult(config, edge)`. **score_fn(config) -> EdgeResult** 주입(테스트는 가짜, 실사용은 pinned_backtest). 선택 기준 = **is_edge(train)** 최대화(옵티마이저가 OOS를 직접 안 봄 → OOS는 심판 검증용으로 남김). random N개 → 최고에서 coordinate descent로 refine.

- [ ] **Step 1: 실패 테스트 (score_fn 주입)**

```python
# ~/corvin_evolution/tests/test_optimizer.py
from evolution.optimizer import optimize
from evolution.types import EdgeResult


def test_optimizer_finds_higher_train_edge():
    # vix_term.green_max가 클수록 train edge 높은 가짜 채점
    def score(cfg):
        v = cfg["vix_term"]["green_max"]
        return EdgeResult(oos_edge=v * 0.4, is_edge=v * 0.5, window_id="w")
    res = optimize(score, n_random=30, seed=7, refine_steps=20)
    assert res.edge.is_edge > 0.0
    # 최고 train edge는 green_max 상한(1.1) 근처여야
    assert res.config["vix_term"]["green_max"] >= 1.0


def test_optimizer_deterministic():
    score = lambda cfg: EdgeResult(oos_edge=0.5, is_edge=cfg["breadth"]["green_min"] / 100, window_id="w")
    a = optimize(score, n_random=10, seed=3, refine_steps=5)
    b = optimize(score, n_random=10, seed=3, refine_steps=5)
    assert a.config == b.config
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
# optimizer.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Callable

from evolution import search_space
from evolution.types import EdgeResult


@dataclass(frozen=True)
class OptResult:
    config: dict[str, Any]
    edge: EdgeResult


def optimize(score_fn: Callable[[dict], EdgeResult], *, n_random: int = 40,
             seed: int = 0, refine_steps: int = 30) -> OptResult:
    rng = random.Random(seed)
    best_cfg = search_space.sample(rng)
    best = score_fn(best_cfg)
    for _ in range(n_random - 1):
        cfg = search_space.sample(rng)
        e = score_fn(cfg)
        if e.is_edge > best.is_edge:           # train edge로 선택 (OOS는 심판 검증용)
            best_cfg, best = cfg, e
    # coordinate descent refine
    for _ in range(refine_steps):
        improved = False
        for nb in search_space.neighbors(best_cfg):
            e = score_fn(nb)
            if e.is_edge > best.is_edge:
                best_cfg, best, improved = nb, e, True
        if not improved:
            break
    return OptResult(config=best_cfg, edge=best)
```

- [ ] **Step 4: Run → PASS** · **Step 5: Commit** (`feat(evolution): 옵티마이저 — random+coordinate descent, train-edge 선택`)

---

## Task 4: ledger + 후보 게이트·저장 (tuner)

**Files:**
- Create: `~/corvin_evolution/evolution/ledger.py`, `~/corvin_evolution/evolution/tuner.py`
- Test: `~/corvin_evolution/tests/test_ledger.py`, `~/corvin_evolution/tests/test_tuner.py`

ledger: SQLite(ts·config_json·oos_edge·is_edge·passed·reasons). tuner.`run_tuning(cfg, *, optimize=..., score=..., baseline_edge, ts) -> TuneResult`: 옵티마이저 실행 → 최고 후보를 게이트(`oos_edge ≥ baseline+ε` AND 과적합 갭 OK) → 통과면 `candidates/<ts>.json` 저장 + ledger. **자동 적용 없음.**

- [ ] **Step 1: ledger 실패 테스트** (E2 plan Task 3의 ledger와 동일 — record/recent)

```python
# ~/corvin_evolution/tests/test_ledger.py
from evolution.ledger import record, recent


def test_record_recent(tmp_path):
    db = tmp_path / "evo.db"
    record(db, config_json='{"vix_term":{"green_max":1.0}}', oos_edge=0.57,
           is_edge=0.6, passed=True, reasons=[], ts="T1")
    rows = recent(db, 5)
    assert rows[0]["passed"] == 1 and rows[0]["ts"] == "T1"
```

- [ ] **Step 2: tuner 실패 테스트 (주입)**

```python
# ~/corvin_evolution/tests/test_tuner.py
import json
from evolution.tuner import run_tuning
from evolution.optimizer import OptResult
from evolution.types import EdgeResult


def test_passing_candidate_saved(tmp_path):
    good = OptResult(config={"vix_term": {"green_max": 1.0}},
                     edge=EdgeResult(oos_edge=0.60, is_edge=0.63, window_id="w"))
    res = run_tuning(
        {"candidates_dir": str(tmp_path), "ledger_db": str(tmp_path / "e.db"),
         "epsilon": 0.02, "overfit_gap_threshold": 0.15},
        baseline_edge=0.50, ts="T9",
        optimize_fn=lambda score_fn: good, score_fn=lambda c: good.edge)
    assert res.passed is True
    assert (tmp_path / "T9.json").exists()         # 후보 저장됨
    assert json.loads((tmp_path / "T9.json").read_text())["vix_term"]["green_max"] == 1.0


def test_overfit_candidate_rejected(tmp_path):
    of = OptResult(config={"x": 1}, edge=EdgeResult(oos_edge=0.60, is_edge=0.90, window_id="w"))
    res = run_tuning({"candidates_dir": str(tmp_path), "ledger_db": str(tmp_path / "e.db"),
                      "epsilon": 0.02, "overfit_gap_threshold": 0.15},
                     baseline_edge=0.50, ts="T8",
                     optimize_fn=lambda s: of, score_fn=lambda c: of.edge)
    assert res.passed is False
    assert not (tmp_path / "T8.json").exists()
```

- [ ] **Step 3: Implement** ledger (E2 plan Task 3 코드 재사용, config_json 컬럼) + tuner:

```python
# tuner.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evolution import ledger
from evolution.optimizer import OptResult


@dataclass(frozen=True)
class TuneResult:
    passed: bool
    oos_edge: float
    reasons: list[str]


def run_tuning(cfg: dict[str, Any], *, baseline_edge: float, ts: str,
               optimize_fn: Callable, score_fn: Callable) -> TuneResult:
    best: OptResult = optimize_fn(score_fn)
    reasons: list[str] = []
    if best.edge.oos_edge < baseline_edge + cfg["epsilon"]:
        reasons.append(f"엣지 부족: OOS {best.edge.oos_edge:.3f} < {baseline_edge:.3f}+ε")
    if best.edge.overfit_gap > cfg["overfit_gap_threshold"]:
        reasons.append(f"과적합: 갭 {best.edge.overfit_gap:.3f}")
    passed = not reasons
    if passed:
        cand_dir = Path(cfg["candidates_dir"]); cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / f"{ts}.json").write_text(
            json.dumps(best.config, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger.record(Path(cfg["ledger_db"]), config_json=json.dumps(best.config, ensure_ascii=False),
                  oos_edge=best.edge.oos_edge, is_edge=best.edge.is_edge,
                  passed=passed, reasons=reasons, ts=ts)
    return TuneResult(passed=passed, oos_edge=best.edge.oos_edge, reasons=reasons)
```

- [ ] **Step 4: Run → PASS** (ledger + tuner) · **Step 5: Commit** (`feat(evolution): ledger + tuner 게이트·후보 저장 (드라이런)`)

---

## Task 5: tune.sh 진입점 + 실행 스모크 (API 0 검증)

**Files:**
- Create: `~/corvin_evolution/tune.sh`, `~/corvin_evolution/evolution/tune_run.py`
- Test: 수동 스모크

`tune_run.py`: config.yaml + baseline.json 로드 → `score_fn = lambda c: pinned_backtest.evaluate(subject, config_override=c)` → `optimize_fn = lambda s: optimizer.optimize(s, n_random=cfg.n_random, seed=cfg.seed)` → `run_tuning(...)` → evolution.log + digest. `tune.sh`: 킬스위치 → `python3 -m evolution.tune_run`. **네트워크·claude 호출 0.**

- [ ] **Step 1: tune.sh + tune_run.py 작성** (config.yaml에 `n_random:40`, `seed:0`, `candidates_dir`, `ledger_db` 추가)

- [ ] **Step 2: 실 스모크** — `chmod +x tune.sh && bash tune.sh`. 확인:
  - 네트워크 호출 0 (claude 미실행), 수초 내 완료
  - evolution.log에 verdict 1줄, baseline 대비 결과
  - 통과 시 `candidates/<ts>.json` 생성 — **subject config.json 무변경**(`git -C <subject> status` 깨끗)
- [ ] **Step 3: 킬스위치 스모크** — `touch ~/.corvin_killswitch && bash tune.sh` → skip 확인 → `rm`.
- [ ] **Step 4: Commit** (`feat(evolution): tune.sh — 무료 자동튜너 진입점 (API 0, dry-run)`)

---

## Task 6: 리뷰 digest + 수동 적용 헬퍼

**Files:**
- Create: `~/corvin_evolution/evolution/review_digest.py`, `~/corvin_evolution/apply_candidate.sh`
- Test: `~/corvin_evolution/tests/test_review_digest.py`

`build_digest(ledger_rows, candidates_dir) -> str`: 통과 후보 목록·OOS·경로 + "수동 적용" 안내. `apply_candidate.sh <candidate.json>`: **폐하가 직접 실행** — 후보 설정을 subject `config.json["early_warning"]`에 병합(백업 생성). E3에선 자동 안 함.

- [ ] **Step 1: digest 실패 테스트**

```python
# ~/corvin_evolution/tests/test_review_digest.py
from evolution.review_digest import build_digest


def test_digest_lists_passing():
    rows = [{"ts": "T1", "oos_edge": 0.58, "passed": 1, "reasons": "[]"},
            {"ts": "T2", "oos_edge": 0.40, "passed": 0, "reasons": '["엣지 부족"]'}]
    out = build_digest(rows, "/c")
    assert "T1" in out and "0.58" in out and ("통과" in out or "apply" in out.lower())
```

- [ ] **Step 2-4: Implement digest** (E2 패턴) + `apply_candidate.sh`(jq/python으로 config.json 병합 + `.bak` 백업, 폐하 수동 실행). 테스트 PASS. **Step 5: Commit** (`feat(evolution): 리뷰 digest + 수동 적용 헬퍼`)

- [ ] **Step 6: supervisor 전체 스위트** `cd ~/corvin_evolution && python3 -m pytest -q` → 전 그린.

---

## Self-Review

**목표 커버리지:** API 0(LLM·worktree·네트워크 없음) → Task 3·5 ✓ / 자동 파라미터 탐색 → Task 2·3 ✓ / walk-forward·과적합 게이트 → Task 4 ✓ / 드라이런·후보 저장만 → Task 4·5 ✓ / 사람 리뷰·수동적용 → Task 6 ✓ / 게임내성(채점·fixture supervisor 소유, 화이트리스트 키) → Task 1·2 ✓.

**Placeholder scan:** 없음. Task 5 Step 2(실 스모크)·Task 6 apply 헬퍼는 실환경/수동이라 의도적 수동 단계.

**Type consistency:** E2 `EdgeResult`(oos_edge·is_edge·overfit_gap) 재사용. `OptResult`(config·edge)·`TuneResult`(passed·oos_edge·reasons) 일관. `sample`/`neighbors`/`optimize`/`run_tuning`/`record`/`recent`/`build_digest` 시그니처 일치.

**안전·한계 (명시):**
- ✅ API 0원 — 옵티마이저는 순수 계산. cron 비용 우려 없음.
- ✅ 화이트리스트 키만 생성 → 임의 설정/위험키 불가. E2 safety도 보조로 유지 가능.
- ✅ 드라이런 — 후보는 json 저장만, 적용은 폐하 수동(`apply_candidate.sh`).
- ⚠️ **과적합 한계**: 옵티마이저는 train으로 선택, 심판은 OOS로 검증하나 **단일 fixture**라 둘 다 같은 데이터 계열. 진짜 일반화 검증은 **E4의 홀드아웃/회전 데이터** 필요. 그 전까지 통과 후보는 *참고*이지 보장 아님 — 폐하 사람 판단 필수.
- ⚠️ fixture 품질이 곧 튜닝 품질. 현 40일 합성 fixture는 거칠어 미세 개선 분해능 낮음(E2 리뷰 지적). E4 전 **실데이터 기반 fixture 확장** 권장.

---

**다음 Phase E4 (무료 유지 가능)**: ① 실데이터 기반 + **홀드아웃/회전 fixture**로 과적합·암기 차단 → ② 드라이런 신뢰 누적 후 통과 후보 **자동 적용**(config.json 병합, 백업·롤백) 활성화. 여전히 LLM 없이 가능. E3는 *튜너가 돌되 아무것도 자동 반영 안 함* — 폐하가 후보를 보는 단계.
