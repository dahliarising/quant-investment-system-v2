# Corvin Evolution Loop — Phase E4: 실데이터 fixture + 홀드아웃 검증 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 40일 합성 fixture를 **수년치 실시장 데이터**로 교체해 자동 튜너가 진짜 개선을 찾게 하고, **홀드아웃(옵티마이저·게이트가 절대 안 본 데이터)** 검증을 추가해 과적합/암기를 차단한다. 전부 **무료**(FDR+FRED). 통과 후보 적용은 기본 폐하 수동, 자동적용은 opt-in 플래그(기본 off).

**Architecture:** subject의 *순수 reading 함수*(`ew_providers.semis_reading`·`vix_term_reading`·`breadth_reading`·`fred_reading`)를 **시점별 과거 fetcher**로 재생(replay)해 진짜 `ew_history.json`을 생성한다(스키마 자동 일치). pinned 백테스트를 **train/oos/holdout 3분할**로 확장: 옵티마이저는 train으로 선택, 게이트는 oos로 1차, **holdout으로 최종 확인**(옵티마이저·게이트가 직접 최적화 안 한 데이터). holdout까지 baseline을 넘는 후보만 "고확신"으로 표시.

**Tech Stack:** Python 3.11, pytest, FinanceDataReader(SOXX·SPY·^VIX·^VIX3M·universe), FRED fredgraph.csv(BAMLH0A0HYM2·T10Y2Y). 기존 E2/E3 supervisor + subject `ew_providers`(순수 함수 재사용).

**선행:** [E3 auto-tuner](./2026-06-12-corvin-evolution-E3-auto-tuner.md)(완료) · [spec §4](../specs/2026-06-12-corvin-evolution-loop-design.md). E2/E3 supervisor가 `~/corvin_evolution/`에 존재.

---

## 핵심 설계 결정 (구현 전 폐하 리뷰)

1. **fixture = subject 순수 함수 replay** — 과거 데이터로 별도 분류 로직 안 짠다. `ew_providers`의 reading 함수를 시점별 fetcher로 재생 → 스키마 자동 일치. **데이터 소스 전부 무료**.
2. **윈도우** — 2018-01 ~ 직전월 (~7년, ≈1,800 거래일). 강세·코로나·2022약세·2026전쟁 등 레짐 다양성 확보.
3. **3분할 + 홀드아웃** — 시간순 train(60%)·oos(20%)·**holdout(20%, 가장 최근)**. 옵티마이저=train, 게이트=oos, **최종확인=holdout**. holdout은 옵티마이저·게이트 선택에 전혀 안 쓰임(진짜 안 본 데이터).
4. **적용 정책** — 기본 **폐하 수동**(`apply_candidate.sh`). `config.yaml`에 `auto_apply: false` opt-in. true여도 **holdout까지 통과한 후보만** + 백업·롤백. 라이브 신호 영향이라 보수적 추천(기본 off 유지).
5. **데이터 리스크 정직 처리** — 결측일·소스 불안정(FRED 간헐)·^VIX3M 가용시작(2011~)·universe 생존편향은 구현 중 확인하고 fixture 메타에 커버리지 기록. 못 채우는 구간은 *버리고 로그*(조용한 보간 금지).

---

## File Structure

| 파일 | 위치 | 책임 |
|------|------|------|
| `evolution/hist_fetchers.py` | supervisor | 시점별 과거 fetcher (FRED 날짜범위·FDR 종가/시세) |
| `evolution/build_fixture.py` | supervisor | reading 함수 replay → 실 `fixtures/ew_history_real.json` + 커버리지 메타 |
| `evolution/pinned_backtest.py` (수정) | supervisor | `evaluate(..., splits=("train","oos","holdout"))` 3분할 edge |
| `evolution/tuner.py` (수정) | supervisor | 게이트에 **holdout 통과** 추가 (oos+holdout 둘 다 baseline+ε) |
| `config.yaml` (수정) | supervisor | `fixture: fixtures/ew_history_real.json`, `auto_apply: false`, split 비율 |
| `evolution/auto_apply.py` (신규) | supervisor | opt-in 자동적용(기본 off): holdout 통과 후보만 백업·병합·롤백 |

---

## Task 1: 시점별 과거 fetcher (무료 소스)

**Files:**
- Create: `~/corvin_evolution/evolution/hist_fetchers.py`
- Test: `~/corvin_evolution/tests/test_hist_fetchers.py`

subject의 reading 함수가 받는 fetcher 시그니처를 **과거 시점용**으로 구현:
- `closes_until(symbol, as_of_date, n) -> list[float]` (FDR, as_of까지 마지막 n 종가)
- `quote_at(symbol, as_of_date) -> float | None` (FDR, as_of 종가)
- `fred_until(code, as_of_date, n) -> list[float]` (fredgraph.csv `cosd/coed` 날짜범위)

라이브 함수(`ew_providers.live_*`)를 참고하되 **as_of로 미래 차단(lookahead 금지)**. 네트워크 호출이라 테스트는 캐시/모킹.

- [ ] **Step 1: 실패 테스트 (fetcher 주입 가능 구조)**

```python
# ~/corvin_evolution/tests/test_hist_fetchers.py
from evolution.hist_fetchers import slice_until


def test_slice_until_no_lookahead():
    series = [("2025-01-01", 1.0), ("2025-01-02", 2.0), ("2025-01-03", 3.0)]
    out = slice_until(series, as_of="2025-01-02", n=5)
    assert out == [1.0, 2.0]          # as_of 이후(01-03) 제외 — lookahead 차단
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement** — `slice_until(series_with_dates, as_of, n)` 순수 함수(테스트 가능) + 그 위에 FDR/FRED 라이브 바인딩(`closes_until`·`quote_at`·`fred_until`). 핵심 순수 로직:

```python
# hist_fetchers.py (핵심 — 순수부)
from __future__ import annotations
from typing import Any


def slice_until(series: list[tuple[str, float]], as_of: str, n: int) -> list[float]:
    """(date,value) 시리즈에서 as_of 이하 날짜만, 마지막 n개. 미래 차단."""
    vals = [v for d, v in series if d <= as_of]
    return vals[-n:]
```
(라이브 바인딩 `closes_until` 등은 FDR `DataReader(sym, end=as_of)` → (date,close) 리스트 → `slice_until`. FRED는 `fredgraph.csv?id=...&cosd=START&coed=as_of`.)

- [ ] **Step 4: Run → PASS** · **Step 5: Commit** (`feat(evolution): 시점별 과거 fetcher (lookahead 차단, FDR/FRED)`)

---

## Task 2: 실 fixture 빌더 (reading 함수 replay)

**Files:**
- Create: `~/corvin_evolution/evolution/build_fixture.py`
- Test: `~/corvin_evolution/tests/test_build_fixture.py`

거래일 리스트 × subject reading 함수(과거 fetcher 주입) → `{date, readings, spx}` 생성. 결측일은 **버리고 카운트**(조용한 보간 금지). 커버리지 메타(`_meta`: 시작/끝/일수/소스별 결측)를 fixture에 동봉. 출력 = `fixtures/ew_history_real.json`.

- [ ] **Step 1: 실패 테스트 (순수 조립부, fetcher/reading 주입)**

```python
# ~/corvin_evolution/tests/test_build_fixture.py
from evolution.build_fixture import build_history


def test_build_history_skips_incomplete_days():
    dates = ["2025-01-02", "2025-01-03"]
    # reading_fn: 1/2은 정상, 1/3은 None(결측) 반환
    def reading_fn(date):
        if date == "2025-01-03":
            return None
        return {"readings": {"vix_term": {"ratio": 0.9}}, "spx": 5000.0}
    hist, meta = build_history(dates, reading_fn)
    assert len(hist) == 1 and hist[0]["date"] == "2025-01-02"
    assert meta["skipped"] == 1 and meta["n_days"] == 1
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement** — `build_history(dates, reading_fn) -> (history, meta)` 순수 조립(테스트 가능). 그 위에 `main()`이 실 fetcher로 reading_fn을 구성(semis/vix_term/breadth/hy/curve 전부 모은 날만 채택), `_classify_all` 기대 키 검증, `fixtures/ew_history_real.json` 기록.

- [ ] **Step 4: 실 빌드 (네트워크, 폐하 승인·시간 소요)** — `python3 -m evolution.build_fixture --start 2018-01-01`. 확인: 거래일 ≥1,000, 결측률 로그, readings 키가 `_classify_all` 기대와 일치(샘플 1일 검증). 커버리지 메타 출력.

- [ ] **Step 5: Commit** (`feat(evolution): 실 fixture 빌더 — reading 함수 replay + 커버리지 메타`)

---

## Task 3: pinned 백테스트 3분할 (train/oos/holdout)

**Files:**
- Modify: `~/corvin_evolution/evolution/pinned_backtest.py`
- Test: `~/corvin_evolution/tests/test_pinned_splits.py`

`evaluate`에 holdout 추가: `EdgeResult`에 `holdout_edge` 필드. 시간순 train(0~t1)·oos(t1~t2)·holdout(t2~끝). 기존 호출 하위호환(holdout 없으면 None/동일).

- [ ] **Step 1: 실패 테스트**

```python
# ~/corvin_evolution/tests/test_pinned_splits.py
from evolution.pinned_backtest import evaluate

SUBJ = "/Users/thethethe/Claude/quant_investment_system_v2"


def test_three_way_split_populates_holdout():
    res = evaluate(SUBJ, fixture="fixtures/ew_history_real.json",
                   train_ratio=0.6, oos_ratio=0.2)
    assert res.holdout_edge is not None
    assert 0.0 <= res.holdout_edge <= 1.0
```

- [ ] **Step 2: Run → FAIL** (`holdout_edge` 없음 / `oos_ratio` 인자 없음)

- [ ] **Step 3: Implement** — `types.EdgeResult`에 `holdout_edge: float | None = None` 추가(frozen, 기본 None=하위호환). `evaluate(..., oos_ratio=0.0)`: oos_ratio>0이면 3분할, 각 구간 edge 산출. 게임내성·채점은 그대로 supervisor 소유.

- [ ] **Step 4: Run → PASS**(실 fixture 필요 — Task 2 후). 기존 `test_pinned_backtest`·`test_pinned_config_override` 회귀 그린 확인.

- [ ] **Step 5: Commit** (`feat(evolution): pinned 백테스트 3분할 — holdout_edge 추가`)

---

## Task 4: 홀드아웃 게이트 (tuner)

**Files:**
- Modify: `~/corvin_evolution/evolution/tuner.py`
- Test: `~/corvin_evolution/tests/test_tuner_holdout.py`

게이트 강화: 후보는 `oos_edge ≥ baseline+ε` **AND `holdout_edge ≥ baseline+ε`** AND 과적합 갭 OK여야 통과. holdout은 옵티마이저가 안 본 데이터 → 진짜 일반화 확인. holdout 없으면(구버전 fixture) 기존 동작.

- [ ] **Step 1: 실패 테스트**

```python
# ~/corvin_evolution/tests/test_tuner_holdout.py
from evolution.tuner import run_tuning
from evolution.optimizer import OptResult
from evolution.types import EdgeResult


def _opt(oos, hold, is_=0.6):
    e = EdgeResult(oos_edge=oos, is_edge=is_, window_id="w", holdout_edge=hold)
    return OptResult(config={"x": 1}, edge=e)


def test_passes_oos_but_fails_holdout_is_rejected(tmp_path):
    cfg = {"candidates_dir": str(tmp_path), "ledger_db": str(tmp_path/"e.db"),
           "epsilon": 0.02, "overfit_gap_threshold": 0.5}
    # oos 통과(0.60) but holdout 미달(0.50=baseline) → 거부 (과적합 의심)
    res = run_tuning(cfg, baseline_edge=0.50, ts="T1",
                     optimize_fn=lambda s: _opt(0.60, 0.50), score_fn=lambda c: _opt(0.60, 0.50).edge)
    assert res.passed is False
    assert any("holdout" in r.lower() or "홀드아웃" in r for r in res.reasons)
    assert not (tmp_path/"T1.json").exists()
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement** — `run_tuning`에 holdout 체크 추가: `if edge.holdout_edge is not None and edge.holdout_edge < baseline+ε: reasons.append("홀드아웃 미달...")`. TuneResult에 holdout_edge 노출(선택).

- [ ] **Step 4: Run → PASS** (기존 tuner 테스트 회귀 그린)

- [ ] **Step 5: Commit** (`feat(evolution): 홀드아웃 게이트 — oos+holdout 둘 다 baseline 넘어야 통과`)

---

## Task 5: re-baseline + 실 튜닝 (정직한 결과 확인)

**Files:**
- Modify: `~/corvin_evolution/evolution/tune_run.py`(실 fixture·3분할 wiring), `snapshot_baseline.py`(holdout 포함)
- Test: 수동 스모크

- [ ] **Step 1** — `snapshot_baseline`이 실 fixture로 baseline(oos·holdout 포함) 기록하도록 수정. `tune_run`이 `fixture=실, oos_ratio=0.2`로 평가.
- [ ] **Step 2: 실 baseline + 튜닝 스모크** — `python3 -m evolution.snapshot_baseline && bash tune.sh && cat evolution.log`. 확인: **API 0 유지**(claude 없음), subject 무변경, 결과 정직(통과 후보 나오면 `candidates/`에 저장+holdout 점수 표시, 안 나오면 "없음" — 둘 다 OK). 통과 후보 patch를 폐하 리뷰.
- [ ] **Step 3: Commit** (`feat(evolution): 실 fixture·3분할로 baseline·tune 재구성`)

---

## Task 6: opt-in 자동적용 (기본 off, holdout 가드)

**Files:**
- Create: `~/corvin_evolution/evolution/auto_apply.py`
- Test: `~/corvin_evolution/tests/test_auto_apply.py`

`maybe_auto_apply(cfg, candidate_path, *, applier) -> bool`: `cfg["auto_apply"]`가 **명시적 true**이고 후보가 holdout까지 통과했을 때만 `apply_candidate.sh` 로직 호출(백업·병합). 기본 false → 아무것도 안 함(폐하 수동). 라이브 신호 영향이라 **기본 off 강력 권장**.

- [ ] **Step 1: 실패 테스트**

```python
# ~/corvin_evolution/tests/test_auto_apply.py
from evolution.auto_apply import maybe_auto_apply


def test_default_off_does_not_apply(tmp_path):
    called = []
    assert maybe_auto_apply({"auto_apply": False}, str(tmp_path/"c.json"),
                            applier=lambda p: called.append(p)) is False
    assert called == []


def test_optin_true_applies(tmp_path):
    called = []
    cand = tmp_path/"c.json"; cand.write_text("{}")
    assert maybe_auto_apply({"auto_apply": True}, str(cand),
                            applier=lambda p: called.append(p)) is True
    assert called == [str(cand)]
```

- [ ] **Step 2: Run → FAIL** · **Step 3: Implement**(주입 applier, 기본 false 분기) · **Step 4: Run → PASS** · **Step 5: Commit** (`feat(evolution): opt-in 자동적용 (기본 off, 수동 우선)`)

- [ ] **Step 6: supervisor 전체 스위트** `cd ~/corvin_evolution && python3 -m pytest -q` → 전 그린.

---

## Self-Review

**목표 커버리지:** 실데이터 fixture(무료 replay) → Task 1·2 ✓ / 홀드아웃(안 본 데이터) → Task 3·4 ✓ / 과적합·암기 차단 → Task 4(holdout 게이트) ✓ / 정직한 결과 → Task 5(승자 없어도 OK) ✓ / 적용 정책(수동 기본·opt-in) → Task 6 ✓ / API 0 유지 → Task 5(스모크 검증) ✓.

**Placeholder scan:** 없음. Task 2 Step 4·Task 5 Step 2는 실 네트워크·시간 소요라 의도적 수동(승인). 데이터 소스 정합(키·결측)은 통합 확인.

**Type consistency:** `EdgeResult`에 `holdout_edge` 추가(기본 None=하위호환). `slice_until`·`build_history`·`evaluate(oos_ratio=)`·`run_tuning`(holdout 게이트)·`maybe_auto_apply` 시그니처 일관.

**안전·한계 (명시):**
- ✅ API 0 유지 — FDR/FRED는 데이터 조회(무료), LLM 없음.
- ✅ holdout = 옵티마이저·게이트가 안 본 데이터 → 암기/과적합 1차 차단.
- ✅ 자동적용 기본 off — 폐하 수동이 기본. opt-in true여도 holdout 통과 + 백업·롤백.
- ⚠️ **잔존 한계**: 백테스트 일반화 ≠ 미래 보장(레짐 변화). holdout도 *과거*다. 통과 후보는 "고확신 후보"이지 보장 아님 — 폐하 사람 판단·소액 검증 권장.
- ⚠️ **데이터 품질**: breadth는 universe 생존편향, FRED 간헐 결측 가능 → fixture 메타에 커버리지 기록, 못 채운 구간 버림(투명).
- ⚠️ ^VIX3M 가용 2011~ / universe 과거 구성 변화 → 시작연도·구성 영향 메타 기록.

---

**다음 (선택)**: E5 — 통과 후보를 폐하가 며칠 사람검토 후 신뢰 누적되면 `auto_apply` 켜기(여전히 무료·holdout 가드). 또는 fixture 주기적 갱신 cron(무료). E4는 *진짜 데이터로 튜너가 의미있게 작동 + 안 본 데이터로 검증*까지.
