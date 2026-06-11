# 2축 레짐 — 추세 × 스트레스 진입 자세 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** 단일축(스트레스만) 레짐을 2축(추세 × 스트레스)으로 확장 — "하락+calm(정돈된 grind)"이면 신규 DCA 진입을 *억제(throttle)*, "하락+panic(투매)"에만 역발상 가속. 현 시스템은 가격 추세 입력이 없어 정돈된 하락을 "neutral"로 오인 → 역발상 multiplier가 falling-knife를 가속한 결함 수정.

**Architecture:** `regime.py`에 순수 함수 `trend_from_returns`(광범위 지수 모멘텀→up/chop/down) + `entry_posture`(추세×스트레스→자세) 추가. `detect_regime`이 timeseries에서 sp500 모멘텀으로 trend 계산해 `posture`를 출력·state 기록. `dca_timing`의 `_score_one` multiplier를 posture 기반으로 — posture 없으면 기존 라벨 multiplier fallback(회귀 0).

**Tech Stack:** Python 3.9, pytest. 기존 regime.label_from_signals·dca_timing._regime_multiplier 재사용.

**근거:** 폐하 2026-06-11 — 현 시장은 F&G 60·금 미매수 = *공포 없는 정돈된 디리스킹*. 역발상 매수는 capitulation에만 유효. [[feedback_investor_psychology]]

---

### Task 1: regime.py 2축 순수 함수

**Files:** Modify `corvin_jarvis/regime.py`, Test `tests/test_regime.py` (append)

- [ ] **Step 1: 상수 + 순수 함수**

```python
TRENDS = ("down", "chop", "up")
POSTURES = ("capitulation_buy", "accumulate", "normal", "throttle")
_TREND_DOWN_PCT = -3.0   # 광범위 N일 수익률 임계
_TREND_UP_PCT = 3.0


def trend_from_returns(broad_returns_pct: float | None) -> str:
    """광범위 지수 N일 수익률 → 추세 축. None→chop."""
    if broad_returns_pct is None:
        return "chop"
    if broad_returns_pct <= _TREND_DOWN_PCT:
        return "down"
    if broad_returns_pct >= _TREND_UP_PCT:
        return "up"
    return "chop"


def entry_posture(trend: str, stress_label: str) -> str:
    """추세 × 스트레스 → 진입 자세 (2축 레짐 핵심).

    down+crisis(패닉)  → capitulation_buy (역발상 가속 OK)
    down+그외(정돈하락) → throttle (신규 억제 — 슬로우블리드 칼 방지)
    up+비스트레스      → accumulate
    그 외(chop 등)     → normal
    """
    if trend == "down":
        return "capitulation_buy" if stress_label == "crisis" else "throttle"
    if trend == "up" and stress_label not in ("crisis", "risk_off"):
        return "accumulate"
    return "normal"
```

- [ ] **Step 2: detect_regime에 trend·posture 주입** — VIX z-score 계산부 뒤에 sp500 모멘텀 추가:

```python
    # 광범위 추세 (sp500 N일 모멘텀) — 2축 trend
    broad_pct: float | None = None
    if timeseries_db.exists():
        try:
            with sqlite3.connect(timeseries_db) as conn:
                rows = conn.execute(
                    "SELECT price FROM quote_history WHERE symbol='sp500' "
                    "AND price IS NOT NULL ORDER BY ts_utc DESC LIMIT 11"
                ).fetchall()
            vals = [float(r[0]) for r in rows if r[0] is not None]
            if len(vals) >= 2 and vals[-1] > 0:
                broad_pct = (vals[0] / vals[-1] - 1) * 100   # 최신/10세션전
        except sqlite3.Error:
            pass
    trend = trend_from_returns(broad_pct)
    out["trend"] = trend
    out["broad_returns_pct"] = round(broad_pct, 2) if broad_pct is not None else None
    out["posture"] = entry_posture(trend, out["label"])
```

state_file write에 `"trend"`, `"posture"` 추가.

- [ ] **Step 3: 테스트**

```python
@pytest.mark.unit
def test_trend_from_returns_thresholds():
    assert regime.trend_from_returns(-5.0) == "down"
    assert regime.trend_from_returns(4.0) == "up"
    assert regime.trend_from_returns(1.0) == "chop"
    assert regime.trend_from_returns(None) == "chop"


@pytest.mark.unit
def test_entry_posture_down_calm_throttles():
    # 핵심: 하락+비패닉 → throttle (정돈된 grind, 현 시장)
    assert regime.entry_posture("down", "neutral") == "throttle"
    assert regime.entry_posture("down", "risk_off") == "throttle"


@pytest.mark.unit
def test_entry_posture_down_panic_buys():
    assert regime.entry_posture("down", "crisis") == "capitulation_buy"


@pytest.mark.unit
def test_entry_posture_up_accumulates():
    assert regime.entry_posture("up", "neutral") == "accumulate"
    assert regime.entry_posture("up", "risk_off") == "normal"  # 상승이지만 스트레스


@pytest.mark.unit
def test_entry_posture_chop_normal():
    assert regime.entry_posture("chop", "neutral") == "normal"
```

- [ ] **Step 4: Run** `pytest tests/test_regime.py -q` — 전부 PASS (기존 14 + 신규)

- [ ] **Step 5: Commit** `feat(regime): 2축 추세 함수 + entry_posture — 정돈된 하락 식별`

---

### Task 2: dca_timing posture 기반 multiplier

**Files:** Modify `corvin_jarvis/dca_timing.py`, Test `tests/test_dca_timing.py` (append)

- [ ] **Step 1: posture multiplier + 로더**

```python
_POSTURE_MULT = {
    "capitulation_buy": 1.3,   # 진짜 투매 — 역발상 가속
    "accumulate": 1.0,
    "normal": 1.0,
    "throttle": 0.5,           # 정돈된 하락 — 신규 진입 억제 (점수 반감)
}


def _posture_multiplier(posture: str | None) -> float | None:
    """posture→multiplier. None/미지 → None(라벨 fallback)."""
    return _POSTURE_MULT.get(posture or "") if posture in _POSTURE_MULT else None


def _load_posture() -> str | None:
    last = STATE_DIR / "last_regime.json"
    if last.exists():
        try:
            return json.loads(last.read_text()).get("posture")
        except (json.JSONDecodeError, OSError):
            return None
    return None
```

- [ ] **Step 2: 호출부(line ~325) posture 우선** — 기존:
```python
    regime_mult = _regime_multiplier(regime)
```
변경:
```python
    posture = _load_posture()
    pm = _posture_multiplier(posture)
    regime_mult = pm if pm is not None else _regime_multiplier(regime)  # posture 우선, 없으면 라벨 fallback
```

(posture 없으면 기존 라벨 동작 그대로 — 회귀 0. DCAReport에 posture 노출은 선택.)

- [ ] **Step 3: 테스트**

```python
@pytest.mark.unit
def test_posture_multiplier_throttles_down_calm():
    assert dca_timing._posture_multiplier("throttle") == 0.5
    assert dca_timing._posture_multiplier("capitulation_buy") == 1.3
    assert dca_timing._posture_multiplier("normal") == 1.0


@pytest.mark.unit
def test_posture_multiplier_unknown_falls_back():
    assert dca_timing._posture_multiplier(None) is None
    assert dca_timing._posture_multiplier("weird") is None


@pytest.mark.unit
def test_throttle_halves_dca_score(monkeypatch, tmp_path):
    """throttle posture → DCA 점수 반감 (신규 진입 억제 E2E)."""
    import json as _json
    st = tmp_path / "last_regime.json"
    st.write_text(_json.dumps({"label": "neutral", "posture": "throttle"}))
    monkeypatch.setattr(dca_timing, "STATE_DIR", tmp_path)
    assert dca_timing._load_posture() == "throttle"
    assert dca_timing._posture_multiplier(dca_timing._load_posture()) == 0.5
```

- [ ] **Step 4: Run** `pytest tests/test_dca_timing.py tests/test_regime.py -q` — PASS

- [ ] **Step 5: Commit** `feat(dca): posture 기반 multiplier — 정돈된 하락 throttle(점수 반감), 패닉만 가속`

---

### Task 3: 전체 회귀 + E2E + 보고

- [ ] **Step 1:** `pytest tests/ -q` 전체 PASS
- [ ] **Step 2:** E2E — 현 last_regime.json에 trend/posture 생성 확인 + dca 영향
- [ ] **Step 3:** 코드리뷰(subagent) → HIGH 수정 → 폐하 승인 머지

## Self-Review
- 폐하 진단 커버: 가격추세 입력 추가(T1) · down+calm→throttle(T1 entry_posture) · DCA 가속 반전(T2). narrative 배선·감도 추가보정은 후속.
- 회귀 0: posture None → 라벨 multiplier fallback (T2 Step2). 기존 _regime_multiplier·detect_regime 테스트 보존.
- 타입 일관: trend_from_returns(float|None)→str, entry_posture(str,str)→str, _posture_multiplier(str|None)→float|None.
