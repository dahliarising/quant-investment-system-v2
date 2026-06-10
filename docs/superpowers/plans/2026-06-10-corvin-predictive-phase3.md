# Corvin 예측 엔진 고도화 (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** predictive_engine의 4가지 약점 제거 — ① 변동성 무시 선형 기울기 → 노이즈 게이트, ② 점추정 → 신뢰구간, ③ 고정 confidence → calibration.json 주입, ④ 고정 RS 임계 → 종목별 적응 임계값.

**Architecture:** 전부 `corvin_jarvis/predictive_engine.py` 내 순수 함수 확장 (시그니처는 기본값 인자로 하위호환). 데이터 제약: `get_stock_daily_closes`는 종가만 반환(OHLC 없음) → **ATR 프록시 = 최근 14일 |일변화| 평균**으로 대체 (문서화된 의도적 절충). calibration은 Phase 1 scorer cron이 매일 쓰는 `state/calibration.json`을 기본 경로 로드(주입 가능). snapshot은 fetch 일수만 25→90으로 확대해 적응 임계값 활성화.

**Tech Stack:** Python 표준 라이브러리 (statistics, math, json). 신규 의존성 0.

**Spec:** `docs/superpowers/specs/2026-06-10-corvin-signal-feedback-loop-design.md` §6

---

## File Structure

```
corvin_jarvis/predictive_engine.py   # 수정 — 4개 개선 전부 (파일 1개 응집)
corvin_jarvis/dashboard/snapshot.py  # 수정 — _predictive_signals fetch 25→90일
tests/test_predictive_engine.py      # 수정 — 신규 테스트 append (기존 13개 무수정 원칙)
tests/test_dashboard_snapshot.py     # 수정 불필요 (fetch 일수는 mock 무관)
```

**기존 테스트 호환성 사전 분석 (2026-06-10 확인):**
- 기존 13 테스트는 message를 **부분 일치**(`in`)로만 단언 — message 확장 안전.
- VELOCITY 픽스처는 짧은 closes 리스트 — ATR 프록시는 15봉(window+1) 미만이면 None → 노이즈 게이트 비활성 → 기존 동작 보존. **단, 구현 후 반드시 기존 13개 무수정 통과 확인. 깨지면 구현을 조정하고, 테스트 수정은 금지** (스펙 §6 "기존 테스트 유지 + 회귀 0").

---

### Task 1: VELOCITY 고도화 — ATR 프록시 노이즈 게이트 + 신뢰구간

**Files:**
- Modify: `corvin_jarvis/predictive_engine.py`
- Test: `tests/test_predictive_engine.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# tests/test_predictive_engine.py 하단에 append

# ── Phase 3: VELOCITY 고도화 ──────────────────────────

def _mk_holding(sym="TSLA", price=100.0):
    return [{"symbol": sym, "market": "US", "price": price, "pnl_pct": -5.0}]


def test_velocity_noise_gate_suppresses_weak_slope_in_choppy_market():
    """변동성 대비 미미한 기울기 — 노이즈 게이트 억제 (15봉 이상에서 활성)."""
    # 일변화 ±5 들쭉날쭉(ATR프록시≈5), 순기울기 -0.5 → strength 0.1 < 0.15 게이트
    closes = [100.0]
    deltas = [+5, -5.5, +5, -5.5, +5, -5.5, +5, -5.5, +5, -5.5, +5, -5.5, +5, -5.5, +5, -6.0]
    for d in deltas:
        closes.append(closes[-1] + d)
    sigs = pe.evaluate_velocity(_mk_holding(price=closes[-1]),
                                {"TSLA": closes[-1] - 5}, {"TSLA": closes})
    assert sigs == []  # 같은 거리·기울기라도 고변동 노이즈면 침묵


def test_velocity_clean_downtrend_still_fires_with_atr_data():
    """저변동 명확한 하락 추세 — 게이트 통과, 신뢰구간 evidence 포함."""
    closes = [float(120 - i) for i in range(16)]  # 일정한 -1/일, 16봉
    price, stop = closes[-1], closes[-1] - 5
    sigs = pe.evaluate_velocity(_mk_holding(price=price), {"TSLA": stop}, {"TSLA": closes})
    assert len(sigs) == 1
    ev = sigs[0].evidence
    assert ev["atr_proxy"] is not None and ev["strength"] >= 0.15
    assert ev["days_lo"] is not None and ev["days_hi"] is not None
    assert ev["days_lo"] <= ev["days_to_stop"] <= ev["days_hi"]
    assert "범위" in sigs[0].message  # "(범위 X–Y일)" 표기


def test_velocity_short_series_skips_gate_backcompat():
    """15봉 미만 — ATR 산출 불가 → 게이트 미적용 (기존 동작 보존)."""
    closes = [110.0, 108.0, 106.0, 104.0]  # 4봉, 기존 테스트 스타일
    sigs = pe.evaluate_velocity(_mk_holding(price=104.0), {"TSLA": 98.0}, {"TSLA": closes})
    assert len(sigs) == 1
    assert sigs[0].evidence["atr_proxy"] is None
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_predictive_engine.py -k "noise_gate or clean_downtrend or short_series" -v`
Expected: FAIL (KeyError 'atr_proxy' / 게이트 부재로 빈 리스트 아님)

- [ ] **Step 3: Modify predictive_engine.py**

상단 import에 `import math` 추가. 상수부 정리 — 현재 22-27행의 `_fmt` 안에 잘못 끼어든 주석을 바로잡고(아래 형태로) 상수 추가:

```python
_VELOCITY_HORIZON = 14     # 이 일수 이내 도달 예상이면 경보
_NOISE_GATE = 0.15         # |기울기|/ATR프록시 미만 = 변동성 노이즈 → 억제
_ATR_WINDOW = 14


def _fmt(v: float) -> str:
    """가격 가독 포맷 — KRW 큰 수는 천단위, USD는 소수 유지. 1e+06 방지."""
    return f"{round(v):,}" if v >= 10000 else f"{v:g}"
```

공통 헬퍼 추가 (`_slope` 아래):

```python
def _atr_proxy(closes: list[float], window: int = _ATR_WINDOW) -> float | None:
    """종가 기반 변동성 프록시 — 최근 window일 |일변화| 평균.

    데이터 제약: quote_provider가 종가만 제공(OHLC 없음) → 진짜 ATR 대신
    close-to-close 변동성으로 대체 (스펙 §6 의도 동일: 변동성 정규화).
    window+1봉 미만이거나 변동 0이면 None.
    """
    if len(closes) < window + 1:
        return None
    tail = closes[-(window + 1):]
    changes = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
    atr = statistics.mean(changes)
    return atr if atr > 0 else None


def _slope_se(closes: list[float]) -> float:
    """일변화량 평균의 표준오차 — 신뢰구간용. 변화 표본<2 → 0."""
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    if len(changes) < 2:
        return 0.0
    return statistics.stdev(changes) / math.sqrt(len(changes))
```

`evaluate_velocity` 본문 교체 (시그니처 동일):

```python
def evaluate_velocity(
    holdings: list[dict[str, Any]],
    stops: dict[str, float],
    closes_by_sym: dict[str, list[float]],
    horizon: int = _VELOCITY_HORIZON,
) -> list[PredictiveSignal]:
    """하락 추세 기울기로 손절선 도달 예상일 경보 (Phase 3: 노이즈 게이트 + 신뢰구간).

    이미 손절선 이하인 경우는 signal_engine(STOP)이 담당 — 여기선 불개입.
    """
    out: list[PredictiveSignal] = []
    for pos in holdings:
        sym = str(pos.get("symbol", ""))
        stop = stops.get(sym)
        if stop is None:
            continue
        closes = closes_by_sym.get(sym, [])
        s = _slope(closes)
        if s is None or s >= 0:
            continue  # 상승·횡보 추세
        price = pos.get("price") or (closes[-1] if closes else None)
        if price is None or price <= stop:
            continue  # 이미 손절 이탈 → signal_engine 담당
        atr = _atr_proxy(closes)
        strength = (-s / atr) if atr else None
        if strength is not None and strength < _NOISE_GATE:
            continue  # 변동성 대비 미미한 기울기 — 노이즈 억제
        dist = price - stop
        days_to = dist / (-s)
        if days_to > horizon:
            continue
        se = _slope_se(closes)
        days_lo = round(dist / (-s + se), 1) if (-s + se) > 0 else None  # 빠른 시나리오
        days_hi = round(dist / (-s - se), 1) if (-s - se) > 0 else None  # 느린 시나리오
        rng = ""
        if days_lo is not None and days_hi is not None:
            rng = f" (범위 {math.floor(days_lo)}–{math.ceil(days_hi)}일)"
        urgency = max(40, min(85, int(85 - (days_to / horizon) * 45)))
        out.append(PredictiveSignal(
            symbol=sym, kind="VELOCITY", urgency=urgency, confidence=65.0,
            horizon_days=round(days_to),
            message=f"하락 속도 기준 손절선({_fmt(stop)}) ~{round(days_to)}일 내 도달 예상{rng}",
            evidence={"slope_per_day": round(s, 4), "days_to_stop": round(days_to, 1),
                      "stop": stop, "current_price": price,
                      "atr_proxy": round(atr, 4) if atr else None,
                      "strength": round(strength, 3) if strength is not None else None,
                      "days_lo": days_lo, "days_hi": days_hi},
        ))
    return out
```

(주의: 신뢰구간 부호 — s<0이므로 빠른 도달=가파른 기울기 `-s+se`, 느린 도달=`-s-se`.
confidence=65.0은 Task 2에서 교체된다 — 이 Task에서는 그대로 둔다.)

- [ ] **Step 4: Run** `python3 -m pytest tests/test_predictive_engine.py -v`
Expected: 16 passed (기존 13 무수정 + 신규 3)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/predictive_engine.py tests/test_predictive_engine.py
git commit -m "feat(predictive): VELOCITY 고도화 — ATR프록시 노이즈 게이트 + 도달일 신뢰구간"
```

---

### Task 2: confidence 캘리브레이션 주입

**Files:**
- Modify: `corvin_jarvis/predictive_engine.py`
- Test: `tests/test_predictive_engine.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# ── Phase 3: confidence 캘리브레이션 주입 ─────────────

def test_confidence_for_uses_calibrated_value():
    cal = {"predictive": {"VELOCITY": {"n": 20, "hit_rate": 0.4,
                                       "calibrated_confidence": 47.5}}}
    assert pe.confidence_for("VELOCITY", calibration=cal) == 47.5


def test_confidence_for_falls_back_to_default():
    assert pe.confidence_for("VELOCITY", calibration={}) == 65.0
    assert pe.confidence_for("RS_WEAK", calibration={}) == 60.0
    assert pe.confidence_for("EVENT", calibration={}) == 90.0


def test_confidence_for_ignores_uncalibrated_none():
    """n<10이라 calibrated_confidence=None — 기본값 유지."""
    cal = {"predictive": {"VELOCITY": {"n": 3, "hit_rate": 1.0,
                                       "calibrated_confidence": None}}}
    assert pe.confidence_for("VELOCITY", calibration=cal) == 65.0


def test_evaluate_injects_calibrated_confidence():
    closes = [110.0, 108.0, 106.0, 104.0]
    cal = {"predictive": {"VELOCITY": {"n": 20, "hit_rate": 0.4,
                                       "calibrated_confidence": 47.5}}}
    sigs = pe.evaluate(_mk_holding(price=104.0), stops={"TSLA": 98.0},
                       closes_by_sym={"TSLA": closes}, calibration=cal)
    vel = [s for s in sigs if s.kind == "VELOCITY"]
    assert vel and vel[0].confidence == 47.5
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_predictive_engine.py -k confidence -v` — FAIL (confidence_for 없음)

- [ ] **Step 3: Implement.** 상단 import에 `import json` + `from pathlib import Path` 추가. 상수부에:

```python
_DEFAULT_CONFIDENCE = {"VELOCITY": 65.0, "RS_WEAK": 60.0, "EVENT": 90.0}
_CALIBRATION_PATH = Path(__file__).resolve().parent / "state" / "calibration.json"


def _load_calibration(path: Path | None = None) -> dict:
    """Phase 1 scorer cron이 쓰는 state/calibration.json 로드. 없으면 {}."""
    try:
        return json.loads(Path(path or _CALIBRATION_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def confidence_for(kind: str, calibration: dict | None = None,
                   engine: str = "predictive") -> float:
    """적중률 보정 confidence — calibrated 없으면(n<10 포함) 기본값 fallback."""
    cal = calibration if calibration is not None else _load_calibration()
    entry = (cal.get(engine) or {}).get(kind) or {}
    c = entry.get("calibrated_confidence")
    return float(c) if c is not None else _DEFAULT_CONFIDENCE.get(kind, 50.0)
```

세 evaluator에 `confidence: float | None = None` 파라미터 추가, 내부에서
`conf = confidence if confidence is not None else _DEFAULT_CONFIDENCE["<KIND>"]`로 기존 고정값 대체:
- `evaluate_velocity(..., confidence=None)` → PredictiveSignal(confidence=conf)
- `evaluate_relative_strength(..., confidence=None)` → 동일 (기본 60.0)
- `evaluate_events(..., confidence=None)` → 동일 (기본 90.0)

`evaluate()`에 `calibration: dict | None = None` 추가, 한 번 resolve 후 주입:

```python
def evaluate(
    holdings: list[dict[str, Any]],
    *,
    stops: dict[str, float] | None = None,
    closes_by_sym: dict[str, list[float]] | None = None,
    bench_closes_by_market: dict[str, list[float]] | None = None,
    as_of: date | None = None,
    calibration: dict | None = None,
) -> list[PredictiveSignal]:
    """세 Pillar 통합 → 긴급도 내림차순. confidence는 적중률 보정값 주입."""
    from corvin_jarvis.signal_engine import load_stops
    _stops = stops if stops is not None else load_stops()
    _closes = closes_by_sym or {}
    _bench = bench_closes_by_market or {}
    _date = as_of or date.today()
    _cal = calibration if calibration is not None else _load_calibration()

    sigs: list[PredictiveSignal] = []
    sigs.extend(evaluate_velocity(holdings, _stops, _closes,
                                  confidence=confidence_for("VELOCITY", _cal)))
    sigs.extend(evaluate_relative_strength(holdings, _closes, _bench,
                                           confidence=confidence_for("RS_WEAK", _cal)))
    sigs.extend(evaluate_events(_date, [str(p.get("symbol", "")) for p in holdings],
                                confidence=confidence_for("EVENT", _cal)))
    return sorted(sigs, key=lambda s: s.urgency, reverse=True)
```

- [ ] **Step 4: Run** `python3 -m pytest tests/test_predictive_engine.py -v` — 20 passed (16+4)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/predictive_engine.py tests/test_predictive_engine.py
git commit -m "feat(predictive): confidence 캘리브레이션 주입 — 고정값 제거, 적중률 기반"
```

---

### Task 3: RS_WEAK 적응 임계값

**Files:**
- Modify: `corvin_jarvis/predictive_engine.py`
- Test: `tests/test_predictive_engine.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# ── Phase 3: RS_WEAK 적응 임계값 ─────────────────────

def test_adaptive_threshold_falls_back_on_short_history():
    """이력 부족(90봉 미만) — 기본 -5.0 유지 (기존 동작 보존)."""
    closes = [100.0 + i * 0.1 for i in range(30)]
    bench = list(closes)
    assert pe._adaptive_rs_threshold(closes, bench) == -5.0


def test_adaptive_threshold_widens_for_volatile_pair():
    """변동 큰 종목 — 하위 10분위가 -5보다 깊어짐 (오탐 억제)."""
    import random
    rng = random.Random(42)
    closes, bench = [100.0], [100.0]
    for _ in range(100):
        closes.append(max(1.0, closes[-1] * (1 + rng.uniform(-0.05, 0.048))))
        bench.append(bench[-1] * 1.001)
    thr = pe._adaptive_rs_threshold(closes, bench)
    assert thr < -5.0          # 더 깊은(느슨한) 임계
    assert thr >= -20.0        # 하한 클램프


def test_adaptive_threshold_clamped_upper():
    """안정 페어 — 임계가 -2보다 얕아지지 않게 클램프 (과민 방지)."""
    closes = [100.0 + i * 0.01 for i in range(100)]
    bench = [100.0 + i * 0.012 for i in range(100)]
    thr = pe._adaptive_rs_threshold(closes, bench)
    assert -5.0 <= thr <= -2.0


def test_rs_weak_uses_adaptive_threshold_with_long_history():
    """90봉 이력 — 적응 임계 적용, evidence에 사용 임계 기록."""
    import random
    rng = random.Random(7)
    closes, bench = [100.0], [100.0]
    for _ in range(100):
        closes.append(max(1.0, closes[-1] * (1 + rng.uniform(-0.05, 0.048))))
        bench.append(bench[-1] * 1.001)
    # 최근 20일 급락 페어 추가 — rs가 적응 임계도 뚫도록
    for _ in range(20):
        closes.append(closes[-1] * 0.93)
        bench.append(bench[-1] * 1.001)
    holdings = [{"symbol": "XXX", "market": "US", "price": closes[-1]}]
    sigs = pe.evaluate_relative_strength(holdings, {"XXX": closes}, {"US": bench})
    assert len(sigs) == 1
    assert "threshold_pct" in sigs[0].evidence
    assert sigs[0].evidence["threshold_pct"] != -5.0  # 적응값 사용됨
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_predictive_engine.py -k adaptive -v` — FAIL

- [ ] **Step 3: Implement.** 상수부에:

```python
_RS_HIST_WINDOWS = 60      # 적응 임계 계산용 과거 rs 표본 수
_RS_MIN_SAMPLES = 30       # 미만이면 기본 임계 fallback
_RS_CLAMP = (-20.0, -2.0)  # 적응 임계 안전 클램프
```

헬퍼 추가:

```python
def _adaptive_rs_threshold(closes: list[float], bench: list[float],
                           n_days: int = _RS_N_DAYS,
                           default: float = _RS_THRESHOLD) -> float:
    """종목별 과거 rs 분포의 하위 10분위 — 변동성 맞춤 임계 (스펙 §6).

    rs_i = (종목 n일 수익률 - 벤치 n일 수익률), 과거 _RS_HIST_WINDOWS개 윈도.
    표본 < _RS_MIN_SAMPLES → default(-5.0). 결과는 _RS_CLAMP로 클램프.
    """
    rs_vals: list[float] = []
    for i in range(_RS_HIST_WINDOWS):
        end_c, end_b = len(closes) - i, len(bench) - i
        h = _n_day_return(closes[:end_c], n_days)
        b = _n_day_return(bench[:end_b], n_days)
        if h is None or b is None:
            break
        rs_vals.append(h - b)
    if len(rs_vals) < _RS_MIN_SAMPLES:
        return default
    rs_sorted = sorted(rs_vals)
    thr = rs_sorted[int(len(rs_sorted) * 0.10)]
    lo, hi = _RS_CLAMP
    return max(lo, min(hi, thr))
```

`evaluate_relative_strength` 변경 — `threshold: float | None = None`으로 바꾸고 (기존 기본 `_RS_THRESHOLD` → None), 루프 내 종목별로:

```python
        thr = threshold if threshold is not None else _adaptive_rs_threshold(closes, bench)
        if rs >= thr:
            continue
        urgency = max(30, min(75, int(30 + (thr - rs) * 4)))
```

evidence에 `"threshold_pct": round(thr, 2)` 추가. (기존 테스트가 `threshold=` 인자를 명시 전달하면 그대로 동작 — None일 때만 적응. 기존 픽스처는 짧은 closes라 적응 함수가 default(-5.0) 반환 → 동작 보존.)

- [ ] **Step 4: Run** `python3 -m pytest tests/test_predictive_engine.py -v` — 24 passed (20+4)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/predictive_engine.py tests/test_predictive_engine.py
git commit -m "feat(predictive): RS_WEAK 적응 임계값 — 종목별 rs 분포 하위 10분위"
```

---

### Task 4: snapshot fetch 90일 확대

**Files:**
- Modify: `corvin_jarvis/dashboard/snapshot.py` (`_predictive_signals` 내부 숫자만)

- [ ] **Step 1:** `_predictive_signals`에서:
- `qp.get_stock_daily_closes(s, days=25, ...)` → `days=90` (보유 종목 + SPY 두 곳)
- pykrx KR 벤치: `timedelta(days=55)` → `timedelta(days=130)`, `df["종가"].tail(25)` → `tail(90)`

(적응 임계값은 90봉 이상에서만 활성 — 25봉이면 영원히 fallback이라 이 변경이 §6 활성화 조건.)

- [ ] **Step 2: Run** `python3 -m pytest tests/test_dashboard_snapshot.py tests/test_predictive_engine.py -q` — 전체 PASS (snapshot 테스트는 fetch를 mock하므로 일수 무관)

- [ ] **Step 3: Commit**

```bash
git add corvin_jarvis/dashboard/snapshot.py
git commit -m "feat(dashboard): predictive fetch 90일 — RS 적응 임계값 활성화"
```

---

### Task 5: 전체 회귀 + E2E + 보고

- [ ] **Step 1:** `python3 -m pytest tests/ -q` — 전체 PASS (예상 683+)

- [ ] **Step 2: E2E 실데이터**

```bash
python3 -c "
from corvin_jarvis.dashboard import snapshot
snapshot._CACHE['data'] = None
s = snapshot.build_snapshot()
for sig in s['predictive_signals']:
    print(sig['kind'], sig['symbol'], 'conf:', sig['confidence'], '|', sig['message'][:80])
    ev = sig.get('evidence', {})
    if sig['kind'] == 'VELOCITY':
        print('   atr:', ev.get('atr_proxy'), 'strength:', ev.get('strength'),
              'range:', ev.get('days_lo'), '-', ev.get('days_hi'))
    if sig['kind'] == 'RS_WEAK':
        print('   threshold:', ev.get('threshold_pct'))
"
```

기대: VELOCITY에 atr/strength/범위, RS_WEAK에 적응 임계 표시. calibration.json이 아직 미보정({})이므로 confidence는 기본값 그대로 — 정상 (표본 누적 후 자동 전환).

- [ ] **Step 3:** 대시보드 재시작 + Discord 보고 (폐하 승인 게이트 — Phase 2+3 일괄 main 머지 여부 포함).

---

## Self-Review 결과

- **스펙 §6 커버리지**: ATR 정규화(T1 — 종가 프록시 절충 문서화) · 신뢰구간(T1) · confidence 주입+fallback(T2) · RS 적응 임계+fallback(T3) · 기존 테스트 무수정 유지(전 Task Step 4) — 전부 매핑.
- **하위호환**: 모든 새 파라미터는 기본값 인자(None) — 기존 호출부 무변경. 짧은 closes(<15봉)에서 게이트·적응 임계 자동 비활성 → 기존 13 테스트 보존 근거.
- **타입 일관성**: confidence_for/_load_calibration/_adaptive_rs_threshold 시그니처가 T2·T3·T5 사용처와 일치 확인. evidence 키(atr_proxy/strength/days_lo/days_hi/threshold_pct)가 T5 E2E 스크립트와 일치.
- **노이즈 정책**: 게이트는 신호를 줄이는 방향(억제) — alert noise aversion과 정합.
