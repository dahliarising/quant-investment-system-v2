# 상방(수익) 레이어 — 손절 대칭화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 또는 executing-plans. Steps use `- [ ]`.

**Goal:** 방어(손절)만 있던 시스템에 대칭 상방 레이어 — 평면 +25% "승자 매도"를 *추적 익절*(let winners run)로, 하락 전용 VELOCITY에 *상방 도달일*을 추가.

**Architecture:** 손절 작업(ATR/버킷)의 거울상. 추적 익절은 고점 PnL 대비 ATR 비례 되돌림으로 판정(승자는 달리되 이익 잠금). peak_pnl은 closes·price·pnl에서 *무상태* 유도(avg = price/(1+pnl/100)). 상방 VELOCITY는 evaluate_velocity의 상승 미러.

**Tech Stack:** Python 3.9, pytest, 기존 predictive_engine._slope/_atr_proxy 재사용.

**근거:** 폐하 2026-06-11 지적 — "왜 수익 없이 손절만?". 현재 `_TAKE_PROFIT=25.0` 평면 1개 + +25% 비중축소(승자 매도) = 손실회피 강화. [[feedback_investor_psychology]]

---

### Task 1: 추적 익절 (verdict.py)

**Files:** Modify `corvin_jarvis/signals/verdict.py`, Test `tests/test_verdict.py` (append)

- [ ] **Step 1: 상수 + 헬퍼**

```python
_TP_ACTIVATE = 15.0        # 이 수익% 넘어야 추적 가동 (그 전엔 달리게)
_TP_GIVEBACK_K = 3.0       # 고점에서 K×ATR% 되돌리면 익절
_TP_MIN_GIVEBACK = 5.0     # ATR 없을 때 최소 되돌림%
```

```python
def _trailing_take_profit(pnl_pct, peak_pnl_pct, atr_pct,
                          activate=_TP_ACTIVATE, k=_TP_GIVEBACK_K):
    """고점 대비 되돌림이 ATR 비례 임계 넘으면 익절. peak None→평면 fallback."""
    if pnl_pct is None:
        return False
    if peak_pnl_pct is None:
        return pnl_pct >= _TAKE_PROFIT
    peak = max(peak_pnl_pct, pnl_pct)
    if peak < activate:
        return False
    trail = max(_TP_MIN_GIVEBACK, k * atr_pct) if atr_pct else _TP_MIN_GIVEBACK
    return (peak - pnl_pct) >= trail
```

- [ ] **Step 2: decide() 익절선 교체** — `if pnl >= _TAKE_PROFIT: 비중축소` → `_trailing_take_profit(...)` 사용, 근거에 "추적 익절 — 고점 +X%" 표기. 양 버킷 공통(승자 보호는 dca도).

- [ ] **Step 3: for_symbol() peak_pnl_pct·atr_pct 주입** — atr_pct를 모든 held로 확대(버킷 무관, 익절도 사용). peak: `avg=price/(1+pnl/100); peak_pnl=(max(closes+[price])/avg-1)*100`.

- [ ] **Step 4: 테스트** (아래) — RED→GREEN. 기존 `test_held_take_profit_is_trim`(pnl=26, peak None) 보존.

```python
def test_trailing_tp_lets_winner_run():
    # +20%, 고점 +22%, ATR2 → 되돌림 2 < max(5,6) → 익절 아님(홀딩)
    v = verdict.decide(_ctx(held=True, pnl_pct=20.0, peak_pnl_pct=22.0,
                            atr_pct=2.0, theme_alive=True))
    assert v.action == "홀딩"

def test_trailing_tp_triggers_on_pullback():
    # +18%, 고점 +30%, ATR2 → 되돌림 12 ≥ 6 → 비중축소
    v = verdict.decide(_ctx(held=True, pnl_pct=18.0, peak_pnl_pct=30.0, atr_pct=2.0))
    assert v.action == "비중축소" and "추적" in v.rationale

def test_trailing_tp_inactive_below_activate():
    # 고점 +12% < 15% 가동선 → 익절 안 함
    v = verdict.decide(_ctx(held=True, pnl_pct=10.0, peak_pnl_pct=12.0,
                            atr_pct=1.0, theme_alive=True))
    assert v.action == "홀딩"

def test_trailing_tp_flat_fallback_no_peak():
    # peak 미제공 → 평면 +25% 보존
    v = verdict.decide(_ctx(held=True, pnl_pct=26.0))
    assert v.action == "비중축소"

def test_trailing_tp_min_giveback_without_atr():
    # ATR 없음 → 최소 되돌림 5%p. +24%, 고점 +30% → 되돌림 6 ≥ 5 → 익절
    v = verdict.decide(_ctx(held=True, pnl_pct=24.0, peak_pnl_pct=30.0))
    assert v.action == "비중축소"

def test_trailing_helper_units():
    assert verdict._trailing_take_profit(20.0, 22.0, 2.0) is False
    assert verdict._trailing_take_profit(18.0, 30.0, 2.0) is True
    assert verdict._trailing_take_profit(26.0, None, None) is True
    assert verdict._trailing_take_profit(10.0, 12.0, 1.0) is False
```

- [ ] **Step 5: Commit** `feat(signals): 추적 익절 — 평면 +25% 매도 → 고점대비 ATR 되돌림(승자 달리게)`

---

### Task 2: 상방 VELOCITY (predictive_engine.py)

**Files:** Modify `corvin_jarvis/predictive_engine.py`, Test `tests/test_predictive_engine.py` (append)

- [ ] **Step 1: evaluate_upside_velocity** — evaluate_velocity 미러. 상승추세(slope>0) + 현재가 < 최근고점(target=max(closes))일 때 도달일 경보. 노이즈 게이트 동일(_NOISE_GATE). kind="VELOCITY_UP".

```python
def evaluate_upside_velocity(holdings, closes_by_sym,
                             horizon=_VELOCITY_HORIZON, confidence=None):
    """상승 속도로 최근 고점(저항) 도달 예상일 — VELOCITY 상방 미러."""
    out = []
    conf = confidence if confidence is not None else _DEFAULT_CONFIDENCE.get("VELOCITY_UP", 60.0)
    for pos in holdings:
        sym = str(pos.get("symbol", ""))
        closes = closes_by_sym.get(sym, [])
        s = _slope(closes)
        if s is None or s <= 0:
            continue
        price = pos.get("price") or (closes[-1] if closes else None)
        if price is None or not closes:
            continue
        target = max(closes)
        if price >= target:
            continue  # 이미 고점 돌파 — 별개 상황
        atr = _atr_proxy(closes)
        strength = (s / atr) if atr else None
        if strength is not None and strength < _NOISE_GATE:
            continue
        days_to = (target - price) / s
        if days_to > horizon:
            continue
        urgency = max(30, min(70, int(30 + (1 - days_to / horizon) * 40)))
        out.append(PredictiveSignal(
            symbol=sym, kind="VELOCITY_UP", urgency=urgency, confidence=conf,
            horizon_days=round(days_to),
            message=f"상승 속도 기준 최근고점({_fmt(target)}) ~{round(days_to)}일 내 도달 가능",
            evidence={"slope_per_day": round(s, 4), "days_to_target": round(days_to, 1),
                      "target": target, "current_price": price,
                      "atr_proxy": round(atr, 4) if atr else None,
                      "strength": round(strength, 3) if strength is not None else None},
        ))
    return out
```

- [ ] **Step 2: _DEFAULT_CONFIDENCE에 "VELOCITY_UP": 60.0 추가**

- [ ] **Step 3: evaluate() 통합** — closes 있으면 upside velocity도 결과에 합류 (기존 velocity와 나란히).

- [ ] **Step 4: 테스트**

```python
@pytest.mark.unit
def test_upside_velocity_fires_on_uptrend_below_high():
    closes = [100.0 + i for i in range(20)]  # +1/일 상승, 고점 119
    h = [{"symbol": "X", "market": "US", "price": 119.0}]
    sigs = pe.evaluate_upside_velocity(h, {"X": closes + [119.0]})
    # 가격 119 == max → skip; 가격 더 낮으면 발화
    h2 = [{"symbol": "X", "price": 115.0}]
    sigs2 = pe.evaluate_upside_velocity(h2, {"X": closes})
    assert any(s.kind == "VELOCITY_UP" for s in sigs2)

@pytest.mark.unit
def test_upside_velocity_skips_downtrend():
    closes = [120.0 - i for i in range(20)]
    sigs = pe.evaluate_upside_velocity([{"symbol": "X", "price": 101.0}], {"X": closes})
    assert sigs == []

@pytest.mark.unit
def test_upside_velocity_noise_gated():
    import random
    rng = random.Random(1)
    closes = [100.0]
    for _ in range(20):
        closes.append(closes[-1] + rng.uniform(-2, 2.1))  # 큰 변동, 미미한 순기울기
    sigs = pe.evaluate_upside_velocity([{"symbol": "X", "price": closes[-1]-1}], {"X": closes})
    # 노이즈 대비 약한 기울기는 억제 (발화해도 strength≥gate 보장 안되면 빈 리스트)
    for s in sigs:
        assert s.evidence["strength"] is None or s.evidence["strength"] >= _NOISE_GATE
```

- [ ] **Step 5: Commit** `feat(predictive): 상방 VELOCITY — 최근고점 도달 예상일(하락 미러)`

---

### Task 3: 전체 회귀 + E2E + 보고

- [ ] **Step 1:** `pytest tests/ -q` 전체 PASS
- [ ] **Step 2:** E2E — verdict 추적익절 시나리오 + snapshot 상방 velocity 노출 확인
- [ ] **Step 3:** 코드리뷰(subagent) → HIGH 수정 → 폐하 승인 후 머지

## Self-Review
- 폐하 지적 커버: 평면 익절→추적(T1) · 상방 도달일(T2) · 승자 달리게(T1 핵심). 불타기/사다리(③④)는 후속.
- 회귀 0: peak None·atr None → 평면 _TAKE_PROFIT fallback (T1 Step1). 기존 take_profit 테스트 보존.
- 타입 일관: _trailing_take_profit(pnl,peak,atr)→bool, ctx 키 peak_pnl_pct·atr_pct.
