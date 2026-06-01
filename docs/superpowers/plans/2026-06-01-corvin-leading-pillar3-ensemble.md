# Corvin 선행 인텔리전스 — Pillar 3 앙상블 스크리너 (Plan 3) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 단일 신호 대신 여러 방법론(Minervini 추세 + 거래량 돌파 + 멀티타임프레임 상대강도)의 복합 점수를 산출하는 Pillar 3을 만든다. xang1234/stock-screener 앙상블 패턴 차용.

**Architecture:** 각 방법론은 순수 함수(지표 입력 → 0–100). `ensemble_score`가 가중 합산 → `build_ensemble_signal`이 LeadingSignal 변환. 라이브 가격/지표 계산은 호출부(기존 dca_timing 헬퍼 재사용).

**Tech Stack:** Python 3, pytest, 기존 `LeadingSignal` 계약.

**테스트 실행:** 루트에서 `python3 -m pytest tests/<file> -v`.

---

## 파일 구조

| 파일 | 책임 | 신규/수정 |
|------|------|-----------|
| `corvin_jarvis/signals/ensemble.py` | Minervini/거래량/RS → 복합 점수 → LeadingSignal | 신규 |
| `tests/test_ensemble.py` | 순수 함수 + 통합 | 신규 |

---

## Task 1: Minervini 추세 템플릿 점수 (순수)

**Files:**
- Create: `corvin_jarvis/signals/ensemble.py`
- Test: `tests/test_ensemble.py`

6개 추세 기준 통과율 → 0–100.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ensemble.py
"""Tests for corvin_jarvis.signals.ensemble."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals import ensemble
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_minervini_all_criteria_pass():
    # 완벽한 상승추세: price > 모든 MA, MA 정렬, 52주 위치 양호
    score = ensemble.minervini_trend_score(
        price=120.0, ma50=110.0, ma150=100.0, ma200=90.0,
        low_52w=70.0, high_52w=125.0,
    )
    assert score == 100.0


@pytest.mark.unit
def test_minervini_downtrend_low_score():
    # 하락추세: price < MA, 역배열
    score = ensemble.minervini_trend_score(
        price=80.0, ma50=90.0, ma150=100.0, ma200=110.0,
        low_52w=78.0, high_52w=160.0,
    )
    assert score <= 35.0


@pytest.mark.unit
def test_minervini_handles_none():
    # 지표 부족 시 None → NEUTRAL 50
    score = ensemble.minervini_trend_score(
        price=100.0, ma50=None, ma150=None, ma200=None,
        low_52w=None, high_52w=None,
    )
    assert score == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ensemble.py -k minervini -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/ensemble.py
"""Corvin 선행 인텔리전스 — Pillar 3: 앙상블 스크리너.

여러 방법론의 복합 점수 (단일 신호 아님). xang1234/stock-screener 패턴 차용:
- Minervini 추세 템플릿 (MA 정렬 + 52주 위치)
- 거래량 돌파
- 멀티타임프레임 상대강도(RS)

순수 함수 (지표 입력 → 0-100). 라이브 계산은 호출부(dca_timing 헬퍼 재사용).
"""
from __future__ import annotations

from typing import Any

from corvin_jarvis.signals.leading_signal import LeadingSignal

NEUTRAL = 50.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def minervini_trend_score(
    price: float,
    ma50: float | None,
    ma150: float | None,
    ma200: float | None,
    low_52w: float | None,
    high_52w: float | None,
) -> float:
    """Minervini 추세 템플릿 6기준 통과율 → 0-100. 지표 전무하면 NEUTRAL."""
    criteria: list[bool] = []
    if ma150 is not None and ma200 is not None:
        criteria.append(price > ma150 and price > ma200)  # 1. price > MA150/200
        criteria.append(ma150 > ma200)                    # 2. MA150 > MA200
    if ma50 is not None and ma150 is not None and ma200 is not None:
        criteria.append(ma50 > ma150 > ma200)             # 3. MA 정배열
    if ma50 is not None:
        criteria.append(price > ma50)                     # 4. price > MA50
    if low_52w is not None and low_52w > 0:
        criteria.append(price >= low_52w * 1.30)          # 5. 52주 저가 +30%↑
    if high_52w is not None and high_52w > 0:
        criteria.append(price >= high_52w * 0.75)         # 6. 52주 고가 -25%내
    if not criteria:
        return NEUTRAL
    return round(sum(criteria) / len(criteria) * 100.0, 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ensemble.py -k minervini -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/ensemble.py tests/test_ensemble.py
git commit -m "feat(corvin): Minervini trend score (Pillar 3)"
```

---

## Task 2: 거래량 돌파 + 멀티타임프레임 RS 점수 (순수)

**Files:**
- Modify: `corvin_jarvis/signals/ensemble.py`
- Test: `tests/test_ensemble.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ensemble.py 에 추가
@pytest.mark.unit
def test_volume_breakthrough_high():
    # 거래량 평균 2배 → 높은 점수
    assert ensemble.volume_breakthrough_score(vol=200.0, vol_avg=100.0) >= 80


@pytest.mark.unit
def test_volume_breakthrough_normal():
    assert 40 <= ensemble.volume_breakthrough_score(vol=100.0, vol_avg=100.0) <= 60


@pytest.mark.unit
def test_volume_breakthrough_none():
    assert ensemble.volume_breakthrough_score(vol=None, vol_avg=100.0) == 50.0


@pytest.mark.unit
def test_multi_timeframe_rs_strong():
    # 모든 타임프레임서 벤치 대비 강세
    rs = {"1w": 3.0, "1m": 5.0, "3m": 8.0, "6m": 12.0}
    assert ensemble.multi_timeframe_rs_score(rs) >= 70


@pytest.mark.unit
def test_multi_timeframe_rs_weak():
    rs = {"1w": -3.0, "1m": -5.0, "3m": -8.0, "6m": -12.0}
    assert ensemble.multi_timeframe_rs_score(rs) <= 30


@pytest.mark.unit
def test_multi_timeframe_rs_empty():
    assert ensemble.multi_timeframe_rs_score({}) == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ensemble.py -k "volume or multi_timeframe" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/ensemble.py 에 추가

def volume_breakthrough_score(vol: float | None, vol_avg: float | None) -> float:
    """거래량/평균 비율 → 0-100. 1배=50, 2배+=100. None이면 NEUTRAL."""
    if vol is None or vol_avg is None or vol_avg <= 0:
        return NEUTRAL
    ratio = vol / vol_avg
    return _clamp(50.0 + (ratio - 1.0) * 50.0)


def multi_timeframe_rs_score(rs_by_window: dict[str, float]) -> float:
    """타임프레임별 상대강도(%p) 평균 → 0-100. ±10%p 기준. 비면 NEUTRAL."""
    if not rs_by_window:
        return NEUTRAL
    avg_rs = sum(rs_by_window.values()) / len(rs_by_window)
    return _clamp(50.0 + avg_rs * 4.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ensemble.py -k "volume or multi_timeframe" -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/ensemble.py tests/test_ensemble.py
git commit -m "feat(corvin): volume breakthrough + multi-timeframe RS (Pillar 3)"
```

---

## Task 3: 앙상블 복합 점수 + LeadingSignal

**Files:**
- Modify: `corvin_jarvis/signals/ensemble.py`
- Test: `tests/test_ensemble.py`

3개 방법론 가중 합산(Minervini 40 / RS 40 / 거래량 20) → 복합 점수 → 신호.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ensemble.py 에 추가
@pytest.mark.unit
def test_ensemble_score_weighted():
    score = ensemble.ensemble_score(minervini=100.0, rs=75.0, volume=50.0)
    # 100*0.4 + 75*0.4 + 50*0.2 = 40 + 30 + 10 = 80
    assert abs(score - 80.0) < 0.01


@pytest.mark.unit
def test_build_ensemble_signal_bull():
    sig = ensemble.build_ensemble_signal(
        symbol="012450", minervini=90.0, rs=80.0, volume=70.0,
    )
    assert isinstance(sig, LeadingSignal)
    assert sig.pillar == "ensemble"
    assert sig.direction == "bull"
    assert sig.horizon == "days"
    assert sig.advisory is False
    assert sig.score is not None and sig.score >= 70
    assert sig.confidence >= 60


@pytest.mark.unit
def test_build_ensemble_signal_bear():
    sig = ensemble.build_ensemble_signal(
        symbol="META", minervini=20.0, rs=25.0, volume=40.0,
    )
    assert sig.direction == "bear"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ensemble.py -k "ensemble_score or build_ensemble" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/ensemble.py 에 추가

W_MINERVINI = 0.4
W_RS = 0.4
W_VOLUME = 0.2


def ensemble_score(minervini: float, rs: float, volume: float) -> float:
    """3방법론 가중 합산 → 0-100."""
    return round(
        minervini * W_MINERVINI + rs * W_RS + volume * W_VOLUME, 2
    )


def _direction(score: float) -> str:
    if score >= 70.0:
        return "bull"
    if score < 40.0:
        return "bear"
    return "neutral"


def build_ensemble_signal(
    symbol: str, minervini: float, rs: float, volume: float
) -> LeadingSignal:
    """앙상블 복합 점수 → LeadingSignal."""
    score = ensemble_score(minervini, rs, volume)
    direction = _direction(score)
    confidence = _clamp(abs(score - NEUTRAL) * 2.0)
    label = {"bull": "강세", "bear": "약세", "neutral": "중립"}[direction]
    return LeadingSignal(
        pillar="ensemble", symbol=symbol, direction=direction,
        confidence=confidence, score=score, horizon="days",
        advisory=False,
        message=f"🎯 {symbol} 앙상블 {label} (복합 {score:.0f}/100)",
        evidence={"minervini": minervini, "rs": rs, "volume": volume,
                  "composite": score},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ensemble.py -k "ensemble_score or build_ensemble" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/ensemble.py tests/test_ensemble.py
git commit -m "feat(corvin): ensemble composite score + signal (Pillar 3)"
```

---

## Task 4: 전체 회귀 확인

- [ ] **Step 1: 전체 테스트**

Run: `python3 -m pytest tests/ -q`
Expected: 전체 PASS, 무회귀.

- [ ] **Step 2: Commit (slice marker)**

```bash
git commit -m "test(corvin): Pillar 3 ensemble screener verified" --allow-empty
```

---

## 자체 검토 (spec 대비)

- **Spec §3 Pillar 3: 다중 방법론 병렬** → Task 1,2 (Minervini/거래량/RS) ✅
- **Spec §3 Pillar 3: 멀티타임프레임 상대강도(1W/1M/3M/6M)** → Task 2 `multi_timeframe_rs_score` ✅
- **Spec §3 Pillar 3: 복합 점수** → Task 3 ✅
- **공통 LeadingSignal 계약** → Task 3 ✅
- **DCA score 재사용**: 기존 `dca_timing._composite_score`는 "눌림 매수(oversold)" 관점, Minervini는 "추세 강도" 관점 — 상반된 시각. 앙상블은 추세 관점을 채택(선행 강세 포착 목적). DCA는 별도 Pillar로 이미 존재 → 중복 통합 안 함(YAGNI).
