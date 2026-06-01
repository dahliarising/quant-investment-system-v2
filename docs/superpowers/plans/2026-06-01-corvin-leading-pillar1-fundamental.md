# Corvin 선행 인텔리전스 — Pillar 1 펀더멘털 성장 엔진 (Plan 2) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 종목의 성장 방향성을 0–100 Growth Score로 산출하는 Pillar 1을 만든다. 재무 수치(50%) + Claude 정성(30%) + 뉴스 감정(20%)을 통합하고, `LeadingSignal`로 변환한다.

**Architecture:** 모든 점수 산출은 **순수 함수**(입력 dict → float)로 만들어 라이브 API 없이 테스트한다. 라이브 fetch(yfinance/Claude/GDELT)는 호출부 책임으로 분리. Claude 정성 점수가 없으면(API 실패) 재무+뉴스만으로 graceful degrade.

**Tech Stack:** Python 3, pytest, dataclass, 기존 `LeadingSignal` 계약 재사용.

**테스트 실행:** 루트에서 `python3 -m pytest tests/<file> -v`.

---

## 파일 구조

| 파일 | 책임 | 신규/수정 |
|------|------|-----------|
| `corvin_jarvis/signals/fundamental.py` | 재무/뉴스/정성 점수 → Growth Score → LeadingSignal | 신규 |
| `tests/test_fundamental.py` | 순수 함수 + 통합 테스트 | 신규 |

---

## Task 1: 재무 성장 점수 (순수 함수)

**Files:**
- Create: `corvin_jarvis/signals/fundamental.py`
- Test: `tests/test_fundamental.py`

재무 지표 4종 → 0–100. 각 지표를 구간 매핑 후 평균. 입력 누락 시 해당 지표 제외(가용 지표 평균).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fundamental.py
"""Tests for corvin_jarvis.signals.fundamental."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals import fundamental


@pytest.mark.unit
def test_financial_growth_score_strong():
    # 고성장: EPS +30%, 영업이익률 추세 +, 부채 낮음, 매출 +20%
    metrics = {
        "eps_growth_yoy": 0.30,
        "op_margin_trend": 0.05,    # 마진 개선
        "debt_ratio": 0.3,
        "revenue_growth_yoy": 0.20,
    }
    score = fundamental.financial_growth_score(metrics)
    assert 70 <= score <= 100


@pytest.mark.unit
def test_financial_growth_score_weak():
    metrics = {
        "eps_growth_yoy": -0.20,
        "op_margin_trend": -0.05,
        "debt_ratio": 2.5,           # 고부채
        "revenue_growth_yoy": -0.10,
    }
    score = fundamental.financial_growth_score(metrics)
    assert 0 <= score <= 35


@pytest.mark.unit
def test_financial_growth_score_partial_metrics():
    # 일부 지표만 있어도 가용 지표로 산출
    score = fundamental.financial_growth_score({"eps_growth_yoy": 0.15})
    assert 0 <= score <= 100


@pytest.mark.unit
def test_financial_growth_score_empty_returns_neutral():
    assert fundamental.financial_growth_score({}) == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fundamental.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corvin_jarvis.signals.fundamental'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/fundamental.py
"""Corvin 선행 인텔리전스 — Pillar 1: 펀더멘털 성장 엔진.

종목 성장 방향성을 0-100 Growth Score로 산출.
구성: 재무 수치(50%) + Claude 정성(30%) + 뉴스 감정(20%).

모든 점수 함수는 순수(입력 dict → float). 라이브 fetch는 호출부 책임.
Claude 정성 점수 없으면 재무+뉴스로 graceful degrade.
"""
from __future__ import annotations

from typing import Any

from corvin_jarvis.signals.leading_signal import LeadingSignal

NEUTRAL = 50.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _score_eps_growth(g: float) -> float:
    # -20% → 10, 0% → 50, +30%+ → 90
    return _clamp(50.0 + g * 133.0)


def _score_margin_trend(t: float) -> float:
    # 마진 개선폭 ±5%p 기준
    return _clamp(50.0 + t * 600.0)


def _score_debt_ratio(d: float) -> float:
    # 부채비율 0 → 90, 1.0 → 50, 2.5+ → 10
    return _clamp(90.0 - d * 32.0)


def _score_revenue_growth(g: float) -> float:
    return _clamp(50.0 + g * 150.0)


def financial_growth_score(metrics: dict[str, Any]) -> float:
    """재무 4지표 → 0-100. 가용 지표만 평균. 전무하면 NEUTRAL."""
    parts: list[float] = []
    if "eps_growth_yoy" in metrics:
        parts.append(_score_eps_growth(float(metrics["eps_growth_yoy"])))
    if "op_margin_trend" in metrics:
        parts.append(_score_margin_trend(float(metrics["op_margin_trend"])))
    if "debt_ratio" in metrics:
        parts.append(_score_debt_ratio(float(metrics["debt_ratio"])))
    if "revenue_growth_yoy" in metrics:
        parts.append(_score_revenue_growth(float(metrics["revenue_growth_yoy"])))
    if not parts:
        return NEUTRAL
    return round(sum(parts) / len(parts), 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_fundamental.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/fundamental.py tests/test_fundamental.py
git commit -m "feat(corvin): financial growth score (Pillar 1)"
```

---

## Task 2: 뉴스 감정 점수 (순수 함수)

**Files:**
- Modify: `corvin_jarvis/signals/fundamental.py`
- Test: `tests/test_fundamental.py`

GDELT/narrative의 sentiment_tone(-1~+1 추정) → 0–100 정규화.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fundamental.py 에 추가
@pytest.mark.unit
def test_news_sentiment_score_positive():
    assert fundamental.news_sentiment_score(0.5) > 60


@pytest.mark.unit
def test_news_sentiment_score_negative():
    assert fundamental.news_sentiment_score(-0.5) < 40


@pytest.mark.unit
def test_news_sentiment_score_none_returns_neutral():
    assert fundamental.news_sentiment_score(None) == 50.0


@pytest.mark.unit
def test_news_sentiment_score_clamps():
    assert fundamental.news_sentiment_score(5.0) == 100.0
    assert fundamental.news_sentiment_score(-5.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fundamental.py -k news_sentiment -v`
Expected: FAIL — `AttributeError: ... has no attribute 'news_sentiment_score'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/fundamental.py 에 추가

def news_sentiment_score(tone: float | None) -> float:
    """뉴스 감정 tone(-1~+1) → 0-100. None이면 NEUTRAL."""
    if tone is None:
        return NEUTRAL
    return _clamp(50.0 + float(tone) * 50.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_fundamental.py -k news_sentiment -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/fundamental.py tests/test_fundamental.py
git commit -m "feat(corvin): news sentiment score (Pillar 1)"
```

---

## Task 3: Growth Score 통합 + 방향 판정 (순수, graceful degrade)

**Files:**
- Modify: `corvin_jarvis/signals/fundamental.py`
- Test: `tests/test_fundamental.py`

가중 평균(재무 50 / 정성 30 / 뉴스 20). 정성 None이면 재무·뉴스 가중치 재정규화.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fundamental.py 에 추가
@pytest.mark.unit
def test_combine_growth_all_present():
    score = fundamental.combine_growth_score(
        financial=80.0, qualitative=70.0, news=60.0
    )
    # 80*0.5 + 70*0.3 + 60*0.2 = 40 + 21 + 12 = 73
    assert abs(score - 73.0) < 0.01


@pytest.mark.unit
def test_combine_growth_qualitative_none_renormalizes():
    # 정성 빠지면 재무 50/70 + 뉴스 20/70 비율로 재정규화
    score = fundamental.combine_growth_score(
        financial=80.0, qualitative=None, news=60.0
    )
    # (80*0.5 + 60*0.2) / 0.7 = (40+12)/0.7 = 74.2857
    assert abs(score - 74.29) < 0.1


@pytest.mark.unit
def test_growth_to_direction():
    assert fundamental.growth_to_direction(75.0) == "bull"
    assert fundamental.growth_to_direction(50.0) == "neutral"
    assert fundamental.growth_to_direction(30.0) == "bear"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fundamental.py -k "combine or direction" -v`
Expected: FAIL — `AttributeError: ... 'combine_growth_score'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/fundamental.py 에 추가

W_FINANCIAL = 0.5
W_QUALITATIVE = 0.3
W_NEWS = 0.2


def combine_growth_score(
    financial: float, qualitative: float | None, news: float
) -> float:
    """가중 평균 Growth Score. qualitative None이면 가중치 재정규화."""
    pairs: list[tuple[float, float]] = [(financial, W_FINANCIAL), (news, W_NEWS)]
    if qualitative is not None:
        pairs.append((qualitative, W_QUALITATIVE))
    total_w = sum(w for _, w in pairs)
    return round(sum(v * w for v, w in pairs) / total_w, 2)


def growth_to_direction(score: float) -> str:
    """Growth Score → bull/neutral/bear."""
    if score >= 70.0:
        return "bull"
    if score < 40.0:
        return "bear"
    return "neutral"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_fundamental.py -k "combine or direction" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/fundamental.py tests/test_fundamental.py
git commit -m "feat(corvin): growth score combine + direction (Pillar 1)"
```

---

## Task 4: 펀더멘털 → LeadingSignal 생성

**Files:**
- Modify: `corvin_jarvis/signals/fundamental.py`
- Test: `tests/test_fundamental.py`

순수 함수: 이미 계산된 sub-score 입력 → LeadingSignal. confidence는 Growth Score가 중립(50)에서 멀수록 높게(확신도) 매핑.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fundamental.py 에 추가
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_build_fundamental_signal_bull():
    sig = fundamental.build_fundamental_signal(
        symbol="012450", financial=85.0, qualitative=80.0, news=70.0,
    )
    assert isinstance(sig, LeadingSignal)
    assert sig.pillar == "fundamental"
    assert sig.symbol == "012450"
    assert sig.direction == "bull"
    assert sig.score is not None and sig.score >= 70
    assert sig.horizon == "weeks"
    assert sig.advisory is False
    # 강한 방향(중립서 멂) → 높은 confidence
    assert sig.confidence >= 60


@pytest.mark.unit
def test_build_fundamental_signal_neutral_low_confidence():
    sig = fundamental.build_fundamental_signal(
        symbol="META", financial=52.0, qualitative=50.0, news=48.0,
    )
    assert sig.direction == "neutral"
    # 중립 근처 → 낮은 confidence (게이트서 무음될 수 있음)
    assert sig.confidence < 60


@pytest.mark.unit
def test_build_fundamental_signal_degrades_without_qualitative():
    sig = fundamental.build_fundamental_signal(
        symbol="NVDA", financial=80.0, qualitative=None, news=70.0,
    )
    assert sig.evidence["qualitative_used"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fundamental.py -k build_fundamental -v`
Expected: FAIL — `AttributeError: ... 'build_fundamental_signal'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/fundamental.py 에 추가

def _confidence_from_score(score: float) -> float:
    """Growth Score가 중립(50)에서 멀수록 confidence 높게. |score-50|*2 → 0-100."""
    return _clamp(abs(score - NEUTRAL) * 2.0)


def build_fundamental_signal(
    symbol: str,
    financial: float,
    qualitative: float | None,
    news: float,
) -> LeadingSignal:
    """sub-score → Growth Score → LeadingSignal."""
    growth = combine_growth_score(financial, qualitative, news)
    direction = growth_to_direction(growth)
    confidence = _confidence_from_score(growth)
    label = {"bull": "성장", "bear": "둔화", "neutral": "중립"}[direction]
    return LeadingSignal(
        pillar="fundamental", symbol=symbol, direction=direction,
        confidence=confidence, score=growth, horizon="weeks",
        advisory=False,
        message=f"📊 {symbol} 펀더멘털 {label} (Growth {growth:.0f}/100)",
        evidence={
            "financial": financial, "news": news,
            "qualitative": qualitative,
            "qualitative_used": qualitative is not None,
            "growth_score": growth,
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_fundamental.py -k build_fundamental -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/fundamental.py tests/test_fundamental.py
git commit -m "feat(corvin): fundamental signal builder (Pillar 1)"
```

---

## Task 5: 전체 회귀 확인

- [ ] **Step 1: 전체 테스트**

Run: `python3 -m pytest tests/ -q`
Expected: 전체 PASS, 무회귀.

- [ ] **Step 2: Commit (slice marker)**

```bash
git commit -m "test(corvin): Pillar 1 fundamental engine verified" --allow-empty
```

---

## 자체 검토 (spec 대비)

- **Spec §3 Pillar 1: 재무 수치** → Task 1 ✅
- **Spec §3 Pillar 1: 뉴스 감정** → Task 2 ✅
- **Spec §3 Pillar 1: 정성 + graceful degrade** → Task 3 (combine None 처리) ✅. ⚠️ Claude haiku 라이브 호출 어댑터는 **라이브 fetch 계층**으로 이 plan 범위 밖(순수 점수만). 정성 점수는 호출부가 주입. 후속 통합(Plan 4 cron 배선)에서 실제 Claude 호출 어댑터 추가.
- **Spec §3 Pillar 1: Growth Score 0-100 + Bull/Neutral/Bear** → Task 3,4 ✅
- **공통 LeadingSignal 계약** → Task 4 ✅
- **노이즈 게이트 연동**: confidence가 중립서 멀수록 높게 → 약한 신호 자동 무음 ✅
