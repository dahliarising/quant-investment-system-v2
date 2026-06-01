# Corvin 선행 인텔리전스 — 재무 metrics 어댑터 (Plan 6) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 펀더멘털 Pillar를 라이브로 켠다. US(yfinance)·KR(pykrx 2시점 EPS) 재무 데이터를 `financial_growth_score` 입력 metrics로 변환하는 파서/어댑터를 만들고 `run_leading`에 배선한다.

**Architecture:** raw API 응답 → metrics dict 변환은 **순수 파서**(mock 데이터로 테스트). 라이브 fetch(yfinance.info / pykrx)는 thin 어댑터. 데이터 없으면 None → 해당 종목 펀더멘털 스킵(추측 금지). KR은 현재/1년전 EPS 2시점으로 성장률 산출.

**Tech Stack:** Python 3, pytest, yfinance(US만 — KR 금지), pykrx(KR).

**메모리 준수:** yfinance는 **US 종목만** (KR 재무는 pykrx). 테스트 실송 0.

---

## 파일 구조

| 파일 | 책임 | 신규/수정 |
|------|------|-----------|
| `corvin_jarvis/financial_metrics.py` | raw API → metrics 파서 + fetch 어댑터 | 신규 |
| `corvin_jarvis/run_leading.py` | `_metrics_fetcher` 실제 배선 | 수정 |
| `tests/test_financial_metrics.py` | 파서 테스트(mock raw) | 신규 |

---

## Task 1: US 재무 파서 (순수)

**Files:**
- Create: `corvin_jarvis/financial_metrics.py`
- Test: `tests/test_financial_metrics.py`

yfinance `.info` dict → metrics. 가용 필드만 매핑.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_financial_metrics.py
"""Tests for corvin_jarvis.financial_metrics (mock raw, no real API)."""
from __future__ import annotations

import pytest

from corvin_jarvis import financial_metrics as fm


@pytest.mark.unit
def test_parse_us_financials_full():
    info = {
        "earningsGrowth": 0.62,
        "revenueGrowth": 0.33,
        "debtToEquity": 35.6,       # percent form
        "operatingMargins": 0.40,
    }
    m = fm.parse_us_financials(info)
    assert abs(m["eps_growth_yoy"] - 0.62) < 1e-9
    assert abs(m["revenue_growth_yoy"] - 0.33) < 1e-9
    assert abs(m["debt_ratio"] - 0.356) < 1e-6   # 35.6/100


@pytest.mark.unit
def test_parse_us_financials_partial():
    m = fm.parse_us_financials({"earningsGrowth": 0.1})
    assert m == {"eps_growth_yoy": 0.1}


@pytest.mark.unit
def test_parse_us_financials_empty():
    assert fm.parse_us_financials({}) == {}


@pytest.mark.unit
def test_parse_us_financials_ignores_none():
    m = fm.parse_us_financials({"earningsGrowth": None, "revenueGrowth": 0.2})
    assert "eps_growth_yoy" not in m
    assert m["revenue_growth_yoy"] == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_financial_metrics.py -k us_financials -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/financial_metrics.py
"""Corvin 선행 인텔리전스 — 재무 metrics 어댑터.

raw API 응답(yfinance.info / pykrx EPS)을 financial_growth_score 입력
metrics dict로 변환. 파서는 순수(테스트 가능), fetch는 thin 어댑터.

⚠️ yfinance는 US 종목만 (메모리 feedback_no_yfinance_kr). KR은 pykrx.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("corvin.financial_metrics")


def parse_us_financials(info: dict[str, Any]) -> dict[str, float]:
    """yfinance .info → metrics. 가용/유효 필드만. debtToEquity는 %→ratio."""
    m: dict[str, float] = {}
    eg = info.get("earningsGrowth")
    if eg is not None:
        m["eps_growth_yoy"] = float(eg)
    rg = info.get("revenueGrowth")
    if rg is not None:
        m["revenue_growth_yoy"] = float(rg)
    dte = info.get("debtToEquity")
    if dte is not None:
        m["debt_ratio"] = float(dte) / 100.0   # yfinance는 percent 형
    return m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_financial_metrics.py -k us_financials -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/financial_metrics.py tests/test_financial_metrics.py
git commit -m "feat(corvin): US financials parser (yfinance info → metrics)"
```

---

## Task 2: KR 재무 파서 (2시점 EPS → 성장률, 순수)

**Files:**
- Modify: `corvin_jarvis/financial_metrics.py`
- Test: `tests/test_financial_metrics.py`

pykrx는 EPS 레벨만 → 현재/1년전 EPS로 성장률 산출.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_financial_metrics.py 에 추가
@pytest.mark.unit
def test_parse_kr_fundamental_growth():
    # EPS 1년전 1000 → 현재 1300 = +30%
    m = fm.parse_kr_fundamental(eps_now=1300.0, eps_year_ago=1000.0)
    assert abs(m["eps_growth_yoy"] - 0.30) < 1e-9


@pytest.mark.unit
def test_parse_kr_fundamental_negative_eps_ago():
    # 흑자전환(이전 적자) → 성장률 정의 불가 → 빈 dict
    assert fm.parse_kr_fundamental(eps_now=500.0, eps_year_ago=-100.0) == {}


@pytest.mark.unit
def test_parse_kr_fundamental_missing():
    assert fm.parse_kr_fundamental(eps_now=None, eps_year_ago=1000.0) == {}
    assert fm.parse_kr_fundamental(eps_now=1300.0, eps_year_ago=None) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_financial_metrics.py -k kr_fundamental -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/financial_metrics.py 에 추가

def parse_kr_fundamental(
    eps_now: float | None, eps_year_ago: float | None
) -> dict[str, float]:
    """현재/1년전 EPS → eps_growth_yoy. 둘 중 누락 또는 과거≤0이면 빈 dict.

    과거 EPS ≤ 0(적자)이면 성장률 정의가 왜곡 → 산출 안 함(추측 금지).
    """
    if eps_now is None or eps_year_ago is None:
        return {}
    if eps_year_ago <= 0:
        return {}
    return {"eps_growth_yoy": round((eps_now - eps_year_ago) / eps_year_ago, 6)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_financial_metrics.py -k kr_fundamental -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/financial_metrics.py tests/test_financial_metrics.py
git commit -m "feat(corvin): KR fundamental parser (2-point EPS → growth)"
```

---

## Task 3: fetch 어댑터 + run_leading 배선

**Files:**
- Modify: `corvin_jarvis/financial_metrics.py` (fetch 어댑터 추가)
- Modify: `corvin_jarvis/run_leading.py`
- Test: 손동작 dry-run

US=yfinance, KR=pykrx 2시점. 실패 시 None.

- [ ] **Step 1: Write fetch adapters**

```python
# corvin_jarvis/financial_metrics.py 에 추가

def _is_kr(symbol: str) -> bool:
    return symbol.isdigit() and len(symbol) == 6


def fetch_metrics(symbol: str) -> dict[str, float] | None:
    """종목 재무 metrics fetch. US=yfinance, KR=pykrx 2시점 EPS. 실패시 None."""
    try:
        if _is_kr(symbol):
            return _fetch_kr(symbol)
        return _fetch_us(symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("재무 fetch 실패 %s: %s", symbol, e)
        return None


def _fetch_us(symbol: str) -> dict[str, float] | None:
    import yfinance as yf
    info = yf.Ticker(symbol).info
    m = parse_us_financials(info or {})
    return m or None


def _fetch_kr(symbol: str) -> dict[str, float] | None:
    from pykrx import stock
    from datetime import datetime, timedelta
    now = datetime.now()
    # 최근 영업일 EPS + 약 1년 전 EPS
    eps_now = _kr_eps_near(stock, symbol, now)
    eps_ago = _kr_eps_near(stock, symbol, now - timedelta(days=365))
    m = parse_kr_fundamental(eps_now, eps_ago)
    return m or None


def _kr_eps_near(stock_mod: Any, code: str, when: Any) -> float | None:
    """when 근처 영업일의 pykrx EPS. 최대 10일 역탐색."""
    from datetime import timedelta
    for delta in range(0, 10):
        d = (when - timedelta(days=delta)).strftime("%Y%m%d")
        df = stock_mod.get_market_fundamental_by_date(d, d, code)
        if not df.empty:
            eps = float(df.iloc[-1].get("EPS", 0) or 0)
            return eps if eps else None
    return None
```

- [ ] **Step 2: Wire into run_leading**

`corvin_jarvis/run_leading.py`에서 `_metrics_fetcher` 교체:

```python
# 기존:
def _metrics_fetcher(_sym: str) -> dict | None:
    # 재무 metrics 소스 배선은 후속(yfinance US financials / pykrx KR). 현재 미연결 → 스킵.
    return None

# 교체 후:
from corvin_jarvis import financial_metrics


def _metrics_fetcher(sym: str) -> dict | None:
    return financial_metrics.fetch_metrics(sym)
```

상단 import 줄(`from corvin_jarvis import (...)`)에 `financial_metrics`를 추가하고, 함수 내부 import는 제거(모듈 상단 import 통일).

- [ ] **Step 3: Dry-run with send blocked (real fetch)**

Run:
```bash
python3 -c "
from corvin_jarvis import financial_metrics as fm
print('META', fm.fetch_metrics('META'))
print('012450', fm.fetch_metrics('012450'))
"
```
Expected: META는 metrics dict 출력, 012450은 dict 또는 None(EPS 2시점 가용 여부). 에러 없음.

- [ ] **Step 4: Commit**

```bash
git add corvin_jarvis/financial_metrics.py corvin_jarvis/run_leading.py
git commit -m "feat(corvin): wire financial metrics into fundamental Pillar"
```

---

## Task 4: 전체 회귀 + 통합 dry-run

- [ ] **Step 1: 전체 테스트**

Run: `python3 -m pytest tests/ -q`
Expected: 전체 PASS, 무회귀.

- [ ] **Step 2: 통합 dry-run (실제 전송 차단)**

Run:
```bash
python3 -c "
from corvin_jarvis import run_leading, leading_orchestrator as orch
cap = {}
orig = orch.dispatch_brief
orch.dispatch_brief = lambda brief, sender: (cap.update(brief=brief) or orig(brief, sender=lambda b: True))
run_leading.main()
print(cap.get('brief'))
"
```
Expected: 펀더멘털 신호가 포함될 수 있는 브리프 출력(US 종목 재무 기반). 실제 전송 0건.

- [ ] **Step 3: Commit (marker)**

```bash
git commit -m "test(corvin): fundamental Pillar live with financial metrics" --allow-empty
```

---

## 자체 검토 (spec 대비)

- **Spec §5 Pillar 1 재무 데이터** → Task 1(US)·2(KR)·3(배선) ✅
- **메모리 yfinance KR 금지** → `_is_kr` 분기로 KR은 pykrx만 ✅
- **추측 금지** → 데이터 누락/적자 과거 EPS면 빈 dict→스킵 ✅
- **op_margin_trend**: yfinance는 현재 마진만(추세 아님) → 매핑 제외. financial_growth_score가 가용 지표만 평균하므로 정상 동작.
- **남은 후속(Plan 7)**: 거래량 fetch(KIS OHLCV avg → ensemble volume) + Claude haiku 정성 어댑터(qualitative_fetcher). 둘 다 DI 자리 마련됨.
