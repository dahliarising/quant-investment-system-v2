# Corvin 선행 인텔리전스 — 라이브 데이터 배선 (Plan 5) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 순수 점수 함수에 실제 데이터를 주입하는 **provider 어댑터**를 만들어 `run_leading`에 배선한다. 4 Pillar가 라이브 데이터로 매일 브리프를 생성하게 한다.

**Architecture:** 각 provider는 **fetcher 콜백을 주입받는다**(의존성 주입). 테스트는 mock fetcher로 외부 호출 없이 검증. 라이브 배선은 기존 `dca_timing.default_fetcher`(종가), `quote_provider.get_quote`(프록시 pct), `narrative.latest_signal`(sentiment_tone)을 실제 fetcher로 연결. 외부 데이터 누락 시 해당 종목/Pillar 스킵(추측 금지).

**Tech Stack:** Python 3, pytest, 기존 모듈 재사용.

**테스트 실행:** 루트에서 `python3 -m pytest tests/<file> -v`.

**메모리 준수:** yfinance KR 금지(pykrx/KIS만) · 테스트 실전송 0 · DCA는 완성봉 기준.

---

## 파일 구조

| 파일 | 책임 | 신규/수정 |
|------|------|-----------|
| `corvin_jarvis/signals/ensemble.py` | 종가→지표 변환 헬퍼 추가 | 수정 |
| `corvin_jarvis/leading_providers.py` | 4 Pillar 라이브 provider (DI) | 신규 |
| `corvin_jarvis/run_leading.py` | 실제 fetcher 배선 | 수정 |
| `tests/test_ensemble.py` | 지표 헬퍼 테스트 추가 | 수정 |
| `tests/test_leading_providers.py` | provider 테스트(mock fetcher) | 신규 |

---

## Task 1: 종가 → 지표 변환 헬퍼 (순수)

**Files:**
- Modify: `corvin_jarvis/signals/ensemble.py`
- Test: `tests/test_ensemble.py`

종가 리스트 → MA/52주 hi-lo, 벤치 대비 RS 윈도우.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ensemble.py 에 추가
@pytest.mark.unit
def test_indicators_from_closes():
    closes = [float(i) for i in range(1, 261)]  # 1..260 상승추세
    ind = ensemble.indicators_from_closes(closes)
    assert ind["price"] == 260.0
    assert ind["ma50"] is not None and ind["ma50"] < 260.0
    assert ind["high_52w"] == 260.0
    assert ind["low_52w"] == 9.0 or ind["low_52w"] == closes[-252]


@pytest.mark.unit
def test_indicators_from_closes_short_series():
    ind = ensemble.indicators_from_closes([100.0, 101.0])
    assert ind["price"] == 101.0
    assert ind["ma50"] is None      # 50개 미만 → None
    assert ind["ma200"] is None


@pytest.mark.unit
def test_rs_windows_from_closes():
    # symbol +10% over window, bench flat → positive RS
    sym = [100.0] * 200 + [110.0]
    bench = [100.0] * 201
    rs = ensemble.rs_windows_from_closes(sym, bench)
    assert rs["1w"] > 0      # 최근 강세
    assert all(k in rs for k in ("1w", "1m", "3m", "6m"))


@pytest.mark.unit
def test_rs_windows_insufficient():
    assert ensemble.rs_windows_from_closes([100.0], [100.0]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ensemble.py -k "indicators or rs_windows" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/ensemble.py 에 추가

_RS_WINDOWS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}


def _ma(closes: list[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def indicators_from_closes(closes: list[float]) -> dict[str, float | None]:
    """종가 시계열 → Minervini 입력 지표. 부족하면 해당 항목 None."""
    if not closes:
        return {"price": None, "ma50": None, "ma150": None,
                "ma200": None, "low_52w": None, "high_52w": None}
    window_52w = closes[-252:] if len(closes) >= 252 else closes
    return {
        "price": closes[-1],
        "ma50": _ma(closes, 50),
        "ma150": _ma(closes, 150),
        "ma200": _ma(closes, 200),
        "low_52w": min(window_52w),
        "high_52w": max(window_52w),
    }


def _pct_return(closes: list[float], window: int) -> float | None:
    if len(closes) <= window:
        return None
    past = closes[-window - 1]
    if past == 0:
        return None
    return (closes[-1] - past) / past * 100.0


def rs_windows_from_closes(
    closes: list[float], bench_closes: list[float]
) -> dict[str, float]:
    """타임프레임별 (종목수익률 - 벤치수익률) %p. 데이터 부족 윈도우는 제외."""
    out: dict[str, float] = {}
    for name, w in _RS_WINDOWS.items():
        sym_r = _pct_return(closes, w)
        bench_r = _pct_return(bench_closes, w)
        if sym_r is not None and bench_r is not None:
            out[name] = round(sym_r - bench_r, 4)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ensemble.py -k "indicators or rs_windows" -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/ensemble.py tests/test_ensemble.py
git commit -m "feat(corvin): close-series indicator helpers (Pillar 3 wiring)"
```

---

## Task 2: 앙상블 라이브 provider (DI)

**Files:**
- Create: `corvin_jarvis/leading_providers.py`
- Test: `tests/test_leading_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leading_providers.py
"""Tests for corvin_jarvis.leading_providers (mock fetcher, no real calls)."""
from __future__ import annotations

import pytest

from corvin_jarvis import leading_providers as lp
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_ensemble_provider_builds_signals():
    # 상승추세 종가 + 평탄 벤치 → bull 앙상블 신호
    closes = {"012450": [float(i) for i in range(1, 261)]}
    bench = [100.0] * 261

    def price_fetcher(sym, days=252):
        return closes.get(sym, [])

    def bench_fetcher():
        return bench

    sigs = lp.ensemble_provider(
        ["012450"], price_fetcher=price_fetcher, bench_fetcher=bench_fetcher
    )
    assert len(sigs) == 1
    assert isinstance(sigs[0], LeadingSignal)
    assert sigs[0].pillar == "ensemble"


@pytest.mark.unit
def test_ensemble_provider_skips_insufficient_history():
    def price_fetcher(sym, days=252):
        return [100.0, 101.0]      # 너무 짧음

    def bench_fetcher():
        return [100.0, 101.0]

    sigs = lp.ensemble_provider(
        ["012450"], price_fetcher=price_fetcher, bench_fetcher=bench_fetcher
    )
    assert sigs == []      # 데이터 부족 → 스킵(추측 금지)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_leading_providers.py -k ensemble -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/leading_providers.py
"""Corvin 선행 인텔리전스 — 라이브 데이터 provider 어댑터.

순수 점수 함수에 실제 데이터를 주입한다. 모든 provider는 fetcher 콜백을
주입받아(의존성 주입) 테스트 가능. 데이터 누락 시 해당 종목 스킵(추측 금지).
"""
from __future__ import annotations

from typing import Any, Callable

from corvin_jarvis.signals import cross_market, ensemble, fundamental
from corvin_jarvis.signals.leading_signal import LeadingSignal

# 앙상블 Minervini 최소 이력 (MA200 필요)
_MIN_ENSEMBLE_HISTORY = 200


def ensemble_provider(
    symbols: list[str],
    price_fetcher: Callable[..., list[float]],
    bench_fetcher: Callable[[], list[float]],
) -> list[LeadingSignal]:
    """종목별 종가→지표→앙상블 신호. 이력 부족 종목은 스킵."""
    bench_closes = bench_fetcher()
    out: list[LeadingSignal] = []
    for sym in symbols:
        closes = price_fetcher(sym, 252)
        if len(closes) < _MIN_ENSEMBLE_HISTORY:
            continue
        ind = ensemble.indicators_from_closes(closes)
        minervini = ensemble.minervini_trend_score(
            price=ind["price"], ma50=ind["ma50"], ma150=ind["ma150"],
            ma200=ind["ma200"], low_52w=ind["low_52w"], high_52w=ind["high_52w"],
        )
        rs_windows = ensemble.rs_windows_from_closes(closes, bench_closes)
        rs = ensemble.multi_timeframe_rs_score(rs_windows)
        # 거래량은 종가 fetcher에 없음 → NEUTRAL(50) 주입(graceful)
        volume = ensemble.volume_breakthrough_score(None, None)
        out.append(ensemble.build_ensemble_signal(sym, minervini, rs, volume))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_leading_providers.py -k ensemble -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/leading_providers.py tests/test_leading_providers.py
git commit -m "feat(corvin): ensemble live provider (DI fetcher)"
```

---

## Task 3: 교차시장 라이브 provider (DI)

**Files:**
- Modify: `corvin_jarvis/leading_providers.py`
- Test: `tests/test_leading_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leading_providers.py 에 추가
@pytest.mark.unit
def test_cross_market_provider():
    # 타깃: (대상종목, 프록시명, 프록시심볼)
    targets = [("012450", "LMT 야간", "LMT")]
    quotes = {"LMT": 3.0}

    def pct_fetcher(proxy_symbol):
        return quotes.get(proxy_symbol)

    sigs = lp.cross_market_provider(targets, pct_fetcher=pct_fetcher)
    assert len(sigs) == 1
    assert sigs[0].pillar == "cross_market"
    assert sigs[0].advisory is True


@pytest.mark.unit
def test_cross_market_provider_skips_missing_proxy():
    targets = [("012450", "LMT 야간", "LMT")]

    def pct_fetcher(proxy_symbol):
        return None      # 프록시 가격 없음

    assert lp.cross_market_provider(targets, pct_fetcher=pct_fetcher) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_leading_providers.py -k cross_market -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/leading_providers.py 에 추가

def cross_market_provider(
    targets: list[tuple[str, str, str]],
    pct_fetcher: Callable[[str], float | None],
) -> list[LeadingSignal]:
    """targets: (대상종목, 프록시명, 프록시심볼). 프록시 pct 없으면 스킵."""
    out: list[LeadingSignal] = []
    for symbol, proxy_name, proxy_symbol in targets:
        pct = pct_fetcher(proxy_symbol)
        if pct is None:
            continue
        out.append(cross_market.build_cross_market_signal(symbol, proxy_name, pct))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_leading_providers.py -k cross_market -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/leading_providers.py tests/test_leading_providers.py
git commit -m "feat(corvin): cross-market live provider (DI fetcher)"
```

---

## Task 4: 펀더멘털 라이브 provider (DI)

**Files:**
- Modify: `corvin_jarvis/leading_providers.py`
- Test: `tests/test_leading_providers.py`

재무 metrics fetcher + sentiment tone fetcher 주입. 정성(Claude)은 선택적(None 허용).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leading_providers.py 에 추가
@pytest.mark.unit
def test_fundamental_provider_builds():
    metrics = {"012450": {"eps_growth_yoy": 0.25, "revenue_growth_yoy": 0.15}}

    def metrics_fetcher(sym):
        return metrics.get(sym)

    def tone_fetcher(sym):
        return 0.3      # 약한 긍정

    sigs = lp.fundamental_provider(
        ["012450"], metrics_fetcher=metrics_fetcher, tone_fetcher=tone_fetcher,
    )
    assert len(sigs) == 1
    assert sigs[0].pillar == "fundamental"
    # 정성 미주입 → degrade
    assert sigs[0].evidence["qualitative_used"] is False


@pytest.mark.unit
def test_fundamental_provider_skips_no_metrics():
    def metrics_fetcher(sym):
        return None      # 재무 없음

    def tone_fetcher(sym):
        return 0.0

    sigs = lp.fundamental_provider(
        ["012450"], metrics_fetcher=metrics_fetcher, tone_fetcher=tone_fetcher,
    )
    assert sigs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_leading_providers.py -k fundamental -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/leading_providers.py 에 추가

def fundamental_provider(
    symbols: list[str],
    metrics_fetcher: Callable[[str], dict[str, Any] | None],
    tone_fetcher: Callable[[str], float | None],
    qualitative_fetcher: Callable[[str], float | None] | None = None,
) -> list[LeadingSignal]:
    """재무 metrics + 뉴스 tone (+선택 정성) → 펀더멘털 신호. 재무 없으면 스킵."""
    out: list[LeadingSignal] = []
    for sym in symbols:
        metrics = metrics_fetcher(sym)
        if not metrics:
            continue
        fin = fundamental.financial_growth_score(metrics)
        news = fundamental.news_sentiment_score(tone_fetcher(sym))
        qual = qualitative_fetcher(sym) if qualitative_fetcher else None
        out.append(fundamental.build_fundamental_signal(sym, fin, qual, news))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_leading_providers.py -k fundamental -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/leading_providers.py tests/test_leading_providers.py
git commit -m "feat(corvin): fundamental live provider (DI fetcher)"
```

---

## Task 5: run_leading 실제 fetcher 배선

**Files:**
- Modify: `corvin_jarvis/run_leading.py`
- Test: 손동작 dry-run (실제 전송 차단)

기존 fetcher를 실제 connect: 종가=`dca_timing.default_fetcher`, 프록시 pct=`quote_provider.get_quote(...).pct_change`, sentiment=`narrative.latest_signal`. 재무 fetcher는 보수적으로 빈 구현(데이터 소스 미완 → 펀더멘털 스킵). KR 벤치=KOSPI.

- [ ] **Step 1: Write the runner wiring**

```python
# corvin_jarvis/run_leading.py — 전체 교체
"""Corvin 선행 인텔리전스 — cron 진입점.

4 Pillar provider를 라이브 데이터로 수집 → 게이트 → 브리프 → Telegram 발송.

⚠️ crontab 자동 등록 금지(정책). 사용자 수동 등록:
    30 8 * * 1-5 cd <repo> && python3 -m corvin_jarvis.run_leading >> state/leading.log 2>&1
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from corvin_jarvis import (
    channels,
    dca_timing,
    leading_orchestrator as orch,
    leading_providers as lp,
    narrative,
    quote_provider,
)
from corvin_jarvis.signals import event_calendar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("corvin.run_leading")

# 보유/관심 종목 (portfolio + universe 핵심). 추후 universe.json 로딩으로 확장.
KR_SYMBOLS = ["012450"]
US_SYMBOLS = ["META", "MSFT", "NVDA", "TSLA"]
# 교차시장 타깃: (대상, 프록시명, 프록시심볼)
CROSS_TARGETS = [
    ("012450", "ITA 방산ETF 야간", "ITA"),
    ("META", "NQ 나스닥선물", "QQQ"),
]
KOSPI_SYMBOL = "KOSPI"


def _today() -> date:
    return date.today()


def _kr_bench_fetcher() -> list[float]:
    try:
        return dca_timing.default_fetcher(KOSPI_SYMBOL, 252)
    except Exception as e:  # noqa: BLE001
        log.warning("KOSPI 벤치 fetch 실패: %s", e)
        return []


def _proxy_pct_fetcher(proxy_symbol: str) -> float | None:
    try:
        q = quote_provider.get_quote(proxy_symbol)
        return q.pct_change if q and q.error is None else None
    except Exception as e:  # noqa: BLE001
        log.warning("프록시 %s fetch 실패: %s", proxy_symbol, e)
        return None


def _tone_fetcher(_sym: str) -> float | None:
    # narrative는 KR 시장 레벨 sentiment_tone만 제공(종목별 미지원)
    try:
        sig = narrative.latest_signal(narrative.DEFAULT_SIGNALS_DB, market="KR")
        return sig.get("sentiment_tone") if sig else None
    except Exception as e:  # noqa: BLE001
        log.warning("sentiment fetch 실패: %s", e)
        return None


def _metrics_fetcher(_sym: str) -> dict | None:
    # 재무 metrics 소스 배선은 후속(yfinance US financials / pykrx KR). 현재 미연결 → 스킵.
    return None


def main() -> int:
    as_of = _today()
    providers = [
        lambda: event_calendar.build_event_signals(
            as_of=as_of, earnings_rows=[], macro_horizon_days=30,
        ),
        lambda: lp.ensemble_provider(
            KR_SYMBOLS + US_SYMBOLS,
            price_fetcher=dca_timing.default_fetcher,
            bench_fetcher=_kr_bench_fetcher,
        ),
        lambda: lp.cross_market_provider(CROSS_TARGETS, pct_fetcher=_proxy_pct_fetcher),
        lambda: lp.fundamental_provider(
            KR_SYMBOLS + US_SYMBOLS,
            metrics_fetcher=_metrics_fetcher,
            tone_fetcher=_tone_fetcher,
        ),
    ]
    signals = orch.collect(providers)
    brief = orch.format_brief(signals, threshold=60.0)
    sent = orch.dispatch_brief(brief, sender=channels.send_telegram)
    log.info("leading brief: signals=%d sent=%s", len(signals), sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Dry-run with send blocked (NO real Telegram)**

Run:
```bash
python3 -c "
from corvin_jarvis import run_leading, leading_orchestrator as orch
cap = {}
orig = orch.dispatch_brief
orch.dispatch_brief = lambda brief, sender: (cap.update(brief=brief) or orig(brief, sender=lambda b: True))
rc = run_leading.main()
print('rc', rc); print(cap.get('brief'))
"
```
Expected: 에러 없이 실행, 브리프 출력(라이브 데이터 기반). 실제 전송 0건.

- [ ] **Step 3: Commit**

```bash
git add corvin_jarvis/run_leading.py
git commit -m "feat(corvin): wire live fetchers into leading runner"
```

---

## Task 6: 전체 회귀 + 마커

- [ ] **Step 1: 전체 테스트**

Run: `python3 -m pytest tests/ -q`
Expected: 전체 PASS, 무회귀.

- [ ] **Step 2: Commit (slice marker)**

```bash
git commit -m "test(corvin): live data wiring verified" --allow-empty
```

---

## 자체 검토 (spec 대비)

- **Spec §5 데이터 소스 배선** → Task 2~5 ✅ (종가/프록시/sentiment 라이브)
- **Spec §3 Pillar 3 거래량**: 종가 fetcher에 거래량 없음 → NEUTRAL graceful. 거래량 fetch는 후속(KIS OHLCV) — 명기.
- **Spec §3 Pillar 1 재무 metrics**: yfinance US financials / pykrx KR fundamental 어댑터는 **후속(Plan 6)** — 현재 `_metrics_fetcher`가 None 반환 → 펀더멘털 자동 스킵(추측 금지). DI 구조라 어댑터만 교체하면 즉시 활성.
- **Spec §3 Pillar 1 정성(Claude haiku)**: `qualitative_fetcher` 선택적 인자로 자리 마련, 미주입 시 degrade. Claude 호출 어댑터는 후속.
- **메모리 yfinance KR 금지**: KR 종가/벤치는 quote_provider(KIS/pykrx 경유) ✅
- **메모리 실전송 금지**: provider 테스트 전부 mock fetcher, run_leading dry-run은 send 차단 ✅
- **추측 금지(CLAUDE.md)**: 데이터 누락 시 종목/Pillar 스킵 ✅
