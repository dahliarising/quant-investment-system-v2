# Corvin 선행 인텔리전스 — 거래량 + Claude 정성 + cron (Plan 7) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 마지막 라이브 데이터 3종을 붙인다 — ①거래량(앙상블 volume) ②Claude haiku 정성(펀더멘털 qualitative) ③crontab 자동 발송(수동 등록). 4 Pillar 완전 가동.

**Architecture:** 거래량/정성 모두 **순수 변환 + DI fetcher/client**로 테스트. Claude 정성은 API 키 없으면 None(graceful degrade). crontab은 정책상 자동 등록 금지 → setup 스크립트 + 안내만.

**Tech Stack:** Python 3, pytest, anthropic SDK(haiku), pykrx/yfinance(거래량).

**메모리 준수:** yfinance US만 / 테스트 실송·실 API콜 0(mock) / crontab 자동등록 금지.

---

## 파일 구조

| 파일 | 책임 | 신규/수정 |
|------|------|-----------|
| `corvin_jarvis/signals/ensemble.py` | `volume_stats` 순수 헬퍼 | 수정 |
| `corvin_jarvis/leading_providers.py` | ensemble_provider에 volume_fetcher 주입 | 수정 |
| `corvin_jarvis/market_data.py` | 거래량 fetch 어댑터(KR/US) | 신규 |
| `corvin_jarvis/qualitative.py` | Claude haiku 정성 어댑터(DI client) | 신규 |
| `corvin_jarvis/run_leading.py` | volume/qualitative 배선 | 수정 |
| `corvin_jarvis/setup_leading_cron.sh` | crontab 등록 안내 스크립트 | 신규 |
| `tests/test_ensemble.py` / `test_leading_providers.py` / `test_qualitative.py` | 테스트 | 수정/신규 |

---

## Task 1: 거래량 순수 헬퍼 + provider 주입

**Files:**
- Modify: `corvin_jarvis/signals/ensemble.py`, `corvin_jarvis/leading_providers.py`
- Test: `tests/test_ensemble.py`, `tests/test_leading_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ensemble.py 에 추가
@pytest.mark.unit
def test_volume_stats_basic():
    vols = [100.0] * 19 + [200.0]   # avg20=105, latest=200
    latest, avg20 = ensemble.volume_stats(vols)
    assert latest == 200.0
    assert abs(avg20 - 105.0) < 1e-9


@pytest.mark.unit
def test_volume_stats_short():
    latest, avg20 = ensemble.volume_stats([])
    assert latest is None and avg20 is None
```

```python
# tests/test_leading_providers.py 에 추가
@pytest.mark.unit
def test_ensemble_provider_uses_volume_fetcher():
    closes = {"012450": [float(i) for i in range(1, 261)]}

    def price_fetcher(sym, days=252):
        return closes.get(sym, [])

    def bench_fetcher():
        return [100.0] * 261

    def volume_fetcher(sym):
        return (300.0, 100.0)   # 3배 돌파

    sigs = lp.ensemble_provider(
        ["012450"], price_fetcher=price_fetcher,
        bench_fetcher=bench_fetcher, volume_fetcher=volume_fetcher,
    )
    assert len(sigs) == 1
    assert sigs[0].evidence["volume"] >= 80   # 거래량 점수 반영
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ensemble.py -k volume_stats tests/test_leading_providers.py -k uses_volume -v`
Expected: FAIL — `AttributeError` / `TypeError` (volume_fetcher 미지원)

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/ensemble.py 에 추가
def volume_stats(volumes: list[float]) -> tuple[float | None, float | None]:
    """거래량 시계열 → (최근, 20일평균). 비면 (None, None)."""
    if not volumes:
        return None, None
    latest = volumes[-1]
    window = volumes[-20:]
    avg20 = sum(window) / len(window)
    return latest, avg20
```

```python
# corvin_jarvis/leading_providers.py — ensemble_provider 시그니처/본문 수정
def ensemble_provider(
    symbols: list[str],
    price_fetcher: Callable[..., list[float]],
    bench_fetcher: Callable[[], list[float]],
    volume_fetcher: Callable[[str], tuple[float | None, float | None]] | None = None,
) -> list[LeadingSignal]:
    """종목별 종가→지표→앙상블 신호. 이력 부족 종목은 스킵.

    volume_fetcher 주입 시 거래량 점수 반영, 없으면 NEUTRAL.
    """
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
        if volume_fetcher is not None:
            vol, vol_avg = volume_fetcher(sym)
        else:
            vol, vol_avg = None, None
        volume = ensemble.volume_breakthrough_score(vol, vol_avg)
        out.append(ensemble.build_ensemble_signal(sym, minervini, rs, volume))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ensemble.py -k volume_stats -v` 그리고 `python3 -m pytest tests/test_leading_providers.py -k "ensemble" -v`
Expected: PASS (모든 ensemble provider 테스트 포함)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/ensemble.py corvin_jarvis/leading_providers.py tests/test_ensemble.py tests/test_leading_providers.py
git commit -m "feat(corvin): volume signal wiring into ensemble Pillar"
```

---

## Task 2: 거래량 fetch 어댑터 + run_leading 배선

**Files:**
- Create: `corvin_jarvis/market_data.py`
- Modify: `corvin_jarvis/run_leading.py`
- Test: `tests/test_market_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market_data.py
"""Tests for corvin_jarvis.market_data (pure parse, no real API)."""
from __future__ import annotations

import pytest

from corvin_jarvis import market_data as md


@pytest.mark.unit
def test_is_kr():
    assert md._is_kr("012450") is True
    assert md._is_kr("META") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_market_data.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/market_data.py
"""Corvin 선행 인텔리전스 — 시장 거래량 fetch 어댑터.

KR=pykrx OHLCV, US=yfinance history. 실패 시 (None, None).
⚠️ yfinance는 US만 (메모리). KR은 pykrx.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

log = logging.getLogger("corvin.market_data")


def _is_kr(symbol: str) -> bool:
    return symbol.isdigit() and len(symbol) == 6


def fetch_volume(symbol: str) -> tuple[float | None, float | None]:
    """(최근 거래량, 20일평균). 실패 시 (None, None)."""
    try:
        if _is_kr(symbol):
            return _fetch_kr_volume(symbol)
        return _fetch_us_volume(symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("거래량 fetch 실패 %s: %s", symbol, e)
        return None, None


def _fetch_kr_volume(code: str) -> tuple[float | None, float | None]:
    from pykrx import stock
    from corvin_jarvis.signals import ensemble
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=40)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv_by_date(start, end, code)
    if df.empty:
        return None, None
    vols = [float(v) for v in df["거래량"].tolist()]
    return ensemble.volume_stats(vols)


def _fetch_us_volume(symbol: str) -> tuple[float | None, float | None]:
    import yfinance as yf
    from corvin_jarvis.signals import ensemble
    hist = yf.Ticker(symbol).history(period="2mo")
    if hist.empty or "Volume" not in hist:
        return None, None
    vols = [float(v) for v in hist["Volume"].tolist()]
    return ensemble.volume_stats(vols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_market_data.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Wire run_leading**

`run_leading.py`: 상단 import에 `market_data` 추가. ensemble provider 호출에 volume_fetcher 주입:

```python
        lambda: lp.ensemble_provider(
            KR_SYMBOLS + US_SYMBOLS,
            price_fetcher=dca_timing.default_fetcher,
            bench_fetcher=_kr_bench_fetcher,
            volume_fetcher=market_data.fetch_volume,
        ),
```

- [ ] **Step 6: Commit**

```bash
git add corvin_jarvis/market_data.py corvin_jarvis/run_leading.py tests/test_market_data.py
git commit -m "feat(corvin): volume fetch adapters (KR pykrx / US yfinance) + wire"
```

---

## Task 3: Claude haiku 정성 어댑터 (DI client)

**Files:**
- Create: `corvin_jarvis/qualitative.py`
- Test: `tests/test_qualitative.py`

verdict 텍스트 → 0-100 매핑은 순수. Claude 호출은 DI client(mock 테스트). 키 없으면 None.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qualitative.py
"""Tests for corvin_jarvis.qualitative (mock client, no real API)."""
from __future__ import annotations

import pytest

from corvin_jarvis import qualitative as ql


@pytest.mark.unit
def test_verdict_to_score():
    assert ql.verdict_to_score("BULL") == 80.0
    assert ql.verdict_to_score("neutral") == 50.0
    assert ql.verdict_to_score("Bear") == 20.0
    assert ql.verdict_to_score("garbage") is None


class _FakeMsg:
    def __init__(self, text):
        self.content = [type("B", (), {"text": text})()]


class _FakeClient:
    def __init__(self, text):
        self._text = text
        self.messages = type("M", (), {"create": lambda _self, **kw: _FakeMsg(self._text)})()


@pytest.mark.unit
def test_qualitative_score_with_client():
    client = _FakeClient("BULL")
    score = ql.qualitative_score("NVDA", "AI GPU 절대강자", client=client)
    assert score == 80.0


@pytest.mark.unit
def test_qualitative_score_no_client_returns_none():
    assert ql.qualitative_score("NVDA", "x", client=None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qualitative.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/qualitative.py
"""Corvin 선행 인텔리전스 — Claude haiku 정성 분석 어댑터.

종목 사업 요약 → bull/neutral/bear verdict → 0-100. DI client로 테스트.
키/클라이언트 없으면 None(graceful degrade). 모델: claude-haiku-4-5 (저비용).
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("corvin.qualitative")

MODEL = "claude-haiku-4-5"
_VERDICT_SCORE = {"bull": 80.0, "neutral": 50.0, "bear": 20.0}

_PROMPT = (
    "다음 종목의 기술 방향성·경영 의사결정·산업 포지셔닝을 평가해 "
    "성장 전망을 한 단어로만 답하라: BULL, NEUTRAL, BEAR 중 하나.\n"
    "종목: {symbol}\n사업: {summary}\n답(한 단어):"
)


def verdict_to_score(verdict: str) -> float | None:
    """BULL/NEUTRAL/BEAR(대소문자 무관) → 점수. 그 외 None."""
    return _VERDICT_SCORE.get(verdict.strip().lower())


def default_client() -> Any | None:
    """ANTHROPIC_API_KEY 있으면 anthropic 클라이언트, 없으면 None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception as e:  # noqa: BLE001
        log.warning("anthropic client 생성 실패: %s", e)
        return None


def qualitative_score(
    symbol: str, summary: str, client: Any | None
) -> float | None:
    """Claude haiku로 정성 verdict → 점수. client None이면 None(degrade)."""
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user",
                       "content": _PROMPT.format(symbol=symbol, summary=summary)}],
        )
        text = msg.content[0].text if msg.content else ""
        return verdict_to_score(text)
    except Exception as e:  # noqa: BLE001
        log.warning("정성 분석 실패 %s: %s", symbol, e)
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qualitative.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/qualitative.py tests/test_qualitative.py
git commit -m "feat(corvin): Claude haiku qualitative adapter (DI client)"
```

---

## Task 4: 정성 어댑터 run_leading 배선 (키 가드)

**Files:**
- Modify: `corvin_jarvis/run_leading.py`
- Test: 손동작 dry-run

- [ ] **Step 1: Wire qualitative into run_leading**

`run_leading.py`: 상단 import에 `qualitative` 추가. universe.json에서 종목 theme를 summary로 사용(없으면 symbol). 모듈 레벨에서 client 1회 생성:

```python
from corvin_jarvis import qualitative

# 종목 사업 요약 (universe.json theme 활용; 간단히 매핑)
_SUMMARIES = {
    "012450": "한화에어로스페이스 — K-방산 글로벌 확산",
    "META": "Meta — AI 광고 + Reality Labs",
    "MSFT": "Microsoft — Azure + Copilot",
    "NVDA": "NVIDIA — AI GPU 절대강자",
    "TSLA": "Tesla — Optimus 휴머노이드",
}
_QUAL_CLIENT = qualitative.default_client()   # 키 없으면 None → degrade


def _qualitative_fetcher(sym: str) -> float | None:
    summary = _SUMMARIES.get(sym, sym)
    return qualitative.qualitative_score(sym, summary, client=_QUAL_CLIENT)
```

fundamental provider 호출에 qualitative_fetcher 주입:

```python
        lambda: lp.fundamental_provider(
            KR_SYMBOLS + US_SYMBOLS,
            metrics_fetcher=_metrics_fetcher,
            tone_fetcher=_tone_fetcher,
            qualitative_fetcher=_qualitative_fetcher,
        ),
```

- [ ] **Step 2: Dry-run (키 없으면 정성 None degrade, 실송 차단)**

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
Expected: 에러 없이 브리프 출력. dev(키 없음)면 정성 미반영(degrade), cron(키 있음)면 정성 가중 반영. 실제 전송 0건.

- [ ] **Step 3: Commit**

```bash
git add corvin_jarvis/run_leading.py
git commit -m "feat(corvin): wire qualitative analysis into fundamental Pillar"
```

---

## Task 5: crontab 등록 스크립트 (수동, 자동등록 금지)

**Files:**
- Create: `corvin_jarvis/setup_leading_cron.sh`

⚠️ auto-mode persistence 금지 → 스크립트는 **출력만** 하고 실제 등록은 사용자가 실행.

- [ ] **Step 1: Write setup script**

```bash
# corvin_jarvis/setup_leading_cron.sh
#!/usr/bin/env bash
# Corvin 선행 인텔리전스 cron 등록 (수동 실행 전용).
# ⚠️ Claude auto-mode는 crontab 자동 등록 금지 → 폐하가 직접 실행.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"

# KR 프리오픈 08:30, US 프리오픈(09:00 ET≈22:30 KST) 평일
CRON_KR="30 8 * * 1-5 cd $REPO_DIR && $PY -m corvin_jarvis.run_leading >> $REPO_DIR/corvin_jarvis/state/leading.log 2>&1"
CRON_US="30 22 * * 1-5 cd $REPO_DIR && $PY -m corvin_jarvis.run_leading >> $REPO_DIR/corvin_jarvis/state/leading.log 2>&1"

echo "다음 두 줄을 crontab에 추가하세요 (crontab -e):"
echo ""
echo "$CRON_KR"
echo "$CRON_US"
echo ""
echo "또는 자동 추가:"
echo "  (crontab -l 2>/dev/null; echo \"\$CRON_KR\"; echo \"\$CRON_US\") | crontab -"
```

- [ ] **Step 2: Verify script runs (prints, does NOT install)**

Run: `bash corvin_jarvis/setup_leading_cron.sh`
Expected: cron 두 줄 출력. crontab 미변경.

- [ ] **Step 3: Commit**

```bash
chmod +x corvin_jarvis/setup_leading_cron.sh
git add corvin_jarvis/setup_leading_cron.sh
git commit -m "feat(corvin): leading cron setup script (manual install)"
```

---

## Task 6: 전체 회귀 + 최종 통합 dry-run

- [ ] **Step 1: 전체 테스트**

Run: `python3 -m pytest tests/ -q`
Expected: 전체 PASS, 무회귀.

- [ ] **Step 2: 최종 통합 dry-run (실송 차단)**

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
Expected: 4 Pillar 신호가 게이트 통과분만 포함된 브리프. 실송 0.

- [ ] **Step 3: Commit (marker)**

```bash
git commit -m "test(corvin): 4-pillar fully live verified" --allow-empty
```

---

## 자체 검토 (spec 대비)

- **Spec §3 Pillar 3 거래량** → Task 1,2 ✅ (KR pykrx / US yfinance)
- **Spec §3 Pillar 1 정성(Claude haiku)** → Task 3,4 ✅ (키 가드 degrade)
- **Spec §4 cron 스케줄** → Task 5 (KR 08:30 / US 22:30 평일) ✅
- **메모리 yfinance KR 금지** → 거래량도 `_is_kr` 분기 ✅
- **메모리 실송/실API 금지(테스트)** → volume mock, qualitative mock client ✅
- **crontab 자동등록 금지** → setup 스크립트 출력만 ✅
- **4 Pillar 완전 가동**: 이벤트·앙상블(추세/RS/거래량)·펀더멘털(재무/뉴스/정성)·교차시장 모두 라이브.
