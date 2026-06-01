# Corvin 선행 인텔리전스 — Pillar 4 교차시장 + DART + 디스패치 통합 (Plan 4) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Pillar 4 교차시장(advisory) 신호 + DART 공시 감지(Pillar 2 확장) + 오케스트레이터 수집/디스패치 통합을 완성해, 4 Pillar가 한 브리프로 모여 Telegram으로 발송되는 end-to-end를 완성한다.

**Architecture:** 교차시장은 검증 약함(0-3 기각) → `advisory=True` + confidence 30% 페널티. DART는 공시 리스트 파서(순수) + 키워드 필터. 오케스트레이터에 `collect()`(provider 콜백 수집) + `dispatch_brief()`(sender 주입, 빈 브리프면 미발송) 추가. 실제 cron 등록은 사용자 수동(auto-mode persistence 금지 정책).

**Tech Stack:** Python 3, pytest, 기존 `LeadingSignal`·`channels.send_telegram`.

**테스트 실행:** 루트에서 `python3 -m pytest tests/<file> -v`.

---

## 파일 구조

| 파일 | 책임 | 신규/수정 |
|------|------|-----------|
| `corvin_jarvis/signals/cross_market.py` | Pillar 4: 프록시 야간 → advisory 신호 | 신규 |
| `corvin_jarvis/signals/event_calendar.py` | DART 공시 파서 + 신호 (Pillar 2 확장) | 수정 |
| `corvin_jarvis/leading_orchestrator.py` | collect(providers) + dispatch_brief(sender) | 수정 |
| `tests/test_cross_market.py` | Pillar 4 테스트 | 신규 |
| `tests/test_event_calendar.py` | DART 테스트 추가 | 수정 |
| `tests/test_leading_orchestrator.py` | collect/dispatch 테스트 추가 | 수정 |

---

## Task 1: 교차시장 advisory 신호 (순수, 신뢰도 페널티)

**Files:**
- Create: `corvin_jarvis/signals/cross_market.py`
- Test: `tests/test_cross_market.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cross_market.py
"""Tests for corvin_jarvis.signals.cross_market."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals import cross_market
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_cross_market_signal_is_advisory():
    sig = cross_market.build_cross_market_signal(
        symbol="012450", proxy_name="LMT/RTX/ITA 야간", proxy_pct=3.0,
    )
    assert isinstance(sig, LeadingSignal)
    assert sig.pillar == "cross_market"
    assert sig.advisory is True            # 항상 참고용
    assert sig.direction == "bull"         # 프록시 강세 → bull 힌트


@pytest.mark.unit
def test_cross_market_confidence_penalized():
    # 검증 약함 → 동일 강도라도 confidence 30% 페널티
    sig = cross_market.build_cross_market_signal(
        symbol="012450", proxy_name="LMT 야간", proxy_pct=5.0,
    )
    # base = clamp(|5|*10)=50, penalized = 50*0.7 = 35
    assert abs(sig.confidence - 35.0) < 0.01


@pytest.mark.unit
def test_cross_market_bear_on_negative_proxy():
    sig = cross_market.build_cross_market_signal(
        symbol="META", proxy_name="NQ 선물", proxy_pct=-4.0,
    )
    assert sig.direction == "bear"


@pytest.mark.unit
def test_cross_market_neutral_small_move():
    sig = cross_market.build_cross_market_signal(
        symbol="012450", proxy_name="ITA 야간", proxy_pct=0.3,
    )
    assert sig.direction == "neutral"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cross_market.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/cross_market.py
"""Corvin 선행 인텔리전스 — Pillar 4: 교차시장 (advisory only).

시간대 시차 활용한 참고용 방향 힌트:
- KR 종목: LMT/RTX/ITA 야간 마감(미국장이 KR보다 먼저 닫힘) → 012450 힌트
- US 종목: NQ/ES 선물 → META/MSFT/NVDA/TSLA 힌트

⚠️ deep-research 검증 약함(0-3 기각) → advisory=True 강제 + confidence 30% 페널티.
순수 함수. 라이브 프록시 가격은 호출부 책임.
"""
from __future__ import annotations

from corvin_jarvis.signals.leading_signal import LeadingSignal

# 검증 약함 반영: 신뢰도 30% 페널티
CONFIDENCE_PENALTY = 0.7
# 방향 판정 임계 (프록시 %change)
NEUTRAL_BAND = 0.5


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def build_cross_market_signal(
    symbol: str, proxy_name: str, proxy_pct: float
) -> LeadingSignal:
    """프록시 %change → advisory 교차시장 신호. confidence 페널티 적용."""
    if proxy_pct >= NEUTRAL_BAND:
        direction = "bull"
    elif proxy_pct <= -NEUTRAL_BAND:
        direction = "bear"
    else:
        direction = "neutral"
    base_conf = _clamp(abs(proxy_pct) * 10.0)
    confidence = round(base_conf * CONFIDENCE_PENALTY, 2)
    label = {"bull": "강세", "bear": "약세", "neutral": "중립"}[direction]
    return LeadingSignal(
        pillar="cross_market", symbol=symbol, direction=direction,
        confidence=confidence, score=None, horizon="intraday",
        advisory=True,
        message=f"🌐 {symbol} {proxy_name} {proxy_pct:+.1f}% → {label} 힌트",
        evidence={"proxy_name": proxy_name, "proxy_pct": proxy_pct,
                  "penalized": True},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cross_market.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/cross_market.py tests/test_cross_market.py
git commit -m "feat(corvin): Pillar 4 cross-market advisory signals"
```

---

## Task 2: DART 공시 파서 + 신호 (Pillar 2 확장)

**Files:**
- Modify: `corvin_jarvis/signals/event_calendar.py`
- Test: `tests/test_event_calendar.py`

DART 공시 리스트(이미 fetch됨)에서 촉매 키워드(수주/계약/증자/실적)만 필터 → event 신호.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_calendar.py 에 추가
@pytest.mark.unit
def test_filter_dart_keeps_catalyst_keywords():
    disclosures = [
        {"report_nm": "단일판매ㆍ공급계약체결", "rcept_dt": "20260601"},
        {"report_nm": "주주총회소집결의", "rcept_dt": "20260601"},  # 비촉매
        {"report_nm": "유상증자결정", "rcept_dt": "20260601"},
    ]
    kept = event_calendar.filter_dart_disclosures(disclosures)
    names = [d["report_nm"] for d in kept]
    assert "단일판매ㆍ공급계약체결" in names
    assert "유상증자결정" in names
    assert "주주총회소집결의" not in names


@pytest.mark.unit
def test_build_dart_signals():
    disclosures = [{"report_nm": "단일판매ㆍ공급계약체결", "rcept_dt": "20260601"}]
    sigs = event_calendar.build_dart_signals("012450", disclosures)
    assert len(sigs) == 1
    s = sigs[0]
    assert s.pillar == "event"
    assert s.symbol == "012450"
    assert s.confidence >= 60.0
    assert "공급계약" in s.message or "계약" in s.message


@pytest.mark.unit
def test_build_dart_signals_empty():
    assert event_calendar.build_dart_signals("012450", []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_event_calendar.py -k dart -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/event_calendar.py 끝에 추가

# DART 공시 촉매 키워드 (수주/계약/증자/실적 등 주가 영향 큰 항목)
DART_CATALYST_KEYWORDS = ("공급계약", "수주", "계약체결", "증자", "실적", "영업정지",
                          "합병", "분할", "자기주식")


def filter_dart_disclosures(disclosures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DART 공시 리스트에서 촉매 키워드 포함 항목만 필터."""
    out: list[dict[str, Any]] = []
    for d in disclosures:
        name = str(d.get("report_nm", ""))
        if any(kw in name for kw in DART_CATALYST_KEYWORDS):
            out.append(d)
    return out


def build_dart_signals(
    symbol: str, disclosures: list[dict[str, Any]]
) -> list[LeadingSignal]:
    """필터된 DART 공시 → event 신호. 촉매 공시 = 확정 신뢰도."""
    signals: list[LeadingSignal] = []
    for d in filter_dart_disclosures(disclosures):
        name = str(d.get("report_nm", ""))
        signals.append(LeadingSignal(
            pillar="event", symbol=symbol, direction="neutral",
            confidence=70.0, score=None, horizon="days",
            advisory=False,
            message=f"📰 {symbol} DART 공시: {name}",
            evidence={"kind": "dart", "report_nm": name,
                      "rcept_dt": d.get("rcept_dt")},
        ))
    return signals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_event_calendar.py -k dart -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/event_calendar.py tests/test_event_calendar.py
git commit -m "feat(corvin): DART disclosure detection for Pillar 2"
```

---

## Task 3: 오케스트레이터 collect + dispatch

**Files:**
- Modify: `corvin_jarvis/leading_orchestrator.py`
- Test: `tests/test_leading_orchestrator.py`

`collect(providers)`: provider 콜백 리스트 실행 → 신호 평탄화 (각 provider 예외 시 스킵). `dispatch_brief(brief, sender)`: 빈 브리프면 미발송(False), 아니면 sender 호출.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leading_orchestrator.py 에 추가
@pytest.mark.unit
def test_collect_flattens_providers():
    p1 = lambda: [_sig("012450", 80.0)]
    p2 = lambda: [_sig("META", 70.0), _sig("MSFT", 65.0)]
    sigs = orch.collect([p1, p2])
    assert len(sigs) == 3


@pytest.mark.unit
def test_collect_skips_failing_provider():
    def boom():
        raise RuntimeError("provider down")
    p_ok = lambda: [_sig("012450", 80.0)]
    sigs = orch.collect([boom, p_ok])
    assert len(sigs) == 1     # 실패 provider 스킵, 나머지 진행


@pytest.mark.unit
def test_dispatch_brief_sends_nonempty():
    sent = {}
    def fake_sender(body: str) -> bool:
        sent["body"] = body
        return True
    ok = orch.dispatch_brief("내용 있음", sender=fake_sender)
    assert ok is True
    assert sent["body"] == "내용 있음"


@pytest.mark.unit
def test_dispatch_brief_skips_empty():
    called = {"n": 0}
    def fake_sender(body: str) -> bool:
        called["n"] += 1
        return True
    ok = orch.dispatch_brief("", sender=fake_sender)
    assert ok is False
    assert called["n"] == 0     # 빈 브리프 → sender 미호출
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_leading_orchestrator.py -k "collect or dispatch" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/leading_orchestrator.py 에 추가 (상단 import 아래)
import logging
from typing import Callable

log = logging.getLogger("corvin.leading")


def collect(
    providers: list[Callable[[], list[LeadingSignal]]]
) -> list[LeadingSignal]:
    """provider 콜백들을 실행해 신호를 평탄화. 실패 provider는 스킵(추측 금지)."""
    out: list[LeadingSignal] = []
    for p in providers:
        try:
            out.extend(p())
        except Exception as e:  # noqa: BLE001 - graceful degrade
            log.warning("leading provider 실패, 스킵: %s", e)
    return out


def dispatch_brief(
    brief: str, sender: Callable[[str], bool]
) -> bool:
    """브리프 발송. 빈 문자열이면 미발송(False). 노이즈 게이트 결과 존중."""
    if not brief:
        return False
    return sender(brief)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_leading_orchestrator.py -k "collect or dispatch" -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/leading_orchestrator.py tests/test_leading_orchestrator.py
git commit -m "feat(corvin): orchestrator collect + dispatch for leading intelligence"
```

---

## Task 4: 통합 — 4 Pillar end-to-end 손동작 확인

**Files:**
- Test: 통합 검증 (실제 전송 없음 — conftest 안전망 + fake sender)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leading_orchestrator.py 에 추가
from datetime import date

from corvin_jarvis.signals import event_calendar, fundamental, ensemble, cross_market


@pytest.mark.unit
def test_end_to_end_four_pillars():
    # 4 Pillar provider → collect → format → dispatch (fake sender)
    providers = [
        lambda: event_calendar.build_event_signals(
            as_of=date(2026, 6, 1),
            earnings_rows=[{"symbol": "012450", "earnings_date": date(2026, 6, 4)}],
            macro_horizon_days=30,
        ),
        lambda: [fundamental.build_fundamental_signal(
            "012450", financial=85.0, qualitative=80.0, news=70.0)],
        lambda: [ensemble.build_ensemble_signal(
            "012450", minervini=90.0, rs=80.0, volume=70.0)],
        lambda: [cross_market.build_cross_market_signal(
            "012450", proxy_name="LMT 야간", proxy_pct=3.0)],
    ]
    signals = orch.collect(providers)
    brief = orch.format_brief(signals, threshold=60.0)

    sent = {}
    ok = orch.dispatch_brief(brief, sender=lambda b: sent.update(body=b) or True)
    assert ok is True
    # 펀더멘털·앙상블·이벤트는 본문, 교차시장은 "참고용" 섹션
    assert "펀더멘털" in sent["body"]
    assert "앙상블" in sent["body"]
    assert "참고용" in sent["body"]
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python3 -m pytest tests/test_leading_orchestrator.py -k end_to_end -v`
Expected: PASS (모든 부품이 이미 구현됨 — 통합 동작 확인용).
만약 FAIL이면 메시지에서 누락 부품 확인 후 수정.

- [ ] **Step 3: Commit**

```bash
git add tests/test_leading_orchestrator.py
git commit -m "test(corvin): 4-pillar end-to-end integration (no real send)"
```

---

## Task 5: cron 러너 스크립트 (수동 등록용)

**Files:**
- Create: `corvin_jarvis/run_leading.py`
- Test: 손동작(import + dry-run)

⚠️ auto-mode persistence 금지 정책 → crontab 자동 등록 안 함. 스크립트만 제공 + 사용자 수동 등록.

- [ ] **Step 1: Write the runner (no test — thin wiring, verified by import)**

```python
# corvin_jarvis/run_leading.py
"""Corvin 선행 인텔리전스 — cron 진입점.

4 Pillar provider를 수집 → 게이트 → 브리프 → Telegram 발송.
실제 라이브 fetch provider 배선은 추후(Plan 5+); 현재는 이벤트 캘린더만 라이브.

⚠️ crontab 자동 등록 금지(정책). 사용자 수동 등록:
    30 8 * * 1-5 cd <repo> && python3 -m corvin_jarvis.run_leading >> state/leading.log 2>&1
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from corvin_jarvis import channels, leading_orchestrator as orch
from corvin_jarvis.signals import event_calendar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("corvin.run_leading")


def _today() -> date:
    """오늘 날짜 (cron 환경 결정성 위해 분리 — 테스트서 monkeypatch 가능)."""
    return date.today()


def main() -> int:
    as_of = _today()
    providers = [
        lambda: event_calendar.build_event_signals(
            as_of=as_of, earnings_rows=[], macro_horizon_days=30,
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

- [ ] **Step 2: Verify import + dry run (telegram disabled in dev → no real send)**

Run: `python3 -m corvin_jarvis.run_leading`
Expected: 로그 출력 `leading brief: signals=N sent=False` (dev 환경 telegram 미설정 → 미발송). 에러 없음.

- [ ] **Step 3: Commit**

```bash
git add corvin_jarvis/run_leading.py
git commit -m "feat(corvin): leading intelligence cron runner (manual register)"
```

---

## Task 6: 전체 회귀 + 마커

- [ ] **Step 1: 전체 테스트**

Run: `python3 -m pytest tests/ -q`
Expected: 전체 PASS, 무회귀.

- [ ] **Step 2: Commit (slice marker)**

```bash
git commit -m "test(corvin): Pillar 4 + integration verified" --allow-empty
```

---

## 자체 검토 (spec 대비)

- **Spec §3 Pillar 4 교차시장 advisory + 30% 페널티** → Task 1 ✅
- **Spec §3 Pillar 2 DART 공시 감지** → Task 2 ✅
- **Spec §2 오케스트레이터 수집/디스패치** → Task 3 ✅
- **Spec §2 4 Pillar 통합** → Task 4 ✅
- **Spec §4 빈 브리프 미발송(노이즈)** → Task 3 `dispatch_brief` ✅
- **Spec §4 Telegram 라우팅** → Task 5 `channels.send_telegram` ✅
- **Spec §7 테스트 실전송 금지** → conftest 안전망 + fake sender ✅
- **cron 수동 등록 정책** → Task 5 주석 명시, 자동 등록 안 함 ✅
- **라이브 fetch provider 배선**: 이벤트 캘린더만 라이브. 펀더멘털/앙상블/교차시장의 실제 데이터 fetch 어댑터(yfinance·KIS·Claude haiku·GDELT 호출)는 **후속(Plan 5)** — 현재는 순수 점수 함수 + 수동 주입까지 완료. spec §5 데이터 소스 배선이 다음 단계임을 명기.
