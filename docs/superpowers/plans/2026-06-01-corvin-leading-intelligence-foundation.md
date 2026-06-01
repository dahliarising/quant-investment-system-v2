# Corvin 선행 인텔리전스 — Foundation + 이벤트 캘린더 (Plan 1) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모든 Pillar가 공유하는 `LeadingSignal` 계약 + 신뢰도 게이트 + 이벤트 지평선 신호(Pillar 2) + 오케스트레이터를 만들어, 실제 동작하는 선행 알림 1개 수직 슬라이스를 완성한다.

**Architecture:** 각 Pillar는 동일한 `LeadingSignal` dataclass를 반환한다. 오케스트레이터가 Pillar들을 수집 → 신뢰도<60 무음 게이트 통과 → 한국어 메시지로 포맷한다. Plan 1은 Pillar 2(이벤트 캘린더, 기존 `earnings.py` 위에 거시 일정 추가)만 연결하여 end-to-end 동작을 검증한다. 나머지 Pillar 1/3/4는 후속 plan.

**Tech Stack:** Python 3, pytest (`@pytest.mark.unit`), dataclass(frozen), 기존 `corvin_jarvis/` 모듈, SQLite(earnings_calendar 재사용).

**전제(스코프 한계):**
- 이 plan은 4 Pillar 중 **Pillar 2 + 공통 인프라**만 구현한다. Pillar 1(펀더멘털)·3(앙상블)·4(교차시장)은 후속 plan에서 동일 계약 위에 추가한다.
- 실제 Telegram 전송 배선은 기존 `channels.send_telegram`을 재사용하므로 이 plan에서는 **오케스트레이터가 메시지 문자열을 반환**하는 지점까지만 만든다 (전송 호출은 기존 jarvis 파이프라인이 담당). 테스트는 실제 전송 0건(conftest 안전망).

**테스트 실행 규약:** 모든 테스트는 저장소 루트(`quant_investment_system_v2/`)에서 `python3 -m pytest tests/<file> -v`로 실행. import는 `from corvin_jarvis...`.

---

## 파일 구조

| 파일 | 책임 | 신규/수정 |
|------|------|-----------|
| `corvin_jarvis/signals/leading_signal.py` | 공통 `LeadingSignal` dataclass + 신뢰도 게이트 함수 | 신규 |
| `corvin_jarvis/signals/event_calendar.py` | Pillar 2: 거시 일정 + 실적 D-N → LeadingSignal | 신규 |
| `corvin_jarvis/leading_orchestrator.py` | Pillar 수집 → 게이트 → 한국어 포맷 | 신규 |
| `tests/test_leading_signal.py` | 계약 + 게이트 테스트 | 신규 |
| `tests/test_event_calendar.py` | Pillar 2 테스트 | 신규 |
| `tests/test_leading_orchestrator.py` | 오케스트레이터 테스트 | 신규 |

각 모듈 200줄 이내 목표, 단일 책임.

---

## Task 1: 공통 계약 `LeadingSignal` dataclass

**Files:**
- Create: `corvin_jarvis/signals/leading_signal.py`
- Test: `tests/test_leading_signal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leading_signal.py
"""Tests for corvin_jarvis.signals.leading_signal."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_leading_signal_is_frozen():
    sig = LeadingSignal(
        pillar="event", symbol="012450", direction="neutral",
        confidence=72.0, score=None, horizon="days",
        advisory=False, message="실적 D-3", evidence={"days_to": 3},
    )
    with pytest.raises(Exception):
        sig.confidence = 10.0  # frozen → 변경 불가


@pytest.mark.unit
def test_leading_signal_rejects_bad_direction():
    with pytest.raises(ValueError):
        LeadingSignal(
            pillar="event", symbol="012450", direction="up",  # invalid
            confidence=50.0, score=None, horizon="days",
            advisory=False, message="x", evidence={},
        )


@pytest.mark.unit
def test_leading_signal_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        LeadingSignal(
            pillar="event", symbol="012450", direction="bull",
            confidence=150.0, score=None, horizon="days",  # >100
            advisory=False, message="x", evidence={},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_leading_signal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corvin_jarvis.signals.leading_signal'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/leading_signal.py
"""Corvin 선행 인텔리전스 — 모든 Pillar 공통 신호 계약.

각 Pillar(fundamental/event/ensemble/cross_market)는 이 LeadingSignal을 반환한다.
오케스트레이터는 Pillar 종류와 무관하게 게이트·포맷·디스패치한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_PILLARS = {"fundamental", "event", "ensemble", "cross_market"}
VALID_DIRECTIONS = {"bull", "neutral", "bear"}
VALID_HORIZONS = {"intraday", "days", "weeks"}


@dataclass(frozen=True)
class LeadingSignal:
    """선행 신호 단일 계약. frozen → 불변(immutability 원칙)."""

    pillar: str
    symbol: str
    direction: str          # bull | neutral | bear
    confidence: float       # 0-100
    score: float | None     # pillar별 원점수 (예: Growth Score). 없으면 None
    horizon: str            # intraday | days | weeks
    advisory: bool          # True면 "참고용" 태그 강제 (검증 약한 Pillar)
    message: str            # 한국어 해석
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pillar not in VALID_PILLARS:
            raise ValueError(f"invalid pillar: {self.pillar}")
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(f"invalid direction: {self.direction}")
        if self.horizon not in VALID_HORIZONS:
            raise ValueError(f"invalid horizon: {self.horizon}")
        if not (0.0 <= self.confidence <= 100.0):
            raise ValueError(f"confidence out of range: {self.confidence}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_leading_signal.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/leading_signal.py tests/test_leading_signal.py
git commit -m "feat(corvin): LeadingSignal common contract for leading intelligence"
```

---

## Task 2: 신뢰도 게이트 (confidence < 60 무음)

**Files:**
- Modify: `corvin_jarvis/signals/leading_signal.py` (함수 추가)
- Test: `tests/test_leading_signal.py` (테스트 추가)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leading_signal.py 에 추가
from corvin_jarvis.signals.leading_signal import apply_confidence_gate


def _sig(conf, advisory=False, direction="bull"):
    return LeadingSignal(
        pillar="event", symbol="012450", direction=direction,
        confidence=conf, score=None, horizon="days",
        advisory=advisory, message="x", evidence={},
    )


@pytest.mark.unit
def test_gate_mutes_below_threshold():
    sigs = [_sig(40.0), _sig(59.9), _sig(60.0), _sig(85.0)]
    passed = apply_confidence_gate(sigs, threshold=60.0)
    assert [s.confidence for s in passed] == [60.0, 85.0]


@pytest.mark.unit
def test_gate_advisory_never_solo():
    # advisory 신호는 게이트를 통과해도 solo=False (브리프 내 한 줄로만)
    sigs = [_sig(90.0, advisory=True)]
    passed = apply_confidence_gate(sigs, threshold=60.0)
    assert len(passed) == 1
    assert passed[0].advisory is True


@pytest.mark.unit
def test_gate_empty_returns_empty():
    assert apply_confidence_gate([], threshold=60.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_leading_signal.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_confidence_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/leading_signal.py 끝에 추가

def apply_confidence_gate(
    signals: list[LeadingSignal], threshold: float = 60.0
) -> list[LeadingSignal]:
    """신뢰도 < threshold 신호를 무음 처리(필터). 노이즈 제어 핵심.

    메모리 feedback_alert_noise_aversion: 알림은 양보다 질. <60 = 로그만.
    advisory 신호는 통과해도 오케스트레이터가 단독 알림으로 만들지 않는다.
    """
    return [s for s in signals if s.confidence >= threshold]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_leading_signal.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/leading_signal.py tests/test_leading_signal.py
git commit -m "feat(corvin): confidence gate (<60 mute) for leading signals"
```

---

## Task 3: 거시 일정 캘린더 (FOMC/BOK) → 날짜 조회

**Files:**
- Create: `corvin_jarvis/signals/event_calendar.py`
- Test: `tests/test_event_calendar.py`

설계 노트: `Date.now()`/실시간 의존을 피하기 위해 모든 함수는 `as_of: date`를 **명시 인자로** 받는다 (테스트 가능성 + 결정성).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_calendar.py
"""Tests for corvin_jarvis.signals.event_calendar."""
from __future__ import annotations

from datetime import date

import pytest

from corvin_jarvis.signals import event_calendar


@pytest.mark.unit
def test_macro_events_returns_known_dates():
    # 하드코딩 거시 캘린더에서 FOMC/BOK 일정 조회
    events = event_calendar.macro_events_within(
        as_of=date(2026, 6, 1), horizon_days=30
    )
    # 각 이벤트는 (name, event_date) 형태
    assert all("name" in e and "event_date" in e for e in events)
    # 30일 내 이벤트만
    for e in events:
        delta = (e["event_date"] - date(2026, 6, 1)).days
        assert 0 <= delta <= 30


@pytest.mark.unit
def test_macro_events_excludes_past():
    events = event_calendar.macro_events_within(
        as_of=date(2026, 6, 1), horizon_days=30
    )
    for e in events:
        assert e["event_date"] >= date(2026, 6, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_event_calendar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corvin_jarvis.signals.event_calendar'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/event_calendar.py
"""Corvin 선행 인텔리전스 — Pillar 2: 이벤트 지평선 캘린더.

가격과 무관하게 "알려진 미래 촉매"를 미리 경보 → 진짜 시장 선행성.
- 거시 일정(FOMC/BOK): 하드코딩 캘린더 (분기 갱신)
- 실적 D-N: 기존 earnings.py 재사용 (Task 4)

모든 함수는 as_of(date)를 명시 인자로 받아 결정성·테스트가능성을 보장한다.
"""
from __future__ import annotations

from datetime import date
from typing import Any

# 분기마다 갱신하는 거시 캘린더. (name, ISO date)
# 2026 FOMC/BOK 일정 — 갱신 시 이 리스트만 수정.
MACRO_CALENDAR: list[tuple[str, str]] = [
    ("FOMC 금리결정", "2026-06-17"),
    ("BOK 금융통화위원회", "2026-06-11"),
    ("FOMC 금리결정", "2026-07-29"),
    ("BOK 금융통화위원회", "2026-07-09"),
]


def macro_events_within(as_of: date, horizon_days: int = 30) -> list[dict[str, Any]]:
    """as_of 기준 horizon_days 이내의 거시 이벤트만 반환."""
    out: list[dict[str, Any]] = []
    for name, iso in MACRO_CALENDAR:
        ev = date.fromisoformat(iso)
        delta = (ev - as_of).days
        if 0 <= delta <= horizon_days:
            out.append({"name": name, "event_date": ev, "days_to": delta})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_event_calendar.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/event_calendar.py tests/test_event_calendar.py
git commit -m "feat(corvin): macro event calendar (FOMC/BOK) for Pillar 2"
```

---

## Task 4: 실적 D-N + 거시 → LeadingSignal 생성

**Files:**
- Modify: `corvin_jarvis/signals/event_calendar.py` (함수 추가)
- Test: `tests/test_event_calendar.py` (테스트 추가)

기존 `earnings.py`에는 `earnings_calendar` 테이블과 `ALERT_OFFSETS=(7,3,1)`이 있다. 여기서는 **이미 조회된 실적일 리스트**를 받아 LeadingSignal로 변환하는 순수 함수를 만든다(DB 조회는 호출부 책임 → 테스트 결정성).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_calendar.py 에 추가
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_build_event_signals_earnings_dn():
    # 실적이 D-3이면 event 신호 생성
    earnings = [{"symbol": "012450", "earnings_date": date(2026, 6, 4)}]
    sigs = event_calendar.build_event_signals(
        as_of=date(2026, 6, 1),
        earnings_rows=earnings,
        macro_horizon_days=30,
    )
    earn_sigs = [s for s in sigs if s.symbol == "012450"]
    assert len(earn_sigs) == 1
    s = earn_sigs[0]
    assert isinstance(s, LeadingSignal)
    assert s.pillar == "event"
    assert s.direction == "neutral"      # 이벤트는 방향성 없음
    assert s.evidence["days_to"] == 3
    assert "D-3" in s.message


@pytest.mark.unit
def test_build_event_signals_skips_far_earnings():
    # D-10은 ALERT_OFFSETS(7,3,1) 밖 → 신호 없음
    earnings = [{"symbol": "012450", "earnings_date": date(2026, 6, 11)}]
    sigs = event_calendar.build_event_signals(
        as_of=date(2026, 6, 1), earnings_rows=earnings, macro_horizon_days=0
    )
    assert [s for s in sigs if s.symbol == "012450"] == []


@pytest.mark.unit
def test_build_event_signals_macro_advisory():
    # 거시 이벤트는 advisory=False(중요)지만 symbol="_MACRO"로 시장 전체 대상
    sigs = event_calendar.build_event_signals(
        as_of=date(2026, 6, 1), earnings_rows=[], macro_horizon_days=30
    )
    macro = [s for s in sigs if s.pillar == "event" and s.symbol == "_MACRO"]
    assert len(macro) >= 1
    assert all(s.confidence >= 60.0 for s in macro)  # 알려진 일정 = 확정 신뢰도
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_event_calendar.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_event_signals'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/signals/event_calendar.py 끝에 추가
from corvin_jarvis.signals.leading_signal import LeadingSignal

# earnings.py와 동일한 알림 시점 (D-7/3/1)
EARNINGS_ALERT_OFFSETS = (7, 3, 1)


def build_event_signals(
    as_of: date,
    earnings_rows: list[dict[str, Any]],
    macro_horizon_days: int = 30,
) -> list[LeadingSignal]:
    """실적 D-N + 거시 일정 → LeadingSignal 리스트.

    earnings_rows: [{"symbol": str, "earnings_date": date}, ...] (호출부가 DB에서 조회).
    이벤트는 방향성 없음 → direction="neutral", horizon="days".
    """
    signals: list[LeadingSignal] = []

    # 실적 D-N
    for row in earnings_rows:
        sym = row["symbol"]
        ev = row["earnings_date"]
        days_to = (ev - as_of).days
        if days_to in EARNINGS_ALERT_OFFSETS:
            signals.append(LeadingSignal(
                pillar="event", symbol=sym, direction="neutral",
                confidence=80.0, score=None, horizon="days",
                advisory=False,
                message=f"📅 {sym} 실적 D-{days_to} ({ev.isoformat()})",
                evidence={"kind": "earnings", "days_to": days_to,
                          "event_date": ev.isoformat()},
            ))

    # 거시 일정 (시장 전체 대상 → symbol="_MACRO")
    for ev in macro_events_within(as_of, macro_horizon_days):
        signals.append(LeadingSignal(
            pillar="event", symbol="_MACRO", direction="neutral",
            confidence=75.0, score=None, horizon="days",
            advisory=False,
            message=f"🏛️ {ev['name']} D-{ev['days_to']} ({ev['event_date'].isoformat()})",
            evidence={"kind": "macro", "days_to": ev["days_to"],
                      "name": ev["name"]},
        ))

    return signals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_event_calendar.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/event_calendar.py tests/test_event_calendar.py
git commit -m "feat(corvin): event signals (earnings D-N + macro) for Pillar 2"
```

---

## Task 5: 오케스트레이터 — 수집 → 게이트 → 한국어 포맷

**Files:**
- Create: `corvin_jarvis/leading_orchestrator.py`
- Test: `tests/test_leading_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leading_orchestrator.py
"""Tests for corvin_jarvis.leading_orchestrator."""
from __future__ import annotations

import pytest

from corvin_jarvis import leading_orchestrator as orch
from corvin_jarvis.signals.leading_signal import LeadingSignal


def _sig(sym, conf, advisory=False, msg="x", direction="bull"):
    return LeadingSignal(
        pillar="event", symbol=sym, direction=direction,
        confidence=conf, score=None, horizon="days",
        advisory=advisory, message=msg, evidence={},
    )


@pytest.mark.unit
def test_format_brief_groups_passed_signals():
    sigs = [_sig("012450", 80.0, msg="📅 012450 실적 D-3"),
            _sig("META", 40.0, msg="무음대상")]  # 40 → 게이트 탈락
    brief = orch.format_brief(sigs, threshold=60.0)
    assert "012450" in brief
    assert "무음대상" not in brief


@pytest.mark.unit
def test_format_brief_advisory_tagged():
    sigs = [_sig("012450", 90.0, advisory=True, msg="LMT 야간 강세")]
    brief = orch.format_brief(sigs, threshold=60.0)
    assert "참고용" in brief
    assert "LMT 야간 강세" in brief


@pytest.mark.unit
def test_format_brief_all_muted_returns_quiet_marker():
    sigs = [_sig("META", 30.0), _sig("MSFT", 50.0)]
    brief = orch.format_brief(sigs, threshold=60.0)
    assert brief == ""  # 전부 무음 → 빈 문자열(전송 안 함 신호)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_leading_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corvin_jarvis.leading_orchestrator'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/leading_orchestrator.py
"""Corvin 선행 인텔리전스 — 오케스트레이터.

4 Pillar가 반환한 LeadingSignal을 수집 → 신뢰도 게이트 → 한국어 브리프 포맷.
Plan 1은 Pillar 2(event_calendar)만 연결. 후속 plan에서 Pillar 1/3/4 추가.

전송(channels.send_telegram)은 기존 jarvis 파이프라인이 담당.
이 모듈은 "보낼 문자열"을 반환하는 데까지만 책임진다.
"""
from __future__ import annotations

from corvin_jarvis.signals.leading_signal import (
    LeadingSignal,
    apply_confidence_gate,
)


def format_brief(signals: list[LeadingSignal], threshold: float = 60.0) -> str:
    """게이트 통과 신호를 한국어 브리프 문자열로 포맷.

    - advisory 신호는 "참고용" 태그를 붙여 본문 하단에 한 줄로만.
    - 통과 신호가 없으면 빈 문자열 반환(= 전송 안 함).
    """
    passed = apply_confidence_gate(signals, threshold)
    if not passed:
        return ""

    primary = [s for s in passed if not s.advisory]
    advisory = [s for s in passed if s.advisory]

    lines: list[str] = ["**🔭 Corvin 선행 인텔리전스 브리프**", ""]
    for s in primary:
        lines.append(f"- {s.message} (신뢰도 {s.confidence:.0f})")

    if advisory:
        lines.append("")
        lines.append("_참고용 (검증 약함):_")
        for s in advisory:
            lines.append(f"- {s.message}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_leading_orchestrator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/leading_orchestrator.py tests/test_leading_orchestrator.py
git commit -m "feat(corvin): leading intelligence orchestrator (gate + KR brief)"
```

---

## Task 6: 전체 회귀 + 슬라이스 통합 확인

**Files:**
- Test: 전체 스위트

- [ ] **Step 1: 전체 테스트 실행 (회귀 없음 확인)**

Run: `python3 -m pytest tests/ -q`
Expected: 신규 테스트 포함 전체 PASS, 기존 테스트 무회귀.

- [ ] **Step 2: 수직 슬라이스 손동작 확인 (실제 전송 없음)**

Run:
```bash
python3 -c "
from datetime import date
from corvin_jarvis.signals import event_calendar
from corvin_jarvis import leading_orchestrator as orch
sigs = event_calendar.build_event_signals(
    as_of=date(2026, 6, 1),
    earnings_rows=[{'symbol': '012450', 'earnings_date': date(2026, 6, 4)}],
    macro_horizon_days=30,
)
print(orch.format_brief(sigs))
"
```
Expected: 실적 D-3 + 거시 일정이 포함된 한국어 브리프 출력 (전송 아님, stdout만).

- [ ] **Step 3: Commit (슬라이스 완료 마커)**

```bash
git add -A
git commit -m "test(corvin): leading intelligence foundation slice verified" --allow-empty
```

---

## 자체 검토 결과 (spec 대비)

- **Spec §2 공통 계약** → Task 1 ✅
- **Spec §4 노이즈 게이트(<60 무음)** → Task 2 ✅
- **Spec §3 Pillar 2 이벤트 캘린더(실적+거시)** → Task 3,4 ✅
- **Spec §2 오케스트레이터(수집→게이트→포맷)** → Task 5 ✅
- **Spec §4 advisory "참고용" 태그** → Task 5 (`format_brief`) ✅
- **Spec §7 테스트 실전송 금지** → conftest autouse 안전망 + format_brief는 문자열만 반환 ✅
- **DART 공시 감지** → Pillar 2의 일부지만 외부 API 의존 → 후속 plan으로 분리(이 slice는 실적+거시로 동작 검증). spec §3에 DART 명시됨 → **후속 plan에서 구현 예정**임을 명기.

**후속 plan (이 plan 범위 밖):**
- Plan 2: Pillar 1 펀더멘털 성장 엔진 (재무+Claude haiku+GDELT)
- Plan 3: Pillar 3 앙상블 스크리너 (leading.py/dca_timing.py 재사용)
- Plan 4: Pillar 4 교차시장 advisory + DART 공시 감지 + cron 배선/디스패치 통합
