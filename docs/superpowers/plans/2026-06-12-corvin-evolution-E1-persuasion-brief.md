# Corvin Evolution Loop — Phase E1: 설득 브리핑 레이어 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 신호 데이터를 결정론적으로 조합해 "다각도 설득 6요소"를 갖춘 일일 실행 브리핑을 생성하고 텔레그램으로 push한다 (진화 인프라 없이 단독 출시).

**Architecture:** `corvin_jarvis/brief/` 신규 모듈. 6개 단일책임 파일(types·position_overlay·evidence·framing·psychology·render·builder)이 frozen dataclass 인터페이스로 통신. 데이터는 전부 기존 소스(`pulse.fetch_portfolio`·`final_actions.json`·`calibration.json`·`market_hours`)에서 읽고, LLM 호출 없음. `notify.py` digest가 brief 블록을 호출.

**Tech Stack:** Python 3.11, dataclasses(frozen), pytest, 기존 `corvin_jarvis` 모듈.

**선행 spec:** [2026-06-12 Corvin Evolution Loop 설계서](../specs/2026-06-12-corvin-evolution-loop-design.md) §3.3.

---

## File Structure

| 파일 | 책임 |
|------|------|
| `corvin_jarvis/brief/__init__.py` | 패키지 마커 |
| `corvin_jarvis/brief/types.py` | frozen dataclass 인터페이스 (PositionLine·EvidenceStack·Framing·PsychGuard·Brief) |
| `corvin_jarvis/brief/position_overlay.py` | `pulse.fetch_portfolio()` → PositionLine[] (평단대비 손익+액션라벨) |
| `corvin_jarvis/brief/evidence.py` | final_actions+calibration → 종목별 EvidenceStack (n·적중률 or '검증부족') |
| `corvin_jarvis/brief/framing.py` | 신호 → Framing (bull/bear/base + counterfactual), 템플릿 기반 |
| `corvin_jarvis/brief/psychology.py` | 포지션/신호 → PsychGuard (4대 패턴 트리거 → 자문 1줄) |
| `corvin_jarvis/brief/render.py` | Brief → 스캔 포맷 텍스트 (구분선·아이콘·가격신선도 라벨) |
| `corvin_jarvis/brief/builder.py` | 위 컴포넌트 조합 → Brief → render. digest 진입점. |
| `tests/test_brief_*.py` | 모듈별 단위 테스트 |

**액션 아이콘 규약** (메모리 `feedback_discord_scannable_format`): `✂️`축소 `✅`유지 `➕`추가존 `👀`관찰.

---

## Task 1: brief 패키지 + 인터페이스 타입

**Files:**
- Create: `corvin_jarvis/brief/__init__.py`
- Create: `corvin_jarvis/brief/types.py`
- Test: `tests/test_brief_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_types.py
import dataclasses
from corvin_jarvis.brief.types import (
    PositionLine, EvidenceStack, Framing, PsychGuard, Brief,
)


def test_position_line_is_frozen():
    pl = PositionLine(symbol="META", pnl_pct=-4.9, action_icon="👀",
                      label="관망존", fresh_label="종가")
    assert pl.symbol == "META"
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        pl.symbol = "X"  # type: ignore[misc]


def test_brief_composes_all_parts():
    brief = Brief(
        headline="테스트", positions=[], evidence={}, framing=None,
        psych=None, as_of="2026-06-12", market_state="장마감",
    )
    assert brief.market_state == "장마감"
    assert brief.positions == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corvin_jarvis.brief'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/brief/__init__.py
"""Corvin 설득 브리핑 레이어 (Evolution Loop Phase E1)."""
```

```python
# corvin_jarvis/brief/types.py
"""설득 브리핑 컴포넌트 간 frozen dataclass 인터페이스."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionLine:
    symbol: str
    pnl_pct: float | None
    action_icon: str          # ✂️ ✅ ➕ 👀
    label: str                # 종목당 한 줄 액션
    fresh_label: str          # "종가"/"전일종가" 등 가격 신선도


@dataclass(frozen=True)
class EvidenceStack:
    validation: str           # "n=12·적중률 58%" 또는 "검증부족"
    edge_summary: str | None  # 백테스트 엣지 요약 (E1엔 보통 None)


@dataclass(frozen=True)
class Framing:
    bull: str
    bear: str
    base: str
    counterfactual: str


@dataclass(frozen=True)
class PsychGuard:
    triggered: bool
    pattern: str | None       # 항복매수/사건전FOMO/드로다운/앵커링
    question: str | None      # "룰 진입 vs 감정 반응?" 자문 1줄


@dataclass(frozen=True)
class Brief:
    headline: str
    positions: list[PositionLine]
    evidence: dict[str, EvidenceStack]
    framing: Framing | None
    psych: PsychGuard | None
    as_of: str
    market_state: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief_types.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/brief/__init__.py corvin_jarvis/brief/types.py tests/test_brief_types.py
git commit -m "feat(brief): E1 설득 브리핑 인터페이스 타입 (frozen dataclasses)"
```

---

## Task 2: 포지션 오버레이 (보유분 평단대비 손익 + 액션라벨)

**Files:**
- Create: `corvin_jarvis/brief/position_overlay.py`
- Test: `tests/test_brief_position_overlay.py`

데이터 소스: `pulse.fetch_portfolio()` → `position_dicts` (키: symbol, pnl_pct, bucket, currency, current_price, error). 액션라벨 규칙:
- `pnl_pct is None` (시세 실패) → `👀` "시세없음"
- `bucket == "dca"` 그리고 `pnl_pct <= -12` → `✂️` "손절존 검토"
- `pnl_pct >= 8` → `✅` "유지(추세양호)"
- `-3 <= pnl_pct < 8` → `✅` "유지"
- `-12 < pnl_pct < -3` → `👀` "관망존"
- else (`pnl_pct <= -12`, non-dca) → `✂️` "비중축소 검토"

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_position_overlay.py
from corvin_jarvis.brief.position_overlay import build_position_lines


def test_winner_held():
    pos = [{"symbol": "MSFT", "pnl_pct": 12.9, "bucket": "trade",
            "currency": "USD", "current_price": 390.3, "error": None}]
    lines = build_position_lines(pos, fresh_label="종가")
    assert lines[0].action_icon == "✅"
    assert lines[0].symbol == "MSFT"
    assert "유지" in lines[0].label


def test_dca_stop_zone():
    pos = [{"symbol": "012450", "pnl_pct": -17.4, "bucket": "dca",
            "currency": "KRW", "current_price": 1014000, "error": None}]
    lines = build_position_lines(pos, fresh_label="종가")
    assert lines[0].action_icon == "✂️"
    assert "손절존" in lines[0].label


def test_quote_failure_is_watch():
    pos = [{"symbol": "X", "pnl_pct": None, "bucket": "trade",
            "currency": "USD", "current_price": None, "error": "no data"}]
    lines = build_position_lines(pos, fresh_label="종가")
    assert lines[0].action_icon == "👀"
    assert "시세없음" in lines[0].label
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief_position_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/brief/position_overlay.py
"""보유분 평단대비 손익 → 액션라벨 오버레이. 데이터: pulse.fetch_portfolio()."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.types import PositionLine


def _classify(pnl_pct: float | None, bucket: str) -> tuple[str, str]:
    if pnl_pct is None:
        return "👀", "시세없음"
    if bucket == "dca" and pnl_pct <= -12:
        return "✂️", "손절존 검토"
    if pnl_pct >= 8:
        return "✅", "유지(추세양호)"
    if pnl_pct >= -3:
        return "✅", "유지"
    if pnl_pct > -12:
        return "👀", "관망존"
    return "✂️", "비중축소 검토"


def build_position_lines(positions: list[dict[str, Any]],
                         fresh_label: str) -> list[PositionLine]:
    lines: list[PositionLine] = []
    for p in positions:
        pnl = p.get("pnl_pct")
        bucket = str(p.get("bucket") or "trade")
        icon, label = _classify(pnl, bucket)
        lines.append(PositionLine(
            symbol=str(p.get("symbol", "")),
            pnl_pct=pnl,
            action_icon=icon,
            label=label,
            fresh_label=fresh_label,
        ))
    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief_position_overlay.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/brief/position_overlay.py tests/test_brief_position_overlay.py
git commit -m "feat(brief): 포지션 오버레이 — 평단대비 손익→액션라벨(✂️✅👀)"
```

---

## Task 3: 근거 스택 (n·적중률 or '검증부족' 정직 라벨)

**Files:**
- Create: `corvin_jarvis/brief/evidence.py`
- Test: `tests/test_brief_evidence.py`

데이터 소스: `calibration.json` 구조 `{engine: {kind: {n, hit_rate, avg_confidence, calibrated_confidence}}}`. 규칙:
- 해당 종목 액션에 매핑되는 (engine, kind) 표본 `n >= 10` → `"n={n}·적중률 {hit_rate*100:.0f}%"`
- `n < 10` 또는 없음 → `"검증부족(n={n})"` (n 없으면 `"검증부족"`) — 메모리 `meaningful_metrics`: 거짓 확신 금지.

`min_samples`는 인자(기본 10)로 노출해 spec §4와 일치.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_evidence.py
from corvin_jarvis.brief.evidence import build_evidence


def test_validated_sample():
    calib = {"leading": {"sector": {"n": 12, "hit_rate": 0.58,
             "avg_confidence": 60, "calibrated_confidence": 59}}}
    ev = build_evidence(calib, engine="leading", kind="sector")
    assert ev.validation == "n=12·적중률 58%"


def test_insufficient_sample_is_honest():
    calib = {"predictive": {"EVENT": {"n": 1, "hit_rate": 0.0,
             "avg_confidence": 90, "calibrated_confidence": None}}}
    ev = build_evidence(calib, engine="predictive", kind="EVENT")
    assert ev.validation == "검증부족(n=1)"


def test_missing_entry_is_honest():
    ev = build_evidence({}, engine="x", kind="y")
    assert ev.validation == "검증부족"
    assert ev.edge_summary is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief_evidence.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/brief/evidence.py
"""신호의 검증 근거 스택. 표본 부족 시 '검증부족' 정직 라벨 (거짓 확신 금지)."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.types import EvidenceStack


def build_evidence(calibration: dict[str, Any], *, engine: str, kind: str,
                   min_samples: int = 10) -> EvidenceStack:
    entry = (calibration.get(engine) or {}).get(kind)
    if not entry:
        return EvidenceStack(validation="검증부족", edge_summary=None)
    n = int(entry.get("n", 0))
    if n < min_samples:
        return EvidenceStack(validation=f"검증부족(n={n})", edge_summary=None)
    hit = float(entry.get("hit_rate", 0.0)) * 100
    return EvidenceStack(validation=f"n={n}·적중률 {hit:.0f}%", edge_summary=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief_evidence.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/brief/evidence.py tests/test_brief_evidence.py
git commit -m "feat(brief): 근거 스택 — n·적중률 / 표본부족 시 '검증부족' 정직 라벨"
```

---

## Task 4: 투자심리 가드 (4대 패턴 → 자문 1줄)

**Files:**
- Create: `corvin_jarvis/brief/psychology.py`
- Test: `tests/test_brief_psychology.py`

메모리 `feedback_investor_psychology` 4대 패턴. E1 트리거(결정론):
- **드로다운 흔들림**: 보유 중 `pnl_pct <= -12`인 종목 존재 → "드로다운에 흔들리는 중? 룰 손절선 vs 감정 반응 먼저 점검."
- **사건전 FOMO / 항복매수**: 액션이 신규 `매수후보`인데 해당 종목 최근 강세(`framing.bull` 우세) → "추격 진입인가? 룰 진입존 vs FOMO 자문."
- 둘 다 아니면 `triggered=False`.

우선순위: 드로다운 > FOMO. 한 번에 1개만.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_psychology.py
from corvin_jarvis.brief.psychology import build_psych_guard


def test_drawdown_trigger():
    positions = [{"symbol": "012450", "pnl_pct": -17.4}]
    g = build_psych_guard(positions, has_new_buy_candidate=False)
    assert g.triggered is True
    assert g.pattern == "드로다운"
    assert "룰 손절선" in g.question


def test_fomo_trigger_when_no_drawdown():
    positions = [{"symbol": "MSFT", "pnl_pct": 12.0}]
    g = build_psych_guard(positions, has_new_buy_candidate=True)
    assert g.triggered is True
    assert g.pattern == "FOMO"


def test_no_trigger():
    positions = [{"symbol": "MSFT", "pnl_pct": 2.0}]
    g = build_psych_guard(positions, has_new_buy_candidate=False)
    assert g.triggered is False
    assert g.question is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief_psychology.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/brief/psychology.py
"""투자심리 가드 — 4대 패턴 트리거 시 '룰 vs 감정' 자문 1줄."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.types import PsychGuard

_DRAWDOWN_PCT = -12.0


def build_psych_guard(positions: list[dict[str, Any]], *,
                      has_new_buy_candidate: bool) -> PsychGuard:
    deep = [p for p in positions
            if p.get("pnl_pct") is not None and p["pnl_pct"] <= _DRAWDOWN_PCT]
    if deep:
        return PsychGuard(
            triggered=True, pattern="드로다운",
            question="드로다운에 흔들리는 중? 룰 손절선 vs 감정 반응 먼저 점검.",
        )
    if has_new_buy_candidate:
        return PsychGuard(
            triggered=True, pattern="FOMO",
            question="추격 진입인가? 룰 진입존 도달 vs FOMO 자문.",
        )
    return PsychGuard(triggered=False, pattern=None, question=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief_psychology.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/brief/psychology.py tests/test_brief_psychology.py
git commit -m "feat(brief): 투자심리 가드 — 드로다운/FOMO 트리거 자문 1줄"
```

---

## Task 5: 다각 프레이밍 (bull/bear/base + counterfactual)

**Files:**
- Create: `corvin_jarvis/brief/framing.py`
- Test: `tests/test_brief_framing.py`

데이터 소스: `final_actions.json` actions(action·rationale·urgency) + summary winners/losers. E1은 **템플릿 기반**(LLM 생성은 E2 진화 몫):
- **bull**: `매수후보`/`유지` 신호 수 + 대표 rationale.
- **bear**: `매도검토`/`비중축소`/방어 신호 수 + 대표 rationale.
- **base**: arbiter 순(net) 한 줄 — "방어 N · 매수후보 M".
- **counterfactual**: 최고 urgency 액션 기준 "지금 {action} 안 하면: {urgency 기반 기회/리스크}".

입력은 정규화된 dict 리스트(아래 시그니처)로 받아 final_actions 파싱과 분리(테스트 용이).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_framing.py
from corvin_jarvis.brief.framing import build_framing


def test_framing_counts_directions():
    actions = [
        {"symbol": "TSLA", "action": "매도검토", "urgency": 95, "rationale": "추세이탈"},
        {"symbol": "NVDA", "action": "유지", "urgency": 40, "rationale": "모멘텀양호"},
        {"symbol": "BWXT", "action": "매수후보", "urgency": 55, "rationale": "원자력테제"},
    ]
    f = build_framing(actions)
    assert "추세이탈" in f.bear
    assert "모멘텀양호" in f.bull or "매수후보" in f.bull
    assert "방어" in f.base
    # 최고 urgency=TSLA 매도검토 기준 counterfactual
    assert "TSLA" in f.counterfactual


def test_framing_empty_is_safe():
    f = build_framing([])
    assert f.base
    assert f.counterfactual
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief_framing.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/brief/framing.py
"""다각 프레이밍 — bull/bear/base + counterfactual (E1 템플릿 기반)."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.types import Framing

_BEAR = {"매도검토", "비중축소"}
_BULL = {"매수후보", "유지", "추가존"}


def _rep_rationale(actions: list[dict[str, Any]], kinds: set[str]) -> str:
    for a in sorted(actions, key=lambda x: x.get("urgency", 0), reverse=True):
        if a.get("action") in kinds and a.get("rationale"):
            return str(a["rationale"])
    return ""


def build_framing(actions: list[dict[str, Any]]) -> Framing:
    bear_n = sum(1 for a in actions if a.get("action") in _BEAR)
    bull_n = sum(1 for a in actions if a.get("action") in _BULL)
    bull_r = _rep_rationale(actions, _BULL)
    bear_r = _rep_rationale(actions, _BEAR)

    bull = f"강세축 {bull_n}건" + (f" — {bull_r}" if bull_r else "")
    bear = f"방어축 {bear_n}건" + (f" — {bear_r}" if bear_r else "")
    base = f"base: 방어 {bear_n} · 매수후보/유지 {bull_n}"

    if actions:
        top = max(actions, key=lambda x: x.get("urgency", 0))
        cf = (f"지금 {top.get('symbol')} {top.get('action')} 미실행 시: "
              f"urgency {top.get('urgency', 0)} 신호 방치 — {top.get('rationale', '')}")
    else:
        cf = "활성 액션 없음 — 관망이 base."
    return Framing(bull=bull, bear=bear, base=base, counterfactual=cf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief_framing.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/brief/framing.py tests/test_brief_framing.py
git commit -m "feat(brief): 다각 프레이밍 — bull/bear/base + counterfactual(템플릿)"
```

---

## Task 6: 렌더러 (스캔 포맷 + 가격 신선도 라벨)

**Files:**
- Create: `corvin_jarvis/brief/render.py`
- Test: `tests/test_brief_render.py`

메모리 규칙: 구분선·여백·상태아이콘·종목당 한 줄·통합 1메시지(`feedback_discord_scannable_format`), 가격은 `종가/전일종가` 라벨+시장상태(`feedback_price_freshness_labeling`). 구분선 = `━━━━━━━━━━━━━━`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_render.py
from corvin_jarvis.brief.types import (
    Brief, PositionLine, EvidenceStack, Framing, PsychGuard,
)
from corvin_jarvis.brief.render import render_brief


def _sample() -> Brief:
    return Brief(
        headline="🦅 Corvin 실행 브리핑 — 06/12",
        positions=[
            PositionLine("012450", -17.4, "✂️", "손절존 검토", "종가"),
            PositionLine("MSFT", 12.9, "✅", "유지", "종가"),
        ],
        evidence={"012450": EvidenceStack("검증부족(n=1)", None)},
        framing=Framing("강세축 2건", "방어축 1건 — 추세이탈",
                        "base: 방어 1 · 매수후보/유지 2", "지금 TSLA 미실행 시…"),
        psych=PsychGuard(True, "드로다운", "룰 손절선 vs 감정 반응 점검."),
        as_of="2026-06-12 05:45 KST", market_state="장마감",
    )


def test_render_has_all_sections_and_scannable():
    out = render_brief(_sample())
    assert "━━━" in out                     # 구분선
    assert "✂️ 012450" in out                # 종목당 한 줄
    assert "-17.4%" in out
    assert "종가" in out and "장마감" in out    # 가격 신선도
    assert "검증부족(n=1)" in out             # 정직 라벨
    assert "드로다운" in out                  # 심리 가드
    assert "base:" in out                    # 프레이밍


def test_render_single_message_under_limit():
    out = render_brief(_sample())
    assert len(out) <= 1900                  # 텔레그램/디스코드 단일 메시지
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief_render.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/brief/render.py
"""Brief → 스캔 가능한 단일 메시지 텍스트 (구분선·신선도 라벨)."""
from __future__ import annotations

from corvin_jarvis.brief.types import Brief

_DIV = "━━━━━━━━━━━━━━"


def _pnl(pl) -> str:
    return f"{pl.pnl_pct:+.1f}%" if pl.pnl_pct is not None else "n/a"


def render_brief(brief: Brief) -> str:
    out: list[str] = [brief.headline,
                      f"_{brief.as_of} · {brief.market_state} 기준_", ""]
    out.append("📊 **포지션**")
    for pl in brief.positions:
        ev = brief.evidence.get(pl.symbol)
        tag = f"  ·{ev.validation}" if ev else ""
        out.append(f"{pl.action_icon} {pl.symbol} {_pnl(pl)} ({pl.fresh_label}) — {pl.label}{tag}")
    if brief.framing:
        out += [_DIV, "🧭 **다각 프레이밍**",
                f"🔼 {brief.framing.bull}", f"🔽 {brief.framing.bear}",
                f"• {brief.framing.base}", f"⚠️ {brief.framing.counterfactual}"]
    if brief.psych and brief.psych.triggered:
        out += [_DIV, f"🧠 **심리 체크** ({brief.psych.pattern})", brief.psych.question or ""]
    text = "\n".join(out)
    return text[:1900]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief_render.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/brief/render.py tests/test_brief_render.py
git commit -m "feat(brief): 렌더러 — 스캔 포맷·구분선·가격 신선도 라벨·단일메시지"
```

---

## Task 7: 빌더 (컴포넌트 조합 → Brief)

**Files:**
- Create: `corvin_jarvis/brief/builder.py`
- Test: `tests/test_brief_builder.py`

조합 책임만. 데이터 로딩은 인자로 주입(테스트 격리). 시그니처:
`build_brief(positions, actions, calibration, market_state, fresh_label, as_of) -> Brief`.
- evidence: 각 보유 종목에 대해, actions에서 그 종목의 (engine='leading' 등) 매핑이 복잡하므로 E1에선 **액션의 sources[0]+action을 (engine, kind) 근사**로 calibration 조회. 매핑 없으면 '검증부족'.
- `has_new_buy_candidate`: actions에 `매수후보`이며 held 아님인 게 있으면 True.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_builder.py
from corvin_jarvis.brief.builder import build_brief


def test_build_brief_end_to_end():
    positions = [
        {"symbol": "012450", "pnl_pct": -17.4, "bucket": "dca",
         "currency": "KRW", "current_price": 1014000, "error": None},
        {"symbol": "MSFT", "pnl_pct": 12.9, "bucket": "trade",
         "currency": "USD", "current_price": 390.3, "error": None},
    ]
    actions = [
        {"symbol": "TSLA", "action": "매도검토", "urgency": 95,
         "rationale": "추세이탈", "sources": ["jarvis"]},
        {"symbol": "BWXT", "action": "매수후보", "urgency": 55,
         "rationale": "원자력", "sources": ["leading"]},
    ]
    calibration = {"leading": {"매수후보": {"n": 12, "hit_rate": 0.6}}}
    brief = build_brief(positions=positions, actions=actions,
                        calibration=calibration, held={"012450", "MSFT"},
                        market_state="장마감", fresh_label="종가",
                        as_of="2026-06-12 05:45 KST")
    syms = {p.symbol for p in brief.positions}
    assert syms == {"012450", "MSFT"}
    # 012450 깊은 손실 → 드로다운 심리 가드
    assert brief.psych.triggered and brief.psych.pattern == "드로다운"
    assert brief.framing is not None
    assert brief.headline.startswith("🦅")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief_builder.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/brief/builder.py
"""설득 브리핑 빌더 — 컴포넌트 조합. 데이터 로딩은 호출자 책임(주입)."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.evidence import build_evidence
from corvin_jarvis.brief.framing import build_framing
from corvin_jarvis.brief.position_overlay import build_position_lines
from corvin_jarvis.brief.psychology import build_psych_guard
from corvin_jarvis.brief.types import Brief, EvidenceStack


def build_brief(*, positions: list[dict[str, Any]], actions: list[dict[str, Any]],
                calibration: dict[str, Any], held: set[str],
                market_state: str, fresh_label: str, as_of: str) -> Brief:
    lines = build_position_lines(positions, fresh_label=fresh_label)

    evidence: dict[str, EvidenceStack] = {}
    for a in actions:
        sym = str(a.get("symbol", ""))
        if not sym:
            continue
        engine = (a.get("sources") or ["?"])[0]
        evidence[sym] = build_evidence(calibration, engine=engine,
                                       kind=str(a.get("action", "")))

    framing = build_framing(actions)

    has_new_buy = any(a.get("action") == "매수후보"
                      and str(a.get("symbol")) not in held for a in actions)
    psych = build_psych_guard(positions, has_new_buy_candidate=has_new_buy)

    headline = f"🦅 Corvin 실행 브리핑 — {as_of.split(' ')[0][5:]}"
    return Brief(headline=headline, positions=lines, evidence=evidence,
                 framing=framing, psych=psych, as_of=as_of,
                 market_state=market_state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief_builder.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/brief/builder.py tests/test_brief_builder.py
git commit -m "feat(brief): 빌더 — 6요소 컴포넌트 조합 → Brief"
```

---

## Task 8: digest 통합 + 진입점

**Files:**
- Create: `corvin_jarvis/brief/__main__.py` (CLI 진입점)
- Modify: `corvin_jarvis/notify.py` (digest에 brief 블록 1줄 추가)
- Test: `tests/test_brief_entrypoint.py`

`__main__.py`는 실데이터를 로딩해 brief를 만들고 stdout 출력(크론이 캡처). 데이터 로딩:
- positions, _ = `pulse.fetch_portfolio()`
- actions = `final_actions.json`의 actions
- calibration = `state/calibration.json`
- held = `notify._held_symbols()`
- market_state/fresh_label = `market_hours`로 판정 (장중→"현재가"/"개장", 그외→"종가"/"장마감")

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_entrypoint.py
from corvin_jarvis.brief.__main__ import compose_from_sources


def test_compose_from_sources_injected(tmp_path, monkeypatch):
    # 데이터 로더를 주입해 외부 IO 없이 검증
    positions = [{"symbol": "MSFT", "pnl_pct": 12.9, "bucket": "trade",
                  "currency": "USD", "current_price": 390.3, "error": None}]
    text = compose_from_sources(
        load_positions=lambda: positions,
        load_actions=lambda: [{"symbol": "BWXT", "action": "매수후보",
                               "urgency": 55, "rationale": "원자력",
                               "sources": ["leading"]}],
        load_calibration=lambda: {},
        load_held=lambda: {"MSFT"},
        market_state="장마감", fresh_label="종가",
        as_of="2026-06-12 05:45 KST",
    )
    assert "MSFT" in text
    assert "━━━" in text
    assert len(text) <= 1900
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief_entrypoint.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/brief/__main__.py
"""brief CLI 진입점. `python -m corvin_jarvis.brief` → 설득 브리핑 stdout."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from corvin_jarvis.brief.builder import build_brief
from corvin_jarvis.brief.render import render_brief


def compose_from_sources(*,
                         load_positions: Callable[[], list[dict[str, Any]]],
                         load_actions: Callable[[], list[dict[str, Any]]],
                         load_calibration: Callable[[], dict[str, Any]],
                         load_held: Callable[[], set[str]],
                         market_state: str, fresh_label: str,
                         as_of: str) -> str:
    brief = build_brief(
        positions=load_positions(), actions=load_actions(),
        calibration=load_calibration(), held=load_held(),
        market_state=market_state, fresh_label=fresh_label, as_of=as_of)
    return render_brief(brief)


def _market_labels() -> tuple[str, str]:
    from corvin_jarvis import market_hours
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    try:
        open_now = market_hours.is_any_open(now)  # 존재 시
    except AttributeError:
        open_now = False
    return ("장중", "현재가") if open_now else ("장마감", "종가")


def main() -> None:
    import json
    from pathlib import Path

    from corvin_jarvis import notify, pulse

    state = Path(__file__).resolve().parent.parent / "state"
    market_state, fresh_label = _market_labels()
    as_of = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")

    def load_actions() -> list[dict[str, Any]]:
        try:
            return json.loads((state / "final_actions.json").read_text()).get("actions", [])
        except OSError:
            return []

    def load_calibration() -> dict[str, Any]:
        try:
            return json.loads((state / "calibration.json").read_text())
        except OSError:
            return {}

    print(compose_from_sources(
        load_positions=lambda: pulse.fetch_portfolio()[0],
        load_actions=load_actions,
        load_calibration=load_calibration,
        load_held=notify._held_symbols,
        market_state=market_state, fresh_label=fresh_label, as_of=as_of))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief_entrypoint.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run full suite (no regression)**

Run: `pytest -q`
Expected: 기존 773 + 신규 brief 테스트 전부 PASS

- [ ] **Step 6: Commit**

```bash
git add corvin_jarvis/brief/__main__.py tests/test_brief_entrypoint.py
git commit -m "feat(brief): CLI 진입점 — 실데이터 조합 → 설득 브리핑 stdout"
```

---

## Task 9: 크론 러너 스크립트 + 텔레그램 push

**Files:**
- Create: `scripts/run_brief.sh`
- Test: 수동 검증 (스크립트)

라우팅: 메모리 `feedback_cron_imessage_only` — **텔레그램 달리아봇만**. 기존 `channels.py send_telegram()` 재사용. 스크립트는 `python -m corvin_jarvis.brief` 출력을 캡처해 `channels`로 전송.

- [ ] **Step 1: Create the runner script**

```bash
# scripts/run_brief.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f corvin_jarvis/.env ] && . corvin_jarvis/.env; set +a
BRIEF="$(python3 -m corvin_jarvis.brief)"
python3 - "$BRIEF" <<'PY'
import sys
from corvin_jarvis import channels
channels.send(sys.argv[1], channels_override=["telegram", "log_only"])
PY
```

- [ ] **Step 2: chmod + dry-run**

Run: `chmod +x scripts/run_brief.sh && python3 -m corvin_jarvis.brief | head -30`
Expected: 설득 브리핑 텍스트가 stdout에 출력 (포지션·프레이밍·구분선 포함)

- [ ] **Step 3: Verify channels signature**

Run: `python3 -c "from corvin_jarvis import channels; import inspect; print([n for n in dir(channels) if 'send' in n])"`
Expected: `send` 또는 `send_telegram` 존재 확인. 시그니처 불일치 시 스크립트의 호출부를 실제 함수에 맞춰 수정 (channels.py:`send`/`send_telegram` 중 존재하는 것 사용).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_brief.sh
git commit -m "feat(brief): 크론 러너 — 설득 브리핑 텔레그램 달리아봇 push"
```

- [ ] **Step 5: 크론 등록 (수동, 폐하 승인 후)**

장마감 후 시각에 등록 (기존 run_digest 16:00과 분리 or 통합 — 폐하 결정):
```
# 예: KR 장마감 직후 15:45 KST
45 15 * * 1-5 /Users/thethethe/Claude/quant_investment_system_v2/scripts/run_brief.sh >> /tmp/corvin_brief.log 2>&1
```

---

## Self-Review

**Spec coverage (§3.3 설득 6요소):**
1. 포지션 오버레이 → Task 2 ✓
2. 확신 근거 스택(n·적중률/검증부족) → Task 3 ✓
3. 다각 프레이밍(bull/bear/base+counterfactual) → Task 5 ✓
4. 투자심리 가드 → Task 4 ✓
5. 스캔 포맷 → Task 6 ✓
6. 가격 신선도 라벨 → Task 6 ✓
- 텔레그램 달리아봇 라우팅 → Task 9 ✓
- 단일 메시지 한계 → Task 6 (≤1900) ✓

**Placeholder scan:** 없음. 모든 step에 실제 코드/명령. (Task 9 Step 3·5는 실환경 시그니처 확인/등록이라 의도적 수동.)

**Type consistency:** `PositionLine`/`EvidenceStack`/`Framing`/`PsychGuard`/`Brief` 필드명이 Task 1 정의와 Task 2·3·4·5·6·7에서 일치. `build_position_lines(positions, fresh_label=)`·`build_evidence(calibration, engine=, kind=, min_samples=)`·`build_framing(actions)`·`build_psych_guard(positions, has_new_buy_candidate=)`·`build_brief(...)`·`render_brief(brief)`·`compose_from_sources(...)` 시그니처 전 task 일관.

**알려진 통합 리스크 (실행 중 확인):**
- `market_hours.is_any_open` 함수명이 다를 수 있음 → Task 8 `_market_labels()`가 AttributeError를 폴백 처리(기본 장마감). 실제 함수명 확인 후 정정.
- `channels.send` vs `send_telegram` 시그니처 → Task 9 Step 3에서 확인 후 맞춤.
- E1 evidence의 (engine,kind) 매핑은 근사(액션→sources[0]). 정밀 매핑은 E2에서 ledger 직조회로 개선 — 현재는 대부분 '검증부족'으로 정직 표기되어 안전.

---

**다음 Phase**: E2(진화 인프라 골격, 드라이런) → 별도 plan. 본 E1은 단독으로 폐하께 더 나은 일일 브리핑을 제공한다.
