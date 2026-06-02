# Corvin Playbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모든 모니터링/보유 종목에 일관된 매수·익절 래더(존+비율)를 자동 부여하고, 매일 텍스트 푸시 + HTML 대시보드 두 화면으로 "타이밍·비율"을 한눈에 보여준다.

**Architecture:** 순수함수 계층(technicals → ladder → position)으로 종목별 `Playbook` 객체를 만들고, `builder`가 지표 fetch + `portfolio.json` stance 결정으로 조립한다. 두 렌더러(text/html)가 같은 `Playbook` 리스트를 소비한다. 진입점 `run_playbook`이 빌드→렌더→푸시/저장한다.

**Tech Stack:** Python 3.9, pytest, dataclasses(frozen), yfinance(US)·pykrx(KR), 기존 `channels.send_telegram`.

**참고 스펙:** `docs/superpowers/specs/2026-06-02-corvin-playbook-design.md`

---

## File Structure

신규 패키지 `corvin_jarvis/playbook/`:
- `__init__.py` — 패키지 마커
- `models.py` — `Technicals`, `Zone`, `Playbook` (frozen dataclass)
- `technicals.py` — 순수 지표: `sma`, `rsi`, `recent_high`
- `ladder.py` — 순수: `build_buy_ladder`, `build_trim_ladder`
- `position.py` — `stance`, `classify` (순수)
- `builder.py` — `load_holdings`, `fetch_closes`, `build_playbooks`
- `render_text.py` — `render_push`
- `render_html.py` — `render_dashboard`
- `run_playbook.py` — `main` 진입점

테스트: `tests/test_playbook_*.py` (모듈별).
운영: `corvin_jarvis/run_playbook.sh` (크론 래퍼, `run_leading.sh` 패턴).

상수(모듈 상단):
- 매수 비율 `(30, 40, 30)`, 익절 비율 `(33, 33, 34)`
- 존 밴드 `BAND = 0.015` (±1.5%)
- stance 임계 `HARVEST_PNL = 5.0` (%)
- 과열 RSI `RSI_HOT = 70.0`, 과매도 `RSI_COLD = 30.0`
- 최소 히스토리 `MIN_BARS = 50`

---

## Task 1: 패키지 스캐폴드 + 데이터 모델

**Files:**
- Create: `corvin_jarvis/playbook/__init__.py`
- Create: `corvin_jarvis/playbook/models.py`
- Test: `tests/test_playbook_models.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_playbook_models.py
"""Tests for corvin_jarvis.playbook.models."""
from __future__ import annotations

import dataclasses

import pytest

from corvin_jarvis.playbook import models


@pytest.mark.unit
def test_zone_is_frozen() -> None:
    z = models.Zone(kind="buy", label="Z1", ratio=30, low=100.0, high=103.0, note="MA20")
    with pytest.raises(dataclasses.FrozenInstanceError):
        z.ratio = 40  # type: ignore[misc]


@pytest.mark.unit
def test_playbook_holds_zones_tuple() -> None:
    tech = models.Technicals(symbol="X", market="US", price=10.0, ma20=11.0,
                             ma50=12.0, rsi=50.0, hi_52w=15.0)
    z = models.Zone(kind="buy", label="Z1", ratio=30, low=9.9, high=10.1, note="")
    pb = models.Playbook(symbol="X", name="X Co", stance="ENTER", tech=tech,
                         zones=(z,), status="BUY_NOW", badge="🟢",
                         pnl_pct=None, active_zone=z)
    assert pb.zones[0].ratio == 30
    assert pb.active_zone is z
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_playbook_models.py -v`
Expected: FAIL — `ModuleNotFoundError: corvin_jarvis.playbook`

- [ ] **Step 3: 최소 구현**

```python
# corvin_jarvis/playbook/__init__.py
"""Corvin Playbook — 종목별 매수/익절 래더 + 시각 대시보드."""
```

```python
# corvin_jarvis/playbook/models.py
"""플레이북 데이터 모델 (불변)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Technicals:
    symbol: str
    market: str  # "US" | "KR"
    price: float
    ma20: float
    ma50: float
    rsi: float
    hi_52w: float


@dataclass(frozen=True)
class Zone:
    kind: str    # "buy" | "trim"
    label: str
    ratio: int   # percent
    low: float
    high: float
    note: str


@dataclass(frozen=True)
class Playbook:
    symbol: str
    name: str
    stance: str           # "ENTER" | "ACCUMULATE" | "HARVEST"
    tech: Technicals
    zones: tuple[Zone, ...]
    status: str           # "BUY_NOW" | "TRIM_NOW" | "WAIT" | "INVALID"
    badge: str
    pnl_pct: float | None
    active_zone: Zone | None
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_playbook_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add corvin_jarvis/playbook/__init__.py corvin_jarvis/playbook/models.py tests/test_playbook_models.py
git commit -m "feat(playbook): scaffold package and frozen data models"
```

---

## Task 2: 순수 지표 (technicals.py)

**Files:**
- Create: `corvin_jarvis/playbook/technicals.py`
- Test: `tests/test_playbook_technicals.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_playbook_technicals.py
"""Tests for corvin_jarvis.playbook.technicals."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import technicals as t


@pytest.mark.unit
def test_sma_simple() -> None:
    assert t.sma([10, 20, 30], 3) == 20.0


@pytest.mark.unit
def test_sma_uses_last_n() -> None:
    assert t.sma([1, 2, 100, 200], 2) == 150.0


@pytest.mark.unit
def test_sma_insufficient_returns_none() -> None:
    assert t.sma([1, 2], 3) is None


@pytest.mark.unit
def test_rsi_all_up_is_100() -> None:
    prices = list(range(1, 30))  # strictly increasing
    assert t.rsi(prices) == 100.0


@pytest.mark.unit
def test_rsi_midrange_for_mixed() -> None:
    prices = [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10]
    r = t.rsi(prices)
    assert 30.0 < r < 70.0


@pytest.mark.unit
def test_recent_high() -> None:
    assert t.recent_high([5, 9, 3, 7], 252) == 9.0
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_playbook_technicals.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 최소 구현**

```python
# corvin_jarvis/playbook/technicals.py
"""순수 기술지표 — 외부 의존 없음, 리스트 in → 스칼라 out."""
from __future__ import annotations


def sma(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def rsi(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for prev, cur in zip(prices[-(period + 1):-1], prices[-period:]):
        diff = cur - prev
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def recent_high(prices: list[float], window: int = 252) -> float | None:
    if not prices:
        return None
    return max(prices[-window:])
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_playbook_technicals.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add corvin_jarvis/playbook/technicals.py tests/test_playbook_technicals.py
git commit -m "feat(playbook): pure technical indicators (sma/rsi/recent_high)"
```

---

## Task 3: 래더 규칙 (ladder.py)

**Files:**
- Create: `corvin_jarvis/playbook/ladder.py`
- Test: `tests/test_playbook_ladder.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_playbook_ladder.py
"""Tests for corvin_jarvis.playbook.ladder."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import ladder
from corvin_jarvis.playbook.models import Technicals


def _tech(price=100.0, ma20=105.0, ma50=110.0, rsi=50.0, hi=130.0) -> Technicals:
    return Technicals(symbol="X", market="US", price=price, ma20=ma20,
                      ma50=ma50, rsi=rsi, hi_52w=hi)


@pytest.mark.unit
def test_buy_ladder_has_three_zones_30_40_30() -> None:
    zones = ladder.build_buy_ladder(_tech())
    assert [z.ratio for z in zones] == [30, 40, 30]
    assert all(z.kind == "buy" for z in zones)


@pytest.mark.unit
def test_buy_ladder_levels_anchor_to_ma() -> None:
    zones = ladder.build_buy_ladder(_tech(ma20=105.0, ma50=110.0))
    z1, z2, z3 = zones
    assert z1.low < 105.0 < z1.high      # Z1 around MA20
    assert z2.low < 110.0 < z2.high      # Z2 around MA50
    assert z3.high < 110.0               # Z3 below MA50 (deep)


@pytest.mark.unit
def test_trim_ladder_has_three_zones_33_33_34() -> None:
    zones = ladder.build_trim_ladder(_tech())
    assert [z.ratio for z in zones] == [33, 33, 34]
    assert all(z.kind == "trim" for z in zones)


@pytest.mark.unit
def test_trim_z2_anchors_to_52w_high() -> None:
    zones = ladder.build_trim_ladder(_tech(hi=130.0))
    z2 = zones[1]
    assert z2.low <= 130.0 <= z2.high
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_playbook_ladder.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 최소 구현**

```python
# corvin_jarvis/playbook/ladder.py
"""고정룰 래더 — 기술지표 → 존(가격밴드+비율). 순수."""
from __future__ import annotations

from corvin_jarvis.playbook.models import Technicals, Zone

BAND = 0.015
BUY_RATIOS = (30, 40, 30)
TRIM_RATIOS = (33, 33, 34)


def _band(level: float) -> tuple[float, float]:
    return level * (1 - BAND), level * (1 + BAND)


def build_buy_ladder(tech: Technicals) -> tuple[Zone, ...]:
    z1_lo, z1_hi = _band(tech.ma20)
    z2_lo, z2_hi = _band(tech.ma50)
    deep = tech.ma50 * 0.95
    z3_lo, z3_hi = _band(deep)
    return (
        Zone("buy", "Z1 1차눌림", BUY_RATIOS[0], z1_lo, z1_hi, "MA20"),
        Zone("buy", "Z2 핵심지지", BUY_RATIOS[1], z2_lo, z2_hi, "MA50"),
        Zone("buy", "Z3 딥밸류", BUY_RATIOS[2], z3_lo, z3_hi, "MA50-5%/과매도"),
    )


def build_trim_ladder(tech: Technicals) -> tuple[Zone, ...]:
    hot = max(tech.ma20 * 1.10, tech.price if tech.rsi >= 70 else 0.0)
    z1_lo, z1_hi = _band(hot)
    z2_lo, z2_hi = _band(tech.hi_52w)
    return (
        Zone("trim", "Z1 과열", TRIM_RATIOS[0], z1_lo, z1_hi, "RSI≥70/MA20+10%"),
        Zone("trim", "Z2 전고", TRIM_RATIOS[1], z2_lo, z2_hi, "52주고"),
        Zone("trim", "Z3 코어홀드", TRIM_RATIOS[2], tech.ma50, tech.hi_52w,
             "MA50 이탈청산"),
    )
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_playbook_ladder.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
git add corvin_jarvis/playbook/ladder.py tests/test_playbook_ladder.py
git commit -m "feat(playbook): fixed-rule buy/trim ladders"
```

---

## Task 4: stance + 상태 분류 (position.py)

**Files:**
- Create: `corvin_jarvis/playbook/position.py`
- Test: `tests/test_playbook_position.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_playbook_position.py
"""Tests for corvin_jarvis.playbook.position."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import ladder, position
from corvin_jarvis.playbook.models import Technicals


def _tech(price=100.0, ma20=105.0, ma50=110.0, rsi=50.0, hi=130.0) -> Technicals:
    return Technicals(symbol="X", market="US", price=price, ma20=ma20,
                      ma50=ma50, rsi=rsi, hi_52w=hi)


@pytest.mark.unit
def test_stance_enter_when_not_held() -> None:
    assert position.stance(pnl_pct=None) == "ENTER"


@pytest.mark.unit
def test_stance_accumulate_when_underwater() -> None:
    assert position.stance(pnl_pct=-7.3) == "ACCUMULATE"


@pytest.mark.unit
def test_stance_harvest_when_in_profit() -> None:
    assert position.stance(pnl_pct=20.2) == "HARVEST"


@pytest.mark.unit
def test_buy_status_buy_now_when_price_in_or_below_deep() -> None:
    tech = _tech(price=100.0, ma20=105.0, ma50=110.0)  # below all buy zones
    zones = ladder.build_buy_ladder(tech)
    status, badge, active = position.classify(tech, zones, "ENTER")
    assert status == "BUY_NOW"
    assert badge == "🟢"
    assert active is not None


@pytest.mark.unit
def test_buy_status_wait_when_price_above_zones() -> None:
    tech = _tech(price=200.0, ma20=105.0, ma50=110.0)
    zones = ladder.build_buy_ladder(tech)
    status, badge, active = position.classify(tech, zones, "ENTER")
    assert status == "WAIT"
    assert active is None


@pytest.mark.unit
def test_trim_status_trim_now_when_overheated() -> None:
    tech = _tech(price=145.0, ma20=120.0, ma50=110.0, rsi=78.0, hi=145.0)
    zones = ladder.build_trim_ladder(tech)
    status, badge, active = position.classify(tech, zones, "HARVEST")
    assert status == "TRIM_NOW"
    assert badge == "✂️"


@pytest.mark.unit
def test_trim_status_invalid_when_below_ma50() -> None:
    tech = _tech(price=100.0, ma20=120.0, ma50=110.0, rsi=40.0, hi=145.0)
    zones = ladder.build_trim_ladder(tech)
    status, badge, active = position.classify(tech, zones, "HARVEST")
    assert status == "INVALID"
    assert badge == "⚠️"
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_playbook_position.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 최소 구현**

```python
# corvin_jarvis/playbook/position.py
"""stance 결정 + 현재가 → 존/상태 분류. 순수."""
from __future__ import annotations

from corvin_jarvis.playbook.models import Technicals, Zone

HARVEST_PNL = 5.0

_BADGE = {"BUY_NOW": "🟢", "TRIM_NOW": "✂️", "WAIT": "⏳", "INVALID": "⚠️"}


def stance(pnl_pct: float | None) -> str:
    if pnl_pct is None:
        return "ENTER"
    if pnl_pct > HARVEST_PNL:
        return "HARVEST"
    return "ACCUMULATE"


def _in_zone(price: float, z: Zone) -> bool:
    return z.low <= price <= z.high


def classify(
    tech: Technicals, zones: tuple[Zone, ...], stance_val: str
) -> tuple[str, str, Zone | None]:
    price = tech.price
    if stance_val == "HARVEST":
        if price < tech.ma50:
            return "INVALID", _BADGE["INVALID"], None
        for z in zones:
            if _in_zone(price, z) or (z.label.startswith("Z1") and tech.rsi >= 70):
                return "TRIM_NOW", _BADGE["TRIM_NOW"], z
        return "WAIT", _BADGE["WAIT"], None
    # ENTER / ACCUMULATE → buy ladder
    deepest = zones[-1]
    if price <= deepest.high:
        active = next((z for z in zones if _in_zone(price, z)), deepest)
        return "BUY_NOW", _BADGE["BUY_NOW"], active
    for z in zones:
        if _in_zone(price, z):
            return "BUY_NOW", _BADGE["BUY_NOW"], z
    return "WAIT", _BADGE["WAIT"], None
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_playbook_position.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
git add corvin_jarvis/playbook/position.py tests/test_playbook_position.py
git commit -m "feat(playbook): stance + zone/status classification"
```

---

## Task 5: 빌더 (builder.py)

**Files:**
- Create: `corvin_jarvis/playbook/builder.py`
- Test: `tests/test_playbook_builder.py`

빌더는 `portfolio.json`에서 보유·평단을 읽고, 주입된 `fetch_prices` 콜러블로 종가 리스트를 받아 `Playbook`을 조립한다. 실제 fetch(yfinance/pykrx)는 기본 구현이되 테스트에서는 mock 주입.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_playbook_builder.py
"""Tests for corvin_jarvis.playbook.builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corvin_jarvis.playbook import builder


@pytest.mark.unit
def test_load_holdings_maps_symbol_to_pnl(tmp_path: Path) -> None:
    f = tmp_path / "portfolio.json"
    f.write_text(json.dumps({"holdings": [
        {"symbol": "MSFT", "shares": 6, "pnlPct": 20.2},
        {"symbol": "012450", "shares": 4, "pnlPct": -7.3},
    ]}))
    h = builder.load_holdings(f)
    assert h["MSFT"]["pnl_pct"] == 20.2
    assert h["012450"]["pnl_pct"] == -7.3


@pytest.mark.unit
def test_build_playbooks_watched_is_enter() -> None:
    universe = [{"symbol": "BWXT", "market": "US", "name": "BWX"}]
    # downward-then-flat closes → price below MAs
    closes = [200.0] * 30 + [190.0] * 30
    pbs = builder.build_playbooks(
        universe, holdings={}, fetch_prices=lambda s, m: closes
    )
    assert len(pbs) == 1
    assert pbs[0].stance == "ENTER"
    assert pbs[0].zones[0].kind == "buy"


@pytest.mark.unit
def test_build_playbooks_held_profit_is_harvest() -> None:
    universe = [{"symbol": "MSFT", "market": "US", "name": "Microsoft"}]
    closes = list(range(100, 200))  # rising → in profit, overbought
    pbs = builder.build_playbooks(
        universe, holdings={"MSFT": {"pnl_pct": 20.2}},
        fetch_prices=lambda s, m: closes,
    )
    assert pbs[0].stance == "HARVEST"
    assert pbs[0].pnl_pct == 20.2
    assert pbs[0].zones[0].kind == "trim"


@pytest.mark.unit
def test_build_playbooks_skips_insufficient_history() -> None:
    universe = [{"symbol": "X", "market": "US", "name": "X"}]
    pbs = builder.build_playbooks(
        universe, holdings={}, fetch_prices=lambda s, m: [1.0, 2.0]
    )
    assert pbs == []


@pytest.mark.unit
def test_build_playbooks_skips_on_fetch_error() -> None:
    universe = [{"symbol": "X", "market": "US", "name": "X"}]

    def boom(s: str, m: str) -> list[float]:
        raise RuntimeError("network")

    pbs = builder.build_playbooks(universe, holdings={}, fetch_prices=boom)
    assert pbs == []
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_playbook_builder.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 최소 구현**

```python
# corvin_jarvis/playbook/builder.py
"""플레이북 조립 — portfolio + 지표 fetch → Playbook 리스트."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from corvin_jarvis.playbook import ladder, position, technicals
from corvin_jarvis.playbook.models import Playbook, Technicals

log = logging.getLogger(__name__)

MIN_BARS = 50
BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_FILE = BASE_DIR.parent / "portfolio.json"

PriceFetcher = Callable[[str, str], list[float]]


def load_holdings(path: Path = PORTFOLIO_FILE) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for h in data.get("holdings", []):
        sym = h.get("symbol")
        if sym:
            out[str(sym)] = {"pnl_pct": h.get("pnlPct"), "shares": h.get("shares")}
    return out


def fetch_closes(symbol: str, market: str) -> list[float]:
    if market == "KR":
        from datetime import datetime, timedelta

        from pykrx import stock

        end = datetime.now()
        start = end - timedelta(days=400)
        df = stock.get_market_ohlcv(
            start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), symbol
        )
        return [float(x) for x in df["종가"].tolist()]
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period="1y")
    return [float(x) for x in hist["Close"].tolist()]


def _technicals(symbol: str, market: str, closes: list[float]) -> Technicals | None:
    if len(closes) < MIN_BARS:
        return None
    ma20 = technicals.sma(closes, 20)
    ma50 = technicals.sma(closes, 50)
    rsi = technicals.rsi(closes)
    hi = technicals.recent_high(closes)
    if None in (ma20, ma50, rsi, hi):
        return None
    return Technicals(symbol=symbol, market=market, price=closes[-1],
                      ma20=ma20, ma50=ma50, rsi=rsi, hi_52w=hi)


def build_playbooks(
    universe: list[dict[str, Any]],
    holdings: dict[str, dict[str, Any]],
    fetch_prices: PriceFetcher = fetch_closes,
) -> list[Playbook]:
    out: list[Playbook] = []
    for entry in universe:
        sym = entry["symbol"]
        market = entry.get("market", "US")
        try:
            closes = fetch_prices(sym, market)
        except Exception as e:  # noqa: BLE001 - graceful per-symbol skip
            log.warning("playbook fetch 실패 %s: %s", sym, e)
            continue
        tech = _technicals(sym, market, closes)
        if tech is None:
            log.warning("playbook 데이터 부족 %s", sym)
            continue
        held = holdings.get(sym)
        pnl = held.get("pnl_pct") if held else None
        st = position.stance(pnl)
        zones = (ladder.build_trim_ladder(tech) if st == "HARVEST"
                 else ladder.build_buy_ladder(tech))
        status, badge, active = position.classify(tech, zones, st)
        out.append(Playbook(symbol=sym, name=entry.get("name", sym), stance=st,
                            tech=tech, zones=zones, status=status, badge=badge,
                            pnl_pct=pnl, active_zone=active))
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_playbook_builder.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add corvin_jarvis/playbook/builder.py tests/test_playbook_builder.py
git commit -m "feat(playbook): builder assembles playbooks from portfolio + fetched prices"
```

---

## Task 6: 텍스트 렌더러 (render_text.py)

**Files:**
- Create: `corvin_jarvis/playbook/render_text.py`
- Test: `tests/test_playbook_render_text.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_playbook_render_text.py
"""Tests for corvin_jarvis.playbook.render_text."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import render_text
from corvin_jarvis.playbook.models import Playbook, Technicals, Zone


def _pb(symbol, status, badge, stance, pnl=None) -> Playbook:
    tech = Technicals(symbol=symbol, market="US", price=100.0, ma20=105.0,
                      ma50=110.0, rsi=50.0, hi_52w=130.0)
    z = Zone("buy", "Z2 핵심지지", 40, 99.0, 101.0, "MA50")
    return Playbook(symbol=symbol, name=symbol, stance=stance, tech=tech,
                    zones=(z, z, z), status=status, badge=badge,
                    pnl_pct=pnl, active_zone=z)


@pytest.mark.unit
def test_push_lists_triggers_first() -> None:
    pbs = [
        _pb("AAA", "WAIT", "⏳", "ENTER"),
        _pb("BWXT", "BUY_NOW", "🟢", "ENTER"),
    ]
    out = render_text.render_push(pbs, date_label="6/2")
    assert "오늘 액션" in out
    assert "BWXT" in out
    # trigger appears before the wait-only summary section
    assert out.index("BWXT") < out.index("관찰")


@pytest.mark.unit
def test_push_counts_watch_and_triggers() -> None:
    pbs = [_pb("AAA", "WAIT", "⏳", "ENTER") for _ in range(5)]
    pbs.append(_pb("BBB", "BUY_NOW", "🟢", "ENTER"))
    out = render_text.render_push(pbs, date_label="6/2")
    assert "트리거1" in out or "트리거 1" in out


@pytest.mark.unit
def test_push_shows_holdings_line() -> None:
    pbs = [_pb("MSFT", "TRIM_NOW", "✂️", "HARVEST", pnl=20.2)]
    out = render_text.render_push(pbs, date_label="6/2")
    assert "보유" in out
    assert "MSFT" in out
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_playbook_render_text.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 최소 구현**

```python
# corvin_jarvis/playbook/render_text.py
"""Playbook → 매일 텍스트 푸시 문자열."""
from __future__ import annotations

from corvin_jarvis.playbook.models import Playbook

_TRIGGER = {"BUY_NOW", "TRIM_NOW"}


def _ccy(market: str) -> str:
    return "₩" if market == "KR" else "$"


def _action_line(pb: Playbook) -> str:
    z = pb.active_zone
    ratio = f"{z.ratio}%" if z else ""
    zlabel = z.label if z else ""
    c = _ccy(pb.tech.market)
    return f"{pb.badge} {pb.name} {zlabel} {c}{pb.tech.price:g} → {ratio}"


def render_push(playbooks: list[Playbook], date_label: str) -> str:
    triggers = [p for p in playbooks if p.status in _TRIGGER]
    held = [p for p in playbooks if p.stance in ("ACCUMULATE", "HARVEST")]
    watch = [p for p in playbooks if p.stance == "ENTER"]

    lines = [f"📊 플레이북 · {date_label} 장마감"]
    lines.append(f"🔔 오늘 액션 ({len(triggers)})")
    if triggers:
        lines += [_action_line(p) for p in triggers]
    else:
        lines.append("— 없음 (전 종목 존 대기)")

    if held:
        chips = " ".join(f"{p.symbol}{p.badge}" for p in held)
        lines.append(f"💼 보유{len(held)}: {chips}")

    watch_trig = sum(1 for p in watch if p.status in _TRIGGER)
    lines.append(
        f"👀 관찰{len(watch)}·트리거{watch_trig}·나머지⏳ → 🔗대시보드"
    )
    return "\n".join(lines)
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_playbook_render_text.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add corvin_jarvis/playbook/render_text.py tests/test_playbook_render_text.py
git commit -m "feat(playbook): daily text-push renderer"
```

---

## Task 7: HTML 대시보드 렌더러 (render_html.py)

**Files:**
- Create: `corvin_jarvis/playbook/render_html.py`
- Test: `tests/test_playbook_render_html.py`

존-바는 매수존 최저 ~ 익절존(또는 전고) 최고 범위에 현재가 ●를 비율 위치로 찍는다.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_playbook_render_html.py
"""Tests for corvin_jarvis.playbook.render_html."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import render_html
from corvin_jarvis.playbook.models import Playbook, Technicals, Zone


def _pb(symbol="NVDA", status="WAIT", badge="⏳", stance="HARVEST", pnl=13.8):
    tech = Technicals(symbol=symbol, market="US", price=224.0, ma20=216.0,
                      ma50=200.0, rsi=54.0, hi_52w=235.0)
    z = Zone("trim", "Z2 전고", 33, 232.0, 238.0, "52주고")
    return Playbook(symbol=symbol, name=symbol, stance=stance, tech=tech,
                    zones=(z, z, z), status=status, badge=badge,
                    pnl_pct=pnl, active_zone=None)


@pytest.mark.unit
def test_dashboard_is_html() -> None:
    out = render_html.render_dashboard([_pb()])
    assert out.lstrip().startswith("<!DOCTYPE html>")
    assert "</html>" in out


@pytest.mark.unit
def test_dashboard_contains_symbol_and_rsi() -> None:
    out = render_html.render_dashboard([_pb("NVDA")])
    assert "NVDA" in out
    assert "RSI" in out


@pytest.mark.unit
def test_dashboard_sorts_triggers_first() -> None:
    pbs = [_pb("WAITER", "WAIT", "⏳"), _pb("ACTOR", "BUY_NOW", "🟢")]
    out = render_html.render_dashboard(pbs)
    assert out.index("ACTOR") < out.index("WAITER")


@pytest.mark.unit
def test_dashboard_empty_is_still_valid() -> None:
    out = render_html.render_dashboard([])
    assert "<!DOCTYPE html>" in out
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_playbook_render_html.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 최소 구현**

```python
# corvin_jarvis/playbook/render_html.py
"""Playbook → HTML 대시보드 문자열."""
from __future__ import annotations

import html

from corvin_jarvis.playbook.models import Playbook

_ORDER = {"BUY_NOW": 0, "TRIM_NOW": 1, "INVALID": 2, "WAIT": 3}

_STYLE = """
body{background:#0a0612;color:#eee;font-family:system-ui,sans-serif;margin:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{background:#160f24;border:1px solid #2a1f40;border-radius:10px;padding:12px}
.sym{font-weight:700;font-size:18px}
.bar{height:14px;border-radius:7px;background:linear-gradient(90deg,#1d6b3a,#6b1d2a);position:relative;margin:8px 0}
.mark{position:absolute;top:-3px;width:3px;height:20px;background:#fff}
.muted{color:#9a90b0;font-size:12px}
"""


def _pct_pos(pb: Playbook) -> float:
    lo = min(z.low for z in pb.zones)
    hi = max(z.high for z in pb.zones)
    if hi <= lo:
        return 50.0
    return max(0.0, min(100.0, (pb.tech.price - lo) / (hi - lo) * 100.0))


def _card(pb: Playbook) -> str:
    t = pb.tech
    pnl = f" {pb.pnl_pct:+.1f}%" if pb.pnl_pct is not None else ""
    nxt = pb.active_zone.note if pb.active_zone else "존 대기"
    return (
        f'<div class="card"><div class="sym">{html.escape(pb.name)} '
        f'{pb.badge}</div>'
        f'<div class="muted">{t.symbol} · {t.price:g} · RSI {t.rsi:.0f}{pnl}</div>'
        f'<div class="bar"><div class="mark" style="left:{_pct_pos(pb):.0f}%">'
        f'</div></div>'
        f'<div class="muted">{pb.stance} → {html.escape(nxt)}</div></div>'
    )


def render_dashboard(playbooks: list[Playbook]) -> str:
    ordered = sorted(playbooks, key=lambda p: _ORDER.get(p.status, 9))
    cards = "\n".join(_card(p) for p in ordered)
    return (
        "<!DOCTYPE html>\n<html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Corvin 플레이북</title><style>{_STYLE}</style></head>"
        f"<body><h2>📊 Corvin 플레이북</h2><div class='grid'>{cards}</div>"
        "</body></html>"
    )
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_playbook_render_html.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
git add corvin_jarvis/playbook/render_html.py tests/test_playbook_render_html.py
git commit -m "feat(playbook): HTML dashboard renderer"
```

---

## Task 8: 진입점 (run_playbook.py)

**Files:**
- Create: `corvin_jarvis/playbook/run_playbook.py`
- Test: `tests/test_playbook_run.py`

진입점은 universe·portfolio 로드 → build → 텍스트 푸시 송신 + HTML 파일 저장. 송신/저장은 주입 가능해 테스트에서 mock.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_playbook_run.py
"""Tests for corvin_jarvis.playbook.run_playbook."""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis.playbook import run_playbook
from corvin_jarvis.playbook.models import Playbook, Technicals, Zone


def _pb():
    tech = Technicals(symbol="BWXT", market="US", price=188.0, ma20=204.0,
                      ma50=212.0, rsi=26.0, hi_52w=238.0)
    z = Zone("buy", "Z3 딥밸류", 30, 198.0, 202.0, "딥")
    return Playbook(symbol="BWXT", name="BWX", stance="ENTER", tech=tech,
                    zones=(z, z, z), status="BUY_NOW", badge="🟢",
                    pnl_pct=None, active_zone=z)


@pytest.mark.unit
def test_run_sends_push_and_writes_html(tmp_path: Path) -> None:
    sent: list[str] = []
    html_path = tmp_path / "playbook.html"
    result = run_playbook.run(
        playbooks=[_pb()],
        date_label="6/2",
        sender=lambda body: sent.append(body) or True,
        html_path=html_path,
    )
    assert result is True
    assert sent and "BWXT" in sent[0]
    assert html_path.exists()
    assert "<!DOCTYPE html>" in html_path.read_text()


@pytest.mark.unit
def test_run_returns_false_when_no_playbooks(tmp_path: Path) -> None:
    result = run_playbook.run(
        playbooks=[], date_label="6/2",
        sender=lambda body: True, html_path=tmp_path / "p.html",
    )
    assert result is False
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_playbook_run.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 최소 구현**

```python
# corvin_jarvis/playbook/run_playbook.py
"""플레이북 진입점 — build → 푸시 + HTML 저장."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from corvin_jarvis import channels
from corvin_jarvis.playbook import builder, render_html, render_text
from corvin_jarvis.playbook.models import Playbook

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UNIVERSE_FILE = BASE_DIR / "monitored_universe.json"
HTML_OUT = BASE_DIR.parent / "strategies" / "playbook.html"


def _load_universe(path: Path = UNIVERSE_FILE) -> list[dict]:
    try:
        return json.loads(path.read_text()).get("tickers", [])
    except (OSError, json.JSONDecodeError):
        return []


def run(
    playbooks: list[Playbook],
    date_label: str,
    sender: Callable[[str], bool],
    html_path: Path,
) -> bool:
    if not playbooks:
        log.warning("playbook 없음 — 송신/저장 생략")
        return False
    html_doc = render_html.render_dashboard(playbooks)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_doc)
    push = render_text.render_push(playbooks, date_label)
    sent = sender(push)
    log.info("playbook: n=%d sent=%s html=%s", len(playbooks), sent, html_path)
    return bool(sent)


def main() -> int:
    universe = _load_universe()
    holdings = builder.load_holdings()
    playbooks = builder.build_playbooks(universe, holdings)
    date_label = datetime.now().strftime("%-m/%-d")
    ok = run(playbooks, date_label, channels.send_telegram, HTML_OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_playbook_run.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 전체 스위트 회귀 확인**

Run: `python3 -m pytest tests/ -q`
Expected: 기존 429 + 신규(약 26) 통과, 실패 0

- [ ] **Step 6: 커밋**

```bash
git add corvin_jarvis/playbook/run_playbook.py tests/test_playbook_run.py
git commit -m "feat(playbook): entrypoint wiring (push + dashboard)"
```

---

## Task 9: 크론 래퍼 + 운영 전환

**Files:**
- Create: `corvin_jarvis/run_playbook.sh`
- Reference: `corvin_jarvis/run_leading.sh` (패턴 복사)

- [ ] **Step 1: 래퍼 작성** (`run_leading.sh` 복사 + 모듈명만 교체)

```bash
# corvin_jarvis/run_playbook.sh
#!/usr/bin/env bash
# Corvin 플레이북 — cron entry script
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
LOG="$SCRIPT_DIR/state/playbook.log"
mkdir -p "$SCRIPT_DIR/state"
PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then PY=/opt/homebrew/bin/python3; fi
if [ -f "$SCRIPT_DIR/.env" ]; then set -a; . "$SCRIPT_DIR/.env"; set +a; fi
echo "[$(date '+%F %T')] Playbook 시작" >> "$LOG"
"$PY" -m corvin_jarvis.playbook.run_playbook >> "$LOG" 2>&1 || echo "[ERROR] run_playbook 실패" >> "$LOG"
echo "[$(date '+%F %T')] Playbook 완료" >> "$LOG"
```

- [ ] **Step 2: 실행 권한**

Run: `chmod +x corvin_jarvis/run_playbook.sh`

- [ ] **Step 3: 수동 dry-run 검증** (실송신 방지 — TELEGRAM 토큰 없이)

Run: `cd /Users/thethethe/Claude/quant_investment_system_v2 && TELEGRAM_BOT_TOKEN= python3 -m corvin_jarvis.playbook.run_playbook; echo "exit=$?"; ls -la strategies/playbook.html`
Expected: HTML 파일 생성됨. send_telegram은 토큰 없으면 False 반환(로그 경고) — 정상 graceful.

- [ ] **Step 4: 크론 전환** (기존 `run_leading.sh` → `run_playbook.sh` 교체; 폐하 확인 후 수동)

```bash
crontab -l | sed 's#run_leading.sh#run_playbook.sh#' | crontab -
crontab -l | grep run_playbook
```

- [ ] **Step 5: 커밋**

```bash
git add corvin_jarvis/run_playbook.sh
git commit -m "feat(playbook): cron wrapper + ops switch from leading brief"
```

---

## Self-Review (작성자 체크 — 완료)

- **스펙 커버리지**: §4 래더=Task3, §4.0 stance=Task4, §5① 텍스트=Task6, §5② HTML=Task7, §6 아키텍처=Task1·5, §7 에러=Task5(skip), §8 테스트=각 Task, §9 운영=Task9. 전 항목 매핑됨.
- **플레이스홀더**: 없음 (모든 스텝 실제 코드/명령).
- **타입 일관성**: `Playbook`/`Zone`/`Technicals` 필드명이 Task1 정의와 Task4~8 사용 일치. `fetch_prices(symbol, market)` 시그니처 builder·테스트 일치. `render_push(playbooks, date_label)`·`render_dashboard(playbooks)`·`run(playbooks,date_label,sender,html_path)` 호출부 일치.
- **알려진 단순화**: BWXT 등 MA50 하단 종목은 레벨기반 매수존에서 BUY_NOW(딥밸류)로 분류 — 스펙의 "MA50 회복확인" 뉘앙스는 접근3에서 보강.
