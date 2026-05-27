# Phase A — Universe & Sector Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 지수를 견인하는 대표 대형주(韓+美)의 개별 급등락 + 섹터 바스켓 급등락을 감지해 `alerts.json`에 병합하고, 하이브리드 phase 라벨(🟡 잠정/✅ 확정)을 붙인다.

**Architecture:** 신규 `corvin_jarvis/signals/` 패키지에 universe 로더·market phase 판정·universe_monitor(감지)를 두고, `pulse.py`가 monitored universe 시세를 snapshot에 적재한다. `jarvis.py`는 기존 merge 패턴(`merge_*_alerts`)을 그대로 따라 `merge_signal_alerts()`로 alerts.json에 병합한다. notify.py는 변경하지 않는다 — phase를 `metric` 문자열에 내장해 기존 dedup이 자연히 잠정/확정을 분리한다.

**Tech Stack:** Python 3.11+, pytest, 기존 `quote_provider` 어댑터(KIS→pykrx/yfinance fallback), `zoneinfo`.

**Scope:** Phase A만. B1(RS·모멘텀)/B2(거래량·52주신고가)/C(cross-market)는 별도 plan.

**Spec:** `docs/superpowers/specs/2026-05-27-proactive-signal-detection-design.md`

---

## File Structure

| 파일 | 책임 | 신규/수정 |
|---|---|---|
| `corvin_jarvis/signals/__init__.py` | 패키지 마커 | Create |
| `corvin_jarvis/monitored_universe.json` | 감시 종목 + sector 태그 (단일 소스) | Create |
| `corvin_jarvis/signals/universe_loader.py` | monitored_universe.json 로드·검증 | Create |
| `corvin_jarvis/signals/market_phase.py` | (market, now) → "provisional"/"confirmed" | Create |
| `corvin_jarvis/signals/universe_monitor.py` | 종목별 임계 + 섹터 바스켓 alert 생성 | Create |
| `corvin_jarvis/config.json` | stock_pct_change·sector_basket_pct 추가 | Modify |
| `corvin_jarvis/pulse.py` | fetch_universe + snapshot.universe 필드 | Modify |
| `corvin_jarvis/jarvis.py` | merge_signal_alerts() + 파이프라인 호출 | Modify |
| `tests/test_universe_loader.py` | 로더 단위테스트 | Create |
| `tests/test_market_phase.py` | phase 판정 단위테스트 | Create |
| `tests/test_universe_monitor.py` | 감지 단위·통합·regression 테스트 | Create |

**Alert dict 스키마** (기존 `earnings`/`narrative` merge 패턴과 동일, `phase`를 metric에 내장):
```python
{
  "category": "universe" | "sector",
  "metric": "universe_005930_provisional",   # phase가 metric에 포함 → dedup 자연 분리
  "severity": "medium" | "high" | "critical",
  "message": "🟡 삼성전자(005930) 급등 +7.02% (임계 ±5.0%, 잠정)",
  "value": 7.02,
  "threshold": 5.0,
  "phase": "provisional" | "confirmed",
}
```

---

### Task 1: monitored_universe.json + universe_loader

**Files:**
- Create: `corvin_jarvis/monitored_universe.json`
- Create: `corvin_jarvis/signals/__init__.py`
- Create: `corvin_jarvis/signals/universe_loader.py`
- Test: `tests/test_universe_loader.py`

- [ ] **Step 1: Create the data file**

Create `corvin_jarvis/monitored_universe.json`:
```json
{
  "_comment": "지수 견인 대형주 감시 유니버스 — alert 전용. DCA universe.json과 별개.",
  "_updated": "2026-05-27",
  "tickers": [
    {"symbol": "005930", "market": "KR", "sector": "semiconductor", "name": "삼성전자"},
    {"symbol": "000660", "market": "KR", "sector": "semiconductor", "name": "SK하이닉스"},
    {"symbol": "373220", "market": "KR", "sector": "battery", "name": "LG에너지솔루션"},
    {"symbol": "207940", "market": "KR", "sector": "bio", "name": "삼성바이오로직스"},
    {"symbol": "005380", "market": "KR", "sector": "auto", "name": "현대차"},
    {"symbol": "000270", "market": "KR", "sector": "auto", "name": "기아"},
    {"symbol": "035420", "market": "KR", "sector": "internet", "name": "NAVER"},
    {"symbol": "035720", "market": "KR", "sector": "internet", "name": "카카오"},
    {"symbol": "068270", "market": "KR", "sector": "bio", "name": "셀트리온"},
    {"symbol": "005490", "market": "KR", "sector": "steel", "name": "POSCO홀딩스"},
    {"symbol": "329180", "market": "KR", "sector": "shipbuilding", "name": "HD현대중공업"},
    {"symbol": "012450", "market": "KR", "sector": "defense", "name": "한화에어로스페이스"},
    {"symbol": "AAPL", "market": "US", "sector": "bigtech", "name": "Apple"},
    {"symbol": "MSFT", "market": "US", "sector": "bigtech", "name": "Microsoft"},
    {"symbol": "NVDA", "market": "US", "sector": "semiconductor", "name": "NVIDIA"},
    {"symbol": "AMZN", "market": "US", "sector": "bigtech", "name": "Amazon"},
    {"symbol": "GOOGL", "market": "US", "sector": "bigtech", "name": "Alphabet"},
    {"symbol": "META", "market": "US", "sector": "bigtech", "name": "Meta"},
    {"symbol": "AVGO", "market": "US", "sector": "semiconductor", "name": "Broadcom"},
    {"symbol": "TSLA", "market": "US", "sector": "auto", "name": "Tesla"},
    {"symbol": "LLY", "market": "US", "sector": "pharma", "name": "Eli Lilly"},
    {"symbol": "JPM", "market": "US", "sector": "finance", "name": "JPMorgan"}
  ]
}
```

- [ ] **Step 2: Create the package marker**

Create `corvin_jarvis/signals/__init__.py`:
```python
"""Corvin Jarvis — Proactive signal detection (Phase A+)."""
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_universe_loader.py`:
```python
from corvin_jarvis.signals import universe_loader


def test_load_returns_all_tickers():
    tickers = universe_loader.load()
    assert len(tickers) >= 20
    symbols = {t.symbol for t in tickers}
    assert "005930" in symbols
    assert "NVDA" in symbols


def test_ticker_has_market_and_sector():
    tickers = universe_loader.load()
    samsung = next(t for t in tickers if t.symbol == "005930")
    assert samsung.market == "KR"
    assert samsung.sector == "semiconductor"


def test_by_sector_groups_symbols():
    tickers = universe_loader.load()
    groups = universe_loader.by_sector(tickers)
    assert "005930" in {t.symbol for t in groups["semiconductor"]}
    assert "000660" in {t.symbol for t in groups["semiconductor"]}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python3 -m pytest tests/test_universe_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: corvin_jarvis.signals.universe_loader`

- [ ] **Step 5: Write minimal implementation**

Create `corvin_jarvis/signals/universe_loader.py`:
```python
"""monitored_universe.json 로드·검증."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

BASE_DIR: Final = Path(__file__).resolve().parent.parent
UNIVERSE_FILE: Final = BASE_DIR / "monitored_universe.json"


@dataclass(frozen=True)
class MonitoredTicker:
    symbol: str
    market: str   # "KR" | "US"
    sector: str
    name: str


def load(path: Path = UNIVERSE_FILE) -> list[MonitoredTicker]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[MonitoredTicker] = []
    for t in data.get("tickers", []):
        out.append(MonitoredTicker(
            symbol=str(t["symbol"]),
            market=str(t["market"]),
            sector=str(t["sector"]),
            name=str(t.get("name", t["symbol"])),
        ))
    return out


def by_sector(tickers: list[MonitoredTicker]) -> dict[str, list[MonitoredTicker]]:
    groups: dict[str, list[MonitoredTicker]] = defaultdict(list)
    for t in tickers:
        groups[t.sector].append(t)
    return dict(groups)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest tests/test_universe_loader.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add corvin_jarvis/monitored_universe.json corvin_jarvis/signals/__init__.py corvin_jarvis/signals/universe_loader.py tests/test_universe_loader.py
git commit -m "feat(corvin): monitored universe loader (Phase A)"
```

---

### Task 2: market_phase helper

**Files:**
- Create: `corvin_jarvis/signals/market_phase.py`
- Test: `tests/test_market_phase.py`

KR 정규장 09:00–15:30 KST. US 정규장 09:30–16:00 America/New_York. 해당 시장이 *열려 있으면* `provisional`(미완성봉), *닫혀 있으면* `confirmed`(완성봉).

- [ ] **Step 1: Write the failing test**

Create `tests/test_market_phase.py`:
```python
from datetime import datetime
from zoneinfo import ZoneInfo

from corvin_jarvis.signals import market_phase

KST = ZoneInfo("Asia/Seoul")


def test_kr_open_is_provisional():
    now = datetime(2026, 5, 27, 9, 13, tzinfo=KST)  # KR 장중
    assert market_phase.phase_for("KR", now) == "provisional"


def test_kr_after_close_is_confirmed():
    now = datetime(2026, 5, 27, 16, 0, tzinfo=KST)  # KR 마감 후
    assert market_phase.phase_for("KR", now) == "confirmed"


def test_us_closed_during_kr_morning_is_confirmed():
    now = datetime(2026, 5, 27, 9, 13, tzinfo=KST)  # 美장 마감 상태
    assert market_phase.phase_for("US", now) == "confirmed"


def test_unknown_market_defaults_confirmed():
    now = datetime(2026, 5, 27, 9, 13, tzinfo=KST)
    assert market_phase.phase_for("XX", now) == "confirmed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_market_phase.py -v`
Expected: FAIL — `ModuleNotFoundError: corvin_jarvis.signals.market_phase`

- [ ] **Step 3: Write minimal implementation**

Create `corvin_jarvis/signals/market_phase.py`:
```python
"""감시 종목의 phase 판정: 시장이 열려 있으면 provisional(미완성봉), 닫혀 있으면 confirmed."""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
NY = ZoneInfo("America/New_York")

# 정규장 시간 (장중 = provisional)
_KR_OPEN, _KR_CLOSE = time(9, 0), time(15, 30)
_US_OPEN, _US_CLOSE = time(9, 30), time(16, 0)


def _is_session_open(local_dt: datetime, open_t: time, close_t: time) -> bool:
    if local_dt.weekday() >= 5:  # 토(5)/일(6)
        return False
    return open_t <= local_dt.time() <= close_t


def phase_for(market: str, now: datetime | None = None) -> str:
    """market 정규장이 열려 있으면 'provisional', 아니면 'confirmed'."""
    now = now or datetime.now(KST)
    if market == "KR":
        local = now.astimezone(KST)
        return "provisional" if _is_session_open(local, _KR_OPEN, _KR_CLOSE) else "confirmed"
    if market == "US":
        local = now.astimezone(NY)
        return "provisional" if _is_session_open(local, _US_OPEN, _US_CLOSE) else "confirmed"
    return "confirmed"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_market_phase.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/market_phase.py tests/test_market_phase.py
git commit -m "feat(corvin): market phase helper for hybrid timing (Phase A)"
```

---

### Task 3: universe_monitor — per-ticker threshold alerts

**Files:**
- Create: `corvin_jarvis/signals/universe_monitor.py`
- Test: `tests/test_universe_monitor.py`

snapshot의 `universe` 리스트(`{symbol, market, sector, price, pct_change, ...}`)를 읽어 종목별 임계 초과 시 alert dict를 생성. severity는 `compare._classify_severity` 재사용.

- [ ] **Step 1: Write the failing test**

Create `tests/test_universe_monitor.py`:
```python
from corvin_jarvis.signals import universe_monitor


def _snapshot(universe):
    return {"timestamp_kst": "2026-05-27T16:00:00+09:00", "universe": universe}


def test_ticker_above_threshold_creates_alert():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor",
         "price": 320000, "pct_change": 7.02},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}}}
    alerts = universe_monitor.check_tickers(snap, cfg, phase="confirmed")
    assert len(alerts) == 1
    a = alerts[0]
    assert a["category"] == "universe"
    assert a["metric"] == "universe_005930_confirmed"
    assert a["phase"] == "confirmed"
    assert a["value"] == 7.02


def test_ticker_below_threshold_no_alert():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor",
         "price": 320000, "pct_change": 1.2},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}}}
    assert universe_monitor.check_tickers(snap, cfg, phase="confirmed") == []


def test_per_ticker_override_threshold():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor",
         "price": 320000, "pct_change": 4.5},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {"005930": 4.0}}}
    alerts = universe_monitor.check_tickers(snap, cfg, phase="confirmed")
    assert len(alerts) == 1  # 4.5 ≥ override 4.0


def test_provisional_phase_label_in_message():
    snap = _snapshot([
        {"symbol": "000660", "market": "KR", "sector": "semiconductor",
         "price": 2262000, "pct_change": 10.23},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}}}
    alerts = universe_monitor.check_tickers(snap, cfg, phase="provisional")
    assert alerts[0]["metric"] == "universe_000660_provisional"
    assert "🟡" in alerts[0]["message"]
    assert "잠정" in alerts[0]["message"]


def test_missing_pct_change_skipped():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor",
         "price": None, "pct_change": None, "error": "no data"},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}}}
    assert universe_monitor.check_tickers(snap, cfg, phase="confirmed") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_universe_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: corvin_jarvis.signals.universe_monitor`

- [ ] **Step 3: Write minimal implementation**

Create `corvin_jarvis/signals/universe_monitor.py`:
```python
"""감시 universe의 종목별 급등락 + 섹터 바스켓 alert 생성."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from compare import Severity, _classify_severity  # noqa: E402

_PHASE_EMOJI = {"provisional": "🟡", "confirmed": "✅"}
_PHASE_LABEL = {"provisional": "잠정", "confirmed": "확정"}


def _name_tag(entry: dict[str, Any]) -> str:
    name = entry.get("name")
    sym = entry.get("symbol")
    return f"{name}({sym})" if name else str(sym)


def check_tickers(snapshot: dict[str, Any], config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    cfg = config.get("stock_pct_change", {})
    default_th = float(cfg.get("default", 5.0))
    overrides = cfg.get("overrides", {})
    emoji, label = _PHASE_EMOJI.get(phase, ""), _PHASE_LABEL.get(phase, phase)

    alerts: list[dict[str, Any]] = []
    for entry in snapshot.get("universe", []):
        sym = entry.get("symbol")
        pct = entry.get("pct_change")
        if pct is None:
            continue
        th = float(overrides.get(sym, default_th))
        if abs(pct) < th:
            continue
        sev = _classify_severity(pct, th)
        direction = "급등" if pct > 0 else "급락"
        alerts.append({
            "category": "universe",
            "metric": f"universe_{sym}_{phase}",
            "severity": sev.value if isinstance(sev, Severity) else str(sev),
            "message": f"{emoji} {_name_tag(entry)} {direction} {pct:+.2f}% (임계 ±{th}%, {label})",
            "value": pct,
            "threshold": th,
            "phase": phase,
        })
    return alerts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_universe_monitor.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/universe_monitor.py tests/test_universe_monitor.py
git commit -m "feat(corvin): per-ticker universe threshold alerts (Phase A)"
```

---

### Task 4: universe_monitor — sector basket alerts (+ 09:13 regression fixture)

**Files:**
- Modify: `corvin_jarvis/signals/universe_monitor.py`
- Test: `tests/test_universe_monitor.py` (append)

섹터 바스켓 % = 해당 sector 구성 종목 `pct_change` 동일가중 평균. 임계 초과 시 alert. 오늘 놓친 케이스(삼성 +7.02·하이닉스 +10.23 → 반도체 ~+8.6%)를 복원하는 회귀 테스트 포함.

- [ ] **Step 1: Write the failing tests (append to tests/test_universe_monitor.py)**

```python
def test_sector_basket_average_triggers_alert():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor", "price": 320000, "pct_change": 7.02},
        {"symbol": "000660", "market": "KR", "sector": "semiconductor", "price": 2262000, "pct_change": 10.23},
    ])
    cfg = {"sector_basket_pct": {"semiconductor": 3.0, "_default": 4.0}}
    alerts = universe_monitor.check_sectors(snap, cfg, phase="provisional")
    assert len(alerts) == 1
    a = alerts[0]
    assert a["category"] == "sector"
    assert a["metric"] == "sector_semiconductor_provisional"
    assert abs(a["value"] - 8.625) < 0.01   # (7.02+10.23)/2
    assert "반도체" in a["message"] or "semiconductor" in a["message"]


def test_sector_below_threshold_no_alert():
    snap = _snapshot([
        {"symbol": "005380", "market": "KR", "sector": "auto", "price": 1, "pct_change": 1.0},
        {"symbol": "000270", "market": "KR", "sector": "auto", "price": 1, "pct_change": 2.0},
    ])
    cfg = {"sector_basket_pct": {"_default": 4.0}}
    assert universe_monitor.check_sectors(snap, cfg, phase="confirmed") == []


def test_sector_default_threshold_used_when_unlisted():
    snap = _snapshot([
        {"symbol": "207940", "market": "KR", "sector": "bio", "price": 1, "pct_change": 5.0},
        {"symbol": "068270", "market": "KR", "sector": "bio", "price": 1, "pct_change": 5.0},
    ])
    cfg = {"sector_basket_pct": {"_default": 4.0}}  # bio 미지정 → default 4.0, 평균 5.0 ≥ 4.0
    alerts = universe_monitor.check_sectors(snap, cfg, phase="confirmed")
    assert len(alerts) == 1


def test_single_constituent_sector_skipped():
    # 바스켓은 2종목 이상일 때만 의미 (개별은 check_tickers가 잡음)
    snap = _snapshot([
        {"symbol": "JPM", "market": "US", "sector": "finance", "price": 1, "pct_change": 9.0},
    ])
    cfg = {"sector_basket_pct": {"_default": 4.0}}
    assert universe_monitor.check_sectors(snap, cfg, phase="confirmed") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_universe_monitor.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'check_sectors'`

- [ ] **Step 3: Add implementation (append to universe_monitor.py)**

```python
_SECTOR_KR_NAME = {
    "semiconductor": "반도체", "battery": "2차전지", "bio": "바이오",
    "auto": "자동차", "internet": "인터넷", "steel": "철강",
    "shipbuilding": "조선", "defense": "방산", "bigtech": "빅테크",
    "finance": "금융", "pharma": "제약",
}


def check_sectors(snapshot: dict[str, Any], config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    cfg = config.get("sector_basket_pct", {})
    default_th = float(cfg.get("_default", 4.0))
    emoji, label = _PHASE_EMOJI.get(phase, ""), _PHASE_LABEL.get(phase, phase)

    buckets: dict[str, list[float]] = {}
    for entry in snapshot.get("universe", []):
        pct = entry.get("pct_change")
        sector = entry.get("sector")
        if pct is None or not sector:
            continue
        buckets.setdefault(sector, []).append(float(pct))

    alerts: list[dict[str, Any]] = []
    for sector, pcts in buckets.items():
        if len(pcts) < 2:  # 바스켓은 2종목 이상
            continue
        avg = sum(pcts) / len(pcts)
        th = float(cfg.get(sector, default_th))
        if abs(avg) < th:
            continue
        sev = _classify_severity(avg, th)
        direction = "급등" if avg > 0 else "급락"
        kr_name = _SECTOR_KR_NAME.get(sector, sector)
        alerts.append({
            "category": "sector",
            "metric": f"sector_{sector}_{phase}",
            "severity": sev.value if isinstance(sev, Severity) else str(sev),
            "message": (f"{emoji} {kr_name} 섹터 {direction} 평균 {avg:+.2f}% "
                        f"({len(pcts)}종, 임계 ±{th}%, {label})"),
            "value": round(avg, 4),
            "threshold": th,
            "phase": phase,
        })
    return alerts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_universe_monitor.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/signals/universe_monitor.py tests/test_universe_monitor.py
git commit -m "feat(corvin): sector basket alerts — recovers index-diluted signal (Phase A)"
```

---

### Task 5: config.json thresholds

**Files:**
- Modify: `corvin_jarvis/config.json` (alert_thresholds 블록 내)

- [ ] **Step 1: Add thresholds to alert_thresholds**

`corvin_jarvis/config.json`의 `"alert_thresholds"` 객체 안, `"portfolio_position"` 항목 뒤에 추가 (직전 항목 끝에 콤마 추가 주의):
```json
    "stock_pct_change": {
      "default": 5.0,
      "overrides": {"005930": 4.0, "000660": 4.0}
    },
    "sector_basket_pct": {
      "semiconductor": 3.0,
      "shipbuilding": 4.0,
      "defense": 4.0,
      "_default": 4.0
    }
```

- [ ] **Step 2: Verify JSON is valid**

Run: `python3 -c "import json; json.load(open('corvin_jarvis/config.json')); print('JSON OK')"`
Expected: `JSON OK`

- [ ] **Step 3: Commit**

```bash
git add corvin_jarvis/config.json
git commit -m "feat(corvin): add universe + sector basket thresholds (Phase A)"
```

---

### Task 6: pulse.py — fetch monitored universe into snapshot

**Files:**
- Modify: `corvin_jarvis/pulse.py`
- Test: `tests/test_universe_monitor.py` (append integration test)

`Snapshot`에 `universe` 필드 추가, `fetch_universe()` 추가, `run_pulse()`에서 호출. watchlist fetch는 그대로 둔다(surgical).

- [ ] **Step 1: Add `universe` field to Snapshot dataclass**

`corvin_jarvis/pulse.py`의 `@dataclass class Snapshot` 내 `watchlist` 필드 다음 줄에 추가:
```python
    universe: list[dict[str, Any]] = field(default_factory=list)
```

- [ ] **Step 2: Add fetch_universe function**

`pulse.py`의 `fetch_watchlist` 함수 바로 다음에 추가:
```python
def fetch_universe() -> list[dict[str, Any]]:
    """monitored_universe.json 종목 시세 fetch (alert 감지용)."""
    from corvin_jarvis.signals import universe_loader
    out: list[dict[str, Any]] = []
    for t in universe_loader.load():
        q = quote_provider.get_stock_quote(t.symbol)
        out.append({
            "symbol": t.symbol,
            "market": t.market,
            "sector": t.sector,
            "name": t.name,
            "price": q.price,
            "pct_change": q.pct_change,
            "source": q.source,
            "error": q.error,
        })
    return out
```

- [ ] **Step 3: Call it in run_pulse**

`pulse.py`의 `run_pulse()`에서 watchlist fetch 라인 다음에 추가:
```python
    log.info("Fetching monitored universe...")
    snapshot.universe = fetch_universe()
```

- [ ] **Step 4: Write integration test (append to tests/test_universe_monitor.py)**

```python
def test_pulse_snapshot_universe_feeds_detection(monkeypatch):
    """pulse가 채운 universe를 universe_monitor가 소비하는 end-to-end 검증."""
    from corvin_jarvis import pulse
    from corvin_jarvis.signals import universe_loader

    class _Q:
        def __init__(self, price, pct):
            self.price, self.pct_change, self.source, self.error = price, pct, "stub", None

    fake = {"005930": _Q(320000, 7.02), "000660": _Q(2262000, 10.23)}
    monkeypatch.setattr(universe_loader, "load", lambda *a, **k: [
        universe_loader.MonitoredTicker("005930", "KR", "semiconductor", "삼성전자"),
        universe_loader.MonitoredTicker("000660", "KR", "semiconductor", "SK하이닉스"),
    ])
    monkeypatch.setattr(pulse.quote_provider, "get_stock_quote", lambda s: fake[s])

    universe = pulse.fetch_universe()
    snap = {"universe": universe}
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}},
           "sector_basket_pct": {"semiconductor": 3.0, "_default": 4.0}}

    ticker_alerts = universe_monitor.check_tickers(snap, cfg, phase="provisional")
    sector_alerts = universe_monitor.check_sectors(snap, cfg, phase="provisional")
    assert len(ticker_alerts) == 2          # 삼성 +7, 하이닉스 +10
    assert len(sector_alerts) == 1          # 반도체 바스켓
    assert sector_alerts[0]["metric"] == "sector_semiconductor_provisional"
```

- [ ] **Step 5: Run the integration test**

Run: `python3 -m pytest tests/test_universe_monitor.py::test_pulse_snapshot_universe_feeds_detection -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add corvin_jarvis/pulse.py tests/test_universe_monitor.py
git commit -m "feat(corvin): pulse fetches monitored universe into snapshot (Phase A)"
```

---

### Task 7: jarvis.py — merge_signal_alerts into pipeline

**Files:**
- Modify: `corvin_jarvis/jarvis.py`
- Test: `tests/test_universe_monitor.py` (append)

기존 `merge_*_alerts` 패턴을 그대로 따른다. phase는 종목의 market별로 판정해야 하므로, KR 종목과 US 종목을 각각의 phase로 검사한다.

- [ ] **Step 1: Add merge_signal_alerts function**

`corvin_jarvis/jarvis.py`의 `merge_predictive_alerts` 함수 다음에 추가:
```python
def merge_signal_alerts() -> int:
    """monitored universe 종목별 + 섹터 바스켓 alert을 phase 라벨과 함께 merge."""
    from corvin_jarvis.signals import market_phase, universe_monitor

    latest = _load(LATEST_FILE) or {}
    if not latest.get("universe"):
        return 0
    cfg = _load(BASE_DIR / "config.json") or {}
    th = cfg.get("alert_thresholds", {})
    now = datetime.now(KST)

    # market별로 universe를 쪼개 각자의 phase로 검사
    by_market: dict[str, list[dict[str, Any]]] = {}
    for e in latest["universe"]:
        by_market.setdefault(e.get("market", "US"), []).append(e)

    s_alerts: list[dict[str, Any]] = []
    for market, entries in by_market.items():
        phase = market_phase.phase_for(market, now)
        sub = {"universe": entries}
        s_alerts.extend(universe_monitor.check_tickers(sub, th, phase))
        s_alerts.extend(universe_monitor.check_sectors(sub, th, phase))

    if not s_alerts:
        log.info("No universe/sector signal alerts")
        return 0

    if ALERTS_FILE.exists():
        data = _load(ALERTS_FILE) or {}
        data.setdefault("alerts", []).extend(s_alerts)
        data["count"] = len(data["alerts"])
    else:
        data = {"alerts": s_alerts, "count": len(s_alerts),
                "generated_at": datetime.now().isoformat(timespec="seconds")}
    ALERTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    log.info("Signal alerts merged: %d", len(s_alerts))
    return len(s_alerts)
```

- [ ] **Step 2: Call it in run_jarvis**

`corvin_jarvis/jarvis.py`의 `run_jarvis()`에서 `merge_predictive_alerts()` 호출 다음 줄에 추가:
```python
    merge_signal_alerts()
```

- [ ] **Step 3: Write the failing test (append to tests/test_universe_monitor.py)**

```python
def test_merge_signal_alerts_appends_to_alerts_file(tmp_path, monkeypatch):
    from corvin_jarvis import jarvis
    from corvin_jarvis.signals import market_phase

    latest = {"timestamp_kst": "2026-05-27T16:00:00+09:00", "universe": [
        {"symbol": "005930", "market": "KR", "sector": "semiconductor", "name": "삼성전자", "price": 320000, "pct_change": 7.02},
        {"symbol": "000660", "market": "KR", "sector": "semiconductor", "name": "SK하이닉스", "price": 2262000, "pct_change": 10.23},
    ]}
    latest_file = tmp_path / "latest.json"
    alerts_file = tmp_path / "alerts.json"
    latest_file.write_text(__import__("json").dumps(latest))
    monkeypatch.setattr(jarvis, "LATEST_FILE", latest_file)
    monkeypatch.setattr(jarvis, "ALERTS_FILE", alerts_file)
    monkeypatch.setattr(market_phase, "phase_for", lambda m, now=None: "confirmed")

    n = jarvis.merge_signal_alerts()
    assert n == 3   # 삼성 + 하이닉스 + 반도체 바스켓
    data = __import__("json").loads(alerts_file.read_text())
    metrics = {a["metric"] for a in data["alerts"]}
    assert "universe_005930_confirmed" in metrics
    assert "sector_semiconductor_confirmed" in metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_universe_monitor.py::test_merge_signal_alerts_appends_to_alerts_file -v`
Expected: PASS

- [ ] **Step 5: Run full suite (no regressions)**

Run: `python3 -m pytest tests/ -q`
Expected: all pass (기존 245+ 테스트 + 신규)

- [ ] **Step 6: Commit**

```bash
git add corvin_jarvis/jarvis.py tests/test_universe_monitor.py
git commit -m "feat(corvin): wire signal detection into jarvis pipeline (Phase A)"
```

---

### Task 8: End-to-end manual verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full pipeline manually (no notify)**

Run: `python3 corvin_jarvis/jarvis.py 2>&1 | tail -30`
Expected: 정상 종료(exit 0), 로그에 "Signal alerts merged: N" 출력.

- [ ] **Step 2: Inspect alerts.json for universe/sector alerts**

Run: `python3 -c "import json; d=json.load(open('corvin_jarvis/state/alerts.json')); [print(a['metric'], a['severity'], a['message']) for a in d['alerts'] if a['category'] in ('universe','sector')]"`
Expected: 현재 시장 상황에 따라 universe_/sector_ alert이 phase 라벨과 함께 출력 (없으면 임계 미달 — 정상).

- [ ] **Step 3: Confirm coverage ≥ 80% on new modules**

Run: `python3 -m pytest tests/test_universe_loader.py tests/test_market_phase.py tests/test_universe_monitor.py --cov=corvin_jarvis/signals --cov-report=term-missing`
Expected: signals 패키지 coverage ≥ 80%.

- [ ] **Step 4: Final commit (if any verification fixups)**

```bash
git add -A corvin_jarvis/ tests/
git commit -m "test(corvin): Phase A universe/sector detection verification"
```

---

## Self-Review (plan author checklist)

- **Spec coverage**: §3.1 A ✅(Task 3·4) · §3.2 scope ✅(Task 1) · §3.3 hybrid phase ✅(Task 2·universe_monitor 라벨·jarvis market별 phase) · §5.1 universe schema ✅(Task 1) · §5.2 config ✅(Task 5) · §6 sector basket ✅(Task 4) · §7 severity reuse ✅(`_classify_severity`) · §9 testing ✅(09:13 fixture Task 6, 단위 Task 1-4). B1/B2/C·notify dedup 정식 phase 키·R1 volume = 의도적으로 별도 plan (scope).
- **Placeholder scan**: 모든 step에 실제 코드/명령/기대출력 포함. placeholder 없음.
- **Type consistency**: `MonitoredTicker(symbol, market, sector, name)`·alert dict 키(category/metric/severity/message/value/threshold/phase)·`check_tickers`/`check_sectors`/`phase_for`/`merge_signal_alerts` 시그니처가 Task 전반에서 일치.

## Notes / Deferred (out of scope for this plan)
- **notify.py phase-aware dedup 정식화**: 현재는 phase를 `metric`에 내장해 분리. spec §5.3의 `category::metric::phase` 키 리팩터는 후속.
- **hourly cron 재가동(R4)**: 사용자 결정 대기. 이 plan은 `jarvis.py` 수동/cron 양쪽에서 동작.
- **B1(RS·모멘텀)·B2(거래량·52주신고가)·C(cross-market)**: 각각 별도 plan.
