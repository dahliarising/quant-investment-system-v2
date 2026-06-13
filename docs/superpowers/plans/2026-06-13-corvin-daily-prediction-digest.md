# Corvin 매일 예측 다이제스트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 매일 09:36 KST에 5개 예측 방법론(통계·확률·모멘텀·지정학·벡터 analog) 결과를 한 통의 텔레그램 다이제스트로 전송한다.

**Architecture:** 새 `corvin_jarvis/prediction/` 패키지. `daily_history` SQLite 테이블에 FDR 일봉을 backfill하고, 5개 모듈이 동일한 `PredictionResult` 계약을 반환하면 조립기가 스캔가능 다이제스트로 합쳐 `channels.send_telegram`으로 전송. launchd가 매일 09:36 KST 오케스트레이터 실행.

**Tech Stack:** Python 3.9+, sqlite3(표준), FinanceDataReader, pykrx, numpy, pytest. 외부 LLM/유료 API 0. 실주문 0.

**Spec:** `docs/superpowers/specs/2026-06-13-corvin-daily-prediction-digest-design.md`

---

## File Structure

| File | 책임 |
|------|------|
| `corvin_jarvis/prediction/__init__.py` | 패키지 마커 |
| `corvin_jarvis/prediction/contract.py` | `PredictionResult` dataclass + `insufficient()` 헬퍼 |
| `corvin_jarvis/prediction/backfill.py` | `daily_history` 테이블 + FDR 적재/증분/조회 |
| `corvin_jarvis/prediction/m_velocity.py` | 시스템1 어댑터 (predictive_engine 재사용) |
| `corvin_jarvis/prediction/m_probability.py` | 시스템2 어댑터 (predict 재사용) |
| `corvin_jarvis/prediction/m_momentum.py` | 시스템3 어댑터 (모멘텀 신호) |
| `corvin_jarvis/prediction/m_geopolitical.py` | 시스템4 어댑터 (geo 신호 주입식) |
| `corvin_jarvis/prediction/m_vector.py` | 시스템5 벡터 analog (신규 순수함수) |
| `corvin_jarvis/prediction/digest_assembler.py` | results → 텔레그램 Markdown |
| `corvin_jarvis/prediction/run_prediction_digest.py` | 오케스트레이터 + `--dry-run` |
| `corvin_jarvis/com.corvin.prediction-digest.plist` | launchd 09:36 KST |
| `tests/test_prediction_contract.py` 등 | 모듈별 단위 테스트 |

테스트는 기존 관례대로 최상위 `tests/`에 둔다. `tests/conftest.py`의 autouse 안전망(실제 전송 차단)을 신뢰한다.

---

### Task 1: 패키지 + 출력 계약 (PredictionResult)

**Files:**
- Create: `corvin_jarvis/prediction/__init__.py`
- Create: `corvin_jarvis/prediction/contract.py`
- Test: `tests/test_prediction_contract.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_contract.py
from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def test_result_fields_and_to_dict():
    r = PredictionResult(system="velocity", scope="META", verdict="손절 근접",
                         confidence=65.0, evidence={"days_to_stop": 4}, data_ok=True)
    d = r.to_dict()
    assert d["system"] == "velocity"
    assert d["scope"] == "META"
    assert d["data_ok"] is True
    assert d["evidence"]["days_to_stop"] == 4


def test_insufficient_helper_sets_data_ok_false():
    r = insufficient(system="vector_analog", scope="market", reason="과거 250일 미만")
    assert r.data_ok is False
    assert r.confidence == 0.0
    assert "데이터 부족" in r.verdict
    assert r.evidence["reason"] == "과거 250일 미만"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: corvin_jarvis.prediction`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/__init__.py
"""Corvin 예측 다이제스트 패키지 (Phase 1: 시스템 1~5)."""
```

```python
# corvin_jarvis/prediction/contract.py
"""모든 예측 모듈이 반환하는 단일 출력 계약.

data_ok=False면 다이제스트에서 '⏸ 보류'로 축약 — 가짜 정밀도 방지(Corvin 철칙).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionResult:
    system: str                 # velocity|probability|momentum|geopolitical|vector_analog
    scope: str                  # "market" 또는 종목 심볼
    verdict: str                # 한 줄 결론 (한국어)
    confidence: float           # 0-100
    evidence: dict[str, Any] = field(default_factory=dict)
    data_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def insufficient(system: str, scope: str, reason: str) -> PredictionResult:
    """데이터 부족 시 표준 '보류' 결과."""
    return PredictionResult(system=system, scope=scope,
                            verdict="데이터 부족 · 보류", confidence=0.0,
                            evidence={"reason": reason}, data_ok=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_contract.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/__init__.py corvin_jarvis/prediction/contract.py tests/test_prediction_contract.py
git commit -m "feat(prediction): PredictionResult 출력 계약 + insufficient 게이트"
```

---

### Task 2: daily_history 테이블 + 조회

**Files:**
- Create: `corvin_jarvis/prediction/backfill.py`
- Test: `tests/test_prediction_backfill.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_backfill.py
import sqlite3
from corvin_jarvis.prediction import backfill


def test_init_and_upsert_and_read(tmp_path):
    db = tmp_path / "daily.db"
    backfill.init_db(db)
    rows = [
        {"symbol": "kospi", "date": "2026-06-10", "open": 1, "high": 2, "low": 0.5,
         "close": 1.5, "volume": 100, "source": "fdr"},
        {"symbol": "kospi", "date": "2026-06-11", "open": 1.5, "high": 2.5, "low": 1,
         "close": 2.0, "volume": 120, "source": "fdr"},
    ]
    backfill.upsert_rows(db, rows)
    # 중복 date upsert → 갱신, 행 증가 없음
    backfill.upsert_rows(db, [dict(rows[1], close=9.9)])
    got = backfill.read_daily(db, "kospi", lookback=10)
    assert [r["date"] for r in got] == ["2026-06-10", "2026-06-11"]  # 시간순
    assert got[-1]["close"] == 9.9  # upsert로 갱신됨


def test_last_date_returns_none_when_empty(tmp_path):
    db = tmp_path / "daily.db"
    backfill.init_db(db)
    assert backfill.last_date(db, "kospi") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_backfill.py -v`
Expected: FAIL — `AttributeError: module 'backfill' has no attribute 'init_db'`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/backfill.py
"""daily_history — 예측 모듈용 일봉 저장소 (intraday quote_history와 분리).

소스 규칙: KR 종목/지수 = FinanceDataReader/pykrx, US = FinanceDataReader.
yfinance는 한국 데이터 stale → KR에 절대 사용 금지.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_history (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    source TEXT,
    PRIMARY KEY (symbol, date)
);
"""


def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as c:
        c.execute(_SCHEMA)


def upsert_rows(db_path: Path, rows: list[dict[str, Any]]) -> int:
    sql = ("INSERT INTO daily_history (symbol,date,open,high,low,close,volume,source) "
           "VALUES (:symbol,:date,:open,:high,:low,:close,:volume,:source) "
           "ON CONFLICT(symbol,date) DO UPDATE SET "
           "open=excluded.open,high=excluded.high,low=excluded.low,"
           "close=excluded.close,volume=excluded.volume,source=excluded.source")
    with sqlite3.connect(db_path) as c:
        c.executemany(sql, rows)
        return c.total_changes


def read_daily(db_path: Path, symbol: str, lookback: int = 250) -> list[dict[str, Any]]:
    """최근 lookback개 일봉을 시간순(오름차순)으로."""
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(
            "SELECT * FROM (SELECT * FROM daily_history WHERE symbol=? "
            "ORDER BY date DESC LIMIT ?) ORDER BY date ASC", (symbol, lookback))
        return [dict(r) for r in cur.fetchall()]


def last_date(db_path: Path, symbol: str) -> str | None:
    with sqlite3.connect(db_path) as c:
        row = c.execute("SELECT MAX(date) FROM daily_history WHERE symbol=?",
                        (symbol,)).fetchone()
        return row[0] if row and row[0] else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_backfill.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/backfill.py tests/test_prediction_backfill.py
git commit -m "feat(prediction): daily_history 테이블 + upsert/read_daily/last_date"
```

---

### Task 3: backfill 적재 (FDR 어댑터, 주입식)

**Files:**
- Modify: `corvin_jarvis/prediction/backfill.py`
- Test: `tests/test_prediction_backfill_fetch.py`

fetcher를 주입(의존성 역전)해서 네트워크 없이 테스트한다. 실제 FDR 호출은 기본 fetcher에만.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_backfill_fetch.py
from corvin_jarvis.prediction import backfill


def test_incremental_update_only_fetches_after_last_date(tmp_path):
    db = tmp_path / "daily.db"
    backfill.init_db(db)
    backfill.upsert_rows(db, [{"symbol": "kospi", "date": "2026-06-10", "open": 1,
        "high": 1, "low": 1, "close": 1, "volume": 0, "source": "seed"}])

    calls = {}
    def fake_fetch(symbol, market, start):
        calls[symbol] = start
        return [{"symbol": symbol, "date": "2026-06-11", "open": 2, "high": 2,
                 "low": 2, "close": 2, "volume": 0, "source": "fake"}]

    n = backfill.incremental_update(db, [("kospi", "KR")], fetcher=fake_fetch)
    assert calls["kospi"] == "2026-06-11"      # last_date + 1일부터
    assert n >= 1
    got = backfill.read_daily(db, "kospi", lookback=10)
    assert [r["date"] for r in got] == ["2026-06-10", "2026-06-11"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_backfill_fetch.py -v`
Expected: FAIL — `AttributeError: ... no attribute 'incremental_update'`

- [ ] **Step 3: Write minimal implementation**

backfill.py 끝에 추가:

```python
from datetime import datetime, timedelta
from typing import Callable

Fetcher = Callable[[str, str, str], list[dict[str, Any]]]


def _default_fetch(symbol: str, market: str, start: str) -> list[dict[str, Any]]:
    """FinanceDataReader 일봉 → daily_history row. KR/US 동일 API."""
    import FinanceDataReader as fdr
    df = fdr.DataReader(symbol, start)
    out: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        out.append({"symbol": symbol, "date": idx.strftime("%Y-%m-%d"),
                    "open": float(row.get("Open", 0)), "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)), "close": float(row.get("Close", 0)),
                    "volume": int(row.get("Volume", 0) or 0), "source": "fdr"})
    return out


def incremental_update(db_path: Path, symbols: list[tuple[str, str]],
                       fetcher: Fetcher | None = None) -> int:
    """symbols = [(symbol, market)]. last_date 다음날부터 fetch 후 upsert."""
    fetch = fetcher or _default_fetch
    total = 0
    for symbol, market in symbols:
        last = last_date(db_path, symbol)
        if last:
            start = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = "2016-01-01"  # 초기 backfill ~10년
        try:
            rows = fetch(symbol, market, start)
        except Exception:
            continue  # 네트워크 실패 → 기존 캐시로 진행
        if rows:
            total += upsert_rows(db_path, rows)
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_backfill_fetch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/backfill.py tests/test_prediction_backfill_fetch.py
git commit -m "feat(prediction): incremental_update (주입식 fetcher, KR/US FDR)"
```

---

### Task 4: 시스템5 벡터 analog — feature 행렬 + 유사도

**Files:**
- Create: `corvin_jarvis/prediction/m_vector.py`
- Test: `tests/test_prediction_vector.py`

신규 핵심. 순수함수 3개를 먼저 TDD로 만든다(어댑터는 Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_vector.py
import numpy as np
from corvin_jarvis.prediction import m_vector


def test_zscore_normalizes_columns():
    m = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    z = m_vector.zscore_columns(m)
    assert abs(z[:, 0].mean()) < 1e-9
    assert abs(z[:, 0].std() - 1.0) < 1e-9


def test_top_k_analogs_finds_most_similar_rows():
    matrix = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [0.0, 1.0]])
    today = np.array([1.0, 0.0])
    idx, sims = m_vector.top_k_analogs(today, matrix, k=2)
    assert list(idx) == [0, 1]          # 가장 닮은 두 행
    assert sims[0] >= sims[1]


def test_forward_distribution_summarizes_returns():
    fwd = [0.02, -0.01, 0.03, 0.00]
    dist = m_vector.forward_distribution(fwd)
    assert abs(dist["mean"] - 0.01) < 1e-9
    assert dist["win_rate"] == 0.5      # >0 비율 (0은 미포함)
    assert dist["n"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_vector.py -v`
Expected: FAIL — `ModuleNotFoundError: ... m_vector`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/m_vector.py
"""시스템5 — 벡터 analog 예측 순수함수.

오늘 시장상태를 벡터로 인코딩 → 코사인 유사 과거 Top-K → forward 분포로 예측.
numpy만 사용 (외부 의존 0).
"""
from __future__ import annotations

import numpy as np


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """열(feature)별 z-score 정규화. std=0 열은 0으로."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std_safe


def top_k_analogs(today: np.ndarray, matrix: np.ndarray, k: int):
    """today 벡터 vs matrix 각 행 코사인 유사도 → 상위 k 인덱스·유사도."""
    eps = 1e-12
    tn = today / (np.linalg.norm(today) + eps)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + eps)
    sims = mn @ tn
    order = np.argsort(-sims)[:k]
    return order, sims[order]


def forward_distribution(forward_returns: list[float]) -> dict:
    """analog 날들의 forward 수익률 요약."""
    arr = np.array(forward_returns, dtype=float)
    wins = int((arr > 0).sum())
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "win_rate": wins / len(arr) if len(arr) else 0.0, "n": len(arr)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_vector.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/m_vector.py tests/test_prediction_vector.py
git commit -m "feat(prediction): 벡터 analog 순수함수 (zscore/cosine/forward 분포)"
```

---

### Task 5: 시스템5 벡터 어댑터 (daily_history → PredictionResult)

**Files:**
- Modify: `corvin_jarvis/prediction/m_vector.py`
- Test: `tests/test_prediction_vector_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_vector_adapter.py
import numpy as np
from corvin_jarvis.prediction import m_vector
from corvin_jarvis.prediction.contract import PredictionResult

FEATURES = ["kospi", "nasdaq", "vix"]


def _series(db_like, sym, vals):
    db_like[sym] = [{"date": f"2026-01-{i+1:02d}", "close": v} for i, v in enumerate(vals)]


def test_predict_returns_insufficient_when_too_few_days():
    closes = {f: [{"date": "2026-01-01", "close": 1.0}] for f in FEATURES}
    r = m_vector.predict(closes, features=FEATURES, min_days=250, k=12, horizon=5)
    assert isinstance(r, PredictionResult)
    assert r.data_ok is False


def test_predict_produces_market_verdict_with_enough_data():
    rng = np.random.default_rng(0)
    closes = {}
    for f in FEATURES:
        prices = (100 + np.cumsum(rng.normal(0, 1, 400))).tolist()
        closes[f] = [{"date": f"d{i}", "close": p} for i, p in enumerate(prices)]
    r = m_vector.predict(closes, features=FEATURES, min_days=250, k=12, horizon=5)
    assert r.system == "vector_analog"
    assert r.scope == "market"
    assert r.data_ok is True
    assert "win_rate" in r.evidence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_vector_adapter.py -v`
Expected: FAIL — `AttributeError: ... no attribute 'predict'`

- [ ] **Step 3: Write minimal implementation**

m_vector.py 끝에 추가:

```python
from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def _returns(closes: list[dict]) -> list[float]:
    out = []
    for prev, cur in zip(closes[:-1], closes[1:]):
        p, c = prev["close"], cur["close"]
        if p and c and p > 0:
            out.append(c / p - 1.0)
        else:
            out.append(0.0)
    return out


def predict(closes_by_feature: dict[str, list[dict]], *, features: list[str],
            min_days: int = 250, k: int = 12, horizon: int = 5,
            min_similarity: float = 0.0) -> PredictionResult:
    """각 feature의 일봉 close → 수익률 행렬 → 오늘 벡터 analog 예측."""
    series = {f: _returns(closes_by_feature.get(f, [])) for f in features}
    n = min((len(s) for s in series.values()), default=0)
    if n < min_days:
        return insufficient("vector_analog", "market",
                            f"공통 과거 {n}일 < 최소 {min_days}일")
    matrix = np.column_stack([np.array(series[f][-n:]) for f in features])
    z = zscore_columns(matrix)
    # 마지막 행 = 오늘, forward horizon 확보 위해 후보는 [0, n-horizon)
    today = z[-1]
    candidates = z[:n - horizon]
    if len(candidates) < k:
        return insufficient("vector_analog", "market", f"analog 후보 {len(candidates)} < k {k}")
    idx, sims = top_k_analogs(today, candidates, k)
    idx = [int(i) for i in idx if sims[list(idx).index(i)] >= min_similarity]
    # forward = analog 날(i) 이후 horizon일 누적수익률 (kospi 기준 = features[0])
    base = np.array(series[features[0]][-n:])
    fwd = []
    for i in idx:
        window = base[i + 1:i + 1 + horizon]
        fwd.append(float(np.prod(1 + window) - 1) if len(window) else 0.0)
    dist = forward_distribution(fwd)
    direction = "상승" if dist["mean"] > 0 else "하락"
    conf = round(min(95.0, 50 + abs(dist["win_rate"] - 0.5) * 90), 1)
    verdict = (f"현재 국면과 닮은 과거 {dist['n']}개 → 다음 {horizon}일 "
               f"평균 {dist['mean']*100:+.1f}%, 승률 {dist['win_rate']*100:.0f}% ({direction} 우위)")
    return PredictionResult("vector_analog", "market", verdict, conf,
                            {**dist, "k": k, "horizon": horizon}, data_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_vector_adapter.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/m_vector.py tests/test_prediction_vector_adapter.py
git commit -m "feat(prediction): 벡터 analog 어댑터 (daily_history→PredictionResult, data_ok 게이트)"
```

---

### Task 6: 시스템1 velocity 어댑터

**Files:**
- Create: `corvin_jarvis/prediction/m_velocity.py`
- Test: `tests/test_prediction_velocity.py`

기존 `predictive_engine.evaluate_velocity(holdings, stops, closes_by_sym, ...)` 재사용. 반환 `PredictiveSignal` → `PredictionResult` 변환.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_velocity.py
from corvin_jarvis.prediction import m_velocity
from corvin_jarvis.prediction.contract import PredictionResult


def test_downtrend_produces_velocity_result():
    holdings = [{"symbol": "AAA", "price": 100.0}]
    stops = {"AAA": 95.0}
    # 명확한 하락 추세 closes
    closes = {"AAA": [120, 116, 112, 108, 104, 100]}
    results = m_velocity.run(holdings, stops, closes, regime_trend="down")
    assert all(isinstance(r, PredictionResult) for r in results)
    assert any(r.system == "velocity" and r.scope == "AAA" for r in results)


def test_no_stop_yields_insufficient_per_symbol():
    holdings = [{"symbol": "BBB", "price": 50.0}]
    results = m_velocity.run(holdings, stops={}, closes_by_sym={"BBB": [50, 50]},
                             regime_trend="down")
    assert results and results[0].data_ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_velocity.py -v`
Expected: FAIL — `ModuleNotFoundError: ... m_velocity`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/m_velocity.py
"""시스템1 어댑터 — predictive_engine.evaluate_velocity 재사용."""
from __future__ import annotations

from typing import Any

from corvin_jarvis import predictive_engine
from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def run(holdings: list[dict[str, Any]], stops: dict[str, float],
        closes_by_sym: dict[str, list[float]], regime_trend: str | None = "down"
        ) -> list[PredictionResult]:
    signals = predictive_engine.evaluate_velocity(
        holdings, stops, closes_by_sym, regime_trend=regime_trend)
    by_sym = {s.symbol for s in signals}
    out: list[PredictionResult] = []
    for s in signals:
        out.append(PredictionResult(
            system="velocity", scope=s.symbol,
            verdict=f"손절선 도달 예상 {s.horizon_days}일 이내" if s.horizon_days else "하락속도 경보",
            confidence=s.confidence, evidence=s.to_dict(), data_ok=True))
    # stop 없는 보유는 '보류'로 가시화
    for pos in holdings:
        sym = str(pos.get("symbol", ""))
        if sym not in by_sym and sym not in stops:
            out.append(insufficient("velocity", sym, "손절선 미설정"))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_velocity.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/m_velocity.py tests/test_prediction_velocity.py
git commit -m "feat(prediction): velocity 어댑터 (engine 재사용 → PredictionResult)"
```

---

### Task 7: 시스템2 probability 어댑터

**Files:**
- Create: `corvin_jarvis/prediction/m_probability.py`
- Test: `tests/test_prediction_probability.py`

기존 `predict.probability_below(current_price, threshold, mu, sigma, horizon_days)` 재사용. mu/sigma는 daily 수익률에서 계산.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_probability.py
import statistics
from corvin_jarvis.prediction import m_probability
from corvin_jarvis.prediction.contract import PredictionResult


def test_probability_result_for_symbol_with_history():
    closes = [100, 101, 99, 102, 98, 103, 97, 104]  # 변동 있음
    r = m_probability.run_symbol("AAA", closes, stop=90.0, horizon_days=5, min_days=5)
    assert isinstance(r, PredictionResult)
    assert r.system == "probability"
    assert r.data_ok is True
    assert 0.0 <= r.evidence["prob_below_stop"] <= 1.0


def test_probability_insufficient_when_short():
    r = m_probability.run_symbol("BBB", [100], stop=90.0, horizon_days=5, min_days=5)
    assert r.data_ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_probability.py -v`
Expected: FAIL — `ModuleNotFoundError: ... m_probability`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/m_probability.py
"""시스템2 어댑터 — predict.probability_below 재사용 (log-normal)."""
from __future__ import annotations

import math
import statistics

from corvin_jarvis import predict
from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def _log_returns(closes: list[float]) -> list[float]:
    out = []
    for p, c in zip(closes[:-1], closes[1:]):
        if p > 0 and c > 0:
            out.append(math.log(c / p))
    return out


def run_symbol(symbol: str, closes: list[float], stop: float,
               horizon_days: int = 5, min_days: int = 20) -> PredictionResult:
    if len(closes) < min_days:
        return insufficient("probability", symbol, f"일봉 {len(closes)} < {min_days}")
    rets = _log_returns(closes)
    if len(rets) < 2:
        return insufficient("probability", symbol, "수익률 표본 부족")
    mu = statistics.mean(rets)
    sigma = statistics.pstdev(rets)
    price = closes[-1]
    prob = predict.probability_below(price, stop, mu, sigma, horizon_days)
    verdict = f"{horizon_days}일 내 손절가({stop:g}) 이탈 확률 {prob*100:.0f}%"
    conf = round(60 + abs(prob - 0.5) * 60, 1)
    return PredictionResult("probability", symbol, verdict, conf,
                            {"prob_below_stop": prob, "mu": mu, "sigma": sigma,
                             "price": price, "horizon_days": horizon_days}, data_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_probability.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/m_probability.py tests/test_prediction_probability.py
git commit -m "feat(prediction): probability 어댑터 (log-normal 손절 이탈 확률)"
```

---

### Task 8: 시스템3 momentum 어댑터

**Files:**
- Create: `corvin_jarvis/prediction/m_momentum.py`
- Test: `tests/test_prediction_momentum.py`

간단한 추세 신호: 단기(예: 20일) vs 장기(예: 120일) 이동평균 교차. 백필 데이터로 의미있음. 자체 순수함수로 구현(스크립트 의존 줄임).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_momentum.py
from corvin_jarvis.prediction import m_momentum
from corvin_jarvis.prediction.contract import PredictionResult


def test_uptrend_signals_bullish():
    closes = list(range(1, 200))  # 꾸준한 상승
    r = m_momentum.run_symbol("AAA", closes, short=20, long=120, min_days=120)
    assert isinstance(r, PredictionResult)
    assert r.evidence["signal"] == "bullish"
    assert r.data_ok is True


def test_short_history_insufficient():
    r = m_momentum.run_symbol("BBB", [1, 2, 3], short=20, long=120, min_days=120)
    assert r.data_ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_momentum.py -v`
Expected: FAIL — `ModuleNotFoundError: ... m_momentum`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/m_momentum.py
"""시스템3 어댑터 — 이동평균 교차 추세 신호."""
from __future__ import annotations

from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def _sma(values: list[float], window: int) -> float:
    return sum(values[-window:]) / window


def run_symbol(symbol: str, closes: list[float], short: int = 20,
               long: int = 120, min_days: int = 120) -> PredictionResult:
    if len(closes) < min_days:
        return insufficient("momentum", symbol, f"일봉 {len(closes)} < {min_days}")
    sma_s = _sma(closes, short)
    sma_l = _sma(closes, long)
    if sma_s > sma_l:
        signal, verdict = "bullish", f"단기 MA({short}) > 장기 MA({long}) → 추세 상방"
    elif sma_s < sma_l:
        signal, verdict = "bearish", f"단기 MA({short}) < 장기 MA({long}) → 추세 하방"
    else:
        signal, verdict = "neutral", "MA 교차 중립"
    gap = (sma_s - sma_l) / sma_l if sma_l else 0.0
    conf = round(min(90.0, 55 + abs(gap) * 300), 1)
    return PredictionResult("momentum", symbol, verdict, conf,
                            {"signal": signal, "sma_short": sma_s, "sma_long": sma_l,
                             "gap_pct": gap * 100}, data_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_momentum.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/m_momentum.py tests/test_prediction_momentum.py
git commit -m "feat(prediction): momentum 어댑터 (MA 교차 추세 신호)"
```

---

### Task 9: 시스템4 geopolitical 어댑터 (주입식)

**Files:**
- Create: `corvin_jarvis/prediction/m_geopolitical.py`
- Test: `tests/test_prediction_geopolitical.py`

MCP 직접 호출은 오케스트레이터가 하고, 어댑터는 risk score dict를 받아 변환(테스트 가능·결합도↓). MCP 미응답 시 보류.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_geopolitical.py
from corvin_jarvis.prediction import m_geopolitical
from corvin_jarvis.prediction.contract import PredictionResult


def test_high_risk_maps_to_caution_verdict():
    payload = {"risk_score": 72, "trend": "rising", "top_event": "중동 긴장"}
    r = m_geopolitical.run(payload)
    assert isinstance(r, PredictionResult)
    assert r.scope == "market"
    assert r.evidence["risk_score"] == 72
    assert "주의" in r.verdict or "경계" in r.verdict


def test_none_payload_insufficient():
    r = m_geopolitical.run(None)
    assert r.data_ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_geopolitical.py -v`
Expected: FAIL — `ModuleNotFoundError: ... m_geopolitical`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/m_geopolitical.py
"""시스템4 어댑터 — Geopolitical MCP risk score → PredictionResult.

MCP 호출은 오케스트레이터 담당. 여기선 dict payload만 변환(주입식, 테스트 가능).
payload 예: {"risk_score": int, "trend": str, "top_event": str}
"""
from __future__ import annotations

from typing import Any

from corvin_jarvis.prediction.contract import PredictionResult, insufficient


def run(payload: dict[str, Any] | None) -> PredictionResult:
    if not payload or "risk_score" not in payload:
        return insufficient("geopolitical", "market", "MCP 응답 없음")
    score = payload["risk_score"]
    trend = payload.get("trend", "?")
    event = payload.get("top_event", "")
    if score >= 65:
        verdict = f"지정학 리스크 높음({score}) — 변동성 경계. 핵심: {event}"
    elif score >= 40:
        verdict = f"지정학 리스크 보통({score}, {trend})"
    else:
        verdict = f"지정학 리스크 낮음({score})"
    conf = round(min(95.0, 50 + abs(score - 50)), 1)
    return PredictionResult("geopolitical", "market", verdict, conf,
                            {"risk_score": score, "trend": trend, "top_event": event},
                            data_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_geopolitical.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/m_geopolitical.py tests/test_prediction_geopolitical.py
git commit -m "feat(prediction): geopolitical 어댑터 (risk score→PredictionResult, 주입식)"
```

---

### Task 10: 다이제스트 조립기

**Files:**
- Create: `corvin_jarvis/prediction/digest_assembler.py`
- Test: `tests/test_prediction_digest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_digest.py
from corvin_jarvis.prediction.contract import PredictionResult, insufficient
from corvin_jarvis.prediction import digest_assembler


def test_assemble_groups_and_marks_holds():
    results = [
        PredictionResult("vector_analog", "market", "다음 5일 평균 +1.8%", 70,
                         {"win_rate": 0.67}, True),
        PredictionResult("geopolitical", "market", "리스크 보통(45)", 60, {}, True),
        PredictionResult("velocity", "META", "손절 4일 이내", 65, {}, True),
        insufficient("probability", "NVDA", "데이터 부족"),
    ]
    text = digest_assembler.assemble(results, date_str="2026-06-15")
    assert "2026-06-15" in text
    assert "장초반 스냅샷" in text          # 시점 라벨 (memory)
    assert "시장 방향" in text              # 섹션 헤더
    assert "META" in text
    assert "⏸" in text                      # 보류 마커
    assert "NVDA" in text


def test_assemble_handles_empty():
    text = digest_assembler.assemble([], date_str="2026-06-15")
    assert "예측 결과 없음" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_digest.py -v`
Expected: FAIL — `ModuleNotFoundError: ... digest_assembler`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/digest_assembler.py
"""PredictionResult 리스트 → 텔레그램 Markdown 다이제스트.

스캔가능 포맷(memory): 구분선·여백·상태아이콘·종목당 한 줄.
헤더에 '장초반 스냅샷' 라벨로 미완성봉 한계 명시(memory: 가격 시점 라벨링).
"""
from __future__ import annotations

from corvin_jarvis.prediction.contract import PredictionResult

_DIV = "━━━━━━━━━━━━"
_MARKET = {"vector_analog", "geopolitical", "momentum"}


def _line(r: PredictionResult) -> str:
    if not r.data_ok:
        return f"⏸ *{r.scope}* — {r.verdict}"
    return f"• *{r.scope}* — {r.verdict} _(신뢰 {r.confidence:.0f})_"


def assemble(results: list[PredictionResult], date_str: str) -> str:
    if not results:
        return f"📅 {date_str} 장초반 스냅샷\n\n예측 결과 없음 (데이터/모듈 점검 필요)"
    market = [r for r in results if r.scope == "market"]
    holdings = [r for r in results if r.scope != "market"]
    parts = [f"📅 *{date_str} 장초반 스냅샷* (09:36 KST · 미완성봉)", _DIV]
    parts.append("📊 *시장 방향*")
    parts += [_line(r) for r in market] or ["• (없음)"]
    parts.append("")
    parts.append(_DIV)
    parts.append("📈 *종목 예측 (보유·감시)*")
    parts += [_line(r) for r in holdings] or ["• (없음)"]
    parts.append(_DIV)
    parts.append("_⚠️ 데이터 기반 advisory · 실매매 판단은 본인 책임_")
    return "\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_digest.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/digest_assembler.py tests/test_prediction_digest.py
git commit -m "feat(prediction): 다이제스트 조립기 (스캔 포맷 + 장초반 라벨)"
```

---

### Task 11: 오케스트레이터 + --dry-run

**Files:**
- Create: `corvin_jarvis/prediction/run_prediction_digest.py`
- Test: `tests/test_prediction_orchestrator.py`

모듈 예외 격리 + 데이터 수집 + 전송. 전송은 `--dry-run`이면 stdout. portfolio/universe read는 주입 가능하게.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_orchestrator.py
from corvin_jarvis.prediction import run_prediction_digest as orch


def test_build_digest_isolates_module_errors(monkeypatch):
    # 한 모듈이 예외를 던져도 전체 다이제스트는 생성된다
    def boom(*a, **k):
        raise RuntimeError("module down")
    monkeypatch.setattr(orch, "_run_velocity", boom)
    text = orch.build_digest(date_str="2026-06-15",
                             holdings=[{"symbol": "META", "price": 100}],
                             universe=[], db_path=None, geo_payload=None,
                             stops={}, closes_by_sym={}, daily_by_feature={})
    assert "2026-06-15" in text     # 생성 성공
    assert isinstance(text, str)


def test_dry_run_does_not_send(monkeypatch, capsys):
    sent = {"called": False}
    monkeypatch.setattr(orch, "_send", lambda body: sent.__setitem__("called", True))
    orch.main(["--dry-run"])
    assert sent["called"] is False   # dry-run은 전송 안 함
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: ... run_prediction_digest`

- [ ] **Step 3: Write minimal implementation**

```python
# corvin_jarvis/prediction/run_prediction_digest.py
"""오케스트레이터 — backfill → 5모듈 → 조립 → 전송. launchd 09:36 KST 호출.

--dry-run: 전송 대신 stdout. 모듈 예외는 격리(한 모듈 실패해도 나머지 진행).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from corvin_jarvis.prediction import (backfill, digest_assembler, m_geopolitical,
                                      m_momentum, m_probability, m_velocity, m_vector)
from corvin_jarvis.prediction.contract import PredictionResult

KST = ZoneInfo("Asia/Seoul")
_DB = Path(__file__).resolve().parent.parent / "state" / "daily_history.db"
_FEATURES = ["kospi", "nasdaq", "vix", "usd_krw", "gold", "copper", "dxy"]


def _safe(label: str, fn, *args, **kwargs) -> list[PredictionResult]:
    try:
        out = fn(*args, **kwargs)
        return out if isinstance(out, list) else [out]
    except Exception as e:  # 모듈 격리
        return [PredictionResult(label, "market", f"모듈 오류·생략 ({e})", 0.0,
                                 {}, data_ok=False)]


def _run_velocity(holdings, stops, closes_by_sym):
    return m_velocity.run(holdings, stops, closes_by_sym)


def build_digest(*, date_str, holdings, universe, db_path, geo_payload,
                 stops, closes_by_sym, daily_by_feature) -> str:
    results: list[PredictionResult] = []
    results += _safe("velocity", _run_velocity, holdings, stops, closes_by_sym)
    for pos in holdings:
        sym = str(pos.get("symbol", ""))
        results += _safe("probability", m_probability.run_symbol, sym,
                         [c for c in closes_by_sym.get(sym, [])], stops.get(sym, 0.0))
    for sym in universe:
        results += _safe("momentum", m_momentum.run_symbol, sym,
                         closes_by_sym.get(sym, []))
    results += _safe("geopolitical", m_geopolitical.run, geo_payload)
    results += _safe("vector_analog", m_vector.predict, daily_by_feature,
                     features=_FEATURES)
    return digest_assembler.assemble(results, date_str)


def _send(body: str) -> None:
    from corvin_jarvis import channels
    channels.send_telegram(body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    date_str = datetime.now(KST).strftime("%Y-%m-%d")
    backfill.init_db(_DB)
    # 실제 데이터 수집은 운영에서 portfolio/universe/MCP로 채운다.
    # (여기서는 골격 — 세부 수집 로직은 운영 진입점에서 주입)
    text = build_digest(date_str=date_str, holdings=[], universe=[], db_path=_DB,
                        geo_payload=None, stops={}, closes_by_sym={},
                        daily_by_feature={})
    if args.dry_run:
        print(text)
    else:
        _send(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_orchestrator.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/run_prediction_digest.py tests/test_prediction_orchestrator.py
git commit -m "feat(prediction): 오케스트레이터 + --dry-run (모듈 예외 격리)"
```

---

### Task 12: 데이터 수집 배선 (portfolio·universe·MCP·daily_history)

**Files:**
- Modify: `corvin_jarvis/prediction/run_prediction_digest.py`
- Test: `tests/test_prediction_wiring.py`

`build_digest`에 넘길 실제 입력을 모으는 `gather_inputs()` 추가. 파일 read는 주입 가능.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_wiring.py
import json
from corvin_jarvis.prediction import run_prediction_digest as orch


def test_gather_inputs_reads_portfolio_and_universe(tmp_path):
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({"holdings": [{"symbol": "META", "price": 100,
                                            "shares": 7}]}))
    uni = tmp_path / "monitored_universe.json"
    uni.write_text(json.dumps({"tickers": [{"symbol": "005930", "market": "KR"}]}))
    inp = orch.gather_inputs(portfolio_path=pf, universe_path=uni,
                             db_path=tmp_path / "daily.db", geo_fetch=lambda: None)
    assert any(h["symbol"] == "META" for h in inp["holdings"])
    assert "005930" in inp["universe"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prediction_wiring.py -v`
Expected: FAIL — `AttributeError: ... no attribute 'gather_inputs'`

- [ ] **Step 3: Write minimal implementation**

run_prediction_digest.py에 추가 + main()에서 사용:

```python
def gather_inputs(*, portfolio_path: Path, universe_path: Path, db_path: Path,
                  geo_fetch) -> dict:
    """portfolio.json + monitored_universe.json + daily_history → build_digest 입력."""
    holdings = []
    if portfolio_path.exists():
        pf = json.loads(portfolio_path.read_text())
        holdings = pf.get("holdings", [])
    universe = []
    if universe_path.exists():
        uni = json.loads(universe_path.read_text())
        universe = [t["symbol"] for t in uni.get("tickers", [])]

    backfill.init_db(db_path)
    syms = [h["symbol"] for h in holdings] + universe
    closes_by_sym = {s: [r["close"] for r in backfill.read_daily(db_path, s, 250)]
                     for s in syms}
    daily_by_feature = {f: backfill.read_daily(db_path, f, 500) for f in _FEATURES}
    stops = {}
    try:
        from corvin_jarvis.signal_engine import load_stops
        stops = load_stops()
    except Exception:
        stops = {}
    geo_payload = None
    try:
        geo_payload = geo_fetch()
    except Exception:
        geo_payload = None
    return {"holdings": holdings, "universe": universe, "stops": stops,
            "closes_by_sym": closes_by_sym, "daily_by_feature": daily_by_feature,
            "geo_payload": geo_payload}
```

main()의 build_digest 호출부를 gather_inputs 사용으로 교체:

```python
    _PF = Path(__file__).resolve().parent.parent.parent / "portfolio.json"
    _UNI = Path(__file__).resolve().parent.parent / "monitored_universe.json"
    inp = gather_inputs(portfolio_path=_PF, universe_path=_UNI, db_path=_DB,
                        geo_fetch=lambda: None)
    text = build_digest(date_str=date_str, holdings=inp["holdings"],
                        universe=inp["universe"], db_path=_DB,
                        geo_payload=inp["geo_payload"], stops=inp["stops"],
                        closes_by_sym=inp["closes_by_sym"],
                        daily_by_feature=inp["daily_by_feature"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prediction_wiring.py tests/test_prediction_orchestrator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/run_prediction_digest.py tests/test_prediction_wiring.py
git commit -m "feat(prediction): gather_inputs 배선 (portfolio·universe·daily_history·stops)"
```

---

### Task 13: 초기 backfill 실행 + 전체 dry-run 검증

**Files:**
- Create: `corvin_jarvis/prediction/seed_backfill.py` (1회용 적재 스크립트)

- [ ] **Step 1: 적재 스크립트 작성**

```python
# corvin_jarvis/prediction/seed_backfill.py
"""1회 실행 — 지수·매크로 + 감시종목 일봉 ~10년 초기 적재."""
from pathlib import Path
import json

from corvin_jarvis.prediction import backfill

_DB = Path(__file__).resolve().parent.parent / "state" / "daily_history.db"
_FEATURES = [("kospi", "KR"), ("kosdaq", "KR"), ("nasdaq", "US"), ("sp500", "US"),
             ("vix", "US"), ("usd_krw", "KR"), ("gold", "US"), ("copper", "US"),
             ("dxy", "US")]

# FDR 심볼 매핑 (내부키 → fdr 심볼)
_FDR_SYMBOL = {"kospi": "KS11", "kosdaq": "KQ11", "nasdaq": "IXIC", "sp500": "US500",
               "vix": "VIX", "usd_krw": "USD/KRW", "gold": "ZG=F" , "copper": "HG=F",
               "dxy": "DX-Y.NYB"}


def _fetch(symbol, market, start):
    import FinanceDataReader as fdr
    fsym = _FDR_SYMBOL.get(symbol, symbol)
    df = fdr.DataReader(fsym, start)
    rows = []
    for idx, row in df.iterrows():
        rows.append({"symbol": symbol, "date": idx.strftime("%Y-%m-%d"),
                     "open": float(row.get("Open", 0) or 0), "high": float(row.get("High", 0) or 0),
                     "low": float(row.get("Low", 0) or 0), "close": float(row.get("Close", 0) or 0),
                     "volume": int(row.get("Volume", 0) or 0), "source": "fdr"})
    return rows


def main():
    backfill.init_db(_DB)
    uni = json.loads((Path(__file__).resolve().parent.parent / "monitored_universe.json").read_text())
    syms = list(_FEATURES) + [(t["symbol"], t.get("market", "KR")) for t in uni.get("tickers", [])]
    n = backfill.incremental_update(_DB, syms, fetcher=_fetch)
    print(f"backfill 완료: {n} rows, db={_DB}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 초기 backfill 실행**

Run: `cd ~/Claude/quant_investment_system_v2 && python3 -m corvin_jarvis.prediction.seed_backfill`
Expected: `backfill 완료: N rows` (N은 수천~수만). 일부 심볼 실패는 허용(continue).

- [ ] **Step 3: 적재 검증**

Run:
```bash
python3 -c "import sqlite3,pathlib;d=pathlib.Path('corvin_jarvis/state/daily_history.db');c=sqlite3.connect(d);print(c.execute('select symbol,count(*),min(date),max(date) from daily_history group by symbol').fetchall())"
```
Expected: kospi/nasdaq 등이 수백~수천 행, 날짜 범위 수년.

- [ ] **Step 4: 전체 dry-run**

Run: `python3 -m corvin_jarvis.prediction.run_prediction_digest --dry-run`
Expected: 스캔가능 다이제스트가 stdout에 출력. 벡터 모듈이 `data_ok=True`(과거 충분) 또는 명확한 보류 사유.

- [ ] **Step 5: Commit**

```bash
git add corvin_jarvis/prediction/seed_backfill.py
git commit -m "feat(prediction): 초기 backfill 시드 스크립트 (FDR 심볼 매핑)"
```

---

### Task 14: launchd 09:36 KST 스케줄러

**Files:**
- Create: `corvin_jarvis/com.corvin.prediction-digest.plist`
- Create: `corvin_jarvis/run_prediction_digest.sh`

`com.corvin.portfolio-watchdog.plist` 패턴 복제.

- [ ] **Step 1: 래퍼 스크립트**

```bash
# corvin_jarvis/run_prediction_digest.sh
#!/bin/sh
cd "$HOME/Claude/quant_investment_system_v2" || exit 1
set -a
[ -f .env ] && . ./.env
set +a
exec python3 -m corvin_jarvis.prediction.run_prediction_digest >> "$HOME/Claude/quant_investment_system_v2/corvin_jarvis/state/prediction_digest.log" 2>&1
```

Run: `chmod +x corvin_jarvis/run_prediction_digest.sh`

- [ ] **Step 2: plist 작성**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.corvin.prediction-digest</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>__HOME__/Claude/quant_investment_system_v2/corvin_jarvis/run_prediction_digest.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key><integer>9</integer>
        <key>Minute</key><integer>36</integer>
    </dict>
    <key>StandardErrorPath</key>
    <string>__HOME__/Claude/quant_investment_system_v2/corvin_jarvis/state/prediction_digest.err</string>
</dict>
</plist>
```

(설치 시 `__HOME__`을 실제 홈으로 치환. launchd는 로컬 시스템 시간=KST 가정 — 폐하 맥 타임존 KST.)

- [ ] **Step 3: 검증 (로드는 폐하 승인 후 수동)**

Run: `plutil -lint corvin_jarvis/com.corvin.prediction-digest.plist`
Expected: `OK`

설치 명령(README에 기록, **자동 실행 금지 — 폐하 승인 후**):
```bash
sed "s|__HOME__|$HOME|g" corvin_jarvis/com.corvin.prediction-digest.plist > ~/Library/LaunchAgents/com.corvin.prediction-digest.plist
launchctl load ~/Library/LaunchAgents/com.corvin.prediction-digest.plist
```

- [ ] **Step 4: Commit**

```bash
git add corvin_jarvis/com.corvin.prediction-digest.plist corvin_jarvis/run_prediction_digest.sh
git commit -m "feat(prediction): launchd 09:36 KST 스케줄러 + 래퍼 (수동 설치)"
```

---

### Task 15: 전체 테스트 + 회귀 확인 + 문서

**Files:**
- Create: `corvin_jarvis/prediction/README.md`

- [ ] **Step 1: 전체 prediction 테스트**

Run: `python3 -m pytest tests/ -k prediction -v`
Expected: 모든 prediction 테스트 PASS

- [ ] **Step 2: 회귀 — 기존 스위트**

Run: `python3 -m pytest tests/test_predictive_engine.py tests/test_predict.py -v`
Expected: 기존 테스트 PASS (재사용한 모듈 무손상 확인)

- [ ] **Step 3: README 작성**

```markdown
# prediction/ — 매일 예측 다이제스트 (Phase 1: 시스템 1~5)

매일 09:36 KST → 텔레그램 달리아봇으로 5개 예측 방법론 다이제스트 전송.

## 모듈
- contract.py — PredictionResult 출력 계약 (data_ok 게이트)
- backfill.py — daily_history (FDR 일봉)
- m_velocity / m_probability / m_momentum / m_geopolitical / m_vector — 5개 예측
- digest_assembler.py — 스캔 포맷 조립
- run_prediction_digest.py — 오케스트레이터 (--dry-run)

## 실행
- 초기 적재: `python3 -m corvin_jarvis.prediction.seed_backfill`
- 수동 미리보기: `python3 -m corvin_jarvis.prediction.run_prediction_digest --dry-run`
- 스케줄 설치(폐하 승인 후): plist를 ~/Library/LaunchAgents에 복사 후 launchctl load

## Phase 2 (예정)
6 ML분류기 · 7 시계열 · 8 센티먼트 · 9 몬테카를로 · 10 앙상블 — 백테스트 검증 통과분만 다이제스트 합류.
```

- [ ] **Step 4: Commit**

```bash
git add corvin_jarvis/prediction/README.md
git commit -m "docs(prediction): Phase 1 README + 실행 가이드"
```

---

## Self-Review

**1. Spec coverage:**
- 데이터 레이어(daily_history+backfill) → Task 2,3,13 ✅
- 출력 계약 → Task 1 ✅
- 5개 모듈(velocity/probability/momentum/geopolitical/vector) → Task 4~9 ✅
- 다이제스트 조립(스캔 포맷+장초반 라벨) → Task 10 ✅
- 오케스트레이터+dry-run+예외격리 → Task 11,12 ✅
- launchd 09:36+텔레그램 라우팅 → Task 14 ✅
- 정직성(data_ok 게이트) → Task 1, 각 모듈 ✅
- 테스트(실제 전송 0) → 각 Task + conftest 안전망, Task 15 회귀 ✅

**2. Placeholder scan:** 모든 step에 실제 코드/명령 포함. Task 13의 데이터 수집은 골격→Task 12에서 gather_inputs로 완성(의도된 순서). FDR 심볼 매핑은 Task 13에 실값.

**3. Type consistency:** `PredictionResult(system, scope, verdict, confidence, evidence, data_ok)` 전 모듈 동일. `insufficient(system, scope, reason)` 일관. `backfill.read_daily/upsert_rows/incremental_update/last_date/init_db` 시그니처 Task 간 일치. `m_vector.predict(closes_by_feature, *, features, ...)` Task 5 정의 = Task 11 호출 일치.

**알려진 한계(운영 검증 필요):** FDR 심볼 매핑(_FDR_SYMBOL)은 실제 가용성에 따라 Task 13 dry-run에서 조정. analog horizon/k 기본값은 Phase 2 auto-tuner로 추후 튜닝.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
