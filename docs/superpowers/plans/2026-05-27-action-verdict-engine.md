# Action Verdict Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** 종목 하나를 넣으면 **매수 / 분할매수 / 홀딩 / 비중축소 / 매도 / 관망** 중 하나 + 신뢰도(상/중/하) + 근거를 돌려주는 행동 판정 엔진. 기존 신호(오늘 등락·RS·DCA 가치점수·보유손익·테마생존)를 융합. advisory·모의 only.

**Architecture:** `corvin_jarvis/signals/verdict.py`. `decide(ctx) -> Verdict` 순수 룰 엔진(테스트 핵심) + `for_symbol(symbol, latest) -> Verdict` 컨텍스트 빌더(quote_provider·dca_timing·portfolio·latest.json에서 ctx 조립). 무어샷/미추적 종목은 보수적으로 강등.

**Tech Stack:** Python 3.11+, pytest. dca_timing의 `_composite_score`/지표 함수 재사용, quote_provider 시세.

**Scope:** v1 = 엔진 + 온디맨드 `for_symbol`. digest 자동 통합은 후속(별도).

---

## File Structure
| 파일 | 책임 | 신규/수정 |
|---|---|---|
| `corvin_jarvis/signals/verdict.py` | Verdict + decide() 룰 + for_symbol() 컨텍스트 | Create |
| `tests/test_verdict.py` | decide 룰 + for_symbol(mock) 테스트 | Create |

---

### Task V1: verdict.decide() pure rule engine

**Files:**
- Create: `corvin_jarvis/signals/verdict.py`
- Test: `tests/test_verdict.py`

- [ ] **Step 1: Write failing tests** — `tests/test_verdict.py`:
```python
from corvin_jarvis.signals import verdict


def _ctx(**kw):
    base = dict(symbol="TST", held=False, pnl_pct=None, dca_score=0,
                rs=None, pct_today=None, theme_alive=False, high_vol=False)
    base.update(kw)
    return base


def test_held_stop_loss_is_sell():
    v = verdict.decide(_ctx(held=True, pnl_pct=-9.0))
    assert v.action == "매도" and v.confidence == "상"


def test_held_take_profit_is_trim():
    v = verdict.decide(_ctx(held=True, pnl_pct=26.0))
    assert v.action == "비중축소"


def test_held_default_is_hold():
    v = verdict.decide(_ctx(held=True, pnl_pct=5.0, theme_alive=True))
    assert v.action == "홀딩" and v.confidence == "상"


def test_held_laggard_dead_theme_trims():
    v = verdict.decide(_ctx(held=True, pnl_pct=3.0, rs=-6.0, theme_alive=False))
    assert v.action == "비중축소"


def test_not_held_spike_is_wait():
    v = verdict.decide(_ctx(held=False, pct_today=12.0, dca_score=70, theme_alive=True))
    assert v.action == "관망"   # 추격 금지가 매수보다 우선


def test_not_held_deep_value_leader_is_buy():
    v = verdict.decide(_ctx(held=False, dca_score=80, theme_alive=True, rs=5.0, pct_today=1.0))
    assert v.action == "매수"


def test_not_held_good_value_is_partial_buy():
    v = verdict.decide(_ctx(held=False, dca_score=62, theme_alive=True, pct_today=1.0))
    assert v.action == "분할매수"


def test_not_held_nothing_is_watch():
    v = verdict.decide(_ctx(held=False, dca_score=20, theme_alive=False))
    assert v.action == "관망"


def test_moonshot_caps_confidence():
    v = verdict.decide(_ctx(held=False, dca_score=80, theme_alive=True, rs=5.0, pct_today=1.0, high_vol=True))
    assert v.action == "매수" and v.confidence == "중"   # 무어샷은 상 안 줌


def test_moonshot_spike_threshold_higher():
    # 무어샷은 +12%로는 관망 안 됨(±15% 기준), 정상 매수 로직 적용
    v = verdict.decide(_ctx(held=False, pct_today=12.0, dca_score=62, theme_alive=True, high_vol=True))
    assert v.action == "분할매수"
```

- [ ] **Step 2: Run — expect FAIL** `python3 -m pytest tests/test_verdict.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `corvin_jarvis/signals/verdict.py`:
```python
"""행동 판정 엔진 — 종목별 매수/홀딩/매도 등 판정 (advisory·모의). 신호 융합."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

_STOP_LOSS = -8.0
_TAKE_PROFIT = 25.0
_RS_LAGGARD = -4.0
_RS_LEADER = 4.0
_DCA_DEEP = 75
_DCA_STRONG = 60
_DCA_OK = 50


@dataclass(frozen=True)
class Verdict:
    symbol: str
    action: str       # 매수 | 분할매수 | 홀딩 | 비중축소 | 매도 | 관망
    confidence: str   # 상 | 중 | 하
    rationale: str


def decide(ctx: dict[str, Any]) -> Verdict:
    sym = ctx.get("symbol", "")
    held = bool(ctx.get("held"))
    pnl = ctx.get("pnl_pct")
    dca = int(ctx.get("dca_score") or 0)
    rs = ctx.get("rs")
    today = ctx.get("pct_today")
    alive = bool(ctx.get("theme_alive"))
    hv = bool(ctx.get("high_vol"))
    cap = "중" if hv else "상"   # 무어샷 신뢰도 상한

    def v(action: str, conf: str, why: str) -> Verdict:
        return Verdict(symbol=sym, action=action, confidence=conf, rationale=why)

    if held:
        if pnl is not None and pnl <= _STOP_LOSS:
            return v("매도", "상", f"손절선({_STOP_LOSS}%) 도달 — 리스크 관리 우선")
        if pnl is not None and pnl >= _TAKE_PROFIT:
            return v("비중축소", cap, f"익절선(+{_TAKE_PROFIT}%) 도달 — 일부 차익실현 검토")
        if rs is not None and rs <= _RS_LAGGARD and not alive:
            return v("비중축소", "중", "지수 대비 약세 + 테마 식음 — 비중 점검")
        return v("홀딩", cap if alive else "중",
                 "보유 논리 유효" + (" · 테마 살아있음" if alive else ""))

    # 미보유
    spike = 15.0 if hv else 8.0
    if today is not None and today >= spike:
        return v("관망", "하" if hv else "중", f"오늘 이미 +{today:.0f}% 급등 — 추격 위험, 눌림 대기")
    if dca >= _DCA_DEEP and alive and (rs or 0) > 0:
        return v("매수", cap, f"DCA 가치점수 {dca}(깊은 저평가)+테마 살아있음+주도주")
    if dca >= _DCA_STRONG and alive:
        return v("분할매수", cap if (rs or 0) > 0 else "중",
                 f"DCA 가치점수 {dca}+테마 살아있음 — 분할 진입")
    if dca >= _DCA_OK and (alive or (rs or 0) > 0):
        return v("분할매수", "하" if hv else "중", f"DCA 가치점수 {dca} — 분할 진입 후보")
    if rs is not None and rs >= _RS_LEADER and alive:
        return v("분할매수", "중", "지수 대비 주도주+테마 살아있음(가치점수는 낮음)")
    return v("관망", "하", "뚜렷한 진입 신호 없음 — 관찰")
```

- [ ] **Step 4: Run — expect PASS** `python3 -m pytest tests/test_verdict.py -v` → 10 passed.

- [ ] **Step 5: Commit**
```bash
git add corvin_jarvis/signals/verdict.py tests/test_verdict.py
git commit -m "feat(corvin): action verdict rule engine (매수/홀딩/매도 etc.)"
```

---

### Task V2: verdict.for_symbol() context builder

**Files:**
- Modify: `corvin_jarvis/signals/verdict.py`
- Test: `tests/test_verdict.py` (append)

컨텍스트 조립: 오늘 등락(quote_provider), DCA 가치점수(dca_timing 지표→_composite_score), RS(등락−벤치마크지수), 보유/손익(latest.json portfolio), 테마생존(latest.json universe의 해당 섹터 바스켓 평균), high_vol(monitored_universe 미포함 종목=보수적 True).

- [ ] **Step 1: Append failing test** — `tests/test_verdict.py`:
```python
def test_for_symbol_builds_context_and_decides(monkeypatch):
    from corvin_jarvis.signals import verdict as V
    from corvin_jarvis import quote_provider, dca_timing

    # 미보유, 깊은 저평가 + 반도체 테마 살아있는 latest
    latest = {
        "indices": {"kospi": {"pct_change": 2.0}, "sp500": {"pct_change": 0.5}},
        "portfolio": [],
        "universe": [
            {"symbol": "000660", "market": "KR", "sector": "semiconductor", "pct_change": 6.0},
            {"symbol": "005930", "market": "KR", "sector": "semiconductor", "pct_change": 5.0},
        ],
    }

    class _Q:
        price, pct_change, source, error = 1000.0, 5.0, "stub", None
    monkeypatch.setattr(quote_provider, "get_stock_quote", lambda s: _Q())
    monkeypatch.setattr(dca_timing, "default_fetcher", lambda s, days=252: [100.0] * 60)
    # _composite_score를 깊은 저평가로 강제
    monkeypatch.setattr(V, "_dca_value_score", lambda prices: 80)

    out = V.for_symbol("000660", latest)
    assert out.symbol == "000660"
    assert out.action in ("매수", "분할매수")     # 저평가+테마+주도주


def test_for_symbol_untracked_is_high_vol(monkeypatch):
    from corvin_jarvis.signals import verdict as V
    from corvin_jarvis import quote_provider, dca_timing
    latest = {"indices": {}, "portfolio": [], "universe": []}

    class _Q:
        price, pct_change, source, error = 50.0, 1.0, "stub", None
    monkeypatch.setattr(quote_provider, "get_stock_quote", lambda s: _Q())
    monkeypatch.setattr(dca_timing, "default_fetcher", lambda s, days=252: [10.0] * 60)
    monkeypatch.setattr(V, "_dca_value_score", lambda prices: 20)

    out = V.for_symbol("277810", latest)   # 미추적 미래기술
    assert out.action == "관망"            # 신호 약함
```

- [ ] **Step 2: Run — expect FAIL** `python3 -m pytest tests/test_verdict.py -k for_symbol -v` → AttributeError (for_symbol/_dca_value_score 없음).

- [ ] **Step 3: Append implementation to `verdict.py`:**
```python
_SECTOR_TH = {"semiconductor": 3.0, "shipbuilding": 4.0, "defense": 4.0}
_SECTOR_TH_DEFAULT = 4.0


def _dca_value_score(prices: list[float]) -> int:
    """DCA 가치/과매도 점수 (0~100, 높을수록 저평가). 임계 게이트 없이 raw."""
    from dca_timing import (MIN_HISTORY_DAYS, _composite_score, _drawdown_52w_pct,
                            _ma_distance_pct, _rsi, _zscore)
    if len(prices) < MIN_HISTORY_DAYS:
        return 0
    score, _ = _composite_score(_rsi(prices), _ma_distance_pct(prices, 50),
                                _zscore(prices, 20), _drawdown_52w_pct(prices))
    return int(score)


def _theme_alive(latest: dict[str, Any], sector: str | None) -> bool:
    if not sector:
        return False
    pcts = [u.get("pct_change") for u in latest.get("universe", [])
            if u.get("sector") == sector and u.get("pct_change") is not None]
    if len(pcts) < 2:
        return False
    avg = sum(pcts) / len(pcts)
    return avg >= _SECTOR_TH.get(sector, _SECTOR_TH_DEFAULT)


def _sector_of(latest: dict[str, Any], symbol: str) -> str | None:
    for u in latest.get("universe", []):
        if u.get("symbol") == symbol:
            return u.get("sector")
    return None


def for_symbol(symbol: str, latest: dict[str, Any]) -> Verdict:
    from dca_timing import _is_kr_symbol, default_fetcher
    from quote_provider import get_stock_quote

    q = get_stock_quote(symbol)
    pct_today = q.pct_change
    dca_score = _dca_value_score(default_fetcher(symbol))

    is_kr = _is_kr_symbol(symbol)
    idx_name = "kospi" if is_kr else "sp500"
    idx_pct = (latest.get("indices", {}).get(idx_name, {}) or {}).get("pct_change")
    rs = (pct_today - idx_pct) if (pct_today is not None and idx_pct is not None) else None

    held, pnl = False, None
    for p in latest.get("portfolio", []):
        if p.get("symbol") == symbol:
            held, pnl = True, p.get("pnl_pct")
            break

    sector = _sector_of(latest, symbol)
    tracked = sector is not None
    ctx = {
        "symbol": symbol, "held": held, "pnl_pct": pnl, "dca_score": dca_score,
        "rs": rs, "pct_today": pct_today, "theme_alive": _theme_alive(latest, sector),
        "high_vol": not tracked,   # monitored_universe 미포함 = 무어샷 취급(보수)
    }
    return decide(ctx)
```

- [ ] **Step 4: Run — expect PASS** `python3 -m pytest tests/test_verdict.py -v` → all pass.

- [ ] **Step 5: Full suite** `python3 -m pytest tests/ -q` → no regressions.

- [ ] **Step 6: Commit**
```bash
git add corvin_jarvis/signals/verdict.py tests/test_verdict.py
git commit -m "feat(corvin): verdict context builder for_symbol (fuses dca/rs/holding/theme)"
```

---

### Task V3: manual verification

- [ ] **Step 1:** On-demand 검증 — 보유주(NVDA) + 미래기술(레인보우로보틱스 277810) 등에 대해 `for_symbol` 호출, 합리적 verdict 나오는지 확인:
```bash
python3 -c "
import json, sys; sys.path.insert(0,'corvin_jarvis')
from corvin_jarvis.signals import verdict
latest = json.load(open('corvin_jarvis/state/latest.json'))
for s in ['NVDA','000660','277810']:
    v = verdict.for_symbol(s, latest)
    print(f'{v.symbol}: {v.action} (신뢰도 {v.confidence}) — {v.rationale}')
"
```
Expected: 합리적 판정 출력(데이터 없는 종목은 보수적). 크래시 없음.

- [ ] **Step 2:** coverage `python3 -m pytest tests/test_verdict.py --cov=corvin_jarvis/signals/verdict --cov-report=term-missing -p no:warnings` → ≥80%.

---

## Self-Review
- Design coverage: 6개 액션 라벨 ✅, 신뢰도 상/중/하 ✅, 근거 ✅, 보유/미보유 분기 ✅, 무어샷 강등 ✅, 추격금지 우선 ✅.
- Placeholders: none.
- Type consistency: `decide(ctx:dict)->Verdict`, `for_symbol(symbol,latest)->Verdict`, `_dca_value_score(prices)->int`, `_theme_alive(latest,sector)->bool`. 일관.

## Notes
- advisory·모의 only — 실매매·단정 금지. "매수"는 dca≥75 깊은저평가+테마+주도주만(고확신), 그 외 분할매수/관망으로 보수.
- digest 자동 통합(보유+오늘 alert 종목에 verdict 줄 추가)은 v2 후속.
- dca_timing 사설 함수 import는 기존 cross-module 패턴(compare._classify_severity)과 동일 — 후속 리팩터 대상.
