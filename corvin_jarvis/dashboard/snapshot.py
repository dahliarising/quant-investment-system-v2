"""대시보드 스냅샷 빌더 — 모든 패널 데이터 오케스트레이션. 절대 예외를 던지지 않는다."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from corvin_jarvis import quote_provider as qp
from corvin_jarvis import market_hours
from corvin_jarvis import signal_engine
from corvin_jarvis.dashboard import equity_curve
from corvin_jarvis.dashboard import polymarket as _pm
from corvin_jarvis.playbook import builder

KST = ZoneInfo("Asia/Seoul")
log = logging.getLogger("corvin.dashboard.snapshot")
ROOT = Path(__file__).resolve().parent.parent.parent
PORTFOLIO = ROOT / "portfolio.json"
UNIVERSE = ROOT / "corvin_jarvis" / "monitored_universe.json"

_NAMES = {"META": "Meta", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla",
          "GOOGL": "Alphabet", "BWXT": "BWX", "012450": "한화에어로", "207940": "삼성바이오"}
_INDICES = [("KS11", "KOSPI", "kr"), ("KQ11", "KOSDAQ", "kr"),
            ("^GSPC", "S&P 500", "us"), ("^IXIC", "NASDAQ", "us"), ("^VIX", "VIX", "us")]
_COMMODITIES = [("BZ=F", "BRENT"), ("GC=F", "GOLD")]

_CACHE: dict[str, Any] = {"data": None, "at": 0.0}


def _polymarket_fetch() -> list:
    return _pm.fetch_trending()


def _safe(fn, default):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — 패널 격리가 목적
        log.warning("snapshot section failed: %s", e)
        return default


def _num(x: float | None) -> float | None:
    """Coerce to a finite float, else None (guards NaN/inf from quote sources)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _sanitize(obj):
    """JSON strict 호환 — NaN/inf를 None으로 (브라우저 JSON.parse 보호)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _load_portfolio() -> dict[str, Any]:
    try:
        return json.loads(PORTFOLIO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"holdings": [], "totals": {}, "cash": {}}


def _positions(pf: dict, fx: float | None = None) -> list[dict]:
    rows = []
    if fx is None:
        fx = _num(_safe(lambda: qp.get_fx_quote("USDKRW=X").price, None)) or 1500.0
    for h in pf.get("holdings", []):
        sym = h["symbol"]
        q = _safe(lambda s=sym: qp.get_stock_quote(s), None)
        price = _num(q.price) if q else None
        day_pct = _num(q.pct_change) if q else None
        if h["currency"] == "USD":
            avg = h.get("avgPriceUSD") or 0
            pnl_pct = (price / avg - 1) * 100 if price and avg else None
            value_krw = price * h["shares"] * fx if price else h.get("valueKRW")
        else:
            avg = h.get("avgPriceKRW") or 0
            pnl_pct = (price / avg - 1) * 100 if price and avg else None
            value_krw = price * h["shares"] if price else h.get("valueKRW")
        value_krw = _num(value_krw)
        pnl_pct = _num(pnl_pct)
        rows.append({"sym": sym, "name": _NAMES.get(sym, sym), "ccy": h["currency"],
                     "price": price, "day_pct": day_pct,
                     "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
                     "value_krw": round(value_krw) if value_krw is not None else None})
    return rows


def _indices() -> list[dict]:
    out = []
    for code, label, kind in _INDICES:
        q = _safe(lambda c=code, k=kind: (qp.get_kr_index_quote(c) if k == "kr" else qp.get_us_index_quote(c)), None)
        if q:
            out.append({"label": label, "price": _num(q.price), "pct": _num(q.pct_change)})
    fx = _safe(lambda: qp.get_fx_quote("USDKRW=X"), None)
    if fx:
        out.append({"label": "USD/KRW", "price": _num(fx.price), "pct": _num(fx.pct_change)})
    return out


def _macro_ticker() -> list[dict]:
    out = []
    for code, label in _COMMODITIES:
        q = _safe(lambda c=code: qp.get_commodity_quote(c), None)
        if q:
            out.append({"label": label, "price": _num(q.price), "pct": _num(q.pct_change)})
    return out


def _allocation(positions: list[dict]) -> list[dict]:
    total = sum(p["value_krw"] for p in positions if p["value_krw"]) or 1
    return sorted(
        [{"label": _NAMES.get(p["sym"], p["sym"]), "pct": round(p["value_krw"] / total * 100, 1)}
         for p in positions if p["value_krw"]],
        key=lambda x: -x["pct"])


def _held_for_engine(pf: dict, positions: list[dict]) -> list[dict]:
    """엔진/페이퍼 공용 입력: 라이브 가격·손익(positions) + 수량·평단(portfolio)."""
    meta = {h["symbol"]: h for h in pf.get("holdings", [])}
    out = []
    for p in positions:
        m = meta.get(p["sym"], {})
        ccy = p.get("ccy")
        out.append({"symbol": p["sym"],
                    "market": "KR" if ccy == "KRW" else "US",
                    "price": p.get("price"), "pnl_pct": p.get("pnl_pct"),
                    "shares": m.get("shares"),
                    "avg": m.get("avgPriceKRW") if ccy == "KRW" else m.get("avgPriceUSD")})
    return out


def _engine_signals(held: list[dict]) -> list[dict]:
    """STAGE① 시그널 엔진 — 보유 포지션을 긴급도순 시그널로 (명시 손절선 포함)."""
    return [s.to_dict() for s in signal_engine.evaluate(held)]


def _predictive_signals(held: list[dict]) -> list[dict]:
    """STAGE①.5 예측 시그널 — VELOCITY·RS_WEAK·EVENT 세 Pillar."""
    from datetime import date as _date
    from corvin_jarvis import predictive_engine as pe

    closes_by_sym: dict[str, list[float]] = {}
    for h in held:
        sym = h["symbol"]
        closes = _safe(lambda s=sym: qp.get_stock_daily_closes(s, days=25, completed_only=True), [])
        if closes:
            closes_by_sym[sym] = closes

    bench: dict[str, list[float]] = {}
    spy = _safe(lambda: qp.get_stock_daily_closes("SPY", days=25, completed_only=True), [])
    if spy:
        bench["US"] = spy
    try:
        from pykrx import stock as _px
        from datetime import datetime, timedelta
        _end = datetime.now(KST).strftime("%Y%m%d")
        _start = (datetime.now(KST) - timedelta(days=55)).strftime("%Y%m%d")
        df = _px.get_index_ohlcv_by_date(_start, _end, "1028")  # KOSPI composite
        if not df.empty:
            bench["KR"] = [float(c) for c in df["종가"].tail(25).tolist() if c > 0]
    except Exception as _e:
        log.debug("KR bench fetch skipped: %s", _e)

    sigs = pe.evaluate(held, closes_by_sym=closes_by_sym, bench_closes_by_market=bench,
                       as_of=_date.today())
    return [s.to_dict() for s in sigs]


def _record_to_ledger(engine_sigs: list[dict], pred_sigs: list[dict]) -> None:
    """Phase 1 원장 기록 — 실패해도 신호 흐름 무영향 (_safe로 호출)."""
    from corvin_jarvis.signals import ledger
    ledger.record_batch("signal_engine", engine_sigs)
    ledger.record_batch("predictive", pred_sigs)


def _scoreboard() -> list[dict]:
    from corvin_jarvis.signals import calibration
    return calibration.scoreboard()


def _paper(held: list[dict]) -> dict:
    """STAGE② 페이퍼 — 룰 트리거 모의청산 '제안' + 원장 실현손익(모의·실주문 0)."""
    from corvin_jarvis import paper_trader as pt
    proposals = [t.to_dict() for t in pt.propose_exits(held)]
    ledger = pt.PaperLedger.load()
    avg = {h["symbol"]: h["avg"] for h in held if h.get("avg")}
    return {"proposals": proposals,
            "realized": pt.realized_pnl(ledger.trades, avg),
            "ledger_count": len(ledger.trades)}


def _build_signals(holdings: dict) -> list[dict]:
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8")).get("tickers", [])
    pbs = builder.build_playbooks(universe, holdings)
    out = []
    for p in pbs:
        if p.status not in ("BUY_NOW", "TRIM_NOW"):
            continue
        az = p.active_zone
        color = "green" if p.status == "BUY_NOW" else "amber"
        out.append({"sym": p.symbol, "zone": az.label if az else p.stance,
                    "stance": p.stance, "color": color})
    return out[:12]


def _action_log(pf: dict) -> list[dict]:
    note = pf.get("updatedBy", "")
    log_rows = [{"ts": pf.get("updatedAt", ""), "text": note}] if note else []
    for h in sorted(pf.get("holdings", []), key=lambda x: x.get("snapshotAt", ""), reverse=True)[:6]:
        log_rows.append({"ts": h.get("snapshotAt", ""),
                         "text": f"{h['symbol']} {h['shares']}주 보유"})
    return log_rows


def _cost_basis_krw(pf: dict, fx: float) -> float:
    """보유 종목 취득원가 합계(₩). USD는 라이브 환율로 환산 — 평가액과 동일 fx라 비율은 순수 가격수익률."""
    cost = 0.0
    for h in pf.get("holdings", []):
        shares = h.get("shares") or 0
        if h.get("currency") == "USD":
            avg = h.get("avgPriceUSD") or 0
            cost += shares * avg * fx
        else:
            avg = h.get("avgPriceKRW") or 0
            cost += shares * avg
    return cost


def _totals(pf: dict, positions: list[dict], fx: float) -> dict:
    """저장값이 아닌 라이브 포지션에서 손익을 재계산한다 (헤더와 포지션표의 정합성 보장)."""
    value = sum(p["value_krw"] for p in positions if p.get("value_krw"))
    cost = _cost_basis_krw(pf, fx)
    pnl = value - cost if cost else None
    pnl_pct = (value / cost - 1) * 100 if cost else None
    cash = pf.get("cash", {}).get("deployableKRW") or 0
    return {"total_assets_krw": round(value + cash) if value else None,
            "equity_pnl_krw": round(pnl) if pnl is not None else None,
            "equity_pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "deployable_krw": cash}


def _equity_curve_live(now: datetime, live_pct: float | None) -> list[dict]:
    """과거 손익률 시계열 + 오늘은 라이브 손익률로 갱신(같은 날짜 중복 제거)."""
    curve = equity_curve.build_series()
    if live_pct is not None:
        today = now.strftime("%Y-%m-%d")
        curve = [p for p in curve if p.get("date") != today]
        curve.append({"date": today, "pnl_pct": round(live_pct, 2)})
    return curve


def build_snapshot() -> dict[str, Any]:
    now = datetime.now(KST)
    pf = _load_portfolio()
    fx = _num(_safe(lambda: qp.get_fx_quote("USDKRW=X").price, None)) or 1500.0
    positions = _safe(lambda: _positions(pf, fx), [])
    totals = _safe(lambda: _totals(pf, positions, fx), {})
    holdings = _safe(builder.load_holdings, {})
    held = _safe(lambda: _held_for_engine(pf, positions), [])
    engine_sigs = _safe(lambda: _engine_signals(held), [])
    pred_sigs = _safe(lambda: _predictive_signals(held), [])
    _safe(lambda: _record_to_ledger(engine_sigs, pred_sigs), None)
    return _sanitize({
        "ts": now.isoformat(timespec="seconds"),
        "market_state": "open" if (market_hours.is_kr_open(now) or market_hours.is_us_open(now)) else "closed",
        "fx_usdkrw": fx,
        "totals": totals,
        "positions": positions,
        "indices": _safe(_indices, []),
        "macro_ticker": _safe(_macro_ticker, []),
        "allocation": _safe(lambda: _allocation(positions), []),
        "signals": _safe(lambda: _build_signals(holdings), []),
        "engine_signals": engine_sigs,
        "predictive_signals": pred_sigs,
        "signal_scoreboard": _safe(_scoreboard, []),
        "paper": _safe(lambda: _paper(held), {}),
        "log": _safe(lambda: _action_log(pf), []),
        "equity_curve": _safe(lambda: _equity_curve_live(now, totals.get("equity_pnl_pct")), []),
        "polymarket": _safe(_polymarket_fetch, []),
    })


def get_snapshot(ttl: float = 30.0, now: float | None = None) -> dict[str, Any]:
    import time
    t = now if now is not None else time.time()
    if _CACHE["data"] is not None and (t - _CACHE["at"]) < ttl:
        return _CACHE["data"]
    _CACHE["data"] = build_snapshot()
    _CACHE["at"] = t
    return _CACHE["data"]
