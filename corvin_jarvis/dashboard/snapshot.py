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
from corvin_jarvis.dashboard import equity_curve
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


def _load_portfolio() -> dict[str, Any]:
    try:
        return json.loads(PORTFOLIO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"holdings": [], "totals": {}, "cash": {}}


def _positions(pf: dict) -> list[dict]:
    rows = []
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
            out.append({"label": label, "price": q.price, "pct": q.pct_change})
    fx = _safe(lambda: qp.get_fx_quote("USDKRW=X"), None)
    if fx:
        out.append({"label": "USD/KRW", "price": fx.price, "pct": fx.pct_change})
    return out


def _macro_ticker() -> list[dict]:
    out = []
    for code, label in _COMMODITIES:
        q = _safe(lambda c=code: qp.get_commodity_quote(c), None)
        if q:
            out.append({"label": label, "price": q.price, "pct": q.pct_change})
    return out


def _allocation(positions: list[dict]) -> list[dict]:
    total = sum(p["value_krw"] for p in positions if p["value_krw"]) or 1
    return sorted(
        [{"label": _NAMES.get(p["sym"], p["sym"]), "pct": round(p["value_krw"] / total * 100, 1)}
         for p in positions if p["value_krw"]],
        key=lambda x: -x["pct"])


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


def _totals(pf: dict) -> dict:
    t = pf.get("totals", {})
    c = pf.get("cash", {})
    return {"total_assets_krw": t.get("totalAssetsKRW") or t.get("valueKRW"),
            "equity_pnl_krw": t.get("equityPnlKRW") or t.get("pnlKRW"),
            "equity_pnl_pct": t.get("equityPnlPct") or t.get("pnlPct"),
            "deployable_krw": c.get("deployableKRW")}


def build_snapshot() -> dict[str, Any]:
    now = datetime.now(KST)
    pf = _load_portfolio()
    positions = _safe(lambda: _positions(pf), [])
    holdings = _safe(builder.load_holdings, {})
    return {
        "ts": now.isoformat(timespec="seconds"),
        "market_state": "open" if (market_hours.is_kr_open(now) or market_hours.is_us_open(now)) else "closed",
        "fx_usdkrw": _safe(lambda: qp.get_fx_quote("USDKRW=X").price, None),
        "totals": _safe(lambda: _totals(pf), {}),
        "positions": positions,
        "indices": _safe(_indices, []),
        "macro_ticker": _safe(_macro_ticker, []),
        "allocation": _safe(lambda: _allocation(positions), []),
        "signals": _safe(lambda: _build_signals(holdings), []),
        "log": _safe(lambda: _action_log(pf), []),
        "equity_curve": _safe(equity_curve.build_series, []),
    }


def get_snapshot(ttl: float = 30.0, now: float | None = None) -> dict[str, Any]:
    import time
    t = now if now is not None else time.time()
    if _CACHE["data"] is not None and (t - _CACHE["at"]) < ttl:
        return _CACHE["data"]
    _CACHE["data"] = build_snapshot()
    _CACHE["at"] = t
    return _CACHE["data"]
