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
