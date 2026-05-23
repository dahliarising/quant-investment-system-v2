"""Corvin Jarvis — What-if Simulator (Tier 2.1)

자연어 가상 거래 입력 → portfolio diff (비중/HHI/currency).

advisory only — 실제 portfolio.json 변경 안 함.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("corvin.whatif")

_TRADE_RE = re.compile(
    r"^\s*(?P<side>\S+)\s+"
    r"(?P<symbol>[A-Za-z0-9.\-_]+)\s+"
    r"(?P<shares>-?\d+(?:\.\d+)?)"
    r"(?:\s*@\s*(?P<price>-?\d+(?:\.\d+)?))?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Trade:
    side: str  # "buy" | "sell"
    symbol: str
    shares: float
    price: float | None  # None = market


def parse_trade(text: str) -> Trade:
    """자연어 거래 → Trade. 형식: "{buy|sell} SYMBOL SHARES [@ PRICE]"."""
    if not text or not text.strip():
        raise ValueError("empty trade text")
    m = _TRADE_RE.match(text)
    if not m:
        raise ValueError(f"unrecognized trade format: {text!r}")
    side = m.group("side").lower()
    if side not in {"buy", "sell"}:
        raise ValueError(f"invalid side {side!r} (expected buy or sell)")
    shares = float(m.group("shares"))
    if shares <= 0:
        raise ValueError(f"shares must be positive: {shares}")
    price_raw = m.group("price")
    price = float(price_raw) if price_raw is not None else None
    return Trade(
        side=side,
        symbol=m.group("symbol").upper(),
        shares=shares,
        price=price,
    )


def apply_trade(
    holdings: list[dict[str, Any]],
    trade: Trade,
    market_prices: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """holdings에 trade 적용 후 새 list 반환 (immutable).

    - buy: 기존 position 있으면 weighted-avg; 없으면 신규 생성
    - sell: shares 감소; full sell이면 제거; over-sell이면 ValueError
    - market 거래 (price=None): market_prices에서 lookup; 없으면 ValueError
    """
    price = trade.price
    if price is None:
        if market_prices is None or trade.symbol not in market_prices:
            raise ValueError(f"market price for {trade.symbol} unavailable")
        price = market_prices[trade.symbol]

    out: list[dict[str, Any]] = []
    matched = False
    for h in holdings:
        if h["symbol"] != trade.symbol:
            out.append(dict(h))  # shallow copy
            continue
        matched = True
        current_shares = float(h["shares"])
        if trade.side == "buy":
            new_shares = current_shares + trade.shares
            cur_avg = float(h.get("avgPriceUSD") or h.get("avgPriceKRW") or 0)
            new_avg = (current_shares * cur_avg + trade.shares * price) / new_shares
            new = dict(h)
            new["shares"] = new_shares
            avg_key = "avgPriceKRW" if h.get("currency") == "KRW" else "avgPriceUSD"
            new[avg_key] = round(new_avg, 4)
            out.append(new)
        else:  # sell
            if trade.shares > current_shares:
                raise ValueError(
                    f"sell {trade.shares} {trade.symbol} exceeds held {current_shares}"
                )
            remaining = current_shares - trade.shares
            if remaining > 0:
                new = dict(h)
                new["shares"] = remaining
                out.append(new)
            # else: drop position

    if not matched:
        if trade.side == "sell":
            raise ValueError(f"{trade.symbol} not held — cannot sell")
        avg_key = "avgPriceUSD"
        out.append({
            "symbol": trade.symbol,
            "shares": trade.shares,
            avg_key: round(price, 4),
            "avgPriceKRW": round(price * 1500, 4),
            "currency": "USD",
        })
    return out
