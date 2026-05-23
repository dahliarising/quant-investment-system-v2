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


# Default FX for currency normalization (advisory only — 정밀 변환은 pulse FX 사용)
_DEFAULT_USD_KRW = 1500.0


def compute_concentration(
    holdings: list[dict[str, Any]],
    market_prices: dict[str, float],
    usd_krw: float = _DEFAULT_USD_KRW,
) -> dict[str, Any]:
    """holdings의 비중 지표 dict.

    - positions, total_value (USD-equivalent)
    - top_symbol, top_weight_pct
    - hhi (Herfindahl-Hirschman Index, 0~1)
    - currency_mix_pct: {USD, KRW, ...}
    """
    if not holdings:
        return {
            "positions": 0, "total_value": 0,
            "top_symbol": None, "top_weight_pct": 0.0,
            "hhi": 0.0, "currency_mix_pct": {},
        }

    values_usd: list[tuple[str, float, str]] = []
    for h in holdings:
        sym = h["symbol"]
        shares = float(h["shares"])
        cur = h.get("currency", "USD")
        price = market_prices.get(sym)
        if price is None:
            continue
        local_value = shares * price
        usd_eq = local_value / usd_krw if cur == "KRW" else local_value
        values_usd.append((sym, usd_eq, cur))

    if not values_usd:
        return {
            "positions": len(holdings), "total_value": 0,
            "top_symbol": None, "top_weight_pct": 0.0,
            "hhi": 0.0, "currency_mix_pct": {},
        }

    total = sum(v for _, v, _ in values_usd)
    weights = [(s, v / total, c) for s, v, c in values_usd]
    weights.sort(key=lambda x: x[1], reverse=True)
    top_sym, top_w, _ = weights[0]
    hhi = sum(w * w for _, w, _ in weights)

    currency_mix: dict[str, float] = {}
    for _, w, cur in weights:
        currency_mix[cur] = currency_mix.get(cur, 0.0) + w * 100

    return {
        "positions": len(values_usd),
        "total_value": round(total, 2),
        "top_symbol": top_sym,
        "top_weight_pct": round(top_w * 100, 2),
        "hhi": round(hhi, 4),
        "currency_mix_pct": {k: round(v, 2) for k, v in currency_mix.items()},
    }
