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
