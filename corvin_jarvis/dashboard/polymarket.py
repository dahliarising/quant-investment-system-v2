"""Polymarket Gamma API — 거래량 상위 트렌딩 마켓 (읽기전용)."""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

log = logging.getLogger("corvin.dashboard.polymarket")
_BASE = "https://gamma-api.polymarket.com/markets"
_PARAMS = "?active=true&closed=false&order=volume&ascending=false&limit=40"


def _http_get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Corvin-Dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read())


def parse_markets(raw: list[dict]) -> list[dict]:
    out = []
    for m in raw:
        try:
            prices = m.get("outcomePrices")
            if isinstance(prices, str):
                prices = json.loads(prices)
            prob = float(prices[0])
            vol = float(m["volume"])
            out.append({"question": m["question"], "prob": prob,
                        "volume_usd": vol,
                        "url": "https://polymarket.com/event/" + m.get("slug", "")})
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return out


def fetch_trending(top: int = 8) -> list[dict]:
    try:
        raw = _http_get(_BASE + _PARAMS)
    except (OSError, ValueError) as e:
        log.warning("polymarket fetch failed: %s", e)
        return []
    rows = parse_markets(raw if isinstance(raw, list) else raw.get("data", []))
    rows.sort(key=lambda r: r["volume_usd"], reverse=True)
    return rows[:top]
