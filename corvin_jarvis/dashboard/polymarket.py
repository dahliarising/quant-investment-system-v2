"""Polymarket Gamma API — 거래량 상위 트렌딩 마켓 (읽기전용)."""
from __future__ import annotations

import http.client
import json
import logging
import urllib.request
from typing import Any

log = logging.getLogger("corvin.dashboard.polymarket")
_BASE = "https://gamma-api.polymarket.com/markets"
_PARAMS = "?active=true&closed=false&order=volume&ascending=false&limit=40"

# Keyword → category mapping (checked against event slug + series ticker)
_CATEGORY_RULES: list[tuple[tuple[str, ...], str]] = [
    (("btc", "bitcoin", "xrp", "eth", "crypto", "coin", "fdv", "token"), "Crypto"),
    (("aapl", "amzn", "nvda", "tsla", "goog", "msft", "stock", "nasdaq", "sp500", "s-p",
      "equity", "ipo", "earnings"), "Stocks"),
    (("fed", "fomc", "rate", "cpi", "gdp", "inflation", "durable-goods", "payroll",
      "macro", "economy", "recession", "treasury"), "Macro"),
    (("election", "president", "senate", "house", "vote", "democrat", "republican",
      "governor", "trump", "biden", "approval", "primary", "poll"), "Politics"),
    (("ai", "gpt", "llm", "openai", "anthropic", "google-ai", "deepseek",
      "artificial-intelligence", "safety-bill"), "AI / Tech"),
    (("soccer", "football", "nba", "nfl", "mlb", "nhl", "f1", "tennis", "golf",
      "world-cup", "fifwc", "wimbledon", "french-open", "us-open", "sports",
      "league", "series", "champion", "draft", "esport", "counter-strike",
      "j2-league", "j1-league", "mrbeast"), "Sports / Entertainment"),
    (("russia", "ukraine", "china", "xi-jinping", "israel", "war", "military",
      "geopolit", "zaporizhia", "taiwan", "nato"), "Geopolitics"),
    (("gpu", "h100", "ornn", "cloud", "tech", "software", "hardware"), "Tech / AI"),
]


def _http_get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Corvin-Dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read())


def _derive_category(m: dict) -> str:
    """Derive a human-readable category from event slug and series ticker."""
    events = m.get("events") or []
    ev = events[0] if events else {}
    ev_slug = (ev.get("slug") or "").lower()
    series_list = ev.get("series") or []
    series_ticker = (series_list[0].get("ticker") or "").lower() if series_list else ""
    q_lower = (m.get("question") or "").lower()
    combined = f"{ev_slug} {series_ticker} {q_lower}"

    for keywords, label in _CATEGORY_RULES:
        if any(kw in combined for kw in keywords):
            return label
    return "Other"


def parse_markets(raw: list[dict]) -> list[dict]:
    out = []
    for m in raw:
        try:
            prices = m.get("outcomePrices")
            if isinstance(prices, str):
                prices = json.loads(prices)
            prob = float(prices[0])
            vol = float(m["volume"])
            cat = _derive_category(m)
            out.append({"question": m["question"], "prob": prob,
                        "volume_usd": vol, "category": cat,
                        "url": "https://polymarket.com/event/" + m.get("slug", "")})
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return out


def fetch_trending(top: int = 8) -> list[dict]:
    try:
        raw = _http_get(_BASE + _PARAMS)
    except (OSError, ValueError, http.client.HTTPException) as e:
        log.warning("polymarket fetch failed: %s", e)
        return []
    rows = parse_markets(raw if isinstance(raw, list) else raw.get("data", []))
    rows.sort(key=lambda r: r["volume_usd"], reverse=True)
    return rows[:top]
