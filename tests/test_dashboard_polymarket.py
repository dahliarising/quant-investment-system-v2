import json
from pathlib import Path
from corvin_jarvis.dashboard import polymarket

FIX = Path(__file__).parent / "fixtures" / "polymarket_gamma_sample.json"


def test_parse_markets():
    raw = json.loads(FIX.read_text())
    rows = polymarket.parse_markets(raw)
    assert rows[0]["question"] == "Fed cuts rates in June?"
    assert abs(rows[0]["prob"] - 0.72) < 1e-6
    assert rows[0]["volume_usd"] == 1200000.0
    assert rows[0]["url"].endswith("fed-june")


def test_parse_skips_malformed():
    rows = polymarket.parse_markets([{"question": "x"}])  # no prices/volume
    assert rows == []


def test_fetch_trending_handles_network_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("no net")
    monkeypatch.setattr(polymarket, "_http_get", boom)
    assert polymarket.fetch_trending() == []


def test_fetch_trending_sorts_by_volume_desc(monkeypatch):
    raw = [
        {"question": "low", "outcomePrices": "[\"0.5\", \"0.5\"]", "volume": "100", "slug": "a", "closed": False},
        {"question": "high", "outcomePrices": "[\"0.5\", \"0.5\"]", "volume": "9000", "slug": "b", "closed": False},
    ]
    monkeypatch.setattr(polymarket, "_http_get", lambda url: raw)
    rows = polymarket.fetch_trending()
    assert [r["question"] for r in rows] == ["high", "low"]
