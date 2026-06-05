from corvin_jarvis.dashboard import snapshot


class _Q:
    def __init__(self, price, pct):
        self.price, self.pct_change, self.source, self.error = price, pct, "test", None


def test_build_snapshot_has_all_sections(monkeypatch):
    monkeypatch.setattr(snapshot.qp, "get_stock_quote", lambda s: _Q(100.0, 1.5))
    monkeypatch.setattr(snapshot.qp, "get_kr_index_quote", lambda s: _Q(8600.0, -1.8))
    monkeypatch.setattr(snapshot.qp, "get_us_index_quote", lambda s: _Q(7500.0, 0.4))
    monkeypatch.setattr(snapshot.qp, "get_fx_quote", lambda s: _Q(1530.0, 1.0))
    monkeypatch.setattr(snapshot.qp, "get_commodity_quote", lambda s: _Q(95.0, -2.0))
    monkeypatch.setattr(snapshot, "_build_signals", lambda holdings: [])
    monkeypatch.setattr(snapshot, "_polymarket_fetch", lambda: [])
    snap = snapshot.build_snapshot()
    for key in ("ts", "market_state", "fx_usdkrw", "totals", "positions",
                "indices", "macro_ticker", "allocation", "signals", "log", "equity_curve",
                "polymarket"):
        assert key in snap
    assert isinstance(snap["positions"], list)


def test_build_snapshot_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(snapshot.qp, "get_stock_quote", boom)
    monkeypatch.setattr(snapshot.qp, "get_kr_index_quote", boom)
    monkeypatch.setattr(snapshot.qp, "get_us_index_quote", boom)
    monkeypatch.setattr(snapshot.qp, "get_fx_quote", boom)
    monkeypatch.setattr(snapshot.qp, "get_commodity_quote", boom)
    monkeypatch.setattr(snapshot, "_build_signals", lambda holdings: [])
    snap = snapshot.build_snapshot()  # must not raise
    # graceful degradation: returns full structure, indices empty (all index quotes failed)
    for key in ("ts", "market_state", "positions", "indices", "equity_curve"):
        assert key in snap
    assert isinstance(snap["positions"], list)
    assert snap["indices"] == []  # every index quote raised -> all skipped


def test_positions_handles_nan_price(monkeypatch):
    class _Qnan:
        price = float("nan"); pct_change = float("nan"); source = "t"; error = None
    monkeypatch.setattr(snapshot.qp, "get_stock_quote", lambda s: _Qnan())
    monkeypatch.setattr(snapshot.qp, "get_fx_quote", lambda s: _Qnan())
    pf = {"holdings": [{"symbol": "BWXT", "shares": 8, "currency": "USD",
                        "avgPriceUSD": 191.38, "valueKRW": 2345461}]}
    rows = snapshot._positions(pf)
    assert len(rows) == 1
    assert rows[0]["price"] is None
    assert rows[0]["pnl_pct"] is None
    assert rows[0]["value_krw"] == 2345461  # graceful fallback to stale value, no crash


def test_get_snapshot_caches(monkeypatch):
    calls = {"n": 0}
    def fake_build():
        calls["n"] += 1
        return {"ts": "x"}
    monkeypatch.setattr(snapshot, "build_snapshot", fake_build)
    snapshot._CACHE["data"] = None
    snapshot._CACHE["at"] = 0.0
    snapshot.get_snapshot(ttl=999, now=1000.0)
    snapshot.get_snapshot(ttl=999, now=1001.0)
    assert calls["n"] == 1
