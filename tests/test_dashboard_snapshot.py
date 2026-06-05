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
    snap = snapshot.build_snapshot()
    for key in ("ts", "market_state", "fx_usdkrw", "totals", "positions",
                "indices", "macro_ticker", "allocation", "signals", "log", "equity_curve"):
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
