import pytest

from corvin_jarvis.dashboard import snapshot
from corvin_jarvis.signals import ledger as _ledger


@pytest.fixture(autouse=True)
def _isolate_signal_ledger(monkeypatch, tmp_path):
    """build_snapshot의 원장 기록 훅이 실제 signal_ledger.db를 오염시키지 않게 격리."""
    monkeypatch.setattr(_ledger, "DB_PATH", tmp_path / "signal_ledger.db")


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


def test_totals_recomputes_live_not_stored():
    # stored totals claims +10%, but live positions are 90 vs 100 cost -> must report -10%
    pf = {
        "holdings": [{"symbol": "AAA", "shares": 10, "currency": "USD", "avgPriceUSD": 100.0}],
        "totals": {"equityPnlPct": 10.0, "totalAssetsKRW": 999_999_999},
        "cash": {"deployableKRW": 1000},
    }
    positions = [{"sym": "AAA", "value_krw": 10 * 90 * 1500.0}]
    t = snapshot._totals(pf, positions, 1500.0)
    assert round(t["equity_pnl_pct"], 2) == -10.0
    assert t["equity_pnl_krw"] == round(10 * 90 * 1500 - 10 * 100 * 1500)
    assert t["total_assets_krw"] == round(10 * 90 * 1500 + 1000)
    assert t["deployable_krw"] == 1000


def test_totals_zero_cost_is_graceful():
    t = snapshot._totals({"holdings": [], "totals": {}, "cash": {}}, [], 1500.0)
    assert t["equity_pnl_pct"] is None
    assert t["equity_pnl_krw"] is None


def test_build_snapshot_equity_curve_is_live_pnl_pct(monkeypatch):
    monkeypatch.setattr(snapshot.qp, "get_stock_quote", lambda s: _Q(100.0, 1.5))
    monkeypatch.setattr(snapshot.qp, "get_fx_quote", lambda s: _Q(1500.0, 0.0))
    monkeypatch.setattr(snapshot.qp, "get_kr_index_quote", lambda s: _Q(1.0, 0.0))
    monkeypatch.setattr(snapshot.qp, "get_us_index_quote", lambda s: _Q(1.0, 0.0))
    monkeypatch.setattr(snapshot.qp, "get_commodity_quote", lambda s: _Q(1.0, 0.0))
    monkeypatch.setattr(snapshot, "_build_signals", lambda holdings: [])
    monkeypatch.setattr(snapshot, "_polymarket_fetch", lambda: [])
    snap = snapshot.build_snapshot()
    curve = snap["equity_curve"]
    assert curve, "curve must not be empty"
    assert all("date" in p and "pnl_pct" in p for p in curve)
    # the final (today) point reflects the LIVE recompute, matching the hero totals
    assert curve[-1]["pnl_pct"] == snap["totals"]["equity_pnl_pct"]


def _mock_quotes(monkeypatch):
    monkeypatch.setattr(snapshot.qp, "get_stock_quote", lambda s: _Q(100.0, 1.5))
    monkeypatch.setattr(snapshot.qp, "get_kr_index_quote", lambda s: _Q(8600.0, -1.8))
    monkeypatch.setattr(snapshot.qp, "get_us_index_quote", lambda s: _Q(7500.0, 0.4))
    monkeypatch.setattr(snapshot.qp, "get_fx_quote", lambda s: _Q(1530.0, 1.0))
    monkeypatch.setattr(snapshot.qp, "get_commodity_quote", lambda s: _Q(95.0, -2.0))
    monkeypatch.setattr(snapshot, "_build_signals", lambda holdings: [])
    monkeypatch.setattr(snapshot, "_polymarket_fetch", lambda: [])


def test_snapshot_has_signal_scoreboard_section(monkeypatch):
    """signal_scoreboard 섹션 존재 — 실패해도 빈 리스트 (panel 격리).

    원장 DB는 autouse 픽스처(_isolate_signal_ledger)로 tmp 경로 격리 — 실제 코드 경로 통과.
    """
    _mock_quotes(monkeypatch)
    snapshot._CACHE["data"] = None  # 캐시 무효화
    s = snapshot.build_snapshot()
    assert "signal_scoreboard" in s
    assert isinstance(s["signal_scoreboard"], list)


def test_snapshot_records_signals_to_ledger(monkeypatch):
    """build_snapshot이 engine/predictive 신호를 원장에 기록."""
    _mock_quotes(monkeypatch)
    recorded = []
    monkeypatch.setattr(_ledger, "record_batch",
                        lambda engine, sigs, **kw: recorded.append((engine, len(sigs))) or 0)
    snapshot._CACHE["data"] = None
    snapshot.build_snapshot()
    engines = {e for e, _ in recorded}
    assert "signal_engine" in engines
    assert "predictive" in engines


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
