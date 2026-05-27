from corvin_jarvis.signals import universe_monitor


def _snapshot(universe):
    return {"timestamp_kst": "2026-05-27T16:00:00+09:00", "universe": universe}


def test_ticker_above_threshold_creates_alert():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor",
         "price": 320000, "pct_change": 7.02},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}}}
    alerts = universe_monitor.check_tickers(snap, cfg, phase="confirmed")
    assert len(alerts) == 1
    a = alerts[0]
    assert a["category"] == "universe"
    assert a["metric"] == "universe_005930_confirmed"
    assert a["phase"] == "confirmed"
    assert a["value"] == 7.02


def test_ticker_below_threshold_no_alert():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor",
         "price": 320000, "pct_change": 1.2},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}}}
    assert universe_monitor.check_tickers(snap, cfg, phase="confirmed") == []


def test_per_ticker_override_threshold():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor",
         "price": 320000, "pct_change": 4.5},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {"005930": 4.0}}}
    alerts = universe_monitor.check_tickers(snap, cfg, phase="confirmed")
    assert len(alerts) == 1  # 4.5 ≥ override 4.0


def test_provisional_phase_label_in_message():
    snap = _snapshot([
        {"symbol": "000660", "market": "KR", "sector": "semiconductor",
         "price": 2262000, "pct_change": 10.23},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}}}
    alerts = universe_monitor.check_tickers(snap, cfg, phase="provisional")
    assert alerts[0]["metric"] == "universe_000660_provisional"
    assert "🟡" in alerts[0]["message"]
    assert "잠정" in alerts[0]["message"]


def test_missing_pct_change_skipped():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor",
         "price": None, "pct_change": None, "error": "no data"},
    ])
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}}}
    assert universe_monitor.check_tickers(snap, cfg, phase="confirmed") == []


def test_sector_basket_average_triggers_alert():
    snap = _snapshot([
        {"symbol": "005930", "market": "KR", "sector": "semiconductor", "price": 320000, "pct_change": 7.02},
        {"symbol": "000660", "market": "KR", "sector": "semiconductor", "price": 2262000, "pct_change": 10.23},
    ])
    cfg = {"sector_basket_pct": {"semiconductor": 3.0, "_default": 4.0}}
    alerts = universe_monitor.check_sectors(snap, cfg, phase="provisional")
    assert len(alerts) == 1
    a = alerts[0]
    assert a["category"] == "sector"
    assert a["metric"] == "sector_semiconductor_provisional"
    assert abs(a["value"] - 8.625) < 0.01   # (7.02+10.23)/2
    assert "반도체" in a["message"] or "semiconductor" in a["message"]


def test_sector_below_threshold_no_alert():
    snap = _snapshot([
        {"symbol": "005380", "market": "KR", "sector": "auto", "price": 1, "pct_change": 1.0},
        {"symbol": "000270", "market": "KR", "sector": "auto", "price": 1, "pct_change": 2.0},
    ])
    cfg = {"sector_basket_pct": {"_default": 4.0}}
    assert universe_monitor.check_sectors(snap, cfg, phase="confirmed") == []


def test_sector_default_threshold_used_when_unlisted():
    snap = _snapshot([
        {"symbol": "207940", "market": "KR", "sector": "bio", "price": 1, "pct_change": 5.0},
        {"symbol": "068270", "market": "KR", "sector": "bio", "price": 1, "pct_change": 5.0},
    ])
    cfg = {"sector_basket_pct": {"_default": 4.0}}  # bio 미지정 → default 4.0, 평균 5.0 ≥ 4.0
    alerts = universe_monitor.check_sectors(snap, cfg, phase="confirmed")
    assert len(alerts) == 1


def test_single_constituent_sector_skipped():
    # 바스켓은 2종목 이상일 때만 의미 (개별은 check_tickers가 잡음)
    snap = _snapshot([
        {"symbol": "JPM", "market": "US", "sector": "finance", "price": 1, "pct_change": 9.0},
    ])
    cfg = {"sector_basket_pct": {"_default": 4.0}}
    assert universe_monitor.check_sectors(snap, cfg, phase="confirmed") == []


def test_pulse_snapshot_universe_feeds_detection(monkeypatch):
    """pulse가 채운 universe를 universe_monitor가 소비하는 end-to-end 검증."""
    from corvin_jarvis import pulse
    from corvin_jarvis.signals import universe_loader

    class _Q:
        def __init__(self, price, pct):
            self.price, self.pct_change, self.source, self.error = price, pct, "stub", None

    fake = {"005930": _Q(320000, 7.02), "000660": _Q(2262000, 10.23)}
    monkeypatch.setattr(universe_loader, "load", lambda *a, **k: [
        universe_loader.MonitoredTicker("005930", "KR", "semiconductor", "삼성전자"),
        universe_loader.MonitoredTicker("000660", "KR", "semiconductor", "SK하이닉스"),
    ])
    monkeypatch.setattr(pulse.quote_provider, "get_stock_quote", lambda s: fake[s])

    universe = pulse.fetch_universe()
    snap = {"universe": universe}
    cfg = {"stock_pct_change": {"default": 5.0, "overrides": {}},
           "sector_basket_pct": {"semiconductor": 3.0, "_default": 4.0}}

    ticker_alerts = universe_monitor.check_tickers(snap, cfg, phase="provisional")
    sector_alerts = universe_monitor.check_sectors(snap, cfg, phase="provisional")
    assert len(ticker_alerts) == 2          # 삼성 +7, 하이닉스 +10
    assert len(sector_alerts) == 1          # 반도체 바스켓
    assert sector_alerts[0]["metric"] == "sector_semiconductor_provisional"
