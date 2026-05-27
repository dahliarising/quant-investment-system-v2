from corvin_jarvis.signals import verdict


def _ctx(**kw):
    base = dict(symbol="TST", held=False, pnl_pct=None, dca_score=0,
                rs=None, pct_today=None, theme_alive=False, high_vol=False)
    base.update(kw)
    return base


def test_held_stop_loss_is_sell():
    v = verdict.decide(_ctx(held=True, pnl_pct=-9.0))
    assert v.action == "매도" and v.confidence == "상"


def test_held_take_profit_is_trim():
    v = verdict.decide(_ctx(held=True, pnl_pct=26.0))
    assert v.action == "비중축소"


def test_held_default_is_hold():
    v = verdict.decide(_ctx(held=True, pnl_pct=5.0, theme_alive=True))
    assert v.action == "홀딩" and v.confidence == "상"


def test_held_laggard_dead_theme_trims():
    v = verdict.decide(_ctx(held=True, pnl_pct=3.0, rs=-6.0, theme_alive=False))
    assert v.action == "비중축소"


def test_not_held_spike_is_wait():
    v = verdict.decide(_ctx(held=False, pct_today=12.0, dca_score=70, theme_alive=True))
    assert v.action == "관망"   # 추격 금지가 매수보다 우선


def test_not_held_deep_value_leader_is_buy():
    v = verdict.decide(_ctx(held=False, dca_score=80, theme_alive=True, rs=5.0, pct_today=1.0))
    assert v.action == "매수"


def test_not_held_good_value_is_partial_buy():
    v = verdict.decide(_ctx(held=False, dca_score=62, theme_alive=True, pct_today=1.0))
    assert v.action == "분할매수"


def test_not_held_nothing_is_watch():
    v = verdict.decide(_ctx(held=False, dca_score=20, theme_alive=False))
    assert v.action == "관망"


def test_moonshot_caps_confidence():
    v = verdict.decide(_ctx(held=False, dca_score=80, theme_alive=True, rs=5.0, pct_today=1.0, high_vol=True))
    assert v.action == "매수" and v.confidence == "중"   # 무어샷은 상 안 줌


def test_moonshot_spike_threshold_higher():
    # 무어샷은 +12%로는 관망 안 됨(±15% 기준), 정상 매수 로직 적용
    v = verdict.decide(_ctx(held=False, pct_today=12.0, dca_score=62, theme_alive=True, high_vol=True))
    assert v.action == "분할매수"


def test_for_symbol_builds_context_and_decides(monkeypatch):
    from corvin_jarvis.signals import verdict as V
    from corvin_jarvis import quote_provider, dca_timing

    latest = {
        "indices": {"kospi": {"pct_change": 2.0}, "sp500": {"pct_change": 0.5}},
        "portfolio": [],
        "universe": [
            {"symbol": "000660", "market": "KR", "sector": "semiconductor", "pct_change": 6.0},
            {"symbol": "005930", "market": "KR", "sector": "semiconductor", "pct_change": 5.0},
        ],
    }

    class _Q:
        price, pct_change, source, error = 1000.0, 5.0, "stub", None
    monkeypatch.setattr(quote_provider, "get_stock_quote", lambda s: _Q())
    monkeypatch.setattr(dca_timing, "default_fetcher", lambda s, days=252: [100.0] * 60)
    monkeypatch.setattr(V, "_dca_value_score", lambda prices: 80)

    out = V.for_symbol("000660", latest)
    assert out.symbol == "000660"
    assert out.action in ("매수", "분할매수")   # 저평가+테마+주도주


def test_for_symbol_untracked_is_high_vol(monkeypatch):
    from corvin_jarvis.signals import verdict as V
    from corvin_jarvis import quote_provider, dca_timing
    latest = {"indices": {}, "portfolio": [], "universe": []}

    class _Q:
        price, pct_change, source, error = 50.0, 1.0, "stub", None
    monkeypatch.setattr(quote_provider, "get_stock_quote", lambda s: _Q())
    monkeypatch.setattr(dca_timing, "default_fetcher", lambda s, days=252: [10.0] * 60)
    monkeypatch.setattr(V, "_dca_value_score", lambda prices: 20)

    out = V.for_symbol("277810", latest)   # 미추적 미래기술
    assert out.action == "관망"            # 신호 약함


def test_verdicts_for_state_covers_held_and_alerted(monkeypatch):
    from corvin_jarvis.signals import verdict as V
    latest = {"portfolio": [{"symbol": "NVDA"}], "indices": {}, "universe": []}
    alerts = [
        {"category": "universe", "metric": "universe_000660_confirmed", "value": 9.3, "severity": "high"},
        {"category": "leading_rs", "metric": "rs_005930_confirmed", "value": 5.0, "severity": "medium"},
        {"category": "narrative", "metric": "foreign_net_buy", "value": 2.4, "severity": "high"},
    ]

    def fake_for_symbol(sym, lt):
        return V.Verdict(symbol=sym, action="관망", confidence="중", rationale="test")
    monkeypatch.setattr(V, "for_symbol", fake_for_symbol)

    out = V.verdicts_for_state(latest, alerts)
    assert set(out.keys()) == {"NVDA", "000660", "005930"}   # held + universe + rs; narrative 제외
    assert out["NVDA"]["action"] == "관망"


def test_jarvis_writes_verdicts_file(tmp_path, monkeypatch):
    import json
    from corvin_jarvis import jarvis
    from corvin_jarvis.signals import verdict as V

    latest = {"portfolio": [{"symbol": "NVDA"}], "indices": {}, "universe": []}
    alerts = {"alerts": [{"category": "universe", "metric": "universe_000660_confirmed",
                          "value": 9.0, "severity": "high"}]}
    lf = tmp_path / "latest.json"
    af = tmp_path / "alerts.json"
    vf = tmp_path / "verdicts.json"
    lf.write_text(json.dumps(latest))
    af.write_text(json.dumps(alerts))
    monkeypatch.setattr(jarvis, "LATEST_FILE", lf)
    monkeypatch.setattr(jarvis, "ALERTS_FILE", af)
    monkeypatch.setattr(jarvis, "VERDICTS_FILE", vf)
    monkeypatch.setattr(V, "for_symbol", lambda s, lt: V.Verdict(s, "홀딩", "중", "t"))

    n = jarvis.compute_and_write_verdicts()
    assert n == 2   # NVDA(held) + 000660(alerted)
    data = json.loads(vf.read_text())
    assert "NVDA" in data and "000660" in data
    assert data["NVDA"]["action"] == "홀딩"
