from corvin_jarvis import ew_providers as ewp


# ── Task 7: DI adapters ──────────────────────────────────────
def test_semis_reading_from_series():
    soxx = [100.0] * 60
    spy = [200.0] * 60
    soxx[-1] = 90.0      # 반도체 약세 → ratio 하락
    r = ewp.semis_reading(lambda s, n: {"SOXX": soxx, "SPY": spy}[s],
                          spx_high_fetcher=lambda: 205.0)
    assert r["ratio"] < r["ratio_ma50"]
    assert r["slope_5d"] < 0
    assert r["spx_dist_from_high_pct"] < 0


def test_semis_skips_on_missing_data():
    assert ewp.semis_reading(lambda s, n: [], spx_high_fetcher=lambda: None) is None


def test_vix_term_reading():
    r = ewp.vix_term_reading(lambda s: {"^VIX": 21.5, "^VIX3M": 21.8}[s])
    assert round(r["ratio"], 3) == round(21.5 / 21.8, 3)


def test_vix_term_skips_on_missing():
    assert ewp.vix_term_reading(lambda s: None) is None


def test_breadth_reading():
    # AAA above its MA200, BBB below
    above = [100.0] * 199 + [120.0]
    below = [100.0] * 199 + [80.0]
    closes = {"AAA": above, "BBB": below}
    r = ewp.breadth_reading(["AAA", "BBB"], lambda s, n: closes[s])
    assert r["pct_above_ma200"] == 50.0


def test_breadth_skips_when_no_history():
    assert ewp.breadth_reading(["AAA"], lambda s, n: [1.0] * 10) is None


def test_fred_reading_value_and_change():
    r = ewp.fred_reading(lambda c, n: [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.74], "X")
    assert r["value"] == 2.74
    assert round(r["chg_5d"], 2) == round(2.74 - 2.2, 2)  # [-1] vs [-6] = 5-day change


def test_fred_skips_on_short_series():
    assert ewp.fred_reading(lambda c, n: [1.0, 2.0], "X") is None


# ── Task 8: live fetcher bindings exist ──────────────────────
def test_live_fetchers_callable():
    assert callable(ewp.live_closes_fetcher)
    assert callable(ewp.live_vix_fetcher)
    assert callable(ewp.live_fred_fetcher)
