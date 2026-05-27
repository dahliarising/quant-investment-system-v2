from corvin_jarvis.signals import leading


def _u(pct, sym="005930", name="삼성전자"):
    return [{"symbol": sym, "market": "KR", "sector": "semiconductor", "name": name, "pct_change": pct}]


def test_outperform_index_creates_strong_alert():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    alerts = leading.check_relative_strength(_u(7.0), index_pct=2.5, config=cfg, phase="confirmed", index_name="kospi")
    assert len(alerts) == 1
    a = alerts[0]
    assert a["category"] == "leading_rs"
    assert a["metric"] == "rs_005930_confirmed"
    assert abs(a["value"] - 4.5) < 0.01      # 7.0 - 2.5
    assert "강세" in a["message"]


def test_underperform_index_creates_weak_alert():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    alerts = leading.check_relative_strength(_u(-1.0), index_pct=2.5, config=cfg, phase="confirmed")
    assert len(alerts) == 1
    assert alerts[0]["value"] < 0
    assert "약세" in alerts[0]["message"]


def test_within_threshold_no_alert():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    assert leading.check_relative_strength(_u(3.0), index_pct=2.5, config=cfg, phase="confirmed") == []


def test_none_index_returns_empty():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    assert leading.check_relative_strength(_u(7.0), index_pct=None, config=cfg, phase="confirmed") == []


def test_missing_pct_skipped():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    entries = [{"symbol": "X", "market": "KR", "sector": "auto", "name": "X", "pct_change": None}]
    assert leading.check_relative_strength(entries, index_pct=2.5, config=cfg, phase="confirmed") == []


def test_provisional_label():
    cfg = {"leading": {"rs_min_pct": 2.0}}
    a = leading.check_relative_strength(_u(7.0), index_pct=2.5, config=cfg, phase="provisional")[0]
    assert a["metric"] == "rs_005930_provisional"
    assert "🟡" in a["message"] and "잠정" in a["message"]
