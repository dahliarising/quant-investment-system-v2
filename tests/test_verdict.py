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
