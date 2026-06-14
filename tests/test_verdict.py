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

    def fake_for_symbol(sym, lt, market_closed=False):
        return V.Verdict(symbol=sym, action="관망", confidence="중", rationale="test")
    monkeypatch.setattr(V, "for_symbol", fake_for_symbol)

    out = V.verdicts_for_state(latest, alerts)
    assert set(out.keys()) == {"NVDA", "000660", "005930"}   # held + universe + rs; narrative 제외
    assert out["NVDA"]["action"] == "관망"


def test_for_symbol_moonshot_sector_capped_confidence(monkeypatch):
    from corvin_jarvis.signals import verdict as V
    from corvin_jarvis import quote_provider, dca_timing

    latest = {
        "indices": {"kospi": {"pct_change": 1.0}},
        "portfolio": [],
        "universe": [
            {"symbol": "277810", "market": "KR", "sector": "humanoid", "pct_change": 9.0},
            {"symbol": "454910", "market": "KR", "sector": "humanoid", "pct_change": 9.0},
        ],
    }

    class _Q:
        price, pct_change, source, error = 100.0, 9.0, "stub", None
    monkeypatch.setattr(quote_provider, "get_stock_quote", lambda s: _Q())
    monkeypatch.setattr(dca_timing, "default_fetcher", lambda s, days=252: [100.0] * 60)
    monkeypatch.setattr(V, "_dca_value_score", lambda prices: 80)   # 깊은 저평가

    out = V.for_symbol("277810", latest)
    # 휴머노이드=무어샷 → high_vol → 깊은저평가+테마+주도주여도 신뢰도 "상" 아닌 "중"
    assert out.action == "매수" and out.confidence == "중"


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
    monkeypatch.setattr(V, "for_symbol", lambda s, lt, market_closed=False: V.Verdict(s, "홀딩", "중", "t"))

    n = jarvis.compute_and_write_verdicts()
    assert n == 2   # NVDA(held) + 000660(alerted)
    data = json.loads(vf.read_text())
    assert "NVDA" in data and "000660" in data
    assert data["NVDA"]["action"] == "홀딩"


# ── ATR 변동성 조정 손절 + A/B 버킷 (2026-06-11) ──────────

def test_atr_stop_threshold_widens_for_volatile():
    """고변동(ATR% 큼) → 더 깊은(넓은) 손절선."""
    thr = verdict._atr_stop_threshold(2.6)   # TSLA류
    assert thr == -13.0   # -5 × 2.6


def test_atr_stop_threshold_tightens_for_calm():
    """저변동 → 더 얕은(타이트한) 손절선."""
    assert verdict._atr_stop_threshold(1.0) == -5.0


def test_atr_stop_threshold_clamped_both_ends():
    """극단 변동성은 클램프 — 너무 넓거나 너무 타이트하지 않게."""
    assert verdict._atr_stop_threshold(6.0) == -25.0   # floor
    assert verdict._atr_stop_threshold(0.5) == -4.0    # ceiling


def test_atr_stop_threshold_none_on_missing():
    """ATR 없음 → None (호출측이 평면 -8% fallback)."""
    assert verdict._atr_stop_threshold(None) is None
    assert verdict._atr_stop_threshold(0) is None


def test_held_uses_atr_stop_not_flat():
    """ATR 손절이 평면보다 넓을 때 — 평면이면 팔릴 손실도 홀딩."""
    # atr_pct 2.6 → 손절 -13%. pnl -10%는 평면(-8)이면 매도지만 ATR이면 홀딩
    v = verdict.decide(_ctx(held=True, pnl_pct=-10.0, atr_pct=2.6, theme_alive=True))
    assert v.action == "홀딩"


def test_held_atr_stop_triggers_when_breached():
    """ATR 손절선 돌파 — 매도 + ATR 근거 명시."""
    v = verdict.decide(_ctx(held=True, pnl_pct=-14.0, atr_pct=2.6))
    assert v.action == "매도"
    assert "ATR" in v.rationale


def test_held_flat_fallback_when_no_atr():
    """ATR 미제공 — 기존 평면 -8% 동작 보존 (회귀 방지)."""
    v = verdict.decide(_ctx(held=True, pnl_pct=-9.0))
    assert v.action == "매도" and v.confidence == "상"


def test_dca_bucket_skips_price_stop():
    """B(dca) 버킷 — 가격 손절 미적용. -20%여도 매도 아님 (예약 추매 보존)."""
    v = verdict.decide(_ctx(held=True, pnl_pct=-20.0, bucket="dca", theme_alive=True))
    assert v.action != "매도"


def test_dca_bucket_still_trims_on_thesis_weakness():
    """B 버킷도 thesis 약화(RS 약세+테마 식음)엔 비중축소 — 가격 아닌 근거 기준."""
    v = verdict.decide(_ctx(held=True, pnl_pct=-20.0, bucket="dca",
                            rs=-6.0, theme_alive=False))
    assert v.action == "비중축소"


def test_dca_bucket_take_profit_still_works():
    """B 버킷도 익절선은 작동."""
    v = verdict.decide(_ctx(held=True, pnl_pct=26.0, bucket="dca"))
    assert v.action == "비중축소"


def test_dca_bucket_hold_rationale_notes_no_price_stop():
    """B 버킷 홀딩 시 근거에 '가격손절 없음' 표기 (투명성)."""
    v = verdict.decide(_ctx(held=True, pnl_pct=-3.0, bucket="dca"))
    assert v.action == "홀딩"
    assert "손절" in v.rationale  # "가격손절 없음" 류 표기


def test_trade_bucket_is_default():
    """버킷 미지정 = trade(A) = 보호적 손절 기본 적용."""
    v = verdict.decide(_ctx(held=True, pnl_pct=-9.0))  # bucket 없음
    assert v.action == "매도"


# ── 추적 익절 (상방 레이어, 2026-06-11) ─────────────────

def test_trailing_tp_lets_winner_run():
    v = verdict.decide(_ctx(held=True, pnl_pct=20.0, peak_pnl_pct=22.0,
                            atr_pct=2.0, theme_alive=True))
    assert v.action == "홀딩"


def test_trailing_tp_triggers_on_pullback():
    v = verdict.decide(_ctx(held=True, pnl_pct=18.0, peak_pnl_pct=30.0, atr_pct=2.0))
    assert v.action == "비중축소" and "추적" in v.rationale


def test_trailing_tp_inactive_below_activate():
    v = verdict.decide(_ctx(held=True, pnl_pct=10.0, peak_pnl_pct=12.0,
                            atr_pct=1.0, theme_alive=True))
    assert v.action == "홀딩"


def test_trailing_tp_flat_fallback_no_peak():
    v = verdict.decide(_ctx(held=True, pnl_pct=26.0))
    assert v.action == "비중축소"


def test_trailing_tp_min_giveback_without_atr():
    v = verdict.decide(_ctx(held=True, pnl_pct=24.0, peak_pnl_pct=30.0))
    assert v.action == "비중축소"


def test_trailing_helper_units():
    assert verdict._trailing_take_profit(20.0, 22.0, 2.0) is False
    assert verdict._trailing_take_profit(18.0, 30.0, 2.0) is True
    assert verdict._trailing_take_profit(26.0, None, None) is True
    assert verdict._trailing_take_profit(10.0, 12.0, 1.0) is False


# ── 리뷰 반영: peak는 최근 스윙 고점만 (진입 전 고점 오염 방지) ──

def test_peak_pnl_ignores_old_pre_entry_high():
    """80봉 전 급등 고점은 무시 — 최근 창만. 거짓 되돌림 익절 방지."""
    closes = [200.0] + [100.0] * 30   # 옛 고점 200, 최근은 100 부근
    # 현재가 110, pnl +10% → 진입가 ~100. 최근 고점도 ~110 → peak ~+10%(과대평가 없음)
    peak = verdict._peak_pnl_pct(closes, 110.0, 10.0)
    assert peak is not None and peak < 15.0   # 옛 200을 안 씀(쓰면 +100%)


def test_peak_pnl_uses_recent_swing_high():
    """최근 창 내 진짜 고점은 반영 — 정당한 되돌림 익절."""
    closes = [100.0] * 10 + [130.0] + [120.0] * 5   # 최근 창에 고점 130
    peak = verdict._peak_pnl_pct(closes, 120.0, 20.0)  # 진입가 ~100
    assert peak is not None and peak >= 28.0   # 130/100-1 ≈ +30%


def test_peak_pnl_none_on_bad_input():
    assert verdict._peak_pnl_pct([100.0], None, 10.0) is None
    assert verdict._peak_pnl_pct([], 100.0, 10.0) is None
    assert verdict._peak_pnl_pct([100.0], 100.0, -100.0) is None


def test_peak_pnl_at_high_no_false_giveback():
    """현재가가 최근 고점 — peak≈pnl, 되돌림 0 → 톱 안 팖."""
    closes = [100.0] * 14
    peak = verdict._peak_pnl_pct(closes, 120.0, 20.0)  # 현재가가 곧 최근 최고
    assert verdict._trailing_take_profit(20.0, peak, 2.0) is False


# ── ③ 불타기(add-to-winners) + ④ 분할익절 사다리 (2026-06-11) ──

def test_add_to_winner_on_confirmed_strength():
    """수익 + 지수 주도(rs≥4) + 테마 살아있음 + 비과열 → 비중확대(불타기)."""
    v = verdict.decide(_ctx(held=True, pnl_pct=10.0, rs=5.0,
                            theme_alive=True, pct_today=2.0))
    assert v.action == "비중확대" and "불타기" in v.rationale


def test_no_add_when_spiked_today():
    """오늘 급등(과열)이면 불타기 보류 — FOMO 추격 방지."""
    v = verdict.decide(_ctx(held=True, pnl_pct=10.0, rs=5.0,
                            theme_alive=True, pct_today=12.0))
    assert v.action != "비중확대"


def test_no_add_when_rs_weak():
    """주도주 아니면(rs<4) 불타기 안 함."""
    v = verdict.decide(_ctx(held=True, pnl_pct=10.0, rs=1.0,
                            theme_alive=True, pct_today=2.0))
    assert v.action != "비중확대"


def test_no_add_when_not_in_profit():
    """손실 중이면 불타기 아님(승자 추가 원칙)."""
    v = verdict.decide(_ctx(held=True, pnl_pct=-3.0, rs=5.0,
                            theme_alive=True, pct_today=2.0))
    assert v.action != "비중확대"


def test_scaleout_small_gain_trims_third():
    """추적 익절 — 작은 고점(+30%, <40)은 1/3 차익실현."""
    v = verdict.decide(_ctx(held=True, pnl_pct=18.0, peak_pnl_pct=30.0, atr_pct=2.0))
    assert v.action == "비중축소" and "1/3" in v.rationale


def test_scaleout_big_gain_trims_half():
    """추적 익절 — 큰 고점(≥+40%)은 절반 차익실현."""
    v = verdict.decide(_ctx(held=True, pnl_pct=30.0, peak_pnl_pct=45.0, atr_pct=2.0))
    assert v.action == "비중축소" and "절반" in v.rationale


def test_no_add_for_dca_bucket():
    """리뷰 반영: dca(가치) 버킷은 불타기(모멘텀) 제외 — 철학 분리."""
    v = verdict.decide(_ctx(held=True, pnl_pct=12.0, rs=6.0, theme_alive=True,
                            pct_today=1.5, bucket="dca"))
    assert v.action != "비중확대"


def test_add_winner_emoji_in_notify_map():
    """리뷰 반영: 비중확대가 다이제스트 이모지 맵에 존재 (아이콘 누락 방지)."""
    from corvin_jarvis import notify
    assert "비중확대" in notify._ACTION_EMOJI
