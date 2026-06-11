"""STAGE①.5 예측 시그널 엔진 — TDD (RED → GREEN).

세 Pillar:
  VELOCITY  — 가격 하락 속도로 손절선 도달일 예측
  RS_WEAK   — 벤치마크 대비 상대강도 약화 탐지
  EVENT     — FOMC/BOK 등 거시 일정 선제 경보
"""
from datetime import date

import pytest

from corvin_jarvis import predictive_engine as pe


# ── VELOCITY ──────────────────────────────────────────────

@pytest.mark.unit
def test_velocity_warns_when_approaching_stop_fast():
    """급락세 — 손절선까지 14일 이내 예상 → VELOCITY 경보."""
    # 450→385 over 13 days (slope=-5), stop=370 → dist=15, days=3
    closes = [450.0 - i * 5.0 for i in range(14)]
    sigs = pe.evaluate_velocity(
        [{"symbol": "TSLA", "price": closes[-1], "shares": 2}],
        stops={"TSLA": 370.0},
        closes_by_sym={"TSLA": closes},
    )
    assert len(sigs) == 1
    assert sigs[0].kind == "VELOCITY"
    assert sigs[0].symbol == "TSLA"
    assert sigs[0].horizon_days is not None and sigs[0].horizon_days <= 14


@pytest.mark.unit
def test_velocity_silent_when_stop_far():
    """미미한 하락세 — 손절선 200까지 400일+ → 신호 없음."""
    closes = [410.0 - i * 0.5 for i in range(14)]  # slope≈-0.5
    sigs = pe.evaluate_velocity(
        [{"symbol": "TSLA", "price": closes[-1], "shares": 2}],
        stops={"TSLA": 200.0},
        closes_by_sym={"TSLA": closes},
    )
    assert sigs == []


@pytest.mark.unit
def test_velocity_silent_without_stop():
    """손절선 미설정 종목 — VELOCITY 없음."""
    closes = [410.0 - i * 3.0 for i in range(14)]
    sigs = pe.evaluate_velocity(
        [{"symbol": "META", "price": closes[-1], "shares": 7}],
        stops={},
        closes_by_sym={"META": closes},
    )
    assert sigs == []


@pytest.mark.unit
def test_velocity_silent_when_price_already_at_stop():
    """이미 손절선 이하 — signal_engine 담당, velocity 불개입."""
    closes = [380.0 - i * 2.0 for i in range(14)]  # closes[-1]=354 < stop=377
    sigs = pe.evaluate_velocity(
        [{"symbol": "TSLA", "price": closes[-1], "shares": 2}],
        stops={"TSLA": 377.0},
        closes_by_sym={"TSLA": closes},
    )
    assert sigs == []


# ── RS_WEAK ────────────────────────────────────────────────

@pytest.mark.unit
def test_rs_weak_when_underperforming_benchmark():
    """보유종목 -18% vs 벤치 +2% → RS -20%p → RS_WEAK."""
    holding = [100.0 * (0.99 ** i) for i in range(21)]   # oldest=100, newest≈81.8
    bench   = [100.0 * (1.001 ** i) for i in range(21)]  # oldest=100, newest≈102.0
    sigs = pe.evaluate_relative_strength(
        [{"symbol": "012450", "market": "KR", "price": holding[-1], "shares": 4}],
        closes_by_sym={"012450": holding},
        bench_closes_by_market={"KR": bench},
        n_days=20,
    )
    assert len(sigs) == 1
    assert sigs[0].kind == "RS_WEAK"
    assert sigs[0].symbol == "012450"
    assert sigs[0].evidence["rs_pct"] < -5.0


@pytest.mark.unit
def test_rs_ok_when_matched_with_benchmark():
    """보유·벤치 동일 하락 → RS=0 → 신호 없음."""
    closes = [100.0 - i * 0.5 for i in range(21)]
    sigs = pe.evaluate_relative_strength(
        [{"symbol": "TSLA", "market": "US", "price": closes[-1], "shares": 2}],
        closes_by_sym={"TSLA": closes},
        bench_closes_by_market={"US": closes},
        n_days=20,
    )
    assert sigs == []


@pytest.mark.unit
def test_rs_skipped_with_insufficient_data():
    """closes 3개 → n_days=20 미달 → 신호 없음."""
    closes = [100.0, 99.0, 98.0]
    sigs = pe.evaluate_relative_strength(
        [{"symbol": "TSLA", "market": "US", "price": 98.0, "shares": 2}],
        closes_by_sym={"TSLA": closes},
        bench_closes_by_market={"US": closes},
        n_days=20,
    )
    assert sigs == []


# ── EVENT ──────────────────────────────────────────────────

@pytest.mark.unit
def test_event_fires_for_upcoming_bok():
    """BOK 금통위 D-2 (2026-07-09, 실제 금리결정 달) — 신호 발생."""
    sigs = pe.evaluate_events(as_of=date(2026, 7, 7), held_symbols=["012450"])
    assert any("금융통화" in s.message or "BOK" in s.message for s in sigs)


@pytest.mark.unit
def test_event_fires_for_upcoming_fomc():
    """FOMC D-8 (2026-06-17) — 신호 발생."""
    sigs = pe.evaluate_events(as_of=date(2026, 6, 9), held_symbols=["META"])
    assert any("FOMC" in s.message for s in sigs)


@pytest.mark.unit
def test_event_silent_when_no_upcoming_events():
    """8월 1일 기준 14일 이내 캘린더 이벤트 없음 → 신호 없음."""
    sigs = pe.evaluate_events(as_of=date(2026, 8, 1), held_symbols=["META"])
    assert sigs == []


@pytest.mark.unit
def test_event_urgency_increases_nearer():
    """D-1은 D-2보다 urgency가 높거나 같아야 한다."""
    d1_sigs = pe.evaluate_events(as_of=date(2026, 6, 10), held_symbols=[])  # BOK D-1
    d2_sigs = pe.evaluate_events(as_of=date(2026, 6, 9),  held_symbols=[])  # BOK D-2
    bok_d1 = [s for s in d1_sigs if "금융통화" in s.message]
    bok_d2 = [s for s in d2_sigs if "금융통화" in s.message]
    if bok_d1 and bok_d2:
        assert bok_d1[0].urgency >= bok_d2[0].urgency


# ── 통합 ──────────────────────────────────────────────────

@pytest.mark.unit
def test_evaluate_returns_urgency_sorted():
    """통합 evaluate → 긴급도 내림차순 정렬."""
    # VELOCITY(dist=8,days=4,slope=-2) + BOK/FOMC events from calendar
    closes = [406.0 - i * 2.0 for i in range(14)]
    sigs = pe.evaluate(
        [{"symbol": "TSLA", "market": "US", "price": 385.0, "shares": 2}],
        stops={"TSLA": 377.0},
        closes_by_sym={"TSLA": closes},
        bench_closes_by_market={"US": [3000.0] * 22},  # flat bench → RS skipped (14<21)
        as_of=date(2026, 6, 9),
    )
    assert sigs  # 최소 이벤트 신호 있어야
    urgencies = [s.urgency for s in sigs]
    assert urgencies == sorted(urgencies, reverse=True)


@pytest.mark.unit
def test_predictive_signal_to_dict():
    """to_dict() 직렬화 — 모든 필드 포함."""
    sigs = pe.evaluate_events(as_of=date(2026, 6, 9), held_symbols=[])
    assert sigs
    d = sigs[0].to_dict()
    assert "symbol" in d and "kind" in d and "urgency" in d
    assert "message" in d and "evidence" in d


# ── Phase 3: VELOCITY 고도화 ──────────────────────────

def _mk_holding(sym="TSLA", price=100.0):
    return [{"symbol": sym, "market": "US", "price": price, "pnl_pct": -5.0}]


@pytest.mark.unit
def test_velocity_noise_gate_suppresses_weak_slope_in_choppy_market():
    """변동성 대비 미미한 기울기 — 노이즈 게이트 억제 (15봉 이상에서 활성)."""
    # 일변화 ±5 들쭉날쭉(ATR프록시≈5), 순기울기 -0.5 → strength 0.1 < 0.15 게이트
    closes = [100.0]
    deltas = [+5, -5.5, +5, -5.5, +5, -5.5, +5, -5.5, +5, -5.5, +5, -5.5, +5, -5.5, +5, -6.0]
    for d in deltas:
        closes.append(closes[-1] + d)
    sigs = pe.evaluate_velocity(_mk_holding(price=closes[-1]),
                                {"TSLA": closes[-1] - 3}, {"TSLA": closes})
    assert sigs == []  # days_to≈10.7 ≤ horizon인데도 게이트(strength≈0.05)가 억제


@pytest.mark.unit
def test_velocity_clean_downtrend_still_fires_with_atr_data():
    """저변동 명확한 하락 추세 — 게이트 통과, 신뢰구간 evidence 포함."""
    # 평균 -1/일 + 미세 변동(se>0) — 완전 선형이면 lo==hi라 범위 표기가 생략됨
    closes = [120.0]
    for i in range(15):
        closes.append(closes[-1] + (-0.9 if i % 2 == 0 else -1.1))
    price, stop = closes[-1], closes[-1] - 5
    sigs = pe.evaluate_velocity(_mk_holding(price=price), {"TSLA": stop}, {"TSLA": closes})
    assert len(sigs) == 1
    ev = sigs[0].evidence
    assert ev["atr_proxy"] is not None and ev["strength"] >= 0.15
    assert ev["days_lo"] is not None and ev["days_hi"] is not None
    assert ev["days_lo"] <= ev["days_to_stop"] <= ev["days_hi"]
    assert "범위" in sigs[0].message  # "(범위 X–Y일)" 표기


@pytest.mark.unit
def test_velocity_short_series_skips_gate_backcompat():
    """15봉 미만 — ATR 산출 불가 → 게이트 미적용 (기존 동작 보존)."""
    closes = [110.0, 108.0, 106.0, 104.0]  # 4봉, 기존 테스트 스타일
    sigs = pe.evaluate_velocity(_mk_holding(price=104.0), {"TSLA": 98.0}, {"TSLA": closes})
    assert len(sigs) == 1
    assert sigs[0].evidence["atr_proxy"] is None


# ── Phase 3: confidence 캘리브레이션 주입 ─────────────

@pytest.mark.unit
def test_confidence_for_uses_calibrated_value():
    cal = {"predictive": {"VELOCITY": {"n": 20, "hit_rate": 0.4,
                                       "calibrated_confidence": 47.5}}}
    assert pe.confidence_for("VELOCITY", calibration=cal) == 47.5


@pytest.mark.unit
def test_confidence_for_falls_back_to_default():
    assert pe.confidence_for("VELOCITY", calibration={}) == 65.0
    assert pe.confidence_for("RS_WEAK", calibration={}) == 60.0
    assert pe.confidence_for("EVENT", calibration={}) == 90.0


@pytest.mark.unit
def test_confidence_for_ignores_uncalibrated_none():
    """n<10이라 calibrated_confidence=None — 기본값 유지."""
    cal = {"predictive": {"VELOCITY": {"n": 3, "hit_rate": 1.0,
                                       "calibrated_confidence": None}}}
    assert pe.confidence_for("VELOCITY", calibration=cal) == 65.0


@pytest.mark.unit
def test_evaluate_injects_calibrated_confidence():
    closes = [110.0, 108.0, 106.0, 104.0]
    cal = {"predictive": {"VELOCITY": {"n": 20, "hit_rate": 0.4,
                                       "calibrated_confidence": 47.5}}}
    sigs = pe.evaluate(_mk_holding(price=104.0), stops={"TSLA": 98.0},
                       closes_by_sym={"TSLA": closes}, calibration=cal)
    vel = [s for s in sigs if s.kind == "VELOCITY"]
    assert vel and vel[0].confidence == 47.5


# ── Phase 3: RS_WEAK 적응 임계값 ─────────────────────

@pytest.mark.unit
def test_adaptive_threshold_falls_back_on_short_history():
    """이력 부족(90봉 미만) — 기본 -5.0 유지 (기존 동작 보존)."""
    closes = [100.0 + i * 0.1 for i in range(30)]
    bench = list(closes)
    assert pe._adaptive_rs_threshold(closes, bench) == -5.0


@pytest.mark.unit
def test_adaptive_threshold_widens_for_volatile_pair():
    """변동 큰 종목 — 하위 10분위가 -5보다 깊어짐 (오탐 억제)."""
    import random
    rng = random.Random(42)
    closes, bench = [100.0], [100.0]
    for _ in range(100):
        closes.append(max(1.0, closes[-1] * (1 + rng.uniform(-0.05, 0.048))))
        bench.append(bench[-1] * 1.001)
    thr = pe._adaptive_rs_threshold(closes, bench)
    assert thr < -5.0          # 더 깊은(느슨한) 임계
    assert thr >= -20.0        # 하한 클램프


@pytest.mark.unit
def test_adaptive_threshold_clamped_upper():
    """안정 페어 — 임계가 -2보다 얕아지지 않게 클램프 (과민 방지)."""
    closes = [100.0 + i * 0.01 for i in range(100)]
    bench = [100.0 + i * 0.012 for i in range(100)]
    thr = pe._adaptive_rs_threshold(closes, bench)
    assert -5.0 <= thr <= -2.0


@pytest.mark.unit
def test_rs_weak_uses_adaptive_threshold_with_long_history():
    """90봉 이력 — 적응 임계 적용, evidence에 사용 임계 기록."""
    import random
    rng = random.Random(7)
    closes, bench = [100.0], [100.0]
    for _ in range(100):
        closes.append(max(1.0, closes[-1] * (1 + rng.uniform(-0.05, 0.048))))
        bench.append(bench[-1] * 1.001)
    # 최근 20일 급락 페어 추가 — rs가 적응 임계도 뚫도록
    for _ in range(20):
        closes.append(closes[-1] * 0.93)
        bench.append(bench[-1] * 1.001)
    holdings = [{"symbol": "XXX", "market": "US", "price": closes[-1]}]
    sigs = pe.evaluate_relative_strength(holdings, {"XXX": closes}, {"US": bench})
    assert len(sigs) == 1
    assert "threshold_pct" in sigs[0].evidence
    assert sigs[0].evidence["threshold_pct"] != -5.0  # 적응값 사용됨


# ── Phase: 상방 VELOCITY (손절 미러, 2026-06-11) ──────────

@pytest.mark.unit
def test_upside_velocity_fires_on_uptrend_below_high():
    """상승추세 + 현재가 < 최근고점 → VELOCITY_UP 발화."""
    closes = [100.0 + i for i in range(20)]  # +1/일, 고점 119
    sigs = pe.evaluate_upside_velocity([{"symbol": "X", "price": 115.0}], {"X": closes})
    assert any(s.kind == "VELOCITY_UP" for s in sigs)
    up = [s for s in sigs if s.kind == "VELOCITY_UP"][0]
    assert "days_to_target" in up.evidence and up.evidence["target"] == 119.0


@pytest.mark.unit
def test_upside_velocity_skips_when_at_high():
    """현재가가 이미 최근고점 이상 → 발화 안 함 (별개 상황)."""
    closes = [100.0 + i for i in range(20)]  # 고점 119
    sigs = pe.evaluate_upside_velocity([{"symbol": "X", "price": 119.0}], {"X": closes})
    assert sigs == []


@pytest.mark.unit
def test_upside_velocity_skips_downtrend():
    """하락추세 → 발화 안 함."""
    closes = [120.0 - i for i in range(20)]
    sigs = pe.evaluate_upside_velocity([{"symbol": "X", "price": 101.0}], {"X": closes})
    assert sigs == []


@pytest.mark.unit
def test_upside_velocity_noise_gated():
    """변동성 대비 미미한 상승 기울기는 억제 (발화 시 strength≥gate)."""
    import random
    rng = random.Random(1)
    closes = [100.0]
    for _ in range(20):
        closes.append(closes[-1] + rng.uniform(-2, 2.1))
    sigs = pe.evaluate_upside_velocity(
        [{"symbol": "X", "price": closes[-1] - 1}], {"X": closes})
    for s in sigs:
        st = s.evidence["strength"]
        assert st is None or st >= pe._NOISE_GATE


# ── VELOCITY 레짐 게이트 — 추세 하락장에서만 발화 (백테스트 검증, 2026-06-11) ──

def _mk_down_closes():
    return [450.0 - i * 5.0 for i in range(14)]   # 급락세


@pytest.mark.unit
def test_velocity_fires_in_down_regime():
    """trend=down → VELOCITY 발화 (백테스트: 하락장 엣지 +5%)."""
    sigs = pe.evaluate_velocity(
        [{"symbol": "TSLA", "price": 385.0}], stops={"TSLA": 370.0},
        closes_by_sym={"TSLA": _mk_down_closes()}, regime_trend="down")
    assert any(s.kind == "VELOCITY" for s in sigs)


@pytest.mark.unit
def test_velocity_suppressed_in_chop_regime():
    """trend=chop/up → VELOCITY 억제 (백테스트: 횡보장 엣지 -4%, 노이즈)."""
    for tr in ("chop", "up"):
        sigs = pe.evaluate_velocity(
            [{"symbol": "TSLA", "price": 385.0}], stops={"TSLA": 370.0},
            closes_by_sym={"TSLA": _mk_down_closes()}, regime_trend=tr)
        assert sigs == [], f"trend={tr} should suppress"


@pytest.mark.unit
def test_velocity_fires_when_regime_unknown():
    """trend=None(레짐 미상) → 기존 동작 보존 (회귀 0)."""
    sigs = pe.evaluate_velocity(
        [{"symbol": "TSLA", "price": 385.0}], stops={"TSLA": 370.0},
        closes_by_sym={"TSLA": _mk_down_closes()}, regime_trend=None)
    assert any(s.kind == "VELOCITY" for s in sigs)
