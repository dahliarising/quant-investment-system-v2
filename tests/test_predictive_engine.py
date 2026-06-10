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
    """BOK 금통위 D-2 (2026-06-11) — 신호 발생."""
    sigs = pe.evaluate_events(as_of=date(2026, 6, 9), held_symbols=["012450"])
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
