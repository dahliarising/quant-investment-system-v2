# tests/test_prediction_digest.py
from corvin_jarvis.prediction.contract import PredictionResult, insufficient
from corvin_jarvis.prediction import digest_assembler


def test_assemble_groups_and_marks_holds():
    results = [
        PredictionResult("vector_analog", "market", "다음 5일 평균 +1.8%", 70,
                         {"win_rate": 0.67}, True),
        PredictionResult("geopolitical", "market", "리스크 보통(45)", 60, {}, True),
        PredictionResult("velocity", "META", "손절 4일 이내", 65, {}, True),
        insufficient("probability", "NVDA", "데이터 부족"),
    ]
    text = digest_assembler.assemble(results, date_str="2026-06-15")
    assert "2026-06-15" in text
    assert "장초반 스냅샷" in text          # 시점 라벨 (memory)
    assert "시장 방향" in text              # 섹션 헤더
    assert "META" in text
    assert "⏸" in text                      # 보류 마커
    assert "NVDA" in text


def test_assemble_handles_empty():
    text = digest_assembler.assemble([], date_str="2026-06-15")
    assert "예측 결과 없음" in text


def test_consolidates_multiple_systems_into_one_line_per_symbol():
    """한 종목이 여러 시스템 결과를 가져도 한 줄로 병합 (종목당 한 줄)."""
    results = [
        PredictionResult("probability", "NVDA", "5일 내 손절가(189) 이탈 확률 4%", 88,
                         {"prob_below_stop": 0.04}, True),
        PredictionResult("momentum", "NVDA", "단기 MA(20) > 장기 MA(120) → 추세 상방",
                         88, {"signal": "bullish", "gap_pct": 12.0}, True),
    ]
    text = digest_assembler.assemble(results, date_str="2026-06-15",
                                     holding_symbols={"NVDA"})
    assert text.count("*NVDA*") == 1


def test_dedup_multiple_holds_into_single_line():
    """손절선 없는 보유가 velocity+probability 양쪽 보류 → 한 줄로 dedup."""
    results = [
        insufficient("velocity", "META", "손절선 미설정"),
        insufficient("probability", "META", "손절가 미설정(stop<=0)"),
    ]
    text = digest_assembler.assemble(results, date_str="2026-06-15",
                                     holding_symbols={"META"})
    assert text.count("*META*") == 1
    assert "⏸" in text


def test_real_signal_suppresses_hold_for_same_symbol():
    """같은 종목에 실신호+보류 공존 시, 실신호가 보류를 덮어씀."""
    results = [
        insufficient("velocity", "NVDA", "손절선 미설정"),
        PredictionResult("momentum", "NVDA", "추세 상방", 80,
                         {"signal": "bullish", "gap_pct": 5.0}, True),
    ]
    text = digest_assembler.assemble(results, date_str="2026-06-15",
                                     holding_symbols={"NVDA"})
    nvda_lines = [ln for ln in text.splitlines() if "*NVDA*" in ln]
    assert len(nvda_lines) == 1
    assert "⏸" not in nvda_lines[0]


def test_near_zero_gap_drops_misleading_minus_zero():
    """gap이 0으로 반올림되면 '-0%' 대신 화살표만 (meaningful-metrics)."""
    results = [PredictionResult("momentum", "MSFT", "MA 교차 중립", 55,
               {"signal": "bearish", "gap_pct": -0.4}, True)]
    text = digest_assembler.assemble(results, date_str="2026-06-15",
                                     holding_symbols={"MSFT"})
    assert "-0%" not in text
    assert "추세↓" in text


def test_universe_momentum_summarized_not_listed():
    """유니버스 모멘텀은 개별 나열 금지 — 카운트 요약."""
    results = [PredictionResult("momentum", f"U{i}",
               "단기 MA(20) > 장기 MA(120) → 추세 상방", 80,
               {"signal": "bullish", "gap_pct": 5.0}, True) for i in range(30)]
    text = digest_assembler.assemble(results, date_str="2026-06-15",
                                     holding_symbols=set())
    assert text.count("추세 상방") < 30
    assert "유니버스" in text
    assert "30" in text
