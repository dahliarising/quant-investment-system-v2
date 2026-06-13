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
