"""긴급 푸시 스캔 포맷 — 🎯판정 묶음/접기 + 🔮예측 + 중복 해석 제거."""
from __future__ import annotations

from corvin_jarvis import notify

VERDICTS = {
    "MSFT": {"action": "비중축소", "confidence": "상", "rationale": "추적 익절(1/3)", "name": "Microsoft"},
    "005490": {"action": "분할매수", "confidence": "상", "rationale": "DCA 60·테마생존", "name": "POSCO홀딩스"},
    "035420": {"action": "관망", "confidence": "중", "rationale": "지난 거래일 급등 — 눌림대기", "name": "NAVER"},
    "004020": {"action": "관망", "confidence": "중", "rationale": "지난 거래일 급등 — 눌림대기", "name": "현대제철"},
    "META": {"action": "홀딩", "confidence": "중", "rationale": "보유 논리 유효", "name": "Meta"},
    "NVDA": {"action": "홀딩", "confidence": "중", "rationale": "보유 논리 유효", "name": "NVIDIA"},
    "NOISE": {"action": "관망", "confidence": "하", "rationale": "신호 없음", "name": "x"},
}


def test_verdicts_block_groups_watch_and_collapses_hold():
    block = "\n".join(notify._format_verdicts_block(VERDICTS))
    assert "Microsoft(MSFT) 비중축소" in block      # 액션성은 개별
    assert "POSCO홀딩스(005490) 분할매수" in block
    assert "035420·004020 관망" in block            # 관망 묶음
    assert "홀딩 2종 안정" in block                  # 홀딩 카운트로 접힘
    assert "META·NVDA" in block
    assert "NOISE" not in block                     # 신뢰 하 관망은 숨김


def test_verdicts_block_empty_when_all_noise():
    assert notify._format_verdicts_block({"X": {"action": "관망", "confidence": "하"}}) == []


def test_prediction_digest_block_strips_chrome(monkeypatch):
    """🔮 = 예측 다이제스트 라이브. 09:36 헤더·푸터·바깥 구분선은 제거."""
    from corvin_jarvis.prediction import backfill, feeds
    from corvin_jarvis.prediction import run_prediction_digest as R

    fake = (
        "📅 *2026-06-14 장초반 스냅샷* (09:36 KST · 미완성봉)\n"
        "━━━━━━━━━━━━\n"
        "📊 *시장 방향*\n"
        "• 과거 12개 → +3.4% 승률83% _(신뢰 80)_\n"
        "📈 *종목 예측 (보유·감시)*\n"
        "• *META* — MC -2% · 추세↓ _(신뢰 69)_\n"
        "━━━━━━━━━━━━\n"
        "_⚠️ advisory_"
    )
    monkeypatch.setattr(backfill, "init_db", lambda db: None)
    monkeypatch.setattr(R, "gather_inputs", lambda **k: {
        "holdings": [], "universe": [], "stops": {}, "closes_by_sym": {},
        "daily_by_feature": {}, "geo_payload": None, "sentiment_payload": None})
    monkeypatch.setattr(R, "build_digest", lambda **k: fake)
    monkeypatch.setattr(feeds, "fetch_sentiment", lambda: None)

    out = notify._prediction_digest_block()
    assert "📅" not in out and "_⚠️" not in out      # 헤더·푸터 제거
    assert "━━━━" not in out                          # 바깥 구분선 제거
    assert "종목 예측" in out and "META" in out        # 본문 보존


def test_prediction_digest_block_graceful_on_error(monkeypatch):
    """예측 블록 실패가 긴급 푸시 전체를 막지 않는다 → ''."""
    from corvin_jarvis.prediction import backfill
    monkeypatch.setattr(backfill, "init_db",
                        lambda db: (_ for _ in ()).throw(RuntimeError("boom")))
    assert notify._prediction_digest_block() == ""


def test_urgent_message_sections_and_no_redundant_interpret():
    alerts = [{"severity": "critical", "message": "파세요 012450 — 손절선 돌파 (-12.2%)"}]
    msg = notify._format_urgent_message(
        alerts, title="🦅 Corvin 긴급", limit=8, verdicts=VERDICTS,
        predictive_block="\n🔮 예측 신호 (선행)\n📅 FOMC 금리결정 D-4 · 변동성↑",
    )
    assert "🎯 판정" in msg
    assert "🔮 예측" in msg
    assert "FOMC 금리결정 D-4" in msg
    assert "예측 신호 (선행)" not in msg     # 원본 헤더는 구분선으로 치환
    assert "📖 해석" not in msg              # 중복 해석 제거
    assert msg.rstrip().endswith("📄 상세 briefing.md")
