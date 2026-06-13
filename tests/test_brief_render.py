from corvin_jarvis.brief.types import (
    Brief, PositionLine, EvidenceStack, Framing, PsychGuard,
)
from corvin_jarvis.brief.render import render_brief


def _sample() -> Brief:
    return Brief(
        headline="🦅 Corvin 실행 브리핑 — 06/12",
        positions=[
            PositionLine("012450", -17.4, "✂️", "손절존 검토", "종가"),
            PositionLine("MSFT", 12.9, "✅", "유지", "종가"),
        ],
        evidence={"012450": EvidenceStack("검증부족(n=1)", None)},
        framing=Framing("강세축 2건", "방어축 1건 — 추세이탈",
                        "base: 방어 1 · 매수후보/유지 2", "지금 TSLA 미실행 시…"),
        psych=PsychGuard(True, "드로다운", "룰 손절선 vs 감정 반응 점검."),
        as_of="2026-06-12 05:45 KST", market_state="장마감",
    )


def test_render_has_all_sections_and_scannable():
    out = render_brief(_sample())
    assert "━━━" in out                     # 구분선
    assert "✂️ 012450" in out                # 종목당 한 줄
    assert "-17.4%" in out
    assert "종가" in out and "장마감" in out    # 가격 신선도
    assert "검증부족(n=1)" in out             # 정직 라벨
    assert "드로다운" in out                  # 심리 가드
    assert "base:" in out                    # 프레이밍


def test_render_single_message_under_limit():
    out = render_brief(_sample())
    assert len(out) <= 1900                  # 텔레그램/디스코드 단일 메시지
