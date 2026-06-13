# corvin_jarvis/prediction/digest_assembler.py
"""PredictionResult 리스트 → 텔레그램 Markdown 다이제스트.

스캔가능 포맷(memory): 구분선·여백·상태아이콘·종목당 한 줄.
헤더에 '장초반 스냅샷' 라벨로 미완성봉 한계 명시(memory: 가격 시점 라벨링).

종목당 한 줄 원칙: 한 종목이 여러 시스템 결과를 가지면 한 줄로 병합한다.
data_ok 실신호가 하나라도 있으면 보류 라인은 덮어쓰고, 보류만 있으면 한 줄로 dedup.
유니버스(보유 밖) 모멘텀은 개별 나열하지 않고 카운트로 요약한다(노이즈 회피 memory).
"""
from __future__ import annotations

from collections import OrderedDict

from corvin_jarvis.prediction.contract import PredictionResult

_DIV = "━━━━━━━━━━━━"
# 시스템 우선순위 — 긴급/액션 신호가 줄 앞에 오도록
_ORDER = {"velocity": 0, "probability": 1, "momentum": 2, "vector_analog": 3}


def _line(r: PredictionResult) -> str:
    """market 섹션용 단일 결과 렌더."""
    if not r.data_ok:
        return f"⏸ *{r.scope}* — {r.verdict}"
    return f"• *{r.scope}* — {r.verdict} _(신뢰 {r.confidence:.0f})_"


def _short(r: PredictionResult) -> str:
    """종목 라인 병합용 압축 표현 (시스템별)."""
    if r.system == "momentum":
        sig = r.evidence.get("signal")
        arrow = {"bullish": "추세↑", "bearish": "추세↓"}.get(sig, "추세→")
        gap = r.evidence.get("gap_pct")
        # gap이 0으로 반올림되면 '-0%' 오해 방지 — 화살표만
        if isinstance(gap, (int, float)) and round(gap) != 0:
            return f"{arrow} {gap:+.0f}%"
        return arrow
    if r.system == "probability":
        p = r.evidence.get("prob_below_stop")
        if isinstance(p, (int, float)):
            return f"손절이탈 {p * 100:.0f}%"
    return r.verdict


def _consolidate(scope: str, rs: list[PredictionResult]) -> str:
    """한 종목의 여러 결과 → 한 줄. 실신호 우선, 없으면 보류 한 줄."""
    real = sorted((r for r in rs if r.data_ok),
                  key=lambda r: _ORDER.get(r.system, 9))
    if real:
        body = " · ".join(_short(r) for r in real)
        conf = max(r.confidence for r in real)
        return f"• *{scope}* — {body} _(신뢰 {conf:.0f})_"
    # 보류만 — 첫 사유 한 줄 (중복 제거)
    return f"⏸ *{scope}* — {rs[0].verdict}"


def _universe_summary(rs: list[PredictionResult]) -> list[str]:
    """유니버스(보유 밖) 모멘텀 data_ok 결과 → 카운트 + |gap| 상위 주목 종목."""
    up = sum(1 for r in rs if r.evidence.get("signal") == "bullish")
    down = sum(1 for r in rs if r.evidence.get("signal") == "bearish")
    neu = len(rs) - up - down
    lines = [f"👀 *유니버스 모멘텀* ({len(rs)}종목): ↑{up} · ↓{down} · →{neu}"]
    notable = sorted(rs, key=lambda r: abs(r.evidence.get("gap_pct", 0) or 0),
                     reverse=True)[:3]
    if notable:
        tags = ", ".join(f"{r.scope}({_short(r)})" for r in notable)
        lines.append(f"  주목: {tags}")
    return lines


def assemble(results: list[PredictionResult], date_str: str,
             holding_symbols: set[str] | None = None) -> str:
    """results → 다이제스트.

    holding_symbols 미지정 시 모든 종목 결과를 보유 섹션에 표기(하위호환).
    지정 시 보유∈set은 인라인 병합, 그 외 종목 모멘텀은 유니버스 요약으로 분리.
    """
    if not results:
        return f"📅 {date_str} 장초반 스냅샷\n\n예측 결과 없음 (데이터/모듈 점검 필요)"

    market = [r for r in results if r.scope == "market"]
    nonmarket = [r for r in results if r.scope != "market"]

    if holding_symbols is None:
        holding_results, universe_results = nonmarket, []
    else:
        holding_results = [r for r in nonmarket if r.scope in holding_symbols]
        universe_results = [r for r in nonmarket if r.scope not in holding_symbols]

    groups: "OrderedDict[str, list[PredictionResult]]" = OrderedDict()
    for r in holding_results:
        groups.setdefault(r.scope, []).append(r)

    parts = [f"📅 *{date_str} 장초반 스냅샷* (09:36 KST · 미완성봉)", _DIV]
    parts.append("📊 *시장 방향*")
    parts += [_line(r) for r in market] or ["• (없음)"]
    parts.append("")
    parts.append(_DIV)
    parts.append("📈 *종목 예측 (보유·감시)*")
    if groups:
        parts += [_consolidate(scope, rs) for scope, rs in groups.items()]
    else:
        parts.append("• (없음)")

    uni_ok = [r for r in universe_results if r.data_ok]
    if uni_ok:
        parts.append("")
        parts += _universe_summary(uni_ok)

    parts.append(_DIV)
    parts.append("_⚠️ 데이터 기반 advisory · 실매매 판단은 본인 책임_")
    return "\n".join(parts)
