# corvin_jarvis/prediction/digest_assembler.py
"""PredictionResult 리스트 → 텔레그램 Markdown 다이제스트.

스캔가능 포맷(memory): 구분선·여백·상태아이콘·종목당 한 줄.
헤더에 '장초반 스냅샷' 라벨로 미완성봉 한계 명시(memory: 가격 시점 라벨링).
"""
from __future__ import annotations

from corvin_jarvis.prediction.contract import PredictionResult

_DIV = "━━━━━━━━━━━━"
_MARKET = {"vector_analog", "geopolitical", "momentum"}


def _line(r: PredictionResult) -> str:
    if not r.data_ok:
        return f"⏸ *{r.scope}* — {r.verdict}"
    return f"• *{r.scope}* — {r.verdict} _(신뢰 {r.confidence:.0f})_"


def assemble(results: list[PredictionResult], date_str: str) -> str:
    if not results:
        return f"📅 {date_str} 장초반 스냅샷\n\n예측 결과 없음 (데이터/모듈 점검 필요)"
    market = [r for r in results if r.scope == "market"]
    holdings = [r for r in results if r.scope != "market"]
    parts = [f"📅 *{date_str} 장초반 스냅샷* (09:36 KST · 미완성봉)", _DIV]
    parts.append("📊 *시장 방향*")
    parts += [_line(r) for r in market] or ["• (없음)"]
    parts.append("")
    parts.append(_DIV)
    parts.append("📈 *종목 예측 (보유·감시)*")
    parts += [_line(r) for r in holdings] or ["• (없음)"]
    parts.append(_DIV)
    parts.append("_⚠️ 데이터 기반 advisory · 실매매 판단은 본인 책임_")
    return "\n".join(parts)
