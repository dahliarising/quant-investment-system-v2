"""Brief → 스캔 가능한 단일 메시지 텍스트 (구분선·신선도 라벨)."""
from __future__ import annotations

from corvin_jarvis.brief.types import Brief

_DIV = "━━━━━━━━━━━━━━"


def _pnl(pl) -> str:
    return f"{pl.pnl_pct:+.1f}%" if pl.pnl_pct is not None else "n/a"


def render_brief(brief: Brief) -> str:
    out: list[str] = [brief.headline,
                      f"_{brief.as_of} · {brief.market_state} 기준_", ""]
    out.append("📊 **포지션**")
    for pl in brief.positions:
        ev = brief.evidence.get(pl.symbol)
        tag = f"  ·{ev.validation}" if ev else ""
        out.append(f"{pl.action_icon} {pl.symbol} {_pnl(pl)} ({pl.fresh_label}) — {pl.label}{tag}")
    if brief.framing:
        out += [_DIV, "🧭 **다각 프레이밍**",
                f"🔼 {brief.framing.bull}", f"🔽 {brief.framing.bear}",
                f"• {brief.framing.base}", f"⚠️ {brief.framing.counterfactual}"]
    if brief.psych and brief.psych.triggered:
        out += [_DIV, f"🧠 **심리 체크** ({brief.psych.pattern})", brief.psych.question or ""]
    text = "\n".join(out)
    return text[:1900]
