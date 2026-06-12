"""보유분 평단대비 손익 → 액션라벨 오버레이. 데이터: pulse.fetch_portfolio()."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.types import PositionLine


def _classify(pnl_pct: float | None, bucket: str) -> tuple[str, str]:
    if pnl_pct is None:
        return "👀", "시세없음"
    if bucket == "dca" and pnl_pct <= -12:
        return "✂️", "손절존 검토"
    if pnl_pct >= 8:
        return "✅", "유지(추세양호)"
    if pnl_pct >= -3:
        return "✅", "유지"
    if pnl_pct > -12:
        return "👀", "관망존"
    return "✂️", "비중축소 검토"


def build_position_lines(positions: list[dict[str, Any]],
                         fresh_label: str) -> list[PositionLine]:
    lines: list[PositionLine] = []
    for p in positions:
        pnl = p.get("pnl_pct")
        bucket = str(p.get("bucket") or "trade")
        icon, label = _classify(pnl, bucket)
        lines.append(PositionLine(
            symbol=str(p.get("symbol", "")),
            pnl_pct=pnl,
            action_icon=icon,
            label=label,
            fresh_label=fresh_label,
        ))
    return lines
