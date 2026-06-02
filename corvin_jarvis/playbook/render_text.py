"""Playbook → 매일 텍스트 푸시 문자열."""
from __future__ import annotations

from corvin_jarvis.playbook.models import Playbook

_TRIGGER = {"BUY_NOW", "TRIM_NOW"}


def _ccy(market: str) -> str:
    return "₩" if market == "KR" else "$"


def fmt_price(market: str, price: float) -> str:
    """KR=콤마 정수(지수표기 방지), US=소수 2자리."""
    if market == "KR":
        return f"₩{price:,.0f}"
    return f"${price:,.2f}"


def _action_line(pb: Playbook) -> str:
    z = pb.active_zone
    ratio = f"{z.ratio}%" if z else ""
    zlabel = z.label if z else ""
    return f"{pb.badge} {pb.name} ({pb.symbol}) {zlabel} {fmt_price(pb.tech.market, pb.tech.price)} → {ratio}"


def _held_line(pb: Playbook) -> str:
    pnl = f" {pb.pnl_pct:+.1f}%" if pb.pnl_pct is not None else ""
    return f"{pb.badge} {pb.symbol} {fmt_price(pb.tech.market, pb.tech.price)}{pnl}"


def render_push(playbooks: list[Playbook], date_label: str) -> str:
    triggers = [p for p in playbooks if p.status in _TRIGGER]
    held = [p for p in playbooks if p.stance in ("ACCUMULATE", "HARVEST")]
    watch = [p for p in playbooks if p.stance == "ENTER"]

    lines = [f"📊 플레이북 · {date_label} 장마감"]
    lines.append(f"🔔 오늘 액션 ({len(triggers)})")
    if triggers:
        lines += [_action_line(p) for p in triggers]
    else:
        lines.append("— 없음 (전 종목 존 대기)")

    if held:
        lines.append(f"💼 보유 ({len(held)})")
        lines += [_held_line(p) for p in held]

    watch_trig = sum(1 for p in watch if p.status in _TRIGGER)
    lines.append(
        f"👀 관찰{len(watch)}·트리거{watch_trig}·나머지⏳ → 🔗대시보드"
    )
    return "\n".join(lines)
