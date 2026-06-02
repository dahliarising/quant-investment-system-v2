"""Playbook → 매일 텍스트 푸시 문자열."""
from __future__ import annotations

from corvin_jarvis.playbook.models import Playbook

_TRIGGER = {"BUY_NOW", "TRIM_NOW"}


def _ccy(market: str) -> str:
    return "₩" if market == "KR" else "$"


def _action_line(pb: Playbook) -> str:
    z = pb.active_zone
    ratio = f"{z.ratio}%" if z else ""
    zlabel = z.label if z else ""
    c = _ccy(pb.tech.market)
    return f"{pb.badge} {pb.name} {zlabel} {c}{pb.tech.price:g} → {ratio}"


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
        chips = " ".join(f"{p.symbol}{p.badge}" for p in held)
        lines.append(f"💼 보유{len(held)}: {chips}")

    watch_trig = sum(1 for p in watch if p.status in _TRIGGER)
    lines.append(
        f"👀 관찰{len(watch)}·트리거{watch_trig}·나머지⏳ → 🔗대시보드"
    )
    return "\n".join(lines)
