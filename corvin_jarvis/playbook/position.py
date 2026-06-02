"""stance 결정 + 현재가 → 존/상태 분류. 순수."""
from __future__ import annotations

from corvin_jarvis.playbook.models import Technicals, Zone

HARVEST_PNL = 5.0

_BADGE = {"BUY_NOW": "🟢", "TRIM_NOW": "✂️", "WAIT": "⏳", "INVALID": "⚠️"}


def stance(pnl_pct: float | None) -> str:
    if pnl_pct is None:
        return "ENTER"
    if pnl_pct > HARVEST_PNL:
        return "HARVEST"
    return "ACCUMULATE"


def _in_zone(price: float, z: Zone) -> bool:
    return z.low <= price <= z.high


def classify(
    tech: Technicals, zones: tuple[Zone, ...], stance_val: str
) -> tuple[str, str, Zone | None]:
    price = tech.price
    if stance_val == "HARVEST":
        if price < tech.ma50:
            return "INVALID", _BADGE["INVALID"], None
        for z in zones:
            if _in_zone(price, z) or (z.label.startswith("Z1") and tech.rsi >= 70):
                return "TRIM_NOW", _BADGE["TRIM_NOW"], z
        return "WAIT", _BADGE["WAIT"], None
    # ENTER / ACCUMULATE → buy ladder
    deepest = zones[-1]
    if price <= deepest.high:
        active = next((z for z in zones if _in_zone(price, z)), deepest)
        return "BUY_NOW", _BADGE["BUY_NOW"], active
    for z in zones:
        if _in_zone(price, z):
            return "BUY_NOW", _BADGE["BUY_NOW"], z
    return "WAIT", _BADGE["WAIT"], None
