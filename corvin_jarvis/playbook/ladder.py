"""고정룰 래더 — 기술지표 → 존(가격밴드+비율). 순수."""
from __future__ import annotations

from corvin_jarvis.playbook.models import Technicals, Zone

BAND = 0.015
BUY_RATIOS = (30, 40, 30)
TRIM_RATIOS = (33, 33, 34)


def _band(level: float) -> tuple[float, float]:
    return level * (1 - BAND), level * (1 + BAND)


def build_buy_ladder(tech: Technicals) -> tuple[Zone, ...]:
    z1_lo, z1_hi = _band(tech.ma20)
    z2_lo, z2_hi = _band(tech.ma50)
    deep = tech.ma50 * 0.95
    z3_lo, z3_hi = _band(deep)
    return (
        Zone("buy", "Z1 1차눌림", BUY_RATIOS[0], z1_lo, z1_hi, "MA20"),
        Zone("buy", "Z2 핵심지지", BUY_RATIOS[1], z2_lo, z2_hi, "MA50"),
        Zone("buy", "Z3 딥밸류", BUY_RATIOS[2], z3_lo, z3_hi, "MA50-5%/과매도"),
    )


def build_trim_ladder(tech: Technicals) -> tuple[Zone, ...]:
    hot = max(tech.ma20 * 1.10, tech.price if tech.rsi >= 70 else 0.0)
    z1_lo, z1_hi = _band(hot)
    z2_lo, z2_hi = _band(tech.hi_52w)
    return (
        Zone("trim", "Z1 과열", TRIM_RATIOS[0], z1_lo, z1_hi, "RSI≥70/MA20+10%"),
        Zone("trim", "Z2 전고", TRIM_RATIOS[1], z2_lo, z2_hi, "52주고"),
        Zone("trim", "Z3 코어홀드", TRIM_RATIOS[2], tech.ma50, tech.hi_52w,
             "MA50 이탈청산"),
    )
