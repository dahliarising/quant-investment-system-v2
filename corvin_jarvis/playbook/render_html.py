"""Playbook → HTML 대시보드 문자열."""
from __future__ import annotations

import html

from corvin_jarvis.playbook.models import Playbook

_ORDER = {"BUY_NOW": 0, "TRIM_NOW": 1, "INVALID": 2, "WAIT": 3}

_STYLE = """
body{background:#0a0612;color:#eee;font-family:system-ui,sans-serif;margin:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{background:#160f24;border:1px solid #2a1f40;border-radius:10px;padding:12px}
.sym{font-weight:700;font-size:18px}
.bar{height:14px;border-radius:7px;background:linear-gradient(90deg,#1d6b3a,#6b1d2a);position:relative;margin:8px 0}
.mark{position:absolute;top:-3px;width:3px;height:20px;background:#fff}
.muted{color:#9a90b0;font-size:12px}
"""


def _pct_pos(pb: Playbook) -> float:
    lo = min(z.low for z in pb.zones)
    hi = max(z.high for z in pb.zones)
    if hi <= lo:
        return 50.0
    return max(0.0, min(100.0, (pb.tech.price - lo) / (hi - lo) * 100.0))


def _card(pb: Playbook) -> str:
    t = pb.tech
    pnl = f" {pb.pnl_pct:+.1f}%" if pb.pnl_pct is not None else ""
    nxt = pb.active_zone.note if pb.active_zone else "존 대기"
    return (
        f'<div class="card"><div class="sym">{html.escape(pb.name)} '
        f'{pb.badge}</div>'
        f'<div class="muted">{t.symbol} · {t.price:g} · RSI {t.rsi:.0f}{pnl}</div>'
        f'<div class="bar"><div class="mark" style="left:{_pct_pos(pb):.0f}%">'
        f'</div></div>'
        f'<div class="muted">{pb.stance} → {html.escape(nxt)}</div></div>'
    )


def render_dashboard(playbooks: list[Playbook]) -> str:
    ordered = sorted(playbooks, key=lambda p: _ORDER.get(p.status, 9))
    cards = "\n".join(_card(p) for p in ordered)
    return (
        "<!DOCTYPE html>\n<html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Corvin 플레이북</title><style>{_STYLE}</style></head>"
        f"<body><h2>📊 Corvin 플레이북</h2><div class='grid'>{cards}</div>"
        "</body></html>"
    )
