"""Phase C — fact-check 패스.

에이전트 발언에서 (종목, 수치) 주장을 뽑아 검증 컨텍스트와 대조.
일치=verified, 어긋남=mismatch, 컨텍스트에 없음=unverified.
환각 수치가 토론에 섞여도 보는 사람이 즉시 구분하게 한다.
"""
from __future__ import annotations

import re

# 종목(미국 티커 2~6자 대문자 / 한국 6자리) + 부호 퍼센트
_CLAIM_RE = re.compile(r"\b([A-Z]{2,6}|\d{6})\b[^%\d]{0,12}?([+\-−]\s?\d+(?:\.\d+)?)\s*%")


def _to_float(token: str) -> float:
    return float(token.replace("−", "-").replace(" ", ""))


def extract_claims(text: str) -> list[dict]:
    """발언에서 (symbol, pnl_pct) 주장 추출."""
    out = []
    for sym, pct in _CLAIM_RE.findall(text):
        out.append({"symbol": sym, "kind": "pnl_pct", "value": _to_float(pct)})
    return out


def verify_turn(text: str, context_facts: set, tol: float = 0.5) -> dict:
    """발언 주장을 컨텍스트 사실과 대조 → {ok, flags}."""
    # symbol → 허용 live_pnl_pct 값
    allowed = {(s, v) for (s, kind, v) in context_facts if kind == "live_pnl_pct"}
    allowed_syms = {s for (s, _) in allowed}
    flags = []
    for c in extract_claims(text):
        sym, val = c["symbol"], c["value"]
        if sym not in allowed_syms:
            flags.append({"kind": "unverified", "symbol": sym, "value": val})
            continue
        if any(abs(val - av) <= tol for (s, av) in allowed if s == sym):
            continue
        flags.append({"kind": "mismatch", "symbol": sym, "value": val})
    return {"ok": not flags, "flags": flags}


def annotate(turn: dict, verdict: dict) -> dict:
    """턴에 fact-check 배지 부착 (UI 표시용)."""
    if verdict["ok"]:
        badge = "verified"
    elif any(f["kind"] == "mismatch" for f in verdict["flags"]):
        badge = "mismatch"
    else:
        badge = "unverified"
    return {**turn, "badge": badge, "flags": verdict["flags"]}
