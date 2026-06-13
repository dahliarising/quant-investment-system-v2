# corvin_jarvis/prediction/fmt.py
"""공유 표기 유틸 — KR 원화 대형값 천단위 콤마, US 소형값 소수 보존.
:g가 1000000을 '1e+06'으로 출력하던 오해 유발 방지."""
from __future__ import annotations


def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.0f}"
    return f"{p:.2f}".rstrip("0").rstrip(".")
