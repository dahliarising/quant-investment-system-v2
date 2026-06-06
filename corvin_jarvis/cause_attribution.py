"""Phase 2 — 촉발원인 규명.

"오늘 왜 빠졌나"를 멀티소스 당일 이상치로 추정 → 패닉 vs 추세 판단 근거.
순수 코어(rank/message) + DI 리더(금리·섹터분해·환율/유가·뉴스/내러티브).
이상치 없으면 스킵(추측 금지). 향후 geo_signal·narrative 어댑터로 리더 바인딩.
"""
from __future__ import annotations

from typing import Callable


def gather_candidates(readers: dict[str, Callable[[], dict | None]]) -> list[dict]:
    """각 소스 리더 호출(예외/None은 스킵) → 이상치 후보 리스트."""
    out = []
    for _, read in readers.items():
        try:
            c = read()
        except Exception:                 # noqa: BLE001 — 소스 격리
            c = None
        if c:
            out.append(c)
    return out


def rank_causes(candidates: list[dict]) -> list[dict]:
    """이상치 크기(|magnitude|) 내림차순."""
    return sorted(candidates, key=lambda c: abs(c.get("magnitude", 0.0)), reverse=True)


def attribution_message(ranked: list[dict], top_n: int = 3) -> str:
    """top-N 원인을 한 줄로 — '오늘 하락 원인(추정): ① ... ② ...'."""
    if not ranked:
        return "오늘 하락 원인 불명 — 데이터 이상치 없음"
    nums = "①②③④⑤"
    parts = []
    for i, c in enumerate(ranked[:top_n]):
        detail = f" ({c['detail']})" if c.get("detail") else ""
        parts.append(f"{nums[i]} {c['factor']}{detail}")
    return "오늘 하락 원인(추정): " + " ".join(parts)
