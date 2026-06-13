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


def candidates_from_snapshot(snapshot: dict, sector_thresh: float = -2.0,
                             vix_thresh: float = 15.0) -> list[dict]:
    """jarvis 스냅샷에서 원인 후보 추출 (순수, 무네트워크 — 이미 수집한 데이터 재활용).

    섹터별 평균 등락(어디서 시작됐나) + VIX 급등(변동성). 임계 미달은 제외.
    """
    out = []

    # 섹터 분해: universe를 섹터별 평균 등락 → 가장 약한 섹터
    by_sector: dict[str, list[float]] = {}
    for e in snapshot.get("universe", []):
        pct = e.get("pct_change")
        sec = e.get("sector")
        if pct is not None and sec:
            by_sector.setdefault(sec, []).append(pct)
    for sec, pcts in by_sector.items():
        avg = sum(pcts) / len(pcts)
        if avg <= sector_thresh:
            out.append({"factor": f"{sec} 약세", "magnitude": avg,
                        "detail": f"{sec} 평균 {avg:+.1f}%"})

    # VIX 급등 = 변동성 충격
    vix = snapshot.get("indices", {}).get("vix", {})
    vchg = vix.get("pct_change")
    if vchg is not None and vchg >= vix_thresh:
        out.append({"factor": "변동성 급등", "magnitude": vchg / 10,
                    "detail": f"VIX {vchg:+.0f}% ({vix.get('price')})"})

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
