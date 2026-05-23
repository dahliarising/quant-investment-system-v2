"""Corvin Jarvis — Counterfactual Coach (Tier 4.3)

wiki sessions 안의 종목 prediction → 실제 결과 비교 → 자기 calibration.

직접 LLM 평가 안 함 (cost). 키워드 + 가격 변동으로 directional verdict만 산출.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from corvin_jarvis import attribution, timeseries

log = logging.getLogger("corvin.counterfactual")

# direction keyword 사전
_BULLISH_PATTERNS = [
    r"매수", r"매입", r"분할매수", r"신규진입", r"신규 진입",
    r"buy", r"add", r"long", r"진입", r"강세", r"상승",
    r"돌파", r"breakout", r"매집",
]
_BEARISH_PATTERNS = [
    r"매도", r"매각", r"분할매도", r"손절", r"청산",
    r"sell", r"short", r"줄이[기다]", r"하락", r"이탈", r"trim",
]
_NEUTRAL_PATTERNS = [
    r"관망", r"보유", r"홀드", r"hold", r"유지", r"대기", r"wait",
]

_BULL_RE = re.compile("|".join(_BULLISH_PATTERNS), re.IGNORECASE)
_BEAR_RE = re.compile("|".join(_BEARISH_PATTERNS), re.IGNORECASE)
_NEUTRAL_RE = re.compile("|".join(_NEUTRAL_PATTERNS), re.IGNORECASE)

_FLAT_THRESHOLD_PCT = 0.5


_CLAUSE_BOUNDARY = re.compile(r"[.。\n;]+")


def extract_predictions(text: str) -> list[dict[str, Any]]:
    """text를 clause(문장)로 split → 각 clause의 symbol + direction keyword 매칭."""
    seen: dict[str, dict[str, Any]] = {}
    for clause in _CLAUSE_BOUNDARY.split(text):
        clause = clause.strip()
        if not clause:
            continue
        clause_symbols = attribution.session_symbols(clause)
        if not clause_symbols:
            continue
        bull = _BULL_RE.search(clause) is not None
        bear = _BEAR_RE.search(clause) is not None
        neutral = _NEUTRAL_RE.search(clause) is not None
        if bull and not bear:
            direction = "bullish"
        elif bear and not bull:
            direction = "bearish"
        elif neutral:
            direction = "neutral"
        else:
            continue
        for sym in clause_symbols:
            if sym not in seen:
                seen[sym] = {"symbol": sym, "direction": direction, "context": clause}
    return list(seen.values())


def score_prediction(
    direction: str,
    actual_pct: float | None,
    flat_threshold: float = _FLAT_THRESHOLD_PCT,
) -> str:
    """direction(bullish/bearish/neutral) + actual move → correct/incorrect/inconclusive."""
    if actual_pct is None:
        return "inconclusive"
    if direction == "neutral":
        return "inconclusive"
    if abs(actual_pct) < flat_threshold:
        return "inconclusive"
    if direction == "bullish":
        return "correct" if actual_pct > 0 else "incorrect"
    if direction == "bearish":
        return "correct" if actual_pct < 0 else "incorrect"
    return "inconclusive"


def monthly_review(
    wiki_dir: Path,
    db_path: Path,
    year: int,
    month: int,
) -> dict[str, Any]:
    """주어진 month의 wiki sessions 모음 → predictions 추출 → 실제 비교 → calibration report."""
    if not wiki_dir.exists():
        return _empty_report(year, month)

    target_prefix = f"{year:04d}-{month:02d}-"
    entries: list[dict[str, Any]] = []
    total = 0
    correct = 0
    incorrect = 0
    inconclusive = 0

    for md in sorted(wiki_dir.glob(f"{target_prefix}*.md")):
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})-", md.name)
        if not m:
            continue
        session_date = date(int(m[1]), int(m[2]), int(m[3]))
        try:
            text = md.read_text(errors="ignore")
        except OSError:
            continue
        preds = extract_predictions(text)
        if not preds:
            continue
        moves = attribution.actual_moves(
            [p["symbol"] for p in preds], db_path, days_back=14,
        )
        moves_by_sym = {m["symbol"]: m["pct_change"] for m in moves}
        scored: list[dict[str, Any]] = []
        for p in preds:
            actual = moves_by_sym.get(p["symbol"])
            verdict = score_prediction(p["direction"], actual)
            scored.append({**p, "actual_pct": actual, "verdict": verdict})
            total += 1
            if verdict == "correct":
                correct += 1
            elif verdict == "incorrect":
                incorrect += 1
            else:
                inconclusive += 1
        entries.append({
            "file": md.name,
            "date": session_date.isoformat(),
            "predictions": scored,
        })

    decisive = correct + incorrect
    accuracy = (correct / decisive * 100) if decisive > 0 else None
    return {
        "year": year,
        "month": month,
        "total_predictions": total,
        "correct": correct,
        "incorrect": incorrect,
        "inconclusive": inconclusive,
        "accuracy_pct": round(accuracy, 2) if accuracy is not None else None,
        "entries": entries,
    }


def _empty_report(year: int, month: int) -> dict[str, Any]:
    return {
        "year": year, "month": month,
        "total_predictions": 0, "correct": 0, "incorrect": 0, "inconclusive": 0,
        "accuracy_pct": None, "entries": [],
    }
