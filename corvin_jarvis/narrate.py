"""Corvin Jarvis — Narrative Continuity (Phase 6)

llm-wiki/corvin-sessions/의 최근 전략 md를 파싱해서 'continuity context blob'을 생성.
LLM 없이 structured extraction만 수행 — Claude Code session이 이 blob을 읽어
narrative를 합성하거나, 별도 Claude API 호출에 prompt로 사용.

사용:
    python corvin_jarvis/narrate.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
LATEST_FILE = STATE_DIR / "latest.json"
ALERTS_FILE = STATE_DIR / "alerts.json"
WIKI_DIR = Path("/Users/thethethe/Claude/llm-wiki/wiki/corvin-sessions")
CONTEXT_FILE = STATE_DIR / "continuity_context.json"
LOG_FILE = STATE_DIR / "narrate.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("corvin.narrate")


@dataclass
class SessionExtract:
    file: str
    created: str | None
    tags: list[str] = field(default_factory=list)
    confidence: str | None = None
    title: str | None = None
    tldr: str | None = None
    tickers_mentioned: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ContinuityContext:
    generated_at: str
    sessions: list[dict[str, Any]]
    open_themes: list[str]
    portfolio_alignment: dict[str, Any]


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TICKER_RE = re.compile(r"\b([A-Z]{1,5}|\d{6}(?:\.KS|\.KQ)?)\b")
TICKER_BLACKLIST = {
    "TLDR", "PnL", "USD", "KRW", "KST", "UTC", "API", "LLM", "MCP", "JSON",
    "ETF", "VIX", "GDP", "CPI", "FX", "OK", "URL", "MD", "RPG", "HTML",
    "AI", "ML", "DL", "RC", "PM", "ID", "PC", "GG", "OS", "IT", "TBD",
}

REC_PATTERNS = [
    re.compile(r"매수|buy\s|진입"),
    re.compile(r"매도|sell\s|청산|차익실현"),
    re.compile(r"손절|stop\s*loss|stop-loss"),
    re.compile(r"비중\s*확대|overweight|underweight"),
    re.compile(r"TAKE\s*PROFIT|TP\s"),
]


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    fm: dict[str, Any] = {}
    for line in raw.split("\n"):
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip().strip("[]")
        if "," in v:
            fm[k] = [x.strip() for x in v.split(",")]
        else:
            fm[k] = v
    body = text[m.end():]
    return fm, body


def _extract_title(body: str) -> str | None:
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _extract_tldr(body: str) -> str | None:
    m = re.search(r"##\s+TL;DR\s*\n+(.+?)(?=\n##|\Z)", body, re.DOTALL)
    if not m:
        return None
    tldr = m.group(1).strip()
    return tldr[:500]


def _extract_tickers(body: str) -> list[str]:
    found: set[str] = set()
    for m in TICKER_RE.finditer(body):
        sym = m.group(1)
        if sym in TICKER_BLACKLIST:
            continue
        if sym.isdigit() and len(sym) != 6:
            continue
        found.add(sym)
    return sorted(found)


def _extract_recommendations(body: str) -> list[str]:
    recs: list[str] = []
    for line in body.split("\n"):
        s = line.strip()
        if len(s) < 15 or len(s) > 250:
            continue
        for pat in REC_PATTERNS:
            if pat.search(s):
                cleaned = re.sub(r"^[\-\*•\s\d\.\)]+", "", s)
                if cleaned and cleaned not in recs:
                    recs.append(cleaned)
                break
    return recs[:10]


def extract_session(path: Path) -> SessionExtract:
    text = path.read_text()
    fm, body = _parse_frontmatter(text)
    return SessionExtract(
        file=path.name,
        created=fm.get("created") if isinstance(fm.get("created"), str) else None,
        tags=fm.get("tags", []) if isinstance(fm.get("tags"), list) else [],
        confidence=fm.get("confidence") if isinstance(fm.get("confidence"), str) else None,
        title=_extract_title(body),
        tldr=_extract_tldr(body),
        tickers_mentioned=_extract_tickers(body),
        recommendations=_extract_recommendations(body),
    )


def _open_themes(sessions: list[SessionExtract]) -> list[str]:
    """최근 세션들의 공통 tag를 빈도 기반으로 추출."""
    tag_count: dict[str, int] = {}
    for s in sessions:
        for tag in s.tags:
            tag_count[tag] = tag_count.get(tag, 0) + 1
    return [tag for tag, _ in sorted(tag_count.items(), key=lambda x: -x[1])[:8]]


def _portfolio_alignment(sessions: list[SessionExtract]) -> dict[str, Any]:
    """최근 세션 ticker 언급과 현재 portfolio 보유 종목의 교집합/차집합."""
    if not LATEST_FILE.exists():
        return {}
    latest = json.loads(LATEST_FILE.read_text())
    holdings = {p["symbol"] for p in latest.get("portfolio", [])}

    mentioned: set[str] = set()
    for s in sessions:
        mentioned.update(s.tickers_mentioned)

    aligned = holdings & mentioned
    held_not_mentioned = holdings - mentioned
    mentioned_not_held = mentioned - holdings

    return {
        "held_and_discussed": sorted(aligned),
        "held_but_not_in_recent_strategy": sorted(held_not_mentioned),
        "mentioned_but_not_held": sorted(mentioned_not_held)[:15],
    }


def build_context(limit: int = 5) -> ContinuityContext:
    if not WIKI_DIR.exists():
        log.warning("wiki dir 없음: %s", WIKI_DIR)
        return ContinuityContext(
            generated_at=datetime.now().isoformat(),
            sessions=[], open_themes=[], portfolio_alignment={},
        )

    files = sorted(WIKI_DIR.glob("*.md"), reverse=True)[:limit]
    extracts = [extract_session(f) for f in files]

    ctx = ContinuityContext(
        generated_at=datetime.now().isoformat(),
        sessions=[asdict(s) for s in extracts],
        open_themes=_open_themes(extracts),
        portfolio_alignment=_portfolio_alignment(extracts),
    )

    CONTEXT_FILE.write_text(json.dumps(asdict(ctx), indent=2, ensure_ascii=False, default=str))
    log.info("Continuity context 저장 — %d 세션, %d 테마", len(extracts), len(ctx.open_themes))
    return ctx


def _print_summary(ctx: ContinuityContext) -> None:
    log.info("=== Continuity Summary ===")
    log.info("Open themes: %s", ", ".join(ctx.open_themes) or "(none)")
    pa = ctx.portfolio_alignment
    log.info("보유 & 최근 전략 일관: %s", pa.get("held_and_discussed", []))
    log.info("보유 but 최근 전략 미언급: %s", pa.get("held_but_not_in_recent_strategy", []))
    log.info("최근 전략 언급 but 미보유: %s", pa.get("mentioned_but_not_held", []))
    for s in ctx.sessions[:3]:
        log.info("  - %s | %s | recs=%d", s["file"], s.get("title", "")[:60], len(s.get("recommendations", [])))


if __name__ == "__main__":
    ctx = build_context()
    _print_summary(ctx)
