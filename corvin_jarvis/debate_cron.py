"""선제 토론 크론 — 폐하가 물을 법한 질문에 미리 토론 → Telegram 다이제스트.

매일 1회(주식 크론 정렬). 각 질문에 5인 투자대가가 답하고, 신뢰도 높은 의견
순으로 다이제스트. cron alert은 Telegram 달리아봇으로만(메모리 규칙).

순수 코어(run_anticipatory·format_digest)는 LLM=DI → 테스트는 실발송 없음.
"""
from __future__ import annotations

from typing import Callable

from corvin_jarvis.live_debate import engine

# 이 세션에서 폐하가 실제로 자주 물은 질문들
LIKELY_QUESTIONS = [
    "지금 전반적으로 매수·매도·홀딩 중 뭐가 맞아?",
    "전량 매도하고 다른 걸로 리밸런싱 해야 하나?",
    "손절선 깨진 종목(예: 하드스톱 돌파)은 지금 정리할까?",
    "오늘 시장이 왜 빠졌고, 패닉인가 추세 전환인가?",
    "지금 신규 진입(반도체 등)하기 좋은 자리인가?",
]


def run_anticipatory(questions: list[str], context: dict,
                     llm: Callable[[str], str], mode: str = "style") -> list[dict]:
    """각 질문에 페르소나들이 답함 → 질문별 답변(신뢰도 내림차순)."""
    out = []
    for q in questions:
        replies = list(engine.respond_to_user(mode, context, q, llm))
        replies.sort(key=lambda r: r.get("credibility", 0), reverse=True)
        out.append({"question": q, "replies": replies})
    return out


def format_digest(results: list[dict], top_n: int = 3) -> str:
    """Telegram 다이제스트 — 질문별 상위 신뢰도 의견. ≤4000자."""
    lines = ["🐦‍⬛ *Corvin 선제 토론 다이제스트*", "_물어보실 법한 질문에 미리 토론했어요_", ""]
    for r in results:
        lines.append(f"❓ *{r['question']}*")
        for rep in r["replies"][:top_n]:
            txt = rep["text"].replace("\n", " ").strip()
            if len(txt) > 110:
                txt = txt[:108] + "…"
            who = f"{rep['name']}({rep['tag']})" if rep.get("tag") else rep["name"]
            lines.append(f"• {who} 🎯{rep.get('credibility', '?')}: {txt}")
        lines.append("")
    lines.append("_advisory only · 신뢰도=track record · 자세히는 /debate_")
    digest = "\n".join(lines)
    return digest[:3990]


def run(context: dict, send: Callable[[str], bool] | None = None,
        llm: Callable[[str], str] | None = None,
        questions: list[str] | None = None) -> str:
    """라이브 실행 — 토론 → 다이제스트 → Telegram 발송. send/llm 미지정 시 실연결."""
    if llm is None:
        from corvin_jarvis import qualitative
        llm = lambda p: qualitative.run_claude_cli(p, timeout=70) or "(무응답)"
    if send is None:
        from corvin_jarvis import channels
        send = channels.send_telegram
    results = run_anticipatory(questions or LIKELY_QUESTIONS, context, llm)
    digest = format_digest(results)
    send(digest)
    return digest


def _live_context() -> dict:
    """live_debate 서버의 라이브 컨텍스트(KIS 스냅샷 + 원인규명) 재사용."""
    from corvin_jarvis.live_debate import server
    return server._live_context()


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    ctx = _live_context()
    if dry:
        out = run(ctx, send=lambda body: (print(body), True)[1])
    else:
        out = run(ctx)   # 라이브 — Telegram 발송
    print(f"\n[debate_cron] {'DRY' if dry else 'SENT'} · {len(out)}자")
