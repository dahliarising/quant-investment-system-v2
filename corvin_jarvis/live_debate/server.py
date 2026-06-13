"""Phase D — 실시간 토론 SSE 서버 (stdlib http.server, 무의존).

GET /debate            → 카톡 라이브 UI(debate_view.html)
GET /debate/stream     → 토론을 SSE로 스트리밍 (?mode=style|stance, &fake=1 오프라인)

LLM = claude CLI 헤드리스(구독 인증, API키 0원). fake=1은 캔드 응답으로 빠른 스모크.
컨텍스트는 코드(data_verify)가 만든다 — fact-check 기준이 환각 안 됨.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from corvin_jarvis.live_debate import context as _ctx
from corvin_jarvis.live_debate import stream as _stream

BASE = Path(__file__).resolve().parent
VIEW = BASE / "debate_view.html"


# ── LLM 백엔드 ────────────────────────────────────────────────
def _live_llm(prompt: str) -> str:
    from corvin_jarvis import qualitative
    out = qualitative.run_claude_cli(prompt, timeout=90)
    return out or "(응답 없음)"


def _fake_llm_factory():
    """오프라인 스모크용 — 페르소나 인식 캔드 응답(검증 배지 시연: TSLA 오인용 1건)."""
    # 페르소나 system 키워드 → 응답. mismatch 시연 위해 trend는 TSLA −5.0%(실제 −10.5%).
    by_keyword = {
        "위험관리": "MSFT +8.7% 수익권은 두고, TSLA −5.0%는 손절선 깨져 정리 🛑",  # mismatch
        "급락은 기회": "급락은 줍줍 기회! NVDA +4.0% 핵심은 분할매수 간다 🚀",
        "흔들리지 말고": "안전선 안 깨졌으면 보유. MSFT +8.7% 굳이 안 던져 🧘",
        "가치투자": "MSFT +8.7% 우량주는 들고 간다, 사업이 안 망가졌어 💰",
        "성장투자": "급락은 줍줍 기회! NVDA +4.0% 분할매수 간다 🚀",
        "추세추종": "TSLA −5.0% 손절선 깨졌으니 던져라 📉",  # mismatch(실제 −10.5%)
        "글로벌 매크로": "포트 베타 줄이고 금·현금 헤지부터 🌍",
        "퀀트": "백테스트상 반도체 신호는 엣지 0, vix_term만 믿어라 🤖",
    }
    def fake(prompt: str) -> str:
        for kw, resp in by_keyword.items():
            if kw in prompt:
                return resp
        return "데이터 보고 판단하자"
    return fake


# ── 라이브 컨텍스트 (코드가 만든다) ───────────────────────────
def _fake_context() -> dict:
    holdings = [
        {"symbol": "MSFT", "avg_price": 383.25, "live_pnl_pct": 8.72, "price": {"value": 416.67, "confidence": "high"}},
        {"symbol": "TSLA", "avg_price": 437.0, "live_pnl_pct": -10.53, "price": {"value": 391.0, "confidence": "high"}},
    ]
    return {"holdings": holdings, "note": "코스피 -5.5%, 나스닥 -4.2%, 공포지수 +40%",
            "low_confidence": []}


LATEST = BASE.parent / "state" / "latest.json"


def _live_context() -> dict:
    """jarvis 스냅샷(latest.json) 재활용 — KIS 라이브 가격(KR 포함), 빠름, 정확.

    + 원인규명(섹터 분해)을 note에 엮어 토론을 더 똑똑하게.
    """
    try:
        latest = json.loads(LATEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _fake_context()

    holdings, low_conf = [], []
    for p in latest.get("portfolio", []):
        if not p.get("symbol"):
            continue
        price = p.get("current_price")
        conf = "high" if p.get("source") == "kis_live" and price else ("medium" if price else "none")
        if conf == "none":
            low_conf.append(p["symbol"])
        holdings.append({"symbol": p["symbol"], "avg_price": p.get("avg_price"),
                         "live_pnl_pct": p.get("pnl_pct"),
                         "price": {"value": price, "confidence": conf}})

    idx = latest.get("indices", {})
    def _pc(k):
        v = idx.get(k, {})
        return f"{v.get('pct_change')}%" if isinstance(v, dict) else "?"
    market = f"코스피 {_pc('kospi')}, 나스닥 {_pc('nasdaq')}, S&P {_pc('sp500')}, VIX {idx.get('vix', {}).get('price')}"

    from corvin_jarvis import cause_attribution as ca
    ranked = ca.rank_causes(ca.candidates_from_snapshot(latest))
    cause = ca.attribution_message(ranked, top_n=3)

    return {"holdings": holdings, "low_confidence": low_conf,
            "note": f"{market} | {cause}"}


# ── 재사용 라우트 핸들러 (대시보드 통합용) ───────────────────
def serve_view(handler) -> None:
    """GET /debate — 카톡 라이브 UI HTML."""
    body = VIEW.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def serve_stream(handler, query: dict) -> None:
    """GET /debate/stream — 토론 SSE. handler.wfile로 emit."""
    mode = query.get("mode", ["style"])[0]
    fake = query.get("fake", ["0"])[0] == "1"
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()

    def emit(s: str):
        handler.wfile.write(s.encode("utf-8"))
        handler.wfile.flush()

    llm = _fake_llm_factory() if fake else _live_llm
    ctx = _fake_context() if fake else _live_context()
    emit(_stream.sse_event({"kind": "header", "mode": mode,
         "low_confidence": ctx.get("low_confidence", []), "note": ctx.get("note", "")}))
    try:
        _stream.run_stream(mode, ctx, llm, emit, ["opening", "rebuttal"])
    except (BrokenPipeError, ConnectionResetError):
        pass


def serve_ask(handler, query: dict) -> None:
    """GET /debate/ask — 폐하 질문에 각 페르소나가 답하는 1라운드 SSE (참여형)."""
    mode = query.get("mode", ["style"])[0]
    fake = query.get("fake", ["0"])[0] == "1"
    question = query.get("q", [""])[0].strip()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()

    def emit(s: str):
        handler.wfile.write(s.encode("utf-8"))
        handler.wfile.flush()

    if not question:
        emit(_stream.sse_event({"kind": "done"}))
        return
    llm = _fake_llm_factory() if fake else _live_llm
    ctx = _fake_context() if fake else _live_context()
    try:
        _stream.run_reply_stream(mode, ctx, question, llm, emit)
    except (BrokenPipeError, ConnectionResetError):
        pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 조용히
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/debate":
            serve_view(self)
        elif u.path == "/debate/stream":
            serve_stream(self, parse_qs(u.query))
        elif u.path == "/debate/ask":
            serve_ask(self, parse_qs(u.query))
        else:
            self.send_error(404)


def serve(host: str = "127.0.0.1", port: int = 8530):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"live-debate on http://{host}:{port}/debate")
    srv.serve_forever()


if __name__ == "__main__":
    serve(port=int(os.environ.get("DEBATE_PORT", "8530")))
