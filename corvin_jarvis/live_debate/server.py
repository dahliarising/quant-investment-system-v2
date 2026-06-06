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
PORTFOLIO = BASE.parent.parent / "portfolio.json"


# ── LLM 백엔드 ────────────────────────────────────────────────
def _live_llm(prompt: str) -> str:
    from corvin_jarvis import qualitative
    out = qualitative.run_claude_cli(prompt, timeout=90)
    return out or "(응답 없음)"


def _fake_llm_factory():
    """오프라인 스모크용 — 페르소나별 캔드 응답(검증 배지 시연 포함)."""
    lines = iter([
        "MSFT +8.7% 우량주는 들고 간다, 사업이 안 망가졌어 💰",   # verified
        "급락은 줍줍 기회! NVDA 분할매수 간다 🚀",
        "TSLA −5.0% 손절선 깨졌으니 던져라 📉",                  # mismatch(실제 −10.5%)
        "포트 베타 줄이고 금·현금 헤지부터 🌍",
        "백테스트상 반도체 신호는 엣지 0, 숫자만 믿어라 🤖",
    ])
    def fake(prompt: str) -> str:
        try:
            return next(lines)
        except StopIteration:
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


def _live_context() -> dict:
    from corvin_jarvis import data_verify  # noqa: F401 (의존 확인)
    try:
        pf = json.loads(PORTFOLIO.read_text(encoding="utf-8"))
        holdings = [{"symbol": h["symbol"], "avg_price": h.get("avgPriceUSD") or h.get("avgPriceKRW"),
                     "stored_pnl_pct": h.get("pnlPct")} for h in pf.get("holdings", [])]
    except (OSError, json.JSONDecodeError):
        return _fake_context()

    def fetchers_for(sym):
        def yf():
            from corvin_jarvis.ew_providers import live_vix_fetcher  # reuse yahoo
            import yfinance as yfin
            h = yfin.Ticker(sym).history(period="5d")["Close"].dropna()
            return float(h.iloc[-1]) if len(h) else None
        return {"yfinance": yf}
    return _ctx.build_context(holdings, fetchers_for, tol_pct=2.0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 조용히
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/debate":
            self._serve_view()
        elif u.path == "/debate/stream":
            self._serve_stream(parse_qs(u.query))
        else:
            self.send_error(404)

    def _serve_view(self):
        body = VIEW.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self, q):
        mode = (q.get("mode", ["style"])[0])
        fake = q.get("fake", ["0"])[0] == "1"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        llm = _fake_llm_factory() if fake else _live_llm
        ctx = _fake_context() if fake else _live_context()
        # 헤더 이벤트: 데이터 신뢰도
        self._emit(_stream.sse_event({"kind": "header", "mode": mode,
                   "low_confidence": ctx.get("low_confidence", []),
                   "note": ctx.get("note", "")}))
        rounds = ["opening", "rebuttal"]
        try:
            _stream.run_stream(mode, ctx, llm, self._emit, rounds)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _emit(self, s: str):
        self.wfile.write(s.encode("utf-8"))
        self.wfile.flush()


def serve(host: str = "127.0.0.1", port: int = 8530):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"live-debate on http://{host}:{port}/debate")
    srv.serve_forever()


if __name__ == "__main__":
    serve(port=int(os.environ.get("DEBATE_PORT", "8530")))
