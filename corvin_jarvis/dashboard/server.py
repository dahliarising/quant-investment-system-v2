"""로컬 대시보드 HTTP 서버 (stdlib)."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from corvin_jarvis.dashboard import snapshot

STATIC = Path(__file__).resolve().parent / "static"
_CTYPE = {".html": "text/html; charset=utf-8", ".css": "text/css",
          ".js": "application/javascript", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 콘솔 스팸 억제
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/debate":
            from corvin_jarvis.live_debate import server as debate
            debate.serve_view(self)
            return
        if path == "/debate/stream":
            from urllib.parse import parse_qs, urlparse
            from corvin_jarvis.live_debate import server as debate
            debate.serve_stream(self, parse_qs(urlparse(self.path).query))
            return
        if path == "/debate/ask":
            from urllib.parse import parse_qs, urlparse
            from corvin_jarvis.live_debate import server as debate
            debate.serve_ask(self, parse_qs(urlparse(self.path).query))
            return
        if path == "/api/snapshot":
            first = snapshot.get_snapshot(ttl=30.0)
            ttl = 30.0 if first.get("market_state") == "open" else 300.0
            snap = snapshot.get_snapshot(ttl=ttl)
            self._send(200, json.dumps(snap, ensure_ascii=False).encode(), _CTYPE[".json"])
            return
        if path in ("/command", "/command_center", "/cc"):
            self._serve_file(STATIC / "command_center.html")
            return
        if path in ("/flow", "/decision"):
            self._serve_file(STATIC / "decision_flow.html")
            return
        if path in ("/", "/index.html"):
            self._serve_file(STATIC / "index.html")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC / rel).resolve()
            if STATIC in target.parents and target.exists():
                self._serve_file(target)
            else:
                self._send(404, b"not found", "text/plain")
            return
        self._send(404, b"not found", "text/plain")

    def _serve_file(self, fp: Path):
        if not fp.exists():
            self._send(404, b"not found", "text/plain")
            return
        ctype = _CTYPE.get(fp.suffix, "application/octet-stream")
        self._send(200, fp.read_bytes(), ctype)


def make_server(port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)
