"""python -m corvin_jarvis.dashboard — 로컬 대시보드 실행."""
from __future__ import annotations

import os
import sys
import threading
import webbrowser

from corvin_jarvis.dashboard import server

PORT = 8765
# 기본 127.0.0.1(로컬 전용). 폰/테일넷 직접 바인드 시 CORVIN_DASHBOARD_HOST=0.0.0.0.
# 권장: 바인드는 그대로 두고 `tailscale serve --bg 8765`로 테일넷에 HTTPS 노출.
HOST = os.environ.get("CORVIN_DASHBOARD_HOST", "127.0.0.1")


def main() -> int:
    httpd = server.make_server(port=PORT, host=HOST)
    local_url = f"http://127.0.0.1:{PORT}/"
    print(f"🐦‍⬛ Corvin Command Center → {local_url}  (Ctrl+C 종료)")
    if HOST not in ("127.0.0.1", "localhost"):
        print(f"   bind={HOST}:{PORT} — 테일넷/LAN 접근 가능. 폰 접속 권장: `tailscale serve --bg {PORT}`")
    threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
