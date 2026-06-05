"""python -m corvin_jarvis.dashboard — 로컬 대시보드 실행."""
from __future__ import annotations

import sys
import threading
import webbrowser

from corvin_jarvis.dashboard import server

PORT = 8765


def main() -> int:
    httpd = server.make_server(port=PORT)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"🐦‍⬛ Corvin Command Center → {url}  (Ctrl+C 종료)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
