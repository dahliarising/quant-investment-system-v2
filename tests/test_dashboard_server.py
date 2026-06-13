import json
import threading
import urllib.request
from corvin_jarvis.dashboard import server


def test_make_server_host_configurable():
    httpd = server.make_server(port=0, host="127.0.0.1")
    assert httpd.server_address[0] == "127.0.0.1"
    httpd.server_close()


def test_snapshot_route_returns_json(monkeypatch):
    monkeypatch.setattr(server.snapshot, "get_snapshot",
                        lambda ttl: {"ts": "x", "market_state": "open", "positions": []})
    httpd = server.make_server(port=0)
    threading.Thread(target=httpd.handle_request, daemon=True).start()
    port = httpd.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/snapshot", timeout=5) as r:
        body = json.loads(r.read())
    assert body["ts"] == "x"
    httpd.server_close()


def test_index_route_serves_html():
    httpd = server.make_server(port=0)
    threading.Thread(target=httpd.handle_request, daemon=True).start()
    port = httpd.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
        body = r.read().decode()
    assert "<html" in body.lower()
    httpd.server_close()
