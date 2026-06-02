"""플레이북 진입점 — build → 푸시 + HTML 저장."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from corvin_jarvis import channels
from corvin_jarvis.playbook import builder, render_html, render_text
from corvin_jarvis.playbook.models import Playbook

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UNIVERSE_FILE = BASE_DIR / "monitored_universe.json"
HTML_OUT = BASE_DIR.parent / "strategies" / "playbook.html"


def _load_universe(path: Path = UNIVERSE_FILE) -> list[dict]:
    try:
        return json.loads(path.read_text()).get("tickers", [])
    except (OSError, json.JSONDecodeError):
        return []


def run(
    playbooks: list[Playbook],
    date_label: str,
    sender: Callable[[str], bool],
    html_path: Path,
) -> bool:
    if not playbooks:
        log.warning("playbook 없음 — 송신/저장 생략")
        return False
    html_doc = render_html.render_dashboard(playbooks)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_doc)
    push = render_text.render_push(playbooks, date_label)
    sent = sender(push)
    log.info("playbook: n=%d sent=%s html=%s", len(playbooks), sent, html_path)
    return bool(sent)


def main() -> int:
    universe = _load_universe()
    holdings = builder.load_holdings()
    playbooks = builder.build_playbooks(universe, holdings)
    date_label = datetime.now().strftime("%-m/%-d")
    ok = run(playbooks, date_label, channels.send_telegram, HTML_OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
