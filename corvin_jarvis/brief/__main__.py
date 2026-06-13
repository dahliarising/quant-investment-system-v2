"""brief CLI 진입점. `python -m corvin_jarvis.brief` → 설득 브리핑 stdout."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from corvin_jarvis.brief.builder import build_brief
from corvin_jarvis.brief.render import render_brief


def compose_from_sources(*,
                         load_positions: Callable[[], list[dict[str, Any]]],
                         load_actions: Callable[[], list[dict[str, Any]]],
                         load_calibration: Callable[[], dict[str, Any]],
                         load_held: Callable[[], set[str]],
                         market_state: str, fresh_label: str,
                         as_of: str) -> str:
    """Pure entry: accepts injected loaders — no IO. Tested directly."""
    brief = build_brief(
        positions=load_positions(), actions=load_actions(),
        calibration=load_calibration(), held=load_held(),
        market_state=market_state, fresh_label=fresh_label, as_of=as_of,
    )
    return render_brief(brief)


def _market_labels() -> tuple[str, str]:
    """Return (market_state, fresh_label). Falls back to 장마감/종가 if no is_any_open."""
    from corvin_jarvis import market_hours
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    try:
        open_now = market_hours.is_any_open(now)  # type: ignore[attr-defined]
    except AttributeError:
        # market_hours has is_kr_open / is_us_open but not is_any_open;
        # derive it manually
        open_now = market_hours.is_kr_open(now) or market_hours.is_us_open(now)
    return ("장중", "현재가") if open_now else ("장마감", "종가")


def main() -> None:
    import json
    from pathlib import Path

    from corvin_jarvis import notify, pulse

    state_dir = Path(__file__).resolve().parent.parent / "state"
    market_state, fresh_label = _market_labels()
    as_of = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")

    def load_actions() -> list[dict[str, Any]]:
        try:
            data = json.loads((state_dir / "final_actions.json").read_text())
            return data.get("actions", [])
        except OSError:
            return []

    def load_calibration() -> dict[str, Any]:
        try:
            return json.loads((state_dir / "calibration.json").read_text())
        except OSError:
            return {}

    print(compose_from_sources(
        load_positions=lambda: pulse.fetch_portfolio()[0],
        load_actions=load_actions,
        load_calibration=load_calibration,
        load_held=notify._held_symbols,
        market_state=market_state,
        fresh_label=fresh_label,
        as_of=as_of,
    ))


if __name__ == "__main__":
    main()
