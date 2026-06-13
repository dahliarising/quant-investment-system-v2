#!/usr/bin/env bash
# Corvin 일일 다이제스트 — 하루 1회 장마감 후. jarvis 파이프라인 갱신 후 digest 모드 push.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
LOG="$SCRIPT_DIR/state/cron.log"
mkdir -p "$SCRIPT_DIR/state"
PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then PY=/opt/homebrew/bin/python3; fi
if [ -f "$SCRIPT_DIR/.env" ]; then set -a; . "$SCRIPT_DIR/.env"; set +a; fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Digest 시작" >> "$LOG"
"$PY" "$SCRIPT_DIR/jarvis.py" >> "$LOG" 2>&1 || echo "[ERROR] jarvis.py 실패" >> "$LOG"
"$PY" -m corvin_jarvis.signals.arbiter_inputs >> "$LOG" 2>&1 || echo "[ERROR] arbiter 실패" >> "$LOG"
"$PY" "$SCRIPT_DIR/notify.py" digest >> "$LOG" 2>&1 || echo "[ERROR] digest 실패" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Digest 완료" >> "$LOG"
