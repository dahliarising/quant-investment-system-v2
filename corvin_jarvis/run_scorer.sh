#!/usr/bin/env bash
# Corvin 신호 채점 — 매일 16:30 KST 장마감 후. 만기 신호 채점 + calibration.json 갱신.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
LOG="$SCRIPT_DIR/state/cron.log"
mkdir -p "$SCRIPT_DIR/state"
PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then PY=/opt/homebrew/bin/python3; fi
if [ -f "$SCRIPT_DIR/.env" ]; then set -a; . "$SCRIPT_DIR/.env"; set +a; fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scorer 시작" >> "$LOG"
"$PY" -m corvin_jarvis.signals.scorer >> "$LOG" 2>&1 || echo "[ERROR] scorer 실패" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scorer 완료" >> "$LOG"
