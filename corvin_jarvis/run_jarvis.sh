#!/usr/bin/env bash
# Corvin Jarvis — cron entry script
#
# 등록 예시 (crontab -e):
#   0 8-23 * * * /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_jarvis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

LOG="$SCRIPT_DIR/state/cron.log"
mkdir -p "$SCRIPT_DIR/state"

PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then
    PY=/opt/homebrew/bin/python3
fi

# Load secrets (Discord webhook etc.) — gitignored .env. cron does not source ~/.zshrc.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    . "$SCRIPT_DIR/.env"
    set +a
fi

echo "============================================" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Jarvis 시작" >> "$LOG"

# Full pipeline: pulse → compare → geo_signal → narrate → briefing
"$PY" "$SCRIPT_DIR/jarvis.py" >> "$LOG" 2>&1 || echo "[ERROR] jarvis.py 실패" >> "$LOG"

# Multi-channel notify: Discord webhook + iMessage + file fallback
"$PY" "$SCRIPT_DIR/notify.py" >> "$LOG" 2>&1 || echo "[ERROR] notify.py 실패" >> "$LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Jarvis 완료" >> "$LOG"
