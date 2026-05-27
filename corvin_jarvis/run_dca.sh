#!/usr/bin/env bash
# Corvin DCA Timing — cron entry script
#
# 등록 예시 (crontab -e):
#   0 8  * * * /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_dca.sh KR
#   0 21 * * * /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_dca.sh US

set -euo pipefail

WINDOW="${1:-ALL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

LOG="$SCRIPT_DIR/state/dca_cron.log"
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DCA Timing 시작 (window=$WINDOW)" >> "$LOG"

"$PY" -m corvin_jarvis.dca_timing --window "$WINDOW" >> "$LOG" 2>&1 || \
    echo "[ERROR] dca_timing 실패 (window=$WINDOW)" >> "$LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DCA Timing 완료 (window=$WINDOW)" >> "$LOG"
