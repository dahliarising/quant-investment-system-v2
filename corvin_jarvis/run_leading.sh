#!/usr/bin/env bash
# Corvin 선행 인텔리전스 — cron entry script
#
# 등록 예시 (crontab -e):
#   30 8 * * 1-5 /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_leading.sh
#   30 22 * * 1-5 /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_leading.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# cron PATH는 최소 → claude CLI(~/.local/bin) 및 homebrew 경로 추가 (정성 분석용)
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

LOG="$SCRIPT_DIR/state/leading.log"
mkdir -p "$SCRIPT_DIR/state"

PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then
    PY=/opt/homebrew/bin/python3
fi

# Load secrets (TELEGRAM_BOT_TOKEN/CHAT_ID, KIS, ANTHROPIC) — gitignored .env.
# cron does NOT source ~/.zshrc, so the wrapper must source .env itself.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    . "$SCRIPT_DIR/.env"
    set +a
fi

echo "============================================" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Leading 시작" >> "$LOG"

"$PY" -m corvin_jarvis.run_leading >> "$LOG" 2>&1 || echo "[ERROR] run_leading 실패" >> "$LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Leading 완료" >> "$LOG"
