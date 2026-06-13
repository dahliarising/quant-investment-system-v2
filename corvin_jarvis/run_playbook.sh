#!/usr/bin/env bash
# Corvin 플레이북 — cron entry script
#
# 등록 예시 (crontab -e):
#   30 8 * * 1-5 /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_playbook.sh
#   30 22 * * 1-5 /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_playbook.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
LOG="$SCRIPT_DIR/state/playbook.log"
mkdir -p "$SCRIPT_DIR/state"
PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then PY=/opt/homebrew/bin/python3; fi
if [ -f "$SCRIPT_DIR/.env" ]; then set -a; . "$SCRIPT_DIR/.env"; set +a; fi
# 무음 모드: HTML은 생성하되 텔레그램 push 차단 (노이즈 감축 2026-06-13). DCA 중복·08:30 충돌 해소.
export CORVIN_SILENT=1
echo "[$(date '+%F %T')] Playbook 시작 (CORVIN_SILENT)" >> "$LOG"
"$PY" -m corvin_jarvis.playbook.run_playbook >> "$LOG" 2>&1 || echo "[ERROR] run_playbook 실패" >> "$LOG"
echo "[$(date '+%F %T')] Playbook 완료" >> "$LOG"
