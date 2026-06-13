#!/usr/bin/env bash
# Corvin 선제 토론 다이제스트 — cron entry script
#
# 폐하가 물을 법한 질문에 5인 투자대가가 미리 토론 → Telegram 다이제스트.
# 무겁다(질문당 5콜 × claude CLI ~수분) → 매일 1회만. 주식 크론(아침) 정렬.
#
# 등록 예시 (crontab -e), 평일 08:00 KST (장 시작 전 아침 질문 선제):
#   0 8 * * 1-5 /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_debate_digest.sh
#
# DRY-RUN(발송 안 함, 다이제스트만 출력):
#   .../run_debate_digest.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# cron PATH는 최소 → claude CLI(~/.local/bin) 및 homebrew 추가
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then
    PY=/opt/homebrew/bin/python3
fi

# .env에서 시크릿 로드(TELEGRAM_BOT_TOKEN/CHAT_ID, KIS, ANTHROPIC) — cron은 ~/.zshrc 안 읽음
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    . "$SCRIPT_DIR/.env"
    set +a
fi

LOG="$SCRIPT_DIR/state/debate_digest.log"
mkdir -p "$SCRIPT_DIR/state"

echo "============================================" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Debate digest 시작" >> "$LOG"

"$PY" -m corvin_jarvis.debate_cron "$@" >> "$LOG" 2>&1 || echo "[ERROR] debate_cron 실패" >> "$LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Debate digest 완료" >> "$LOG"
