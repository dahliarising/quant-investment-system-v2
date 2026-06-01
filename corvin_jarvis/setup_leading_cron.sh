#!/usr/bin/env bash
# Corvin 선행 인텔리전스 cron 등록 (수동 실행 전용).
# ⚠️ Claude auto-mode는 crontab 자동 등록 금지 → 폐하가 직접 실행.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
LOG="$REPO_DIR/corvin_jarvis/state/leading.log"

# KR 프리오픈 08:30, US 프리오픈(09:00 ET≈22:30 KST) 평일
CRON_KR="30 8 * * 1-5 cd $REPO_DIR && $PY -m corvin_jarvis.run_leading >> $LOG 2>&1"
CRON_US="30 22 * * 1-5 cd $REPO_DIR && $PY -m corvin_jarvis.run_leading >> $LOG 2>&1"

echo "다음 두 줄을 crontab에 추가하세요 (crontab -e):"
echo ""
echo "$CRON_KR"
echo "$CRON_US"
echo ""
echo "또는 자동 추가:"
echo "  (crontab -l 2>/dev/null; echo \"$CRON_KR\"; echo \"$CRON_US\") | crontab -"
