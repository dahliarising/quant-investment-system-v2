#!/usr/bin/env bash
# Corvin 선행 인텔리전스 cron 등록 (수동 실행 전용).
# ⚠️ Claude auto-mode는 crontab 자동 등록 금지 → 폐하가 직접 실행.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$SCRIPT_DIR/run_leading.sh"

# run_leading.sh 래퍼가 .env(토큰/KIS/ANTHROPIC)를 source하므로 cron에서도 발송 가능.
# KR 프리오픈 08:30, US 프리오픈(09:00 ET≈22:30 KST) 평일
CRON_KR="30 8 * * 1-5 $WRAPPER"
CRON_US="30 22 * * 1-5 $WRAPPER"

echo "다음 두 줄을 crontab에 추가하세요 (crontab -e):"
echo ""
echo "$CRON_KR"
echo "$CRON_US"
echo ""
echo "또는 자동 추가:"
echo "  (crontab -l 2>/dev/null; echo \"$CRON_KR\"; echo \"$CRON_US\") | crontab -"
