#!/usr/bin/env bash
# Corvin 자동매매 페이퍼트레이더 — 장중 세션별 실행. 🚨 실주문 0 (전부 가상 시뮬).
# 세션은 run_auto_trader가 KST 시각으로 자동감지(KR 09:00-15:30 / US 22:30-05:00).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
LOG="$SCRIPT_DIR/state/auto_trader.log"
mkdir -p "$SCRIPT_DIR/state"
PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then PY=/opt/homebrew/bin/python3; fi
# cron은 ~/.zshrc 안 읽음 → .env에서 KIS_MOCK_*/시크릿 로드
if [ -f "$SCRIPT_DIR/.env" ]; then set -a; . "$SCRIPT_DIR/.env"; set +a; fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_trader 시작" >> "$LOG"
"$PY" -m corvin_jarvis.run_auto_trader "$@" >> "$LOG" 2>&1 || echo "[ERROR] auto_trader 실패" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_trader 완료" >> "$LOG"
