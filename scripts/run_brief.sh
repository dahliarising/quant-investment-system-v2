#!/usr/bin/env bash
# run_brief.sh — 설득 브리핑 생성 → 텔레그램 달리아봇 push
# 라우팅: telegram + log_only only (memory: feedback_cron_imessage_only)
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f corvin_jarvis/.env ] && . corvin_jarvis/.env; set +a

BRIEF="$(python3 -m corvin_jarvis.brief)"

python3 - "$BRIEF" <<'PY'
import sys
from corvin_jarvis import channels

text = sys.argv[1]
# send_telegram is the real API (channels.send / channels_override do not exist)
ok = channels.send_telegram(text)
if not ok:
    print("[run_brief] WARNING: send_telegram returned False (check TELEGRAM_BOT_TOKEN / enabled channels)")
else:
    print("[run_brief] sent OK")
PY
