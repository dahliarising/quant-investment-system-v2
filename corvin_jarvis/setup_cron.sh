#!/usr/bin/env bash
# Corvin Jarvis — One-shot 활성화 스크립트
#
# 폐하가 직접 실행:
#   bash /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/setup_cron.sh
#
# 동작:
#   1. 현재 crontab 백업
#   2. Jarvis 시간별 entry 추가 (KST 08-23, 매시간 정각)
#   3. 첫 실행 트리거

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP="/tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"

echo "[1/4] 현재 crontab 백업..."
crontab -l > "$BACKUP" 2>/dev/null || touch "$BACKUP"
echo "      → $BACKUP"

if crontab -l 2>/dev/null | grep -q "corvin_jarvis"; then
    echo "[2/4] Jarvis 이미 등록되어 있음 — DCA entry만 확인..."
    if crontab -l 2>/dev/null | grep -q "run_dca.sh"; then
        echo "      DCA cron도 등록되어 있음 — skip"
    else
        echo "      DCA cron entry 추가..."
        (
            crontab -l 2>/dev/null
            echo ""
            echo "# Corvin DCA Timing — 일 2회 장마감 후 (등록: $(date +%Y-%m-%d))"
            echo "# 15:40 KST = KR 장마감 후(완성봉), 06:30 KST = US 장마감 후(완성봉)"
            echo "40 15 * * * $SCRIPT_DIR/run_dca.sh KR"
            echo "30 6  * * * $SCRIPT_DIR/run_dca.sh US"
        ) | crontab -
    fi
else
    echo "[2/4] Jarvis + DCA cron entry 추가..."
    (
        cat "$BACKUP"
        echo ""
        echo "# Corvin Jarvis — 자율 자산관리 (등록: $(date +%Y-%m-%d))"
        echo "# 매시간 정각, KST 08-23"
        echo "0 8-23 * * * $SCRIPT_DIR/run_jarvis.sh"
        echo ""
        echo "# Corvin DCA Timing — 일 2회 장마감 후 (등록: $(date +%Y-%m-%d))"
        echo "# 15:40 KST = KR 장마감 후(완성봉), 06:30 KST = US 장마감 후(완성봉)"
        echo "40 15 * * * $SCRIPT_DIR/run_dca.sh KR"
        echo "30 6  * * * $SCRIPT_DIR/run_dca.sh US"
    ) | crontab -
fi

echo "[3/4] 현재 crontab 확인..."
crontab -l | grep -A1 "Corvin Jarvis" || echo "      (확인 실패 — 수동 점검)"

echo "[4/4] 첫 실행 트리거..."
bash "$SCRIPT_DIR/run_jarvis.sh"

echo ""
echo "✅ Corvin Jarvis 활성화 완료."
echo "   다음 실행: 매시간 정각 (KST 08-23)"
echo "   로그: $SCRIPT_DIR/state/cron.log"
echo "   브리핑: $SCRIPT_DIR/state/briefing.md"
