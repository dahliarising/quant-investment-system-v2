# Smart Alert Filter (Priority + Daily Digest) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 알림이 "멍청"해지지 않도록 push를 똑똑하게 거른다. 매시간엔 긴급(HIGH/CRITICAL)만, 하루 1회는 전체를 중요도순 다이제스트로. 보유종목 관련 신호를 우선하고 건수를 제한한다.

**Architecture:** `notify.py`에 `mode`("urgent"|"digest") 파라미터를 도입. urgent=높은 severity 게이트 + dedup + cap, digest=낮은 게이트 + dedup 없음 + ranked top N. 랭킹 = (severity → 보유종목 관련성 → 변동폭). 임계/개수는 config로.

**Tech Stack:** Python 3.11+, pytest. 기존 notify.py 구조(_filter_severity, _filter_dedup, _format_message) 재사용.

**Design 합의:** Discord 2026-05-27 — 매시간 HIGH+, 하루1회 16:00 KST top8 digest(MEDIUM+), 우선순위 severity→보유→변동폭, cap 8.

---

## File Structure
| 파일 | 책임 | 신규/수정 |
|---|---|---|
| `corvin_jarvis/config.json` | notification: urgent/digest 임계·개수 | Modify |
| `corvin_jarvis/notify.py` | mode 분기 + ranking/actionability + format title/limit | Modify |
| `corvin_jarvis/run_jarvis.sh` | notify를 urgent 모드로 호출 | Modify |
| `corvin_jarvis/run_digest.sh` | 일일 digest 실행 스크립트 | Create |
| `tests/test_notify.py` | ranking/mode 단위테스트 | Create |

---

### Task C1: config — notification 임계/개수

**Files:** Modify `corvin_jarvis/config.json`

- [ ] **Step 1:** `notification` 객체에 키 추가 (기존 `minimum_alert_severity` 다음, 콤마 주의):
```json
    "urgent_min_severity": "high",
    "digest_min_severity": "medium",
    "digest_max_items": 8,
    "max_per_push": 8
```

- [ ] **Step 2: Validate** `python3 -c "import json; n=json.load(open('corvin_jarvis/config.json'))['notification']; print(n['urgent_min_severity'], n['digest_min_severity'], n['digest_max_items'], n['max_per_push'])"` → `high medium 8 8`

- [ ] **Step 3: Commit**
```bash
git add corvin_jarvis/config.json
git commit -m "feat(corvin): add urgent/digest notification thresholds (smart filter)"
```

---

### Task C2: notify.py — ranking + actionability helpers

**Files:**
- Modify: `corvin_jarvis/notify.py`
- Test: `tests/test_notify.py`

- [ ] **Step 1: Write failing tests** — `tests/test_notify.py`:
```python
import json
from corvin_jarvis import notify


def test_actionability_flags_held_symbol():
    held = {"NVDA", "META", "MSFT"}
    a = {"metric": "pnl_NVDA", "message": "NVDA 수익 +16%", "value": 16.0, "severity": "medium"}
    b = {"metric": "universe_005930_confirmed", "message": "삼성전자 급등", "value": 7.0, "severity": "medium"}
    assert notify._actionability(a, held) == 1
    assert notify._actionability(b, held) == 0


def test_rank_orders_by_severity_then_action_then_magnitude():
    held = {"NVDA"}
    alerts = [
        {"metric": "x", "message": "m", "value": 3.0, "severity": "medium"},
        {"metric": "rs_NVDA_confirmed", "message": "NVDA RS", "value": 5.0, "severity": "medium"},
        {"metric": "y", "message": "y", "value": 1.0, "severity": "critical"},
    ]
    ranked = notify._rank_alerts(alerts, held)
    assert ranked[0]["severity"] == "critical"          # severity 최우선
    assert ranked[1]["metric"] == "rs_NVDA_confirmed"    # 같은 medium 중 보유종목 우선


def test_held_symbols_reads_portfolio(tmp_path, monkeypatch):
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({"holdings": [{"symbol": "NVDA"}, {"symbol": "META"}]}))
    monkeypatch.setattr(notify, "PORTFOLIO_FILE", pf)
    assert notify._held_symbols() == {"NVDA", "META"}
```

- [ ] **Step 2: Run — expect FAIL** `python3 -m pytest tests/test_notify.py -v` → AttributeError (_actionability/_rank_alerts/_held_symbols/PORTFOLIO_FILE missing).

- [ ] **Step 3: Implement.** In `corvin_jarvis/notify.py`, add `PORTFOLIO_FILE` next to the other path constants (after `CONFIG_FILE = BASE_DIR / "config.json"`):
```python
PORTFOLIO_FILE = BASE_DIR.parent / "portfolio.json"
```
Add these helpers after `_filter_dedup`:
```python
def _held_symbols() -> set[str]:
    pf = _load_json(PORTFOLIO_FILE)
    return {str(h["symbol"]) for h in pf.get("holdings", []) if h.get("symbol")}


def _actionability(alert: dict[str, Any], held: set[str]) -> int:
    blob = f"{alert.get('metric', '')} {alert.get('message', '')}"
    return 1 if any(sym in blob for sym in held) else 0


def _rank_alerts(alerts: list[dict[str, Any]], held: set[str]) -> list[dict[str, Any]]:
    return sorted(
        alerts,
        key=lambda a: (SEV_RANK.get(a["severity"], 0), _actionability(a, held), abs(a.get("value") or 0)),
        reverse=True,
    )
```

- [ ] **Step 4: Run — expect PASS** `python3 -m pytest tests/test_notify.py -v` → 3 passed.

- [ ] **Step 5: Commit**
```bash
git add corvin_jarvis/notify.py tests/test_notify.py
git commit -m "feat(corvin): alert ranking + actionability helpers (smart filter)"
```

---

### Task C3: notify.py — mode-aware notify() + format title/limit

**Files:**
- Modify: `corvin_jarvis/notify.py`
- Test: `tests/test_notify.py` (append)

- [ ] **Step 1: Append failing tests** to `tests/test_notify.py`:
```python
def _write_state(tmp_path, monkeypatch, alerts):
    af = tmp_path / "alerts.json"
    af.write_text(json.dumps({"alerts": alerts}))
    cf = tmp_path / "config.json"
    cf.write_text(json.dumps({"notification": {
        "urgent_min_severity": "high", "digest_min_severity": "medium",
        "digest_max_items": 8, "max_per_push": 8,
        "discord_webhook_url": None, "imessage_recipient": None,
    }}))
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({"holdings": []}))
    dd = tmp_path / "push_dedup.json"
    monkeypatch.setattr(notify, "ALERTS_FILE", af)
    monkeypatch.setattr(notify, "CONFIG_FILE", cf)
    monkeypatch.setattr(notify, "PORTFOLIO_FILE", pf)
    monkeypatch.setattr(notify, "DEDUP_FILE", dd)
    monkeypatch.setattr(notify, "PENDING_FILE", tmp_path / "pending.json")


def test_urgent_mode_drops_medium(tmp_path, monkeypatch):
    _write_state(tmp_path, monkeypatch, [
        {"category": "x", "metric": "m1", "severity": "medium", "message": "med", "value": 3.0},
        {"category": "x", "metric": "m2", "severity": "high", "message": "hi", "value": 9.0},
    ])
    res = notify.notify(mode="urgent")
    # medium은 severity 게이트(high)에서 제외 → pushed=1 (high만), 채널 없으니 file queue
    assert res.skipped_severity == 1


def test_digest_mode_includes_medium(tmp_path, monkeypatch):
    _write_state(tmp_path, monkeypatch, [
        {"category": "x", "metric": "m1", "severity": "medium", "message": "med", "value": 3.0},
        {"category": "x", "metric": "m2", "severity": "high", "message": "hi", "value": 9.0},
    ])
    res = notify.notify(mode="digest")
    assert res.skipped_severity == 0     # medium도 포함
    assert res.skipped_dedup == 0        # digest는 dedup 안 함
```

- [ ] **Step 2: Run — expect FAIL** `python3 -m pytest tests/test_notify.py -k "mode" -v` → notify() has no `mode` param.

- [ ] **Step 3: Implement.** Replace the `_format_message` signature/header and the `notify()` function.

Change `_format_message` header line and signature. Replace:
```python
def _format_message(alerts: list[dict[str, Any]], compact: bool = False) -> str:
    if not alerts:
        return ""
    header = f"🦅 Corvin Jarvis — {datetime.now().strftime('%H:%M KST')}"
```
with:
```python
def _format_message(alerts: list[dict[str, Any]], compact: bool = False,
                    title: str = "", limit: int = 5) -> str:
    if not alerts:
        return ""
    header = title or f"🦅 Corvin Jarvis — {datetime.now().strftime('%H:%M KST')}"
```
And in the compact branch change `alerts[:5]` → `alerts[:limit]` and `len(alerts) > 5` → `len(alerts) > limit` and `len(alerts) - 5` → `len(alerts) - limit`. In the non-compact branch wrap the loop to `for a in alerts[:limit]:` and append `…외 {len(alerts)-limit}건` if `len(alerts) > limit`.

Replace `notify()`:
```python
def notify(mode: str = "urgent", cooldown_s: int = DEFAULT_COOLDOWN) -> NotifyResult:
    alerts = _load_json(ALERTS_FILE).get("alerts", [])
    cfg = _config().get("notification", {})
    held = _held_symbols()

    if mode == "digest":
        min_sev = cfg.get("digest_min_severity", "medium")
        limit = int(cfg.get("digest_max_items", 8))
        title = f"📋 Corvin 일일 다이제스트 — {datetime.now().strftime('%m/%d %H:%M KST')}"
    else:
        min_sev = cfg.get("urgent_min_severity", "high")
        limit = int(cfg.get("max_per_push", 8))
        title = f"🦅 Corvin 긴급 — {datetime.now().strftime('%H:%M KST')}"

    sev_filtered = _filter_severity(alerts, min_sev)
    skipped_sev = len(alerts) - len(sev_filtered)

    if mode == "digest":
        fresh, skipped_dedup = sev_filtered, 0
    else:
        fresh, skipped_dedup = _filter_dedup(sev_filtered, cooldown_s)

    fresh = _rank_alerts(fresh, held)
    log.info("mode=%s alerts=%d sev_pass=%d fresh=%d (skip_sev=%d, skip_dedup=%d)",
             mode, len(alerts), len(sev_filtered), len(fresh), skipped_sev, skipped_dedup)

    if not fresh:
        return NotifyResult(pushed=0, skipped_dedup=skipped_dedup, skipped_severity=skipped_sev, channels_delivered=[])

    msg_long = _format_message(fresh, compact=False, title=title, limit=limit)
    msg_short = _format_message(fresh, compact=True, title=title, limit=limit)
    delivered: list[str] = []

    webhook = _webhook_url()
    if webhook and _send_discord(webhook, msg_long):
        delivered.append("discord")
        log.info("Discord 전송 성공")

    imsg = _imessage_recipient()
    if imsg and _send_imessage(imsg, msg_short):
        delivered.append("imessage")
        log.info("iMessage 전송 성공 → %s", imsg)

    if not delivered:
        _queue_file(msg_long, len(fresh))
        log.warning("어떤 채널도 전송 실패 — pending file에 적재")

    return NotifyResult(pushed=len(fresh), skipped_dedup=skipped_dedup,
                        skipped_severity=skipped_sev, channels_delivered=delivered)
```

Update `__main__`:
```python
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "urgent"
    result = notify(mode=mode)
    log.info("결과(%s): %s", mode, result)
```

- [ ] **Step 4: Run — expect PASS** `python3 -m pytest tests/test_notify.py -v` → all pass.

- [ ] **Step 5: Full suite** `python3 -m pytest tests/ -q` → no regressions.

- [ ] **Step 6: Commit**
```bash
git add corvin_jarvis/notify.py tests/test_notify.py
git commit -m "feat(corvin): mode-aware notify (urgent push vs daily digest) (smart filter)"
```

---

### Task C4: cron — urgent hourly + daily digest scripts

**Files:**
- Modify: `corvin_jarvis/run_jarvis.sh`
- Create: `corvin_jarvis/run_digest.sh`

- [ ] **Step 1:** In `corvin_jarvis/run_jarvis.sh`, change the notify invocation to urgent mode. Replace:
```bash
"$PY" "$SCRIPT_DIR/notify.py" >> "$LOG" 2>&1 || echo "[ERROR] notify.py 실패" >> "$LOG"
```
with:
```bash
"$PY" "$SCRIPT_DIR/notify.py" urgent >> "$LOG" 2>&1 || echo "[ERROR] notify.py 실패" >> "$LOG"
```

- [ ] **Step 2:** Create `corvin_jarvis/run_digest.sh` (mirrors run_jarvis.sh env-loading, runs jarvis pipeline then digest notify):
```bash
#!/usr/bin/env bash
# Corvin 일일 다이제스트 — 하루 1회 장마감 후. jarvis 파이프라인 갱신 후 digest 모드 push.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
LOG="$SCRIPT_DIR/state/cron.log"
mkdir -p "$SCRIPT_DIR/state"
PY=/usr/bin/python3
if command -v /opt/homebrew/bin/python3 >/dev/null 2>&1; then PY=/opt/homebrew/bin/python3; fi
if [ -f "$SCRIPT_DIR/.env" ]; then set -a; . "$SCRIPT_DIR/.env"; set +a; fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Digest 시작" >> "$LOG"
"$PY" "$SCRIPT_DIR/jarvis.py" >> "$LOG" 2>&1 || echo "[ERROR] jarvis.py 실패" >> "$LOG"
"$PY" "$SCRIPT_DIR/notify.py" digest >> "$LOG" 2>&1 || echo "[ERROR] digest 실패" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Digest 완료" >> "$LOG"
```

- [ ] **Step 3:** `chmod +x corvin_jarvis/run_digest.sh`

- [ ] **Step 4: Commit**
```bash
git add corvin_jarvis/run_jarvis.sh corvin_jarvis/run_digest.sh
git commit -m "feat(corvin): urgent hourly + daily digest run scripts (smart filter)"
```

> **NOTE:** Actual crontab re-enable (hourly urgent + 16:00 digest) is performed by the controller AFTER merge, with user confirmation — not in this plan.

---

## Self-Review
- Design coverage: HIGH+ hourly gate ✅(C1/C3), daily digest MEDIUM+ ranked ✅(C3), actionability(보유) weighting ✅(C2), count cap ✅(C3 limit). cron split ✅(C4).
- Placeholders: none.
- Type consistency: `_held_symbols()->set`, `_actionability(alert,held)->int`, `_rank_alerts(alerts,held)->list`, `notify(mode,cooldown_s)`, `_format_message(alerts,compact,title,limit)` — consistent across tasks.

## Notes
- 숫자(임계/개수/디제스트 시각)는 config·crontab에서 조정 가능.
- crontab 재가동은 머지 후 사용자 확인하에 controller가 수행.
