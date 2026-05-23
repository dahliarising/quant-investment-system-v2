# Tier 4 Moonshot 설계 노트 (구현 없음)

2026-05-23 작성. Tier 1~3 완전 구현 후 Tier 4 항목을 추후 구현하기 위한 architecture sketch.

## 4.1 Multi-Agent Debate

### 목적
매수/매도 결정 전 Bull agent vs Bear agent 토론 → 폐하 moderation → 최종 결정.

### Architecture
```
trigger (e.g., !corvin debate META buy)
   │
   ├─→ Bull Agent (Haiku) — 매수 근거 5개 enumerate
   │     └ tools: qa.explain_move, narrative.latest_signal,
   │              earnings.pending_earnings (긍정 신호 강조)
   │
   ├─→ Bear Agent (Haiku) — 매도/보류 근거 5개 enumerate
   │     └ tools: 동일 (부정 신호 강조)
   │
   └─→ Moderator (Opus) — 양쪽 주장 정리 + 결정 권고
```

### 구현 모듈
- `corvin_jarvis/debate.py`
  - `bull_arguments(symbol, context)` → list[str]
  - `bear_arguments(symbol, context)` → list[str]
  - `moderate(bull_args, bear_args, context)` → decision dict

### 비용 고려
- 한 번 호출 = Haiku × 2 + Opus moderation ≈ $0.05-0.10
- 매일 사용 시 월 ~$3 (감당 가능)
- 추가 가드: cooldown (같은 종목 1일 1회 limit)

### 의존성
- 이미 구현된 retrieval (qa, narrative, earnings) 재사용
- Anthropic Messages API (이미 사용 중)

---

## 4.2 Voice Interface

### 목적
휴대폰 → Whisper STT → Corvin → ElevenLabs TTS → Discord 음성/오디오 전달.

### Architecture
```
phone mic → Whisper API (transcription)
   ↓
Discord text message OR direct Corvin call
   ↓
Corvin processes (Claude)
   ↓
ElevenLabs TTS → audio file
   ↓
Discord voice channel OR file attachment
```

### 구현 모듈
- `corvin_jarvis/voice.py`
  - `transcribe(audio_path) → text` (Whisper)
  - `synthesize(text, voice_id) → audio_path` (ElevenLabs)
- Discord bot 측: audio file을 voice channel에 post

### 제약
- Discord bot이 voice channel join하려면 추가 권한 + voice library (FFmpeg 의존성)
- ElevenLabs API 비용 (글자당)
- CLAUDE.md 정책: **ElevenLabs는 폐하 명시적 허가 필요**

### 우선순위
낮음 — text Discord로 이미 충분히 작동 중. voice는 운전 중 등 특수 상황 한정.

---

## 4.3 Counterfactual Coach

### 목적
"5/19 청산 안 했으면 -₩XXX, 청산 잘한 결정 (확률 87%로 정확)" — 매월 1회 회고 자동 push.

### Architecture
```
monthly trigger (1st of each month)
   ↓
attribution.weekly_report × 4 → 한 달 분 reasoning 통합
   ↓
counterfactual analysis:
   for each session decision:
     - what actually happened (timeseries lookup)
     - what would have happened if opposite choice
     - confidence calibration (Brier score 등)
   ↓
narrative summary → wiki + Discord push
```

### 구현 모듈
- `corvin_jarvis/counterfactual.py`
  - `simulate_alternative(decision, timeseries_db) → outcome dict`
  - `calibrate_confidence(decisions, outcomes) → Brier score`
  - `monthly_review(wiki_dir, db_path, month) → markdown report`

### 의존성
- attribution.py (Tier 2.4) ✅
- whatif.py (Tier 2.1) — alternative simulation ✅
- 시계열 30일+ 누적 (Tier 1.1) ✅

### 측정
- Brier score < 0.25 (well-calibrated)
- 사용자 신뢰도 self-report

---

## 우선순위 (실제 implementation 시점에서 재평가)

1. **4.3 Counterfactual Coach** — 의존성 모두 충족, 가장 즉시 가치 (calibration → 신뢰)
2. **4.1 Multi-Agent Debate** — 비용 감당 가능, 의사결정 quality 향상
3. **4.2 Voice Interface** — 우선순위 낮음, 특수 use case
