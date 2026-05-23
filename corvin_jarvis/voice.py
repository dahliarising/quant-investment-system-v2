"""Corvin Jarvis — Voice Interface Stub (Tier 4.2)

⚠️ NOT IMPLEMENTED — interface stub만.

CLAUDE.md 정책:
- "ElevenLabs/Kling API 사용 시 반드시 사용자 허가 필요"
- 따라서 실제 API call은 wiring 시 별도 PR + 폐하 명시적 승인 필요.

향후 implementation 시 의존성:
- transcribe: OpenAI Whisper API (또는 local whisper.cpp)
- synthesize: ElevenLabs TTS API

interface는 정해두어 호출 측 코드는 미리 작성 가능.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("corvin.voice")


def transcribe(audio_path: Path) -> str:
    """STT: audio → text. Whisper API 구현 미수행.

    향후: OpenAI Whisper API 또는 local whisper.cpp.
    """
    raise NotImplementedError(
        "Whisper API integration not wired. "
        "구현 시 OpenAI client + audio file upload. "
        "CLAUDE.md에 추가 정책 확인 필요."
    )


def synthesize(
    text: str,
    voice_id: str = "default",
    confirmed_paid_use: bool = False,
) -> Path:
    """TTS: text → audio file. ElevenLabs API 구현 미수행.

    ⚠️ ElevenLabs는 CLAUDE.md에 따라 사용자 명시 허가 필요.
    confirmed_paid_use=True로 호출해도 현재는 wiring 미완으로 NotImplementedError.
    """
    if not confirmed_paid_use:
        raise NotImplementedError(
            "ElevenLabs API requires explicit user permission per CLAUDE.md. "
            "Pass confirmed_paid_use=True after 폐하 승인."
        )
    raise NotImplementedError(
        "ElevenLabs TTS integration not yet wired even with permission flag. "
        "TODO: elevenlabs.generate(text, voice=voice_id) → save → return Path."
    )
