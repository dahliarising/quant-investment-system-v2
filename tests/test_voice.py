"""Tests for corvin_jarvis.voice — stub (no API calls per CLAUDE.md policy)."""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis import voice


@pytest.mark.unit
def test_transcribe_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Whisper"):
        voice.transcribe(Path("audio.wav"))


@pytest.mark.unit
def test_synthesize_raises_not_implemented_without_permission() -> None:
    """ElevenLabs API는 CLAUDE.md에서 사용자 명시 허가 필요."""
    with pytest.raises(NotImplementedError, match="ElevenLabs"):
        voice.synthesize("hello", voice_id="default")


@pytest.mark.unit
def test_synthesize_with_explicit_permission_still_stub_until_wired() -> None:
    """현재는 explicit permission flag 있어도 미구현 (TODO 명시)."""
    with pytest.raises(NotImplementedError):
        voice.synthesize("hello", voice_id="default", confirmed_paid_use=True)
