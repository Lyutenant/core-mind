from __future__ import annotations

from coremind.stt.base import SpeechToText


class MockSTT(SpeechToText):
    def transcribe(self, wav_path: str) -> str:
        return "[mock transcript]"


class WhisperLocalSTT(SpeechToText):
    def __init__(self, model: str = "base", language: str = "en") -> None:
        self.model = model
        self.language = language

    def transcribe(self, wav_path: str) -> str:
        raise NotImplementedError("Install faster-whisper: pip install faster-whisper")
