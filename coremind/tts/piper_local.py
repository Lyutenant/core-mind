from __future__ import annotations

from coremind.tts.base import TextToSpeech


class MockTTS(TextToSpeech):
    def synthesize(self, text: str, output_path: str) -> str:
        return output_path


class PiperLocalTTS(TextToSpeech):
    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice

    def synthesize(self, text: str, output_path: str) -> str:
        raise NotImplementedError("Install piper-tts: pip install piper-tts")
