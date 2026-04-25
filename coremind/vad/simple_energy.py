from __future__ import annotations

from coremind.vad.base import VoiceActivityDetector


class SimpleEnergyVAD(VoiceActivityDetector):
    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def is_speech(self, audio_chunk: bytes) -> bool:
        raise NotImplementedError
