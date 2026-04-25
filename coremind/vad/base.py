from __future__ import annotations

from abc import ABC, abstractmethod


class VoiceActivityDetector(ABC):
    @abstractmethod
    def is_speech(self, audio_chunk: bytes) -> bool: ...
