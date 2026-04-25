from __future__ import annotations

from abc import ABC, abstractmethod


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> str: ...
