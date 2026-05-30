from __future__ import annotations

from abc import ABC, abstractmethod


class WakeWordDetector(ABC):
    @property
    def trigger_prompt(self) -> str:
        return "Waiting for trigger..."

    @abstractmethod
    def listen_until_wake_word(self) -> None: ...
