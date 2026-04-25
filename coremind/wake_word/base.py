from __future__ import annotations

from abc import ABC, abstractmethod


class WakeWordDetector(ABC):
    @abstractmethod
    def listen_until_wake_word(self) -> None: ...
