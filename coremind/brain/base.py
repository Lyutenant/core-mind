from __future__ import annotations

from abc import ABC, abstractmethod


class BrainClient(ABC):
    @abstractmethod
    def ask(self, messages: list[dict]) -> str: ...
