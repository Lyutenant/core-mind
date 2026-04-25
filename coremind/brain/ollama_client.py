from __future__ import annotations

from coremind.brain.base import BrainClient


class OllamaClient(BrainClient):
    def __init__(self, base_url: str, model: str, timeout: int = 60) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def ask(self, messages: list[dict]) -> str:
        raise NotImplementedError


class MockBrainClient(BrainClient):
    def ask(self, messages: list[dict]) -> str:
        return "[mock response]"
