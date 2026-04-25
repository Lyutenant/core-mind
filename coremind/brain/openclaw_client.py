from __future__ import annotations

from coremind.brain.base import BrainClient


class OpenClawClient(BrainClient):
    def __init__(self, base_url: str, agent: str, timeout: int = 60) -> None:
        self.base_url = base_url
        self.agent = agent
        self.timeout = timeout

    def ask(self, messages: list[dict]) -> str:
        raise NotImplementedError
