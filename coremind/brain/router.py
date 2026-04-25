from __future__ import annotations

from coremind.brain.base import BrainClient


class BrainRouter(BrainClient):
    def __init__(self, clients: list[BrainClient]) -> None:
        self._clients = clients

    def ask(self, messages: list[dict]) -> str:
        raise NotImplementedError
