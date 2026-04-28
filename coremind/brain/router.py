from __future__ import annotations

import logging

from coremind import BrainError
from coremind.brain.base import BrainClient

logger = logging.getLogger(__name__)


class BrainRouter(BrainClient):
    def __init__(self, primary: BrainClient, *, fallback: BrainClient | None = None) -> None:
        self._primary = primary
        self._fallback = fallback

    def ask(self, messages: list[dict]) -> str:
        try:
            return self._primary.ask(messages)
        except BrainError as e:
            if self._fallback is not None:
                logger.warning(
                    "PRIMARY BRAIN FAILED — falling back to %s. Error: %s",
                    type(self._fallback).__name__,
                    e,
                )
                return self._fallback.ask(messages)
            raise
