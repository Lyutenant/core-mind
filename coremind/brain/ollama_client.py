from __future__ import annotations

import logging

import httpx

from coremind import BrainError
from coremind.brain.base import BrainClient

logger = logging.getLogger(__name__)


def _inject_no_think(messages: list[dict]) -> list[dict]:
    """Append /no_think to the system message to suppress Qwen3 chain-of-thought."""
    result = list(messages)
    for i, msg in enumerate(result):
        if msg["role"] == "system":
            result[i] = {**msg, "content": msg["content"] + "\n/no_think"}
            return result
    return [{"role": "system", "content": "/no_think"}] + result


class OllamaClient(BrainClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 60,
        no_think: bool = False,
        options: dict | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.no_think = no_think
        self.options = options or {}

    def ask(self, messages: list[dict]) -> str:
        msgs = _inject_no_think(messages) if self.no_think else messages
        url = f"{self.base_url}/api/chat"
        payload: dict = {"model": self.model, "messages": msgs, "stream": False}
        if self.options:
            payload["options"] = self.options
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except httpx.ConnectError as e:
            raise BrainError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Check that Ollama is running on your Mac Mini and the Tailscale IP is correct."
            ) from e
        except httpx.TimeoutException as e:
            raise BrainError(
                f"Ollama request timed out after {self.timeout}s. "
                "The model may still be loading or the Mac Mini is busy."
            ) from e
        except httpx.HTTPStatusError as e:
            raise BrainError(
                f"Ollama returned HTTP {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.TransportError as e:
            raise BrainError(f"Network error communicating with Ollama: {e}") from e
        except (KeyError, ValueError) as e:
            raise BrainError(f"Unexpected Ollama response format: {e}") from e


class MockBrainClient(BrainClient):
    def ask(self, messages: list[dict]) -> str:
        return "[mock response]"
