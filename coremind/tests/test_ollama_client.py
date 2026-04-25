from __future__ import annotations

import pytest

from coremind.brain.ollama_client import MockBrainClient, OllamaClient


def test_mock_brain_client_returns_string():
    client = MockBrainClient()
    result = client.ask([{"role": "user", "content": "hello"}])
    assert isinstance(result, str)
    assert len(result) > 0


def test_ollama_client_raises_not_implemented():
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    with pytest.raises(NotImplementedError):
        client.ask([{"role": "user", "content": "hello"}])
