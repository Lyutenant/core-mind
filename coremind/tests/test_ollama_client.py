from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from coremind import BrainError
from coremind.brain.ollama_client import MockBrainClient, OllamaClient
from coremind.brain.router import BrainRouter


def test_mock_brain_client_returns_string():
    client = MockBrainClient()
    result = client.ask([{"role": "user", "content": "hello"}])
    assert isinstance(result, str)
    assert len(result) > 0


def test_ollama_client_success(mocker):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "Hello back!"}}
    mock_resp.raise_for_status.return_value = None
    mocker.patch("httpx.post", return_value=mock_resp)

    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    result = client.ask([{"role": "user", "content": "hello"}])
    assert result == "Hello back!"


def test_ollama_client_strips_trailing_slash(mocker):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "ok"}}
    mock_resp.raise_for_status.return_value = None
    patched = mocker.patch("httpx.post", return_value=mock_resp)

    client = OllamaClient(base_url="http://localhost:11434/", model="test")
    client.ask([{"role": "user", "content": "hi"}])
    called_url = patched.call_args[0][0]
    assert called_url == "http://localhost:11434/api/chat"


def test_ollama_client_connect_error_raises_brain_error(mocker):
    mocker.patch("httpx.post", side_effect=httpx.ConnectError("Connection refused"))
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    with pytest.raises(BrainError, match="Cannot reach Ollama"):
        client.ask([{"role": "user", "content": "hello"}])


def test_ollama_client_timeout_raises_brain_error(mocker):
    mocker.patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    with pytest.raises(BrainError, match="timed out"):
        client.ask([{"role": "user", "content": "hello"}])


def test_ollama_client_http_error_raises_brain_error(mocker):
    err_response = MagicMock(status_code=500, text="Internal server error")
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=err_response
    )
    mocker.patch("httpx.post", return_value=mock_resp)

    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    with pytest.raises(BrainError, match="HTTP 500"):
        client.ask([{"role": "user", "content": "hello"}])


def test_no_think_appended_to_existing_system_message():
    from coremind.brain.ollama_client import _inject_no_think
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hi"},
    ]
    result = _inject_no_think(msgs)
    assert result[0]["content"].endswith("/no_think")
    assert result[1] == {"role": "user", "content": "hi"}


def test_no_think_prepends_system_when_none_present():
    from coremind.brain.ollama_client import _inject_no_think
    msgs = [{"role": "user", "content": "hi"}]
    result = _inject_no_think(msgs)
    assert result[0] == {"role": "system", "content": "/no_think"}
    assert result[1] == {"role": "user", "content": "hi"}


def test_no_think_sent_to_ollama_when_enabled(mocker):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "ok"}}
    mock_resp.raise_for_status.return_value = None
    patched = mocker.patch("httpx.post", return_value=mock_resp)

    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b", no_think=True)
    client.ask([{"role": "system", "content": "You are a bot."}, {"role": "user", "content": "hi"}])

    sent_messages = patched.call_args[1]["json"]["messages"]
    assert "/no_think" in sent_messages[0]["content"]


def test_options_passed_to_ollama(mocker):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "ok"}}
    mock_resp.raise_for_status.return_value = None
    patched = mocker.patch("httpx.post", return_value=mock_resp)

    client = OllamaClient(
        base_url="http://localhost:11434", model="qwen3:8b",
        options={"num_predict": 150, "temperature": 0.5},
    )
    client.ask([{"role": "user", "content": "hi"}])

    sent = patched.call_args[1]["json"]
    assert sent["options"] == {"num_predict": 150, "temperature": 0.5}


def test_empty_options_not_sent_to_ollama(mocker):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "ok"}}
    mock_resp.raise_for_status.return_value = None
    patched = mocker.patch("httpx.post", return_value=mock_resp)

    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b", options={})
    client.ask([{"role": "user", "content": "hi"}])

    sent = patched.call_args[1]["json"]
    assert "options" not in sent


def test_brain_router_uses_primary_when_available(mocker):
    primary = MockBrainClient()
    mocker.patch.object(primary, "ask", return_value="primary response")
    router = BrainRouter(primary=primary)
    assert router.ask([{"role": "user", "content": "hi"}]) == "primary response"


def test_brain_router_falls_back_when_primary_fails(mocker):
    primary = OllamaClient(base_url="http://localhost:11434", model="test")
    mocker.patch.object(primary, "ask", side_effect=BrainError("Ollama down"))
    fallback = MockBrainClient()
    router = BrainRouter(primary=primary, fallback=fallback)
    result = router.ask([{"role": "user", "content": "hi"}])
    assert result == "[mock response]"


def test_brain_router_raises_without_fallback(mocker):
    primary = OllamaClient(base_url="http://localhost:11434", model="test")
    mocker.patch.object(primary, "ask", side_effect=BrainError("Ollama down"))
    router = BrainRouter(primary=primary, fallback=None)
    with pytest.raises(BrainError):
        router.ask([{"role": "user", "content": "hi"}])
