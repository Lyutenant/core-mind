from __future__ import annotations

import pytest

from coremind.memory.session_memory import SessionMemory
from coremind.stt.whisper_local import MockSTT
from coremind.tts.piper_local import MockTTS


def test_mock_stt_returns_string():
    stt = MockSTT()
    result = stt.transcribe("dummy.wav")
    assert isinstance(result, str)


def test_mock_tts_returns_path():
    tts = MockTTS()
    result = tts.synthesize("hello world", "/tmp/out.wav")
    assert result == "/tmp/out.wav"


def test_session_memory_stores_turns():
    mem = SessionMemory(max_turns=3)
    mem.add("user", "hello")
    mem.add("assistant", "hi there")
    messages = mem.get_messages()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1] == {"role": "assistant", "content": "hi there"}


def test_session_memory_truncates_old_turns():
    mem = SessionMemory(max_turns=2)
    for i in range(10):
        mem.add("user", f"message {i}")
    messages = mem.get_messages()
    assert len(messages) <= 4  # max_turns * 2


def test_session_memory_clear():
    mem = SessionMemory()
    mem.add("user", "hello")
    mem.clear()
    assert mem.get_messages() == []
