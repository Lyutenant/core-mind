from __future__ import annotations

import pytest

from coremind import AudioInputError
from coremind.brain.ollama_client import MockBrainClient
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


# ---------------------------------------------------------------------------
# VoiceLoop tests
# ---------------------------------------------------------------------------

def _make_loop(mocker, transcript="hello world", response="[mock response]"):
    from coremind.audio_input.recorder import Recorder
    from coremind.voice_loop import VoiceLoop

    recorder = Recorder()
    mocker.patch.object(recorder, "record")

    stt = MockSTT()
    mocker.patch.object(stt, "transcribe", return_value=transcript)

    brain = MockBrainClient()
    mocker.patch.object(brain, "ask", return_value=response)

    memory = SessionMemory(max_turns=5)
    return VoiceLoop(name="TestBot", recorder=recorder, stt=stt, brain=brain, memory=memory, record_seconds=3)


def test_voice_loop_run_once_returns_transcript_and_response(mocker):
    loop = _make_loop(mocker, transcript="what time is it", response="I don't know the time.")
    t, r = loop.run_once()
    assert t == "what time is it"
    assert r == "I don't know the time."


def test_voice_loop_run_once_updates_memory(mocker):
    loop = _make_loop(mocker, transcript="hello", response="hi back")
    loop.run_once()
    messages = loop._memory.get_messages()
    assert any(m["role"] == "user" and m["content"] == "hello" for m in messages)
    assert any(m["role"] == "assistant" and m["content"] == "hi back" for m in messages)


def test_voice_loop_empty_transcript_returns_empty(mocker):
    loop = _make_loop(mocker, transcript="   ")
    t, r = loop.run_once()
    assert t == ""
    assert r == ""
    assert loop._memory.get_messages() == []


def test_voice_loop_memory_not_mutated_on_brain_failure(mocker):
    from coremind import BrainError
    from coremind.audio_input.recorder import Recorder
    from coremind.voice_loop import VoiceLoop

    recorder = Recorder()
    mocker.patch.object(recorder, "record")
    stt = MockSTT()
    mocker.patch.object(stt, "transcribe", return_value="a question")
    brain = MockBrainClient()
    mocker.patch.object(brain, "ask", side_effect=BrainError("Ollama down"))
    memory = SessionMemory()

    loop = VoiceLoop(name="Bot", recorder=recorder, stt=stt, brain=brain, memory=memory)
    with pytest.raises(BrainError):
        loop.run_once()

    assert memory.get_messages() == []


def test_voice_loop_propagates_audio_error(mocker):
    from coremind.audio_input.recorder import Recorder
    from coremind.voice_loop import VoiceLoop

    recorder = Recorder()
    mocker.patch.object(recorder, "record", side_effect=AudioInputError("no device"))
    loop = VoiceLoop(
        name="Bot", recorder=recorder, stt=MockSTT(), brain=MockBrainClient(),
        memory=SessionMemory(), record_seconds=3,
    )
    with pytest.raises(AudioInputError):
        loop.run_once()


def test_voice_loop_includes_system_prompt_in_messages(mocker):
    from coremind.audio_input.recorder import Recorder
    from coremind.voice_loop import VoiceLoop

    recorder = Recorder()
    mocker.patch.object(recorder, "record")

    stt = MockSTT()
    mocker.patch.object(stt, "transcribe", return_value="hi")

    brain = MockBrainClient()
    captured = []
    mocker.patch.object(brain, "ask", side_effect=lambda msgs: captured.extend(msgs) or "ok")

    loop = VoiceLoop(name="Jarvis", recorder=recorder, stt=stt, brain=brain, memory=SessionMemory())
    loop.run_once()

    assert captured[0]["role"] == "system"
    assert "Jarvis" in captured[0]["content"]
