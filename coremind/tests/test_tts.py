from __future__ import annotations

import subprocess

import pytest

from coremind import TTSError
from coremind.tts.piper_local import EspeakTTS, MockTTS


# ---------------------------------------------------------------------------
# MockTTS
# ---------------------------------------------------------------------------

def test_mock_tts_returns_path():
    tts = MockTTS()
    assert tts.synthesize("hello world", "/tmp/out.wav") == "/tmp/out.wav"


# ---------------------------------------------------------------------------
# PiperLocalTTS
# ---------------------------------------------------------------------------

def test_piper_tts_raises_if_package_not_installed(mocker):
    mocker.patch.dict("sys.modules", {"piper": None, "piper.voice": None})
    import importlib
    import coremind.tts.piper_local as mod
    importlib.reload(mod)

    with pytest.raises(TTSError, match="piper-tts"):
        mod.PiperLocalTTS(model_path="/some/model.onnx")


def test_piper_tts_raises_on_bad_model_path(mocker):
    mock_voice_cls = mocker.MagicMock()
    mock_voice_cls.load.side_effect = Exception("File not found")
    mocker.patch.dict("sys.modules", {"piper": mocker.MagicMock(), "piper.voice": mocker.MagicMock(PiperVoice=mock_voice_cls)})

    import importlib
    import coremind.tts.piper_local as mod
    importlib.reload(mod)

    with pytest.raises(TTSError, match="Failed to load"):
        mod.PiperLocalTTS(model_path="/bad/path.onnx")


def test_piper_tts_synthesize_calls_voice(mocker, tmp_path):
    import numpy as np

    mock_chunk = mocker.MagicMock()
    mock_chunk.sample_channels = 1
    mock_chunk.sample_width = 2
    mock_chunk.sample_rate = 22050
    mock_chunk.audio_float_array = np.zeros(100, dtype=np.float32)

    mock_voice = mocker.MagicMock()
    mock_voice.synthesize.return_value = [mock_chunk]
    mock_voice_cls = mocker.MagicMock()
    mock_voice_cls.load.return_value = mock_voice

    mocker.patch.dict(
        "sys.modules",
        {"piper": mocker.MagicMock(), "piper.voice": mocker.MagicMock(PiperVoice=mock_voice_cls)},
    )

    import importlib
    import coremind.tts.piper_local as mod
    importlib.reload(mod)

    out = tmp_path / "out.wav"
    tts = mod.PiperLocalTTS(model_path="/model.onnx")
    result = tts.synthesize("hello", str(out))

    assert result == str(out)
    mock_voice.synthesize.assert_called_once_with("hello")


# ---------------------------------------------------------------------------
# EspeakTTS
# ---------------------------------------------------------------------------

def test_espeak_tts_calls_subprocess(mocker):
    mock_run = mocker.patch("subprocess.run")
    tts = EspeakTTS(voice="en")
    tts.synthesize("Hello world", "/tmp/out.wav")
    mock_run.assert_called_once_with(
        ["espeak-ng", "-v", "en", "-w", "/tmp/out.wav", "Hello world"],
        check=True,
        capture_output=True,
    )


def test_espeak_tts_raises_if_not_installed(mocker):
    mocker.patch("subprocess.run", side_effect=FileNotFoundError)
    tts = EspeakTTS()
    with pytest.raises(TTSError, match="espeak-ng not found"):
        tts.synthesize("test", "/tmp/out.wav")


def test_espeak_tts_raises_on_nonzero_exit(mocker):
    mocker.patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "espeak-ng", stderr=b"error msg"),
    )
    tts = EspeakTTS()
    with pytest.raises(TTSError, match="espeak-ng failed"):
        tts.synthesize("test", "/tmp/out.wav")


def test_espeak_tts_default_voice_is_en():
    tts = EspeakTTS()
    assert tts.voice == "en"


# ---------------------------------------------------------------------------
# VoiceLoop + TTS integration
# ---------------------------------------------------------------------------

def test_voice_loop_speaks_when_tts_configured(mocker):
    from coremind.audio_input.recorder import Recorder
    from coremind.audio_output.player import Player
    from coremind.brain.ollama_client import MockBrainClient
    from coremind.memory.session_memory import SessionMemory
    from coremind.stt.whisper_local import MockSTT
    from coremind.voice_loop import VoiceLoop

    recorder = Recorder()
    mocker.patch.object(recorder, "record")

    stt = MockSTT()
    mocker.patch.object(stt, "transcribe", return_value="hello")

    brain = MockBrainClient()
    mocker.patch.object(brain, "ask", return_value="hi there")

    tts = MockTTS()
    mock_synthesize = mocker.patch.object(tts, "synthesize", return_value="/tmp/fake.wav")

    player = Player()
    mock_play = mocker.patch.object(player, "play")

    loop = VoiceLoop(
        name="Bot", recorder=recorder, stt=stt, brain=brain,
        memory=SessionMemory(), tts=tts, player=player,
    )
    t, r = loop.run_once()

    assert t == "hello"
    assert r == "hi there"
    mock_synthesize.assert_called_once_with("hi there", mocker.ANY)
    mock_play.assert_called_once()


def test_voice_loop_tts_failure_does_not_raise(mocker):
    from coremind.audio_input.recorder import Recorder
    from coremind.audio_output.player import Player
    from coremind.brain.ollama_client import MockBrainClient
    from coremind.memory.session_memory import SessionMemory
    from coremind.stt.whisper_local import MockSTT
    from coremind.voice_loop import VoiceLoop

    recorder = Recorder()
    mocker.patch.object(recorder, "record")

    stt = MockSTT()
    mocker.patch.object(stt, "transcribe", return_value="hello")

    brain = MockBrainClient()
    mocker.patch.object(brain, "ask", return_value="response")

    tts = MockTTS()
    mocker.patch.object(tts, "synthesize", side_effect=TTSError("synth failed"))

    player = Player()

    loop = VoiceLoop(
        name="Bot", recorder=recorder, stt=stt, brain=brain,
        memory=SessionMemory(), tts=tts, player=player,
    )
    t, r = loop.run_once()
    assert r == "response"


def test_voice_loop_no_tts_skips_playback(mocker):
    from coremind.audio_input.recorder import Recorder
    from coremind.brain.ollama_client import MockBrainClient
    from coremind.memory.session_memory import SessionMemory
    from coremind.stt.whisper_local import MockSTT
    from coremind.voice_loop import VoiceLoop

    recorder = Recorder()
    mocker.patch.object(recorder, "record")

    stt = MockSTT()
    mocker.patch.object(stt, "transcribe", return_value="hello")

    brain = MockBrainClient()
    mocker.patch.object(brain, "ask", return_value="ok")

    loop = VoiceLoop(
        name="Bot", recorder=recorder, stt=stt, brain=brain,
        memory=SessionMemory(), tts=None, player=None,
    )
    t, r = loop.run_once()
    assert r == "ok"
