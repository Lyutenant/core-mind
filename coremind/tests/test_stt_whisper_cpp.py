"""Tests for the whisper.cpp STT provider and STT provider selection.

No model is ever loaded: pywhispercpp is replaced by a fake module, and the
kwargs-building logic is tested on instances built without __init__.
"""
from __future__ import annotations

import logging
import sys
import textwrap
import types

import numpy as np
import pytest

from coremind import ConfigError, STTError
from coremind.config.settings import STTConfig, load_settings
from coremind.stt import make_stt, stt_model_label
from coremind.stt import whisper_cpp
from coremind.stt.whisper_cpp import WhisperCppSTT


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_whisper_cpp_config_defaults():
    cfg = STTConfig()
    assert cfg.provider == "whisper_local"  # faster-whisper stays the default
    assert cfg.whisper_cpp.model == "large-v3-turbo"
    assert cfg.whisper_cpp.model_path is None
    assert cfg.whisper_cpp.n_threads is None
    assert cfg.whisper_cpp.vad_model_path is None


def test_whisper_cpp_config_loads_from_yaml(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(textwrap.dedent("""\
        stt:
          provider: whisper_cpp
          model: distil-large-v3
          whisper_cpp:
            model: medium.en
            model_path: ~/models/ggml-custom.bin
            n_threads: 6
    """))
    settings = load_settings(str(config))
    assert settings.stt.provider == "whisper_cpp"
    # faster-whisper's model choice is untouched by the whisper.cpp block.
    assert settings.stt.model == "distil-large-v3"
    assert settings.stt.whisper_cpp.model == "medium.en"
    assert settings.stt.whisper_cpp.n_threads == 6
    assert "~" not in settings.stt.whisper_cpp.model_path
    assert settings.stt.whisper_cpp.model_path.endswith("models/ggml-custom.bin")


def test_dashboard_save_preserves_unsent_whisper_cpp_keys():
    from coremind.config.settings import merge_config_text

    existing = "stt:\n  provider: whisper_local\n  whisper_cpp:\n    n_threads: 6\n"
    merged = merge_config_text(
        existing, {"stt": {"provider": "whisper_cpp", "whisper_cpp": {"model": "small.en"}}}
    )
    assert "n_threads: 6" in merged
    assert "model: small.en" in merged


# ---------------------------------------------------------------------------
# Decode parameters
# ---------------------------------------------------------------------------

def _make_stt_without_loading_model(**overrides) -> WhisperCppSTT:
    stt = WhisperCppSTT.__new__(WhisperCppSTT)
    stt.language = overrides.get("language", "en")
    stt.n_threads = overrides.get("n_threads")
    stt.beam_size = overrides.get("beam_size", 5)
    stt.vad_filter = overrides.get("vad_filter", False)
    stt.vad_model_path = overrides.get("vad_model_path")
    stt.initial_prompt = overrides.get("initial_prompt")
    stt.hotwords = overrides.get("hotwords")
    return stt


def test_transcribe_kwargs_minimal():
    kwargs = _make_stt_without_loading_model()._transcribe_kwargs()
    assert kwargs == {
        "language": "en",
        "print_progress": False,
        "print_realtime": False,
        "beam_search": {"beam_size": 5, "patience": -1.0},
    }


def test_transcribe_kwargs_greedy_when_beam_size_1():
    kwargs = _make_stt_without_loading_model(beam_size=1)._transcribe_kwargs()
    assert "beam_search" not in kwargs


def test_transcribe_kwargs_folds_hotwords_into_prompt():
    kwargs = _make_stt_without_loading_model(
        initial_prompt="ctx", hotwords="KJYO METAR"
    )._transcribe_kwargs()
    assert kwargs["initial_prompt"] == "ctx KJYO METAR"
    assert "hotwords" not in kwargs


def test_transcribe_kwargs_hotwords_only():
    kwargs = _make_stt_without_loading_model(hotwords="KJYO")._transcribe_kwargs()
    assert kwargs["initial_prompt"] == "KJYO"


def test_transcribe_kwargs_vad_needs_model_path():
    no_path = _make_stt_without_loading_model(vad_filter=True)._transcribe_kwargs()
    assert "vad" not in no_path
    with_path = _make_stt_without_loading_model(
        vad_filter=True, vad_model_path="/m/silero.bin"
    )._transcribe_kwargs()
    assert with_path["vad"] is True
    assert with_path["vad_model_path"] == "/m/silero.bin"


def test_transcribe_kwargs_n_threads():
    kwargs = _make_stt_without_loading_model(n_threads=6)._transcribe_kwargs()
    assert kwargs["n_threads"] == 6


# ---------------------------------------------------------------------------
# Construction + transcription against a fake pywhispercpp
# ---------------------------------------------------------------------------

class _FakeModel:
    instances: list["_FakeModel"] = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.audio = None
        _FakeModel.instances.append(self)

    def transcribe(self, audio):
        self.audio = audio
        return [types.SimpleNamespace(text=" hello "), types.SimpleNamespace(text="world ")]


@pytest.fixture
def fake_pywhispercpp(monkeypatch, tmp_path):
    _FakeModel.instances = []
    pkg = types.ModuleType("pywhispercpp")
    model_mod = types.ModuleType("pywhispercpp.model")
    model_mod.Model = _FakeModel
    const_mod = types.ModuleType("pywhispercpp.constants")
    const_mod.AVAILABLE_MODELS = ["tiny", "large-v3-turbo"]
    const_mod.MODELS_DIR = tmp_path / "models"
    monkeypatch.setitem(sys.modules, "pywhispercpp", pkg)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", model_mod)
    monkeypatch.setitem(sys.modules, "pywhispercpp.constants", const_mod)
    return const_mod


def test_missing_pywhispercpp_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", None)
    with pytest.raises(STTError, match=r"coremind\[stt-cpp\]"):
        WhisperCppSTT()


def test_explicit_model_path_is_used(fake_pywhispercpp, tmp_path):
    model_file = tmp_path / "ggml-custom.bin"
    model_file.write_bytes(b"x")
    WhisperCppSTT(model="tiny", model_path=str(model_file), beam_size=5)
    fake = _FakeModel.instances[-1]
    assert fake.model == str(model_file)
    assert fake.kwargs["params_sampling_strategy"] == 1  # beam search
    assert fake.kwargs["language"] == "en"


def test_missing_model_path_raises(fake_pywhispercpp, tmp_path):
    with pytest.raises(STTError, match="model_path not found"):
        WhisperCppSTT(model_path=str(tmp_path / "nope.bin"))


def test_cached_model_is_used_without_download(fake_pywhispercpp, monkeypatch):
    cached = fake_pywhispercpp.MODELS_DIR / "ggml-tiny.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"x")
    monkeypatch.setattr(whisper_cpp, "_download", lambda *a: pytest.fail("downloaded"))
    WhisperCppSTT(model="tiny", beam_size=1)
    fake = _FakeModel.instances[-1]
    assert fake.model == str(cached)
    assert fake.kwargs["params_sampling_strategy"] == 0  # greedy


def test_uncached_model_is_downloaded(fake_pywhispercpp, monkeypatch):
    calls = []

    def fake_download(url, target):
        calls.append((url, target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")

    monkeypatch.setattr(whisper_cpp, "_download", fake_download)
    WhisperCppSTT(model="large-v3-turbo")
    assert calls[0][0].endswith("/ggml-large-v3-turbo.bin")
    assert calls[0][1] == fake_pywhispercpp.MODELS_DIR / "ggml-large-v3-turbo.bin"


def test_unknown_model_name_raises(fake_pywhispercpp):
    with pytest.raises(STTError, match="Unknown whisper.cpp model"):
        WhisperCppSTT(model="distil-large-v3")


def test_hotwords_warning_logged(fake_pywhispercpp, tmp_path, caplog):
    model_file = tmp_path / "m.bin"
    model_file.write_bytes(b"x")
    with caplog.at_level(logging.WARNING):
        WhisperCppSTT(model_path=str(model_file), hotwords="KJYO")
    assert "no hotword biasing" in caplog.text


def test_transcribe_joins_segments_and_resamples(fake_pywhispercpp, tmp_path):
    import soundfile as sf

    model_file = tmp_path / "m.bin"
    model_file.write_bytes(b"x")
    stt = WhisperCppSTT(model_path=str(model_file))

    # A 48 kHz stereo clip — what a USB mic might record at.
    wav = tmp_path / "in.wav"
    sf.write(str(wav), np.zeros((48000, 2), dtype=np.float32), 48000)
    assert stt.transcribe(str(wav)) == "hello world"

    audio = _FakeModel.instances[-1].audio
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 16000) <= 1


def test_transcribe_wraps_errors(fake_pywhispercpp, tmp_path):
    model_file = tmp_path / "m.bin"
    model_file.write_bytes(b"x")
    stt = WhisperCppSTT(model_path=str(model_file))
    with pytest.raises(STTError, match="Transcription failed"):
        stt.transcribe(str(tmp_path / "missing.wav"))


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def test_make_stt_mock():
    from coremind.stt.whisper_local import MockSTT

    assert isinstance(make_stt(STTConfig(provider="mock")), MockSTT)


def test_make_stt_whisper_local_passes_current_args(monkeypatch):
    captured = {}

    class FakeLocal:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import coremind.stt.whisper_local as wl

    monkeypatch.setattr(wl, "WhisperLocalSTT", FakeLocal)
    make_stt(STTConfig(model="distil-large-v3", compute_type="int8_float32", beam_size=8))
    assert captured == {
        "model": "distil-large-v3",
        "language": "en",
        "compute_type": "int8_float32",
        "beam_size": 8,
        "vad_filter": False,
        "initial_prompt": None,
        "hotwords": None,
    }


def test_make_stt_whisper_cpp(fake_pywhispercpp, tmp_path):
    model_file = tmp_path / "m.bin"
    model_file.write_bytes(b"x")
    cfg = STTConfig(
        provider="whisper_cpp",
        beam_size=3,
        initial_prompt="ctx",
        whisper_cpp={"model_path": str(model_file), "n_threads": 4},
    )
    stt = make_stt(cfg)
    assert isinstance(stt, WhisperCppSTT)
    fake = _FakeModel.instances[-1]
    assert fake.model == str(model_file)
    assert fake.kwargs["n_threads"] == 4
    assert fake.kwargs["beam_search"]["beam_size"] == 3
    assert fake.kwargs["initial_prompt"] == "ctx"


def test_make_stt_unknown_provider():
    with pytest.raises(ConfigError, match="whisper_cpp"):
        make_stt(STTConfig(provider="nope"))


def test_stt_model_label():
    assert stt_model_label(STTConfig(model="small")) == "small"
    assert stt_model_label(STTConfig(provider="whisper_cpp")) == "large-v3-turbo"
    assert stt_model_label(
        STTConfig(provider="whisper_cpp", whisper_cpp={"model_path": "/m/x.bin"})
    ) == "/m/x.bin"


# ---------------------------------------------------------------------------
# Hub: /v1/process honors stt.provider and logs per-stage timing
# ---------------------------------------------------------------------------

@pytest.fixture
def hub(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import coremind.server.app as app
    from coremind.config.settings import Settings

    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(app, "_SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(app, "_INDEX_PATH", sessions_dir / "index.json")
    monkeypatch.setattr(app, "_OLD_SESSIONS_PATH", tmp_path / "sessions.json")
    monkeypatch.setattr(app, "_sessions_registry", {})
    monkeypatch.setattr(app, "_active_session_id", None)
    monkeypatch.setattr(app, "_sessions", {})
    monkeypatch.setattr(app, "_settings", Settings(stt={"provider": "mock"}))
    monkeypatch.setattr(app, "_stt", None)
    monkeypatch.setattr(app, "_get_tts", lambda: None)

    async def fake_llm(messages):
        return "it is sunny", []

    monkeypatch.setattr(app, "_run_llm_with_tools", fake_llm)
    return app, TestClient(app.app)


def test_process_uses_configured_provider_and_logs_timing(hub, caplog):
    import urllib.parse

    from coremind.stt.whisper_local import MockSTT

    app, client = hub
    with caplog.at_level(logging.INFO, logger="coremind.server.app"):
        resp = client.post("/v1/process", files={"audio": ("a.wav", b"RIFF", "audio/wav")})
    assert resp.status_code == 200
    # The Hub used to ignore stt.provider and always build faster-whisper.
    assert isinstance(app._stt, MockSTT)
    assert urllib.parse.unquote(resp.headers["X-Transcript"]) == "[mock transcript]"
    timing = [r.getMessage() for r in caplog.records if "turn timing" in r.getMessage()]
    assert len(timing) == 1
    assert "stt=" in timing[0] and "llm=" in timing[0]
    assert "stt_engine=mock/mock" in timing[0]
    assert "outcome=ok" in timing[0]


def test_hub_stt_calls_are_serialized(monkeypatch):
    """whisper.cpp contexts aren't thread-safe: concurrent turns must not overlap."""
    pytest.importorskip("fastapi")
    import threading
    import time

    import coremind.server.app as app

    active, peak = [0], [0]

    class SlowSTT:
        def transcribe(self, path):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            active[0] -= 1
            return path

    monkeypatch.setattr(app, "_stt", SlowSTT())
    threads = [threading.Thread(target=app._transcribe_blocking, args=(f"{i}.wav",))
               for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1
