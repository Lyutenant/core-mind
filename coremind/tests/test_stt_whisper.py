from __future__ import annotations

import textwrap

from coremind.config.settings import STTConfig, load_settings
from coremind.stt.whisper_local import WhisperLocalSTT


def test_stt_config_defaults():
    cfg = STTConfig()
    assert cfg.compute_type == "int8"
    assert cfg.beam_size == 5
    assert cfg.vad_filter is False
    assert cfg.initial_prompt is None
    assert cfg.hotwords is None


def test_stt_tuning_loads_from_yaml(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(textwrap.dedent("""\
        stt:
          model: distil-large-v3
          compute_type: int8_float32
          beam_size: 8
          vad_filter: true
          hotwords: "CoreMind, KJYO, METAR"
          initial_prompt: "Aviation voice commands."
    """))
    settings = load_settings(str(config))
    assert settings.stt.model == "distil-large-v3"
    assert settings.stt.compute_type == "int8_float32"
    assert settings.stt.beam_size == 8
    assert settings.stt.vad_filter is True
    assert settings.stt.hotwords == "CoreMind, KJYO, METAR"
    assert settings.stt.initial_prompt == "Aviation voice commands."


def _make_stt_without_loading_model(**overrides) -> WhisperLocalSTT:
    """Build a WhisperLocalSTT without invoking faster-whisper.

    The constructor downloads/loads a model, which we don't want in unit
    tests. We bypass __init__ and set only the attributes _transcribe_kwargs
    reads, so the kwargs-building logic can be tested in isolation.
    """
    stt = WhisperLocalSTT.__new__(WhisperLocalSTT)
    stt.language = overrides.get("language", "en")
    stt.beam_size = overrides.get("beam_size", 5)
    stt.vad_filter = overrides.get("vad_filter", False)
    stt.initial_prompt = overrides.get("initial_prompt")
    stt.hotwords = overrides.get("hotwords")
    stt._supports_hotwords = overrides.get("supports_hotwords", True)
    return stt


def test_transcribe_kwargs_minimal():
    stt = _make_stt_without_loading_model()
    kwargs = stt._transcribe_kwargs()
    assert kwargs == {"language": "en", "beam_size": 5, "vad_filter": False}
    # No biasing params unless configured.
    assert "initial_prompt" not in kwargs
    assert "hotwords" not in kwargs


def test_transcribe_kwargs_passes_hotwords_when_supported():
    stt = _make_stt_without_loading_model(
        hotwords="KJYO METAR", initial_prompt="ctx", supports_hotwords=True
    )
    kwargs = stt._transcribe_kwargs()
    assert kwargs["hotwords"] == "KJYO METAR"
    assert kwargs["initial_prompt"] == "ctx"


def test_transcribe_kwargs_folds_hotwords_when_unsupported():
    stt = _make_stt_without_loading_model(
        hotwords="KJYO METAR", initial_prompt="ctx", supports_hotwords=False
    )
    kwargs = stt._transcribe_kwargs()
    assert "hotwords" not in kwargs
    assert kwargs["initial_prompt"] == "ctx KJYO METAR"
