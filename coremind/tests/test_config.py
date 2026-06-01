from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from coremind import ConfigError
from coremind.config.settings import (
    AppConfig,
    AudioConfig,
    BrainConfig,
    MemoryConfig,
    OllamaConfig,
    Settings,
    load_settings,
)


def test_defaults_load_without_config_file(tmp_path):
    settings = load_settings(str(tmp_path / "nonexistent.yaml"))
    assert settings.app.name == "CoreMind"
    assert settings.app.log_level == "INFO"
    assert settings.audio.sample_rate == 16000
    assert settings.brain.provider == "ollama"


def test_load_valid_config_file(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(textwrap.dedent("""\
        app:
          name: test-jarvis
          log_level: DEBUG
        audio:
          sample_rate: 44100
          record_seconds: 3
        ollama:
          base_url: http://10.0.0.1:11434
          model: llama3.2
    """))
    settings = load_settings(str(config))
    assert settings.app.name == "test-jarvis"
    assert settings.app.log_level == "DEBUG"
    assert settings.audio.sample_rate == 44100
    assert settings.audio.record_seconds == 3
    assert settings.ollama.base_url == "http://10.0.0.1:11434"
    assert settings.ollama.model == "llama3.2"


def test_empty_config_file_uses_defaults(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("")
    settings = load_settings(str(config))
    assert settings.app.name == "CoreMind"


def test_invalid_yaml_raises_config_error(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("app: [unclosed bracket")
    with pytest.raises(ConfigError):
        load_settings(str(config))


def test_non_mapping_yaml_raises_config_error(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("- item1\n- item2\n")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_settings(str(config))


def test_invalid_log_level_raises_error():
    with pytest.raises(Exception):
        AppConfig(log_level="INVALID")


def test_log_level_is_case_insensitive():
    cfg = AppConfig(log_level="debug")
    assert cfg.log_level == "DEBUG"


def test_audio_device_defaults_to_none():
    cfg = AudioConfig()
    assert cfg.input_device is None
    assert cfg.output_device is None


def test_memory_max_turns():
    cfg = MemoryConfig(max_turns=5)
    assert cfg.max_turns == 5


def test_settings_partial_override(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("memory:\n  max_turns: 20\n")
    settings = load_settings(str(config))
    assert settings.memory.max_turns == 20
    assert settings.memory.enabled is True


def test_env_var_overrides_yaml(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("ollama:\n  base_url: http://yaml-value:11434\n")
    monkeypatch.setenv("COREMIND_OLLAMA__BASE_URL", "http://env-value:11434")
    settings = load_settings(str(config))
    assert settings.ollama.base_url == "http://env-value:11434"


def test_env_var_overrides_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("COREMIND_AUDIO__SAMPLE_RATE", "44100")
    settings = load_settings(str(tmp_path / "nonexistent.yaml"))
    assert settings.audio.sample_rate == 44100
