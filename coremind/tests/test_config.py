from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from coremind import ConfigError
from coremind.config.settings import (
    AppConfig,
    AudioConfig,
    BrainConfig,
    MCPServerConfig,
    MemoryConfig,
    OllamaConfig,
    Settings,
    expand_env_refs,
    load_settings,
)


def test_defaults_load_without_config_file(tmp_path):
    settings = load_settings(str(tmp_path / "nonexistent.yaml"))
    assert settings.app.name == "CoreMind"
    assert settings.app.log_level == "INFO"
    assert settings.audio.sample_rate == 16000
    assert settings.brain.provider == "ollama"
    # Wake-confirmation gate is on by default with a small terminator set.
    assert settings.runtime.wake_confirm_words == ["over", "go ahead", "confirm"]


def test_wake_confirm_words_override_and_disable(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(textwrap.dedent("""\
        runtime:
          wake_confirm_words: ["roger"]
    """))
    assert load_settings(str(config)).runtime.wake_confirm_words == ["roger"]

    disabled = tmp_path / "disabled.yaml"
    disabled.write_text(textwrap.dedent("""\
        runtime:
          wake_confirm_words: null
    """))
    assert load_settings(str(disabled)).runtime.wake_confirm_words is None


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


# --- MCP server config: headers/env + ${VAR} secret expansion -----------------


def test_mcp_server_headers_env_default_none():
    cfg = MCPServerConfig(name="node", transport="http", url="http://pi:8767")
    assert cfg.headers is None
    assert cfg.env is None
    assert cfg.resolved_headers() is None
    assert cfg.resolved_env() is None


def test_expand_env_refs_substitutes_and_passes_through(monkeypatch):
    monkeypatch.setenv("CM_TEST_TOKEN", "s3cret")
    assert expand_env_refs("Bearer ${CM_TEST_TOKEN}") == "Bearer s3cret"
    assert expand_env_refs("no refs here") == "no refs here"
    # Multiple references in one value all expand.
    monkeypatch.setenv("CM_TEST_B", "x")
    assert expand_env_refs("${CM_TEST_TOKEN}/${CM_TEST_B}") == "s3cret/x"


def test_expand_env_refs_missing_var_fails_closed():
    with pytest.raises(ConfigError, match="CM_TEST_MISSING"):
        expand_env_refs("Bearer ${CM_TEST_MISSING}")


def test_resolved_headers_expands_but_model_keeps_literal(monkeypatch):
    """The secret is only in resolved_headers(); model_dump keeps the ${VAR} literal.

    GET /api/config serves a model_dump to every dashboard client, so an expanded
    token inside the model would leak to any browser on the tailnet.
    """
    monkeypatch.setenv("CM_TEST_HA_TOKEN", "abc123")
    cfg = MCPServerConfig(
        name="homeassistant",
        transport="streamable-http",
        url="http://ha:8123/api/mcp",
        headers={"Authorization": "Bearer ${CM_TEST_HA_TOKEN}"},
    )
    assert cfg.resolved_headers() == {"Authorization": "Bearer abc123"}
    assert cfg.model_dump()["headers"] == {"Authorization": "Bearer ${CM_TEST_HA_TOKEN}"}


def test_resolved_env_expands(monkeypatch):
    monkeypatch.setenv("CM_TEST_HA_TOKEN", "abc123")
    cfg = MCPServerConfig(
        name="hass-stdio",
        transport="stdio",
        command=["uvx", "some-server"],
        env={"HA_TOKEN": "${CM_TEST_HA_TOKEN}", "PLAIN": "value"},
    )
    assert cfg.resolved_env() == {"HA_TOKEN": "abc123", "PLAIN": "value"}


def test_mcp_server_yaml_roundtrip(tmp_path):
    """A streamable-http server entry with headers loads from YAML unexpanded."""
    config = tmp_path / "config.yaml"
    config.write_text(
        textwrap.dedent(
            """
            tools:
              mcp_servers:
                - name: homeassistant
                  transport: streamable-http
                  url: http://ha:8123/api/mcp
                  headers:
                    Authorization: "Bearer ${HA_TOKEN}"
            """
        )
    )
    settings = load_settings(str(config))
    (srv,) = settings.tools.mcp_servers
    assert srv.transport == "streamable-http"
    assert srv.headers == {"Authorization": "Bearer ${HA_TOKEN}"}
