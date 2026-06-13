"""Tests for the dashboard config-save merge — no server/hardware required.

Regression guard for the bug where the Hub dashboard's "Save Configuration"
overwrote the whole config.yaml with the partial payload it builds, wiping
unmanaged sections (tools.mcp_servers, node_mcp) back to defaults and breaking
all MCP tools. merge_config_text() must preserve those sections.
"""
from __future__ import annotations

import yaml

from coremind.config.settings import Settings, merge_config_text

# A realistic Hub config with MCP wiring the dashboard does NOT render.
_EXISTING = """\
# CoreMind Hub config
mode: hub

app:
  name: CoreMind
  log_level: INFO
  home_airport: KIAD
  user_timezone: America/New_York

ollama:
  base_url: http://localhost:11434
  model: qwen2.5:7b

tools:
  enabled: true
  built_in: [time, weather, aviation_weather, airport]
  mcp_servers:
    - name: node
      transport: http
      url: http://100.64.0.5:8767

node_mcp:
  enabled: true
  host: 127.0.0.1
  port: 8767
"""

# What the dashboard's buildConfig() actually POSTs: no tools, no node_mcp,
# app stripped to name + log_level.
_DASHBOARD_PAYLOAD = {
    "mode": "hub",
    "app": {"name": "CoreMind", "log_level": "INFO"},
    "ollama": {"base_url": "http://localhost:11434", "model": "qwen2.5:7b"},
}


def _loads(text: str) -> dict:
    return yaml.safe_load(text)


def test_partial_save_preserves_unmanaged_sections():
    merged = _loads(merge_config_text(_EXISTING, _DASHBOARD_PAYLOAD))
    # tools.mcp_servers — the section whose loss broke play_atc — survives.
    assert merged["tools"]["mcp_servers"][0]["url"] == "http://100.64.0.5:8767"
    assert merged["tools"]["built_in"] == ["time", "weather", "aviation_weather", "airport"]
    assert merged["node_mcp"]["enabled"] is True
    assert merged["node_mcp"]["port"] == 8767


def test_partial_save_preserves_unmanaged_app_subfields():
    merged = _loads(merge_config_text(_EXISTING, _DASHBOARD_PAYLOAD))
    # app keys the dashboard omits must not revert to model defaults.
    assert merged["app"]["home_airport"] == "KIAD"
    assert merged["app"]["user_timezone"] == "America/New_York"


def test_managed_field_overwrite_takes_effect():
    payload = dict(_DASHBOARD_PAYLOAD)
    payload["ollama"] = {"base_url": "http://localhost:11434", "model": "llama3.1:8b"}
    merged = _loads(merge_config_text(_EXISTING, payload))
    assert merged["ollama"]["model"] == "llama3.1:8b"


def test_merged_result_validates_as_settings():
    merged = _loads(merge_config_text(_EXISTING, _DASHBOARD_PAYLOAD))
    settings = Settings.model_validate(merged)
    assert len(settings.tools.mcp_servers) == 1
    assert settings.tools.mcp_servers[0].name == "node"
    assert settings.node_mcp.enabled is True


def test_empty_existing_file_writes_payload():
    merged = _loads(merge_config_text("", _DASHBOARD_PAYLOAD))
    assert merged["mode"] == "hub"
    assert merged["ollama"]["model"] == "qwen2.5:7b"


def test_comments_preserved_when_ruamel_available():
    import pytest

    pytest.importorskip("ruamel.yaml")
    merged_text = merge_config_text(_EXISTING, _DASHBOARD_PAYLOAD)
    assert "# CoreMind Hub config" in merged_text
