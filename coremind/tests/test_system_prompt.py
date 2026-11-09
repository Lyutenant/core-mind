"""Tests for the Hub system-prompt builder — no server/hardware required.

Covers the configurable personality clause: when app.personality is set, the
text is appended to the system prompt used by both the voice (/v1/process) and
chat (/v1/chat) endpoints; when unset, no persona clause appears.
"""
from __future__ import annotations

from coremind.config.settings import AppConfig, Settings
from coremind.server.app import _build_system_prompt


def test_personality_defaults_to_none():
    assert AppConfig().personality is None


def test_personality_survives_validation():
    s = Settings.model_validate(
        {"app": {"name": "Jarvis", "personality": "A refined British butler."}}
    )
    assert s.app.personality == "A refined British butler."


def test_system_prompt_includes_personality_when_set():
    s = Settings(app={"name": "Jarvis", "personality": "A refined British butler."})
    prompt = _build_system_prompt(s)
    assert "Jarvis" in prompt
    assert "A refined British butler." in prompt
    assert "persona" in prompt.lower()


def test_system_prompt_omits_personality_when_unset():
    prompt = _build_system_prompt(Settings())
    assert "persona" not in prompt.lower()


def test_system_prompt_location_clause_warns_about_mis_transcription():
    # A garbled STT token must not count as "specifying a place" — the prompt
    # tells the model to prefer the configured location over an unfamiliar name.
    s = Settings(app={"name": "Jarvis", "user_location": "Newark, NJ"})
    prompt = _build_system_prompt(s)
    assert "Newark, NJ" in prompt
    assert "mis-transcription" in prompt
    assert "speech recognition" in prompt


def test_system_prompt_omits_location_clause_when_unset():
    prompt = _build_system_prompt(Settings())
    assert "mis-transcription" not in prompt
    assert "located in" not in prompt


# --- Tool-use rules -----------------------------------------------------------------

def test_system_prompt_has_no_tool_rules_without_tools():
    prompt = _build_system_prompt(Settings())
    assert "tool" not in prompt.lower()


def test_system_prompt_tool_rules_demand_a_call_and_honest_outcome():
    prompt = _build_system_prompt(Settings(), tool_names=["get_weather"])
    assert "Call the tool every time" in prompt
    assert "never answer from memory" in prompt
    assert "returned a success result" in prompt
    assert "the action did not happen" in prompt
    # The Home Assistant hint only appears when HA tools are registered.
    assert "HassTurnOn" not in prompt


def test_system_prompt_adds_home_assistant_targeting_hint():
    prompt = _build_system_prompt(
        Settings(), tool_names=["get_weather", "HassTurnOn", "HassTurnOff"]
    )
    assert "area plus domain" in prompt
    assert "never means light" in prompt
    assert "GetLiveContext" in prompt


def test_system_prompt_appends_mcp_server_instructions():
    prompt = _build_system_prompt(
        Settings(),
        tool_names=["HassTurnOn"],
        tool_instructions={"homeassistant": "Prefer area and domain.", "node": ""},
    )
    assert "Instructions from the 'homeassistant' tool server: Prefer area and domain." in prompt
    assert "'node' tool server" not in prompt   # empty text is skipped
