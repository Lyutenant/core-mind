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
