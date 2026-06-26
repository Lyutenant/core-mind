"""Tests for the Hub session store — no server/hardware required.

A Session groups a conversation's rounds under a hash id, persisted to
~/.coremind/sessions.json. The server keeps many sessions and marks one active;
turns land in the active session, and the LLM context is rebuilt from a session's
rounds. These tests cover the module-level helpers that the routes rely on:
persistence round-trip, cap enforcement, active-pointer repair, and memory rebuild.
"""
from __future__ import annotations

import pytest

import coremind.server.app as app
from coremind.config.settings import Settings


@pytest.fixture
def clean_sessions(tmp_path, monkeypatch):
    """Isolate the session store in a temp file with empty in-memory state."""
    monkeypatch.setattr(app, "_SESSIONS_PATH", tmp_path / "sessions.json")
    monkeypatch.setattr(app, "_sessions_registry", {})
    monkeypatch.setattr(app, "_active_session_id", None)
    monkeypatch.setattr(app, "_sessions", {})
    monkeypatch.setattr(app, "_settings", Settings())  # avoid loading config.yaml
    return app


def _round(i, transcript="hello", response="hi"):
    return {
        "turn": i,
        "timestamp": "2026-06-25T00:00:00",
        "transcript": transcript,
        "response": response,
        "tool_calls": [],
        "session_id": "x",
        "source": "chat",
    }


def test_get_active_creates_a_session(clean_sessions):
    sess = clean_sessions._get_active_session()
    assert sess["id"] in clean_sessions._sessions_registry
    assert clean_sessions._active_session_id == sess["id"]
    assert sess["rounds"] == []


def test_append_round_sets_title_from_first_transcript(clean_sessions):
    sid = clean_sessions._get_active_session()["id"]
    clean_sessions._append_round(sid, _round(1, transcript="what's the weather"))
    sess = clean_sessions._sessions_registry[sid]
    assert sess["title"] == "what's the weather"
    # A second turn does not overwrite the title.
    clean_sessions._append_round(sid, _round(2, transcript="and tomorrow?"))
    assert sess["title"] == "what's the weather"


def test_persist_and_reload_round_trip(clean_sessions):
    sid = clean_sessions._get_active_session()["id"]
    clean_sessions._append_round(sid, _round(1, transcript="hello", response="hi"))

    # Wipe in-memory state and reload from disk.
    clean_sessions._sessions_registry = {}
    clean_sessions._active_session_id = None
    clean_sessions._load_sessions()

    assert sid in clean_sessions._sessions_registry
    assert clean_sessions._active_session_id == sid
    reloaded = clean_sessions._sessions_registry[sid]["rounds"]
    assert reloaded[0]["transcript"] == "hello"
    assert reloaded[0]["response"] == "hi"


def test_round_cap_trims_oldest(clean_sessions):
    sid = clean_sessions._get_active_session()["id"]
    for i in range(clean_sessions._MAX_ROUNDS + 10):
        clean_sessions._append_round(sid, _round(i, transcript=f"t{i}"))
    rounds = clean_sessions._sessions_registry[sid]["rounds"]
    assert len(rounds) == clean_sessions._MAX_ROUNDS
    assert rounds[0]["transcript"] == "t10"   # first 10 dropped


def test_session_cap_keeps_active(clean_sessions):
    # Create more than the cap; make an OLD session the active one.
    ids = []
    for i in range(clean_sessions._MAX_SESSIONS + 5):
        s = clean_sessions._new_session()
        s["updated_at"] = f"2026-01-01T00:{i:02d}:00"
        clean_sessions._sessions_registry[s["id"]] = s
        ids.append(s["id"])
    clean_sessions._active_session_id = ids[0]   # oldest

    clean_sessions._prune_sessions()

    assert ids[0] in clean_sessions._sessions_registry          # active never dropped
    assert len(clean_sessions._sessions_registry) <= clean_sessions._MAX_SESSIONS + 1


def test_delete_active_reactivates_survivor(clean_sessions):
    s1 = clean_sessions._get_active_session()
    s2 = clean_sessions._new_session()
    clean_sessions._sessions_registry[s2["id"]] = s2

    # Simulate the DELETE route body for the active session.
    del clean_sessions._sessions_registry[s1["id"]]
    clean_sessions._active_session_id = None
    new_active = clean_sessions._get_active_session()

    assert new_active["id"] == s2["id"]


def test_set_active_does_not_lose_data(clean_sessions):
    s1 = clean_sessions._get_active_session()
    clean_sessions._append_round(s1["id"], _round(1))
    s2 = clean_sessions._new_session()
    clean_sessions._sessions_registry[s2["id"]] = s2

    clean_sessions._set_active(s2["id"])
    assert clean_sessions._active_session_id == s2["id"]
    # Switching active doesn't drop the old session's rounds.
    assert clean_sessions._sessions_registry[s1["id"]]["rounds"][0]["transcript"] == "hello"


def test_memory_rebuilds_from_rounds(clean_sessions):
    sid = clean_sessions._get_active_session()["id"]
    clean_sessions._append_round(sid, _round(1, transcript="what is 2+2", response="4"))
    clean_sessions._sessions.pop(sid, None)   # drop the live context cache

    mem = clean_sessions._get_session(sid)
    msgs = mem.get_messages()
    assert {"role": "user", "content": "what is 2+2"} in msgs
    assert {"role": "assistant", "content": "4"} in msgs
