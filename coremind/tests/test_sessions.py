"""Tests for the Hub session store — no server/hardware required.

A Session groups a conversation's rounds under a hash id, persisted under
~/.coremind/sessions/ as one `<id>.json` per session plus an `index.json` (active
pointer + metadata). The server keeps many sessions and marks one active; turns land
in the active session, and the LLM context is rebuilt from a session's rounds. These
tests cover the module-level helpers that the routes rely on: persistence round-trip,
per-file layout, legacy migration, cap enforcement, active-pointer repair, and
memory rebuild.
"""
from __future__ import annotations

import json

import pytest

import coremind.server.app as app
from coremind.config.settings import Settings


@pytest.fixture
def clean_sessions(tmp_path, monkeypatch):
    """Isolate the session store in a temp dir with empty in-memory state."""
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(app, "_SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(app, "_INDEX_PATH", sessions_dir / "index.json")
    monkeypatch.setattr(app, "_OLD_SESSIONS_PATH", tmp_path / "sessions.json")
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


def test_per_session_files_written(clean_sessions):
    sid = clean_sessions._get_active_session()["id"]
    clean_sessions._append_round(sid, _round(1, transcript="hello"))

    # A round writes the session's own file plus the index.
    session_file = clean_sessions._session_path(sid)
    assert session_file.exists()
    assert json.loads(session_file.read_text())["rounds"][0]["transcript"] == "hello"

    # The index carries metadata (round_count) but never the rounds themselves.
    index = json.loads(clean_sessions._INDEX_PATH.read_text())
    assert index["active"] == sid
    entry = index["sessions"][sid]
    assert entry["round_count"] == 1
    assert "rounds" not in entry


def test_delete_removes_file(clean_sessions):
    s1 = clean_sessions._get_active_session()
    clean_sessions._append_round(s1["id"], _round(1))
    s2 = clean_sessions._new_session()
    clean_sessions._sessions_registry[s2["id"]] = s2
    clean_sessions._save_session_file(s2)
    assert clean_sessions._session_path(s2["id"]).exists()

    # Simulate the DELETE route body for a non-active session.
    del clean_sessions._sessions_registry[s2["id"]]
    clean_sessions._sessions.pop(s2["id"], None)
    clean_sessions._delete_session_file(s2["id"])
    clean_sessions._save_index()

    assert not clean_sessions._session_path(s2["id"]).exists()
    index = json.loads(clean_sessions._INDEX_PATH.read_text())
    assert s2["id"] not in index["sessions"]
    assert s1["id"] in index["sessions"]


def test_migrates_legacy_sessions_json(clean_sessions):
    # Seed an old-format single-file store and ensure the new layout is absent.
    legacy = {
        "active": "aaa",
        "sessions": {
            "aaa": {
                "id": "aaa",
                "title": "old one",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-02T00:00:00",
                "rounds": [_round(1, transcript="legacy turn")],
            },
            "bbb": {
                "id": "bbb",
                "title": "old two",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "rounds": [],
            },
        },
    }
    clean_sessions._OLD_SESSIONS_PATH.write_text(json.dumps(legacy))

    clean_sessions._sessions_registry = {}
    clean_sessions._active_session_id = None
    clean_sessions._load_sessions()

    # In-memory state matches the legacy blob.
    assert clean_sessions._active_session_id == "aaa"
    assert set(clean_sessions._sessions_registry) == {"aaa", "bbb"}
    assert clean_sessions._sessions_registry["aaa"]["rounds"][0]["transcript"] == "legacy turn"

    # Split files + index now exist on disk; legacy file renamed to .bak.
    assert clean_sessions._session_path("aaa").exists()
    assert clean_sessions._session_path("bbb").exists()
    assert clean_sessions._INDEX_PATH.exists()
    assert not clean_sessions._OLD_SESSIONS_PATH.exists()
    assert clean_sessions._OLD_SESSIONS_PATH.with_suffix(".json.bak").exists()

    # A second load is a no-op (no legacy file to re-migrate) and preserves state.
    clean_sessions._sessions_registry = {}
    clean_sessions._active_session_id = None
    clean_sessions._load_sessions()
    assert set(clean_sessions._sessions_registry) == {"aaa", "bbb"}
    assert clean_sessions._active_session_id == "aaa"
