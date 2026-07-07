"""Voice-turn isolation for the shared mpv slot (terminate + relaunch).

A SIGSTOP-suspended process keeps the single-open USB speaker, so during a
voice turn playback is fully terminated (freeing the device for the chime/TTS)
and relaunched afterwards. These tests pin that contract with a fake Popen.
"""
from __future__ import annotations

import pytest

from coremind.node_mcp import playback


class FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self._returncode = None
        self.terminated = False

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = 0

    def wait(self, timeout=None):
        return self._returncode

    def kill(self):
        self._returncode = -9


@pytest.fixture(autouse=True)
def _fake_mpv(monkeypatch):
    """Replace subprocess.Popen and reset module state around each test."""
    spawned: list[FakePopen] = []

    def _fake(cmd, **kwargs):
        p = FakePopen(cmd, **kwargs)
        spawned.append(p)
        return p

    monkeypatch.setattr(playback.subprocess, "Popen", _fake)
    playback._current_process = None
    playback._current_cmd = None
    playback._paused = False
    yield spawned
    playback._current_process = None
    playback._current_cmd = None
    playback._paused = False


ATC = ["mpv", "--no-terminal", "atc-url"]
MUSIC = ["mpv", "--no-terminal", "track.mp3"]


def test_pause_terminates_then_resume_relaunches(_fake_mpv):
    proc = playback.start(ATC)
    playback.pause()
    assert proc.terminated
    assert playback._current_process is None  # device freed
    assert playback._paused is True

    playback.resume()
    assert playback._current_process is not None
    assert playback._current_process.cmd == ATC
    assert playback._paused is False


def test_resume_does_not_relaunch_when_stopped_during_turn(_fake_mpv):
    playback.start(ATC)
    playback.pause()
    # User said "stop ATC" mid-turn — even though pause() already terminated the
    # process, the stream is still "playing" from the user's view, so the tool
    # must report it as stopped (not "Nothing was playing.").
    assert playback.stop_current() == "Playback stopped."
    playback.resume()
    assert playback._current_process is None


def test_stop_current_reports_nothing_when_idle(_fake_mpv):
    assert playback.stop_current() == "Nothing was playing."


def test_resume_does_not_relaunch_when_new_playback_started_during_turn(_fake_mpv):
    playback.start(ATC)
    playback.pause()
    # LLM started a different source mid-turn
    music = playback.start(MUSIC)
    playback.resume()
    assert playback._current_process is music
    assert playback._current_process.cmd == MUSIC


def test_resume_is_noop_when_nothing_was_playing(_fake_mpv):
    playback.pause()
    playback.resume()
    assert playback._current_process is None
    assert playback._paused is False


# --- play_stream: the universal primitive for resolver MCPs -------------------


def test_play_stream_starts_mpv_with_url(_fake_mpv):
    from coremind.node_mcp.tools import music_player

    msg = music_player.play_stream("http://example.net/kjfk_twr ", "KJFK Tower")
    assert playback._current_process is not None
    assert playback._current_process.cmd == [
        "mpv", "--no-terminal", "--quiet", "http://example.net/kjfk_twr",
    ]
    assert msg == "Streaming KJFK Tower."


def test_play_stream_default_title_and_no_url_echo(_fake_mpv):
    from coremind.node_mcp.tools import music_player

    msg = music_player.play_stream("https://example.net/feed")
    # The URL must never appear in the reply — the LLM tends to speak it.
    assert "example.net" not in msg
    assert msg == "Streaming audio stream."


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "/home/pi/Music/track.mp3",
        "file:///etc/passwd",
        "ftp://example.net/feed",
        "rtsp://example.net/feed",
        "http://example.net/a b",  # embedded whitespace
        "mpv --input-ipc-server=/tmp/x",
    ],
)
def test_play_stream_rejects_non_http_input(_fake_mpv, bad):
    from coremind.node_mcp.tools import music_player

    msg = music_player.play_stream(bad)
    assert playback._current_process is None
    assert "http(s)" in msg


def test_play_stream_reports_missing_mpv(_fake_mpv, monkeypatch):
    from coremind.node_mcp.tools import music_player

    def _raise(cmd):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(playback, "start", _raise)
    msg = music_player.play_stream("http://example.net/feed")
    assert "mpv is not installed" in msg
