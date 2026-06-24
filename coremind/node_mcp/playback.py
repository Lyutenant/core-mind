"""Shared mpv process state for all audio output (music + ATC).

Only one mpv process runs at a time. Both music_player and atc_player
delegate here so the single shared speaker is coordinated and starting one
source automatically stops the other.

Voice-turn isolation: the USB speaker allows only one ALSA open at a time,
and a SIGSTOP-suspended process keeps the device handle open. So during a
voice turn we fully *terminate* mpv (freeing the speaker for the wake chime
and the spoken reply) and *relaunch* the same command when the turn ends.
"""
from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_current_process: subprocess.Popen | None = None
# Command of the running process, kept so a voice-turn pause can relaunch it.
_current_cmd: list[str] | None = None
# True while playback is suspended for a voice turn and awaiting relaunch.
_paused: bool = False


def _terminate(proc: subprocess.Popen) -> None:
    """Terminate a process and wait briefly, killing if it doesn't exit."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def start(cmd: list[str]) -> subprocess.Popen:
    """Stop any current process, then spawn cmd. Returns the new process."""
    stop_current()
    global _current_process, _current_cmd, _paused
    try:
        _current_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        raise
    _current_cmd = list(cmd)
    _paused = False
    return _current_process


def stop_current() -> str:
    """Terminate the running mpv process. Returns a status string."""
    global _current_process, _current_cmd, _paused
    # A turn-paused stream has no live process but is still "playing" from the
    # user's view (it would relaunch on resume), so report it as stopped.
    was_playing = _paused and _current_cmd is not None
    _paused = False
    _current_cmd = None
    if _current_process is not None and _current_process.poll() is None:
        _terminate(_current_process)
        _current_process = None
        return "Playback stopped."
    _current_process = None
    return "Playback stopped." if was_playing else "Nothing was playing."


def pause() -> None:
    """Free the audio device for a voice turn by terminating the current process.

    Unlike SIGSTOP, terminating releases the ALSA output device so the wake
    chime and the TTS reply can open the (single-open) USB speaker. The stream
    is relaunched by resume() when the turn ends. No-op if nothing is playing.
    """
    global _current_process, _paused
    if _current_process is not None and _current_process.poll() is None:
        _terminate(_current_process)
        _current_process = None
        _paused = True


def resume() -> None:
    """Relaunch the stream that pause() terminated for a voice turn.

    No-op unless a turn-pause is pending and nothing else has started or
    stopped playback in the meantime (e.g. the turn issued a stop/play tool).
    """
    global _paused
    if _paused and _current_cmd is not None and _current_process is None:
        _paused = False
        try:
            start(_current_cmd)
        except FileNotFoundError:
            pass
    else:
        _paused = False
