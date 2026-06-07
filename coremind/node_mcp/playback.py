"""Shared mpv process state for all audio output (music + ATC).

Only one mpv process runs at a time. Both music_player and atc_player
delegate here so SIGSTOP/SIGCONT mic isolation covers whichever source
is active, and starting one automatically stops the other.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess

logger = logging.getLogger(__name__)

_current_process: subprocess.Popen | None = None


def start(cmd: list[str]) -> subprocess.Popen:
    """Stop any current process, then spawn cmd. Returns the new process."""
    stop_current()
    global _current_process
    try:
        _current_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        raise
    return _current_process


def stop_current() -> str:
    """Terminate the running mpv process. Returns a status string."""
    global _current_process
    if _current_process is not None and _current_process.poll() is None:
        _current_process.terminate()
        try:
            _current_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _current_process.kill()
        _current_process = None
        return "Playback stopped."
    _current_process = None
    return "Nothing was playing."


def pause() -> None:
    """Suspend the current process with SIGSTOP (mic isolation)."""
    if _current_process is not None and _current_process.poll() is None:
        try:
            os.kill(_current_process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass


def resume() -> None:
    """Resume a SIGSTOP-suspended process with SIGCONT."""
    if _current_process is not None and _current_process.poll() is None:
        try:
            os.kill(_current_process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
