from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

_current_process: subprocess.Popen | None = None
_music_dir: Path = Path.home() / "Music"

_AUDIO_EXTENSIONS = ("*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aac", "*.wav", "*.opus")


def search_music(query: str) -> str:
    """Search _music_dir recursively for audio files matching query in filename or parent dir."""
    q = query.lower()
    matches: list[Path] = []
    for ext in _AUDIO_EXTENSIONS:
        matches.extend(_music_dir.rglob(ext))
    results = [
        str(p) for p in matches
        if q in p.stem.lower() or q in p.parent.name.lower()
    ][:10]
    if not results:
        return f"No music matching '{query}' found in {_music_dir}"
    return "\n".join(results)


def play_music(path: str) -> str:
    """Stop any current playback and start mpv with the given file path."""
    global _current_process
    stop_playback()
    try:
        _current_process = subprocess.Popen(
            ["mpv", "--no-terminal", "--quiet", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return "mpv is not installed. Run: sudo apt install mpv"
    return f"Now playing: {Path(path).name}"


def stop_playback() -> str:
    """Terminate the current mpv process if one is running."""
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


def pause_mpv() -> None:
    """Suspend mpv with SIGSTOP so mic captures only the user's voice, not music."""
    if _current_process is not None and _current_process.poll() is None:
        try:
            os.kill(_current_process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass


def resume_mpv() -> None:
    """Resume a SIGSTOP-suspended mpv process after the voice turn ends."""
    if _current_process is not None and _current_process.poll() is None:
        try:
            os.kill(_current_process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
