from __future__ import annotations

import subprocess


def set_volume(percent: int) -> str:
    """Set system audio volume to percent (0–100). Tries pactl then amixer."""
    pct = max(0, min(100, percent))
    for cmd in (
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
        ["amixer", "sset", "Master", f"{pct}%"],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return f"Volume set to {pct}%."
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as exc:
            # Command found but failed — still try the next one
            _ = exc
            continue
    return "Could not set volume: neither pactl nor amixer is available."
