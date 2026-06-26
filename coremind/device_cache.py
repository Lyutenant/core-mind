from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Persisted choices for auto-selected audio/camera devices. Pinning by stable
# name/path (not enumeration index) keeps the same physical device selected when
# the USB topology changes — e.g. plugging everything into a powered hub, or a
# plain reboot that re-orders ALSA/V4L enumeration.
_CACHE_PATH = Path.home() / ".coremind" / "device-cache.json"

# Keys stored in the cache file.
INPUT_KEY = "input_device_name"
OUTPUT_KEY = "output_device_name"
CAMERA_KEY = "camera_device_path"


def _path() -> Path:
    return _CACHE_PATH


def load() -> dict:
    """Return the device cache, or an empty dict if missing/corrupt.

    Best-effort: a missing or unreadable cache must never be fatal — callers
    fall back to fresh auto-selection.
    """
    try:
        path = _path()
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001 - cache is advisory only
        logger.debug("Could not read device cache: %s", e)
        return {}


def get(key: str) -> Optional[str]:
    value = load().get(key)
    return value if isinstance(value, str) and value else None


def remember(key: str, value: str) -> None:
    """Persist a single device choice, leaving other keys intact."""
    try:
        path = _path()
        path.parent.mkdir(exist_ok=True)
        data = load()
        if data.get(key) == value:
            return
        data[key] = value
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:  # noqa: BLE001 - cache is advisory only
        logger.debug("Could not write device cache: %s", e)
