from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from coremind import device_cache

logger = logging.getLogger(__name__)

# Stable per-device symlinks created by udev on Linux. Unlike /dev/videoN (which
# is assigned by V4L enumeration order and shuffles when the USB topology changes),
# a by-id path is derived from the camera's USB vendor/product/serial, so it keeps
# pointing at the same physical webcam across hub replugs and reboots.
_BY_ID_DIR = Path("/dev/v4l/by-id")


def list_camera_devices() -> list[str]:
    """Return stable camera device paths (/dev/v4l/by-id/*), sorted. Empty if none."""
    try:
        if not _BY_ID_DIR.is_dir():
            return []
        return sorted(str(p) for p in _BY_ID_DIR.iterdir())
    except Exception as e:  # noqa: BLE001 - discovery is best-effort
        logger.debug("Could not list camera devices: %s", e)
        return []


def _name_present(path: str, devices: list[str]) -> bool:
    return path in devices


def auto_select_camera() -> Optional[str]:
    """Pick a sensible default camera by stable path. None = no by-id camera found."""
    devices = list_camera_devices()
    if not devices:
        return None
    cached = device_cache.get(device_cache.CAMERA_KEY)
    if cached and _name_present(cached, devices):
        return cached
    # A camera exposes several nodes; index0 is its primary capture node.
    primary = [p for p in devices if p.endswith("index0")]
    return (primary or devices)[0]


def resolve_camera_source(
    camera_device: Optional[str],
    camera_index: Optional[int],
) -> Union[int, str]:
    """Resolve the camera to a cv2.VideoCapture source.

    Precedence: an explicit ``camera_device`` path wins, then an explicit
    ``camera_index``, then name-based auto-selection (logged + persisted),
    falling back to index ``0``.
    """
    if camera_device and camera_device != "auto":
        return camera_device
    if camera_index is not None:
        return camera_index
    path = auto_select_camera()
    if path is not None:
        logger.info("Auto-selected camera: '%s'", path)
        device_cache.remember(device_cache.CAMERA_KEY, path)
        return path
    return 0
