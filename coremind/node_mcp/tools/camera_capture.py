from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)

# Module-level config, set by init_camera() at server startup (mirrors music_player/atc_player).
_provider: str = "opencv"
_camera_index: "int | None" = None
_camera_device: "str | None" = None
_width: int = 1280
_height: int = 720

# Returned (as a plain string) when capture fails, so the Hub-side `look` tool can
# detect the failure instead of trying to decode it as image data.
ERROR_PREFIX = "CAMERA_ERROR:"


def init_camera(
    provider: str = "opencv",
    camera_index: "int | None" = None,
    width: int = 1280,
    height: int = 720,
    camera_device: "str | None" = None,
) -> None:
    global _provider, _camera_index, _camera_device, _width, _height
    _provider = provider
    _camera_index = camera_index
    _camera_device = camera_device
    _width = width
    _height = height


def _build_camera():
    if _provider == "mock":
        from coremind.vision.base import MockCamera
        return MockCamera()
    from coremind.vision.camera_select import resolve_camera_source
    from coremind.vision.opencv_camera import OpenCVCamera

    source = resolve_camera_source(_camera_device, _camera_index)
    if isinstance(source, str):
        return OpenCVCamera(camera_device=source, width=_width, height=_height)
    return OpenCVCamera(camera_index=source, width=_width, height=_height)


def capture_image() -> str:
    """Capture a still frame from the Node's camera and return it as a base64 JPEG string.

    The raw image is held only in memory and never written to disk; on success the
    base64 payload is returned, on failure a short ``CAMERA_ERROR:`` string is returned.
    """
    from coremind import VisionError

    try:
        camera = _build_camera()
        jpeg = camera.capture_jpeg()
    except VisionError as e:
        logger.warning("Camera capture failed: %s", e)
        return f"{ERROR_PREFIX} {e}"
    except Exception as e:  # noqa: BLE001 — never let a capture failure crash the MCP server
        logger.warning("Camera capture failed: %s", e)
        return f"{ERROR_PREFIX} {e}"

    logger.info("Captured frame: %d bytes JPEG", len(jpeg))
    return base64.b64encode(jpeg).decode("ascii")
