from __future__ import annotations

import base64
from abc import ABC, abstractmethod


class Camera(ABC):
    """A still-image source on the Node. ``capture_jpeg`` returns JPEG-encoded bytes."""

    @abstractmethod
    def capture_jpeg(self) -> bytes:
        """Capture one frame and return it as JPEG bytes. Raises VisionError on failure."""
        ...


# A real (tiny) 1x1 JPEG. MockCamera returns this so tests need neither a webcam nor
# opencv — the vision model is mocked separately, so the pixels never need to be meaningful.
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAAR"
    "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAA"
    "AAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oA"
    "DAMBAAIRAxEAPwCdABmX/9k="
)


class MockCamera(Camera):
    """Returns a fixed tiny JPEG. Used for tests and the ``vision.provider: mock`` setting."""

    def capture_jpeg(self) -> bytes:
        return base64.b64decode(_TINY_JPEG_B64)
