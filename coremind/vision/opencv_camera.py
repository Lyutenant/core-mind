from __future__ import annotations

import logging
import time
from typing import Optional

from coremind import VisionError
from coremind.vision.base import Camera

logger = logging.getLogger(__name__)


class OpenCVCamera(Camera):
    """Capture a still frame from a USB webcam via opencv-python.

    Capturing a frame is plain I/O — it stays light enough for the Pi. The frame is
    downscaled to ``max_dimension`` on its long edge and JPEG-encoded so the base64
    payload that travels to the Hub stays small, and so the vision model isn't fed a
    needlessly huge image.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        max_dimension: int = 768,
        jpeg_quality: int = 80,
        warmup_seconds: float = 2.0,
        dark_threshold: float = 10.0,
        camera_device: Optional[str] = None,
    ) -> None:
        try:
            import cv2
        except ImportError as e:
            raise VisionError(
                "opencv is not installed. Run: pip install 'coremind[vision]'"
            ) from e
        self._cv2 = cv2
        self.camera_index = camera_index
        # A stable /dev path (e.g. /dev/v4l/by-id/...); when set, takes precedence
        # over the enumeration index so it survives USB re-enumeration.
        self.camera_device = camera_device
        self.width = width
        self.height = height
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.warmup_seconds = warmup_seconds
        self.dark_threshold = dark_threshold

    def capture_jpeg(self) -> bytes:
        cv2 = self._cv2
        source = self.camera_device if self.camera_device else self.camera_index
        cap = cv2.VideoCapture(source)
        try:
            if not cap.isOpened():
                raise VisionError(
                    f"Could not open camera {source!r}. "
                    "Check the USB webcam connection (try: coremind vision test)."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            frame = self._read_settled_frame(cap)
            if frame is None:
                raise VisionError("Camera opened but returned no frame.")
        finally:
            cap.release()

        frame = self._downscale(frame)
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise VisionError("Failed to JPEG-encode the captured frame.")
        return bytes(buf.tobytes())

    def _read_settled_frame(self, cap):
        """Read frames until auto-exposure settles or warmup elapses.

        Many UVC webcams emit black frames for the first ~1 s after opening while
        auto-exposure/auto-gain converge. Reading a few back-to-back frames isn't
        enough — they all arrive before the sensor settles. So we keep grabbing
        for up to ``warmup_seconds``, returning early once a frame is no longer
        dark, and falling back to the last frame read (which may legitimately be
        dark in a dark room) when the warmup window expires.
        """
        deadline = time.monotonic() + self.warmup_seconds
        last = None
        while True:
            ok, f = cap.read()
            if ok and f is not None:
                last = f
                if not self._is_dark(f):
                    return f
            if time.monotonic() >= deadline:
                return last
            time.sleep(0.05)

    def _is_dark(self, frame) -> bool:
        """True if the frame is essentially black (mean brightness below threshold)."""
        return float(frame.mean()) < self.dark_threshold

    def _downscale(self, frame):
        cv2 = self._cv2
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= self.max_dimension:
            return frame
        scale = self.max_dimension / longest
        return cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
