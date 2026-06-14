from __future__ import annotations

import logging

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
    ) -> None:
        try:
            import cv2
        except ImportError as e:
            raise VisionError(
                "opencv is not installed. Run: pip install 'coremind[vision]'"
            ) from e
        self._cv2 = cv2
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

    def capture_jpeg(self) -> bytes:
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.camera_index)
        try:
            if not cap.isOpened():
                raise VisionError(
                    f"Could not open camera index {self.camera_index}. "
                    "Check the USB webcam connection (try: coremind vision test)."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Many webcams need a few frames to auto-expose; keep the last good frame.
            frame = None
            for _ in range(5):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
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
