from __future__ import annotations

import logging
import queue

from coremind import WakeWordError
from coremind.wake_word.base import WakeWordDetector

logger = logging.getLogger(__name__)

# openwakeword requires 80 ms chunks at 16 kHz
_CHUNK_SAMPLES = 1280


class OpenWakeWordDetector(WakeWordDetector):
    def __init__(
        self,
        model: str = "hey_jarvis_v0.1",
        threshold: float = 0.5,
        device: int | None = None,
        sample_rate: int = 16000,
        inference_framework: str = "onnx",
    ) -> None:
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeWordError(
                "openwakeword is not installed. Run: pip install openwakeword"
            ) from e
        try:
            self._oww = Model(wakeword_models=[model], inference_framework=inference_framework)
        except Exception as e:
            raise WakeWordError(
                f"Failed to load wake word model {model!r}: {e}"
            ) from e
        self._threshold = threshold
        self._device = device
        self._sample_rate = sample_rate
        self._model_name = model

    @property
    def trigger_prompt(self) -> str:
        return f"Listening for wake word ({self._model_name})..."

    def listen_until_wake_word(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as e:
            raise WakeWordError(f"Missing audio dependency: {e}") from e

        # Reset model prediction state so activation from the previous detection
        # does not immediately re-fire on the next call.
        if hasattr(self._oww, "reset"):
            self._oww.reset()

        chunk_queue: queue.Queue = queue.Queue()

        def _callback(indata, frame_count, time_info, status):
            chunk_queue.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
                blocksize=_CHUNK_SAMPLES,
                callback=_callback,
            ):
                while True:
                    try:
                        chunk = chunk_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    pcm = (chunk.flatten() * 32767).clip(-32768, 32767).astype(np.int16)
                    prediction = self._oww.predict(pcm)
                    for score in prediction.values():
                        if score >= self._threshold:
                            logger.debug(
                                "Wake word detected (score=%.3f, model=%s)",
                                score, self._model_name,
                            )
                            return
        except WakeWordError:
            raise
        except Exception as e:
            raise WakeWordError(f"Wake word listening failed: {e}") from e
