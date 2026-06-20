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
        vad_threshold: float = 0.0,
    ) -> None:
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeWordError(
                "openwakeword is not installed. Run: pip install openwakeword"
            ) from e

        vad_threshold = max(0.0, float(vad_threshold))
        base_kwargs = {"wakeword_models": [model], "inference_framework": inference_framework}

        # The Silero VAD pre-gate is only loaded when explicitly enabled (vad_threshold > 0).
        # With the default 0.0 the model is built exactly as before — no Silero dependency,
        # so a node whose wake model works but lacks/can't init the VAD never regresses.
        # When the gate IS requested, a VAD-load failure degrades to no-VAD (warn, don't crash).
        # If the gate is enabled later (Hub override / live slider), set_vad_threshold() loads
        # Silero on demand, so the default-0.0 node can be turned on without a restart.
        self._oww = None
        self._vad_loaded = False
        vad_unsupported = False  # this openwakeword build rejects vad_threshold outright
        if vad_threshold > 0.0:
            try:
                self._oww = Model(**base_kwargs, vad_threshold=vad_threshold)
                self._vad_loaded = True
            except TypeError:
                logger.warning(
                    "openwakeword does not support vad_threshold; wake VAD gate disabled "
                    "(upgrade openwakeword to enable it)."
                )
                vad_unsupported = True
                vad_threshold = 0.0
            except Exception as e:
                logger.warning(
                    "Failed to initialize the Silero VAD gate (%s); continuing without it.", e
                )
                vad_threshold = 0.0

        if self._oww is None:
            # Default path (gate off) or a graceful VAD-load fallback: build as before.
            try:
                self._oww = Model(**base_kwargs)
            except Exception as e:
                raise WakeWordError(
                    f"Failed to load wake word model {model!r}: {e}"
                ) from e

        # Whether this openwakeword build actually honors a VAD threshold in predict().
        # A TypeError above proved it doesn't. Otherwise infer from the model exposing a
        # `vad_threshold` attribute, which openWakeWord sets in __init__ when supported —
        # so we never let the live slider report an active gate predict() would ignore.
        self._vad_supported = (not vad_unsupported) and hasattr(self._oww, "vad_threshold")
        self._threshold = threshold
        self._vad_threshold = vad_threshold if self._vad_loaded else 0.0  # snapshot/hot-reload
        self._device = device
        self._sample_rate = sample_rate
        self._model_name = model

    def set_vad_threshold(self, value: float) -> bool:
        """Update the Silero VAD pre-gate at runtime, loading Silero on demand.

        Used by the hot-reload path so a gate enabled from the Hub dashboard (on a node
        that started with the default 0.0) actually takes effect without a restart.
        Setting 0.0 disables the gate (nothing is unloaded). openWakeWord re-reads
        ``vad_threshold`` on every ``predict()``, so the change applies on the next frame.
        Returns True when the gate is active at ``value`` (or off), False if it could not
        be enabled (Silero unavailable, or this openwakeword build has no VAD support).
        """
        value = max(0.0, float(value))
        if value > 0.0 and not self._vad_supported:
            # predict() on this build would ignore vad_threshold — never report it active.
            logger.warning(
                "This openwakeword build does not support the VAD gate; ignoring "
                "(upgrade openwakeword to enable it)."
            )
            self._vad_threshold = 0.0
            return False
        if value > 0.0 and not self._vad_loaded:
            try:
                from openwakeword.vad import VAD
                # Attach the VAD before exposing a >0 threshold so predict() never sees
                # vad_threshold>0 without a VAD object (ordering matters: see below).
                if getattr(self._oww, "vad", None) is None:
                    self._oww.vad = VAD()
                self._vad_loaded = True
            except Exception as e:
                logger.warning(
                    "Could not enable the Silero VAD gate at runtime (%s); leaving it off.", e
                )
                self._vad_threshold = 0.0
                if hasattr(self._oww, "vad_threshold"):
                    self._oww.vad_threshold = 0.0
                return False
        self._vad_threshold = value
        if hasattr(self._oww, "vad_threshold"):
            self._oww.vad_threshold = value
        return True

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
