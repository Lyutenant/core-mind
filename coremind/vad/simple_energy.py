from __future__ import annotations

import numpy as np

from coremind.vad.base import VoiceActivityDetector


class SimpleEnergyVAD(VoiceActivityDetector):
    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Return True if the RMS energy of int16 PCM bytes exceeds the threshold."""
        if not audio_chunk:
            return False
        pcm = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        return rms > self.threshold
