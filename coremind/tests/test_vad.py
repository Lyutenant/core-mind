from __future__ import annotations

import numpy as np
import pytest

from coremind.vad.simple_energy import SimpleEnergyVAD


def _pcm_bytes(amplitude: float, num_samples: int = 480) -> bytes:
    """Generate int16 PCM bytes at the given amplitude (0.0–1.0)."""
    samples = (np.ones(num_samples, dtype=np.float32) * amplitude * 32767).astype(np.int16)
    return samples.tobytes()


def test_silence_is_not_speech():
    vad = SimpleEnergyVAD(threshold=0.01)
    assert vad.is_speech(_pcm_bytes(0.0)) is False


def test_loud_audio_is_speech():
    vad = SimpleEnergyVAD(threshold=0.01)
    assert vad.is_speech(_pcm_bytes(0.5)) is True


def test_threshold_boundary_below():
    vad = SimpleEnergyVAD(threshold=0.5)
    assert vad.is_speech(_pcm_bytes(0.49)) is False


def test_threshold_boundary_above():
    vad = SimpleEnergyVAD(threshold=0.5)
    assert vad.is_speech(_pcm_bytes(0.51)) is True


def test_empty_bytes_returns_false():
    vad = SimpleEnergyVAD(threshold=0.01)
    assert vad.is_speech(b"") is False


def test_default_threshold():
    vad = SimpleEnergyVAD()
    assert vad.threshold == 0.01
