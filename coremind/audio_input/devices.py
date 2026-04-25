from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from coremind import AudioInputError


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float

    def is_input(self) -> bool:
        return self.max_input_channels > 0

    def is_output(self) -> bool:
        return self.max_output_channels > 0


def _query_all() -> list[AudioDevice]:
    try:
        import sounddevice as sd
    except ImportError as e:
        raise AudioInputError("sounddevice is not installed: pip install sounddevice") from e

    try:
        raw = sd.query_devices()
    except Exception as e:
        raise AudioInputError(f"Failed to query audio devices: {e}") from e

    if isinstance(raw, dict):
        raw = [raw]

    return [
        AudioDevice(
            index=i,
            name=d["name"],
            max_input_channels=int(d["max_input_channels"]),
            max_output_channels=int(d["max_output_channels"]),
            default_sample_rate=float(d["default_samplerate"]),
        )
        for i, d in enumerate(raw)
    ]


def list_input_devices() -> list[AudioDevice]:
    return [d for d in _query_all() if d.is_input()]


def list_output_devices() -> list[AudioDevice]:
    return [d for d in _query_all() if d.is_output()]


def get_default_input_device() -> Optional[AudioDevice]:
    try:
        import sounddevice as sd
        idx = sd.default.device[0]
        if idx < 0:
            return None
        return next((d for d in _query_all() if d.index == idx and d.is_input()), None)
    except Exception:
        return None


def get_default_output_device() -> Optional[AudioDevice]:
    try:
        import sounddevice as sd
        idx = sd.default.device[1]
        if idx < 0:
            return None
        return next((d for d in _query_all() if d.index == idx and d.is_output()), None)
    except Exception:
        return None
