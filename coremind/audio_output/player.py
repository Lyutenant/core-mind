from __future__ import annotations

from typing import Optional

from coremind import AudioOutputError


class Player:
    def __init__(self, device: Optional[int] = None) -> None:
        self.device = device

    def play(self, wav_path: str) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as e:
            raise AudioOutputError(
                "Missing dependency: pip install sounddevice soundfile"
            ) from e

        try:
            data, samplerate = sf.read(wav_path, dtype="float32")
        except Exception as e:
            raise AudioOutputError(f"Failed to read {wav_path}: {e}") from e

        try:
            sd.play(data, samplerate, device=self.device)
            sd.wait()
        except Exception as e:
            raise AudioOutputError(f"Playback failed: {e}") from e
