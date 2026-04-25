from __future__ import annotations

from pathlib import Path
from typing import Optional

from coremind import AudioInputError


class Recorder:
    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels

    def record(self, seconds: float, output_path: str) -> Path:
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
        except ImportError as e:
            raise AudioInputError(
                "Missing dependency: pip install sounddevice soundfile numpy"
            ) from e

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            frames = int(seconds * self.sample_rate)
            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
            )
            sd.wait()
        except Exception as e:
            raise AudioInputError(f"Recording failed: {e}") from e

        try:
            sf.write(str(out), audio, self.sample_rate)
        except Exception as e:
            raise AudioInputError(f"Failed to save audio to {output_path}: {e}") from e

        return out
