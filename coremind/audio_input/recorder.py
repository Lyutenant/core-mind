from __future__ import annotations

import threading
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
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except ImportError as e:
            raise AudioInputError(
                "Missing dependency: pip install sounddevice soundfile numpy"
            ) from e

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        target_frames = int(seconds * self.sample_rate)
        chunks: list = []
        done = threading.Event()

        def _callback(indata, frame_count, time_info, status):
            chunks.append(indata.copy())
            if sum(c.shape[0] for c in chunks) >= target_frames:
                done.set()
                raise sd.CallbackStop()

        # Callback mode avoids PortAudio's blocking Pa_ReadStream, which stalls
        # indefinitely on some Pi audio backends (PipeWire, PulseAudio over ALSA).
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=_callback,
            ):
                finished = done.wait(timeout=seconds + 3.0)
        except Exception as e:
            raise AudioInputError(f"Recording failed: {e}") from e

        if not finished or not chunks:
            raise AudioInputError(
                f"Recording timed out — no audio received after {seconds + 3.0:.0f}s. "
                "Check that the microphone is connected and the correct device index is set."
            )

        try:
            audio = np.concatenate(chunks)[:target_frames]
            sf.write(str(out), audio, self.sample_rate)
        except Exception as e:
            raise AudioInputError(f"Failed to save audio to {output_path}: {e}") from e

        return out
