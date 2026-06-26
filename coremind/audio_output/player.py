from __future__ import annotations

from math import gcd
from typing import Optional, Union

from coremind import AudioOutputError


class Player:
    def __init__(self, device: Optional[Union[int, str]] = None) -> None:
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

        # Query the output device's native sample rate and resample if needed.
        # PortAudio does not do automatic resampling — mismatched rates cause
        # paInvalidSampleRate errors on devices like USB speakers that only
        # support 44100 or 48000 Hz.
        try:
            device_info = sd.query_devices(self.device, kind="output")
            device_rate = int(device_info["default_samplerate"])
        except Exception:
            device_rate = samplerate

        if samplerate != device_rate:
            try:
                import numpy as np
                from scipy.signal import resample_poly
                g = gcd(device_rate, samplerate)
                up, down = device_rate // g, samplerate // g
                if data.ndim == 1:
                    data = resample_poly(data, up, down).astype("float32")
                else:
                    import numpy as np
                    data = np.stack(
                        [resample_poly(data[:, i], up, down) for i in range(data.shape[1])],
                        axis=1,
                    ).astype("float32")
            except ImportError:
                # scipy not available — play at original rate and let PortAudio decide
                device_rate = samplerate

        try:
            sd.play(data, device_rate, device=self.device)
            sd.wait()
        except Exception as e:
            raise AudioOutputError(f"Playback failed: {e}") from e
