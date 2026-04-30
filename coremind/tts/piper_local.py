from __future__ import annotations

import logging
import subprocess

from coremind import TTSError
from coremind.tts.base import TextToSpeech

logger = logging.getLogger(__name__)


class MockTTS(TextToSpeech):
    def synthesize(self, text: str, output_path: str) -> str:
        return output_path


class PiperLocalTTS(TextToSpeech):
    def __init__(self, model_path: str) -> None:
        try:
            from piper.voice import PiperVoice
        except ImportError as e:
            raise TTSError(
                "piper-tts is not installed. Run: pip install piper-tts"
            ) from e
        try:
            self._voice = PiperVoice.load(model_path)
        except Exception as e:
            raise TTSError(f"Failed to load Piper voice model from {model_path!r}: {e}") from e

    def synthesize(self, text: str, output_path: str) -> str:
        import wave
        import numpy as np
        try:
            with wave.open(output_path, "wb") as wav_file:
                header_written = False
                for chunk in self._voice.synthesize(text):
                    if not header_written:
                        wav_file.setnchannels(chunk.sample_channels)
                        wav_file.setsampwidth(chunk.sample_width)
                        wav_file.setframerate(chunk.sample_rate)
                        header_written = True
                    pcm = (chunk.audio_float_array * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                    wav_file.writeframes(pcm)
            if not header_written:
                raise TTSError("Piper returned no audio for the given text")
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"Piper synthesis failed: {e}") from e
        return output_path


class EspeakTTS(TextToSpeech):
    def __init__(self, voice: str = "en") -> None:
        self.voice = voice

    def synthesize(self, text: str, output_path: str) -> str:
        try:
            subprocess.run(
                ["espeak-ng", "-v", self.voice, "-w", output_path, text],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as e:
            raise TTSError(
                "espeak-ng not found. Install it: sudo apt install espeak-ng"
            ) from e
        except subprocess.CalledProcessError as e:
            raise TTSError(
                f"espeak-ng failed: {e.stderr.decode(errors='replace')[:200]}"
            ) from e
        return output_path
