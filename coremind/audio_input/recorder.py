from __future__ import annotations

import collections
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from coremind import AudioInputError

if TYPE_CHECKING:
    from coremind.vad.base import VoiceActivityDetector


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

    def record_with_vad(
        self,
        vad: "VoiceActivityDetector",
        silence_seconds: float = 1.2,
        max_record_seconds: float = 20.0,
        min_speech_seconds: float = 0.3,
        onset_timeout: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Record until VAD detects end-of-speech.

        Streams audio in 30 ms chunks using callback mode (Pi-compatible).
        Returns the WAV path on success, or None if no speech was detected.
        """
        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except ImportError as e:
            raise AudioInputError(
                "Missing dependency: pip install sounddevice soundfile numpy"
            ) from e

        CHUNK_SECONDS = 0.03  # 30 ms per chunk
        chunk_frames = int(self.sample_rate * CHUNK_SECONDS)
        silence_chunks_needed = max(1, int(silence_seconds / CHUNK_SECONDS))
        min_speech_chunks = max(1, int(min_speech_seconds / CHUNK_SECONDS))
        PRESPEECH_BUF = 5  # ~150 ms pre-speech context

        chunk_queue: queue.Queue = queue.Queue()

        def _callback(indata, frame_count, time_info, status):
            chunk_queue.put(indata.copy())

        speech_chunks: list = []
        prespeech_buf: collections.deque = collections.deque(maxlen=PRESPEECH_BUF)
        silence_count = 0
        speech_started = False
        total_speech_chunks = 0

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                blocksize=chunk_frames,
                callback=_callback,
            ):
                now = time.monotonic()
                # Onset phase: wait up to onset_timeout for speech to begin.
                # Recording phase: once speech starts, allow max_record_seconds.
                # The deadline is reset at speech-start so the two budgets are
                # independent — a long onset window does not inflate recording time.
                if onset_timeout is not None:
                    deadline = now + onset_timeout + 0.5  # small buffer past onset
                else:
                    deadline = now + max_record_seconds + 2.0
                onset_deadline = (now + onset_timeout) if onset_timeout is not None else None
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    try:
                        chunk = chunk_queue.get(timeout=min(0.5, remaining))
                    except queue.Empty:
                        break  # deadline reached

                    pcm_bytes = (
                        (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                    )
                    is_speech = vad.is_speech(pcm_bytes)

                    # No speech yet and onset window expired → bail early
                    if onset_deadline is not None and not speech_started and time.monotonic() > onset_deadline:
                        break

                    if is_speech:
                        if not speech_started:
                            speech_started = True
                            # Reset deadline so the full recording budget starts
                            # from when speech actually begins, not from now.
                            deadline = time.monotonic() + max_record_seconds + 2.0
                            speech_chunks.extend(prespeech_buf)
                        speech_chunks.append(chunk)
                        silence_count = 0
                        total_speech_chunks += 1
                    elif speech_started:
                        speech_chunks.append(chunk)
                        silence_count += 1
                        if silence_count >= silence_chunks_needed:
                            break
                    else:
                        prespeech_buf.append(chunk)
        except Exception as e:
            raise AudioInputError(f"VAD recording failed: {e}") from e

        if total_speech_chunks < min_speech_chunks:
            return None

        if output_path is None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            audio = np.concatenate(speech_chunks)
            sf.write(str(out), audio, self.sample_rate)
        except Exception as e:
            raise AudioInputError(f"Failed to save VAD-recorded audio: {e}") from e

        return out
