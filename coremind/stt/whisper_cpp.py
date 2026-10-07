from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from coremind import STTError
from coremind.stt.base import SpeechToText

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
_DOWNLOAD_TIMEOUT = (10, 60)  # (connect, read) seconds


class WhisperCppSTT(SpeechToText):
    """Speech-to-text via whisper.cpp (pywhispercpp bindings).

    On Apple Silicon the pywhispercpp wheel runs on the Metal GPU, which is the
    point of this provider: faster-whisper (CTranslate2) is CPU-only on macOS.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        language: str = "en",
        *,
        model_path: Optional[str] = None,
        n_threads: Optional[int] = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        vad_model_path: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> None:
        try:
            from pywhispercpp.model import Model
        except ImportError as e:
            raise STTError(
                "pywhispercpp is not installed. Run: pip install 'coremind[stt-cpp]'"
            ) from e
        self.language = language
        self.n_threads = n_threads
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_model_path = vad_model_path or None
        self.initial_prompt = initial_prompt or None
        self.hotwords = hotwords or None

        if self.hotwords:
            logger.warning(
                "whisper.cpp has no hotword biasing — folding stt.hotwords into "
                "the initial prompt instead."
            )
        if self.vad_filter and not self.vad_model_path:
            logger.warning(
                "stt.vad_filter is set but stt.whisper_cpp.vad_model_path is not — "
                "whisper.cpp VAD needs a Silero ggml model file; VAD disabled."
            )

        resolved = _resolve_model(model, model_path)
        logger.info("whisper.cpp model: %s", resolved)
        self._model = Model(
            resolved,
            params_sampling_strategy=1 if beam_size > 1 else 0,
            redirect_whispercpp_logs_to=None,  # keep Hub logs readable
            **self._transcribe_kwargs(),
        )

    def _transcribe_kwargs(self) -> dict:
        """Build the decode parameters passed to pywhispercpp's ``Model``."""
        kwargs: dict = {
            "language": self.language,
            "print_progress": False,
            "print_realtime": False,
        }
        if self.n_threads:
            kwargs["n_threads"] = self.n_threads
        if self.beam_size > 1:
            kwargs["beam_search"] = {"beam_size": self.beam_size, "patience": -1.0}
        if self.vad_filter and self.vad_model_path:
            kwargs["vad"] = True
            kwargs["vad_model_path"] = self.vad_model_path
        # No hotword support in whisper.cpp: fold them into the prompt, the same
        # best-effort fallback WhisperLocalSTT uses on old faster-whisper.
        initial_prompt = " ".join(p for p in (self.initial_prompt, self.hotwords) if p)
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        return kwargs

    def transcribe(self, wav_path: str) -> str:
        try:
            segments = self._model.transcribe(_load_audio_16k(wav_path))
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e


def _resolve_model(model: str, model_path: Optional[str]) -> str:
    """Return a local ggml model file: explicit path, cached file, or a fresh download."""
    if model_path:
        if not Path(model_path).is_file():
            raise STTError(f"stt.whisper_cpp.model_path not found: {model_path}")
        return model_path

    from pywhispercpp.constants import AVAILABLE_MODELS, MODELS_DIR

    if model not in AVAILABLE_MODELS:
        raise STTError(
            f"Unknown whisper.cpp model {model!r}. Use one of: {', '.join(AVAILABLE_MODELS)} "
            "— or set stt.whisper_cpp.model_path to a ggml .bin file."
        )
    target = Path(MODELS_DIR) / f"ggml-{model}.bin"
    if not target.is_file():
        _download(_MODEL_URL.format(name=model), target)
    return str(target)


def _download(url: str, target: Path) -> None:
    """Download to ``target`` atomically (``.part`` then rename), with timeouts."""
    import requests

    logger.info("Downloading whisper.cpp model %s → %s", url, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        part.replace(target)
    except Exception as e:
        part.unlink(missing_ok=True)
        raise STTError(f"Failed to download whisper.cpp model from {url}: {e}") from e


def _load_audio_16k(wav_path: str) -> np.ndarray:
    """Read a WAV as mono float32 at 16 kHz, whatever rate the Node recorded at.

    pywhispercpp's own WAV loader rejects anything but 16 kHz / 16-bit, while
    faster-whisper resamples transparently — this keeps the providers swappable.
    """
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sr != WHISPER_SAMPLE_RATE:
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(WHISPER_SAMPLE_RATE, sr)
            audio = resample_poly(audio, WHISPER_SAMPLE_RATE // g, sr // g)
        except ImportError:
            n_out = int(round(len(audio) * WHISPER_SAMPLE_RATE / sr))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio
            )
    return np.ascontiguousarray(audio, dtype=np.float32)
