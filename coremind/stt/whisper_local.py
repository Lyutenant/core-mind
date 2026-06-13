from __future__ import annotations

import inspect
import logging
from typing import Optional

from coremind import STTError
from coremind.stt.base import SpeechToText

logger = logging.getLogger(__name__)


class MockSTT(SpeechToText):
    def transcribe(self, wav_path: str) -> str:
        return "[mock transcript]"


class WhisperLocalSTT(SpeechToText):
    def __init__(
        self,
        model: str = "base",
        language: str = "en",
        *,
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = False,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise STTError(
                "faster-whisper is not installed. Run: pip install 'coremind[stt]'"
            ) from e
        self._model = WhisperModel(model, device="cpu", compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt or None
        self.hotwords = hotwords or None

        # `hotwords` is only supported in faster-whisper >= 1.0. Detect it once
        # so older installs degrade gracefully instead of raising per-call.
        self._supports_hotwords = "hotwords" in inspect.signature(
            self._model.transcribe
        ).parameters
        if self.hotwords and not self._supports_hotwords:
            logger.warning(
                "stt.hotwords is set but this faster-whisper version doesn't "
                "support it — folding hotwords into initial_prompt instead. "
                "Upgrade with: pip install -U faster-whisper"
            )

    def _transcribe_kwargs(self) -> dict:
        """Build the keyword arguments passed to ``WhisperModel.transcribe``."""
        kwargs: dict = {
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
        }
        initial_prompt = self.initial_prompt
        if self.hotwords:
            if self._supports_hotwords:
                kwargs["hotwords"] = self.hotwords
            else:
                # Fold the hotword list into the prompt as a best-effort fallback.
                initial_prompt = " ".join(
                    p for p in (initial_prompt, self.hotwords) if p
                )
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        return kwargs

    def transcribe(self, wav_path: str) -> str:
        try:
            segments, _ = self._model.transcribe(wav_path, **self._transcribe_kwargs())
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e
