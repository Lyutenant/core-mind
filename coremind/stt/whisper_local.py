from __future__ import annotations

from coremind import STTError
from coremind.stt.base import SpeechToText


class MockSTT(SpeechToText):
    def transcribe(self, wav_path: str) -> str:
        return "[mock transcript]"


class WhisperLocalSTT(SpeechToText):
    def __init__(self, model: str = "base", language: str = "en") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise STTError(
                "faster-whisper is not installed. Run: pip install 'coremind[stt]'"
            ) from e
        self._model = WhisperModel(model, device="cpu", compute_type="int8")
        self.language = language

    def transcribe(self, wav_path: str) -> str:
        try:
            segments, _ = self._model.transcribe(wav_path, language=self.language)
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e
