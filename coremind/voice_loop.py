from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable

from coremind.audio_input.recorder import Recorder
from coremind.brain.base import BrainClient
from coremind.memory.session_memory import SessionMemory
from coremind.stt.base import SpeechToText

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are {name}, a voice assistant running on a Raspberry Pi. "
    "Keep responses concise and conversational. "
    "Avoid using markdown — your response will be spoken aloud."
)


class VoiceLoop:
    def __init__(
        self,
        name: str,
        recorder: Recorder,
        stt: SpeechToText,
        brain: BrainClient,
        memory: SessionMemory,
        record_seconds: int = 6,
        status_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._name = name
        self._recorder = recorder
        self._stt = stt
        self._brain = brain
        self._memory = memory
        self._record_seconds = record_seconds
        self._status = status_fn or (lambda _: None)

    def _system_messages(self) -> list[dict]:
        return [{"role": "system", "content": _SYSTEM_PROMPT.format(name=self._name)}]

    def run_once(self) -> tuple[str, str]:
        """Record → transcribe → ask brain. Returns (transcript, response)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            self._recorder.record(seconds=self._record_seconds, output_path=tmp_path)
            self._status("Recording complete.")
            self._status("Transcribing...")
            transcript = self._stt.transcribe(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not transcript.strip():
            return "", ""

        # Build messages without mutating memory yet — only commit after ask() succeeds
        # so a transient brain failure doesn't leave an orphaned user turn in history.
        pending = {"role": "user", "content": transcript}
        messages = self._system_messages() + self._memory.get_messages() + [pending]
        self._status("Sending to LLM...")
        response = self._brain.ask(messages)
        self._memory.add("user", transcript)
        self._memory.add("assistant", response)
        return transcript, response
