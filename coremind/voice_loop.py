from __future__ import annotations

import logging
import tempfile
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from coremind import AudioOutputError, BrainError, TTSError
from coremind.audio_input.recorder import Recorder
from coremind.audio_output.player import Player
from coremind.brain.base import BrainClient
from coremind.memory.session_memory import SessionMemory
from coremind.stt.base import SpeechToText
from coremind.tts.base import TextToSpeech

if TYPE_CHECKING:
    from coremind.vad.base import VoiceActivityDetector
    from coremind.wake_word.base import WakeWordDetector

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
        stt: SpeechToText | None = None,
        brain: BrainClient | None = None,
        memory: SessionMemory | None = None,
        record_seconds: int = 6,
        status_fn: Callable[[str], None] | None = None,
        tts: TextToSpeech | None = None,
        player: Player | None = None,
        wake_word: "WakeWordDetector | None" = None,
        wake_fn: Callable[[], None] | None = None,
        vad: "VoiceActivityDetector | None" = None,
        vad_silence_seconds: float = 1.2,
        vad_max_record_seconds: float = 20.0,
        vad_min_speech_seconds: float = 0.3,
        remote_url: str | None = None,
        remote_timeout: float = 90.0,
    ) -> None:
        self._name = name
        self._recorder = recorder
        self._stt = stt
        self._brain = brain
        self._memory = memory
        self._record_seconds = record_seconds
        self._status = status_fn or (lambda _: None)
        self._tts = tts
        self._player = player
        self._wake_word = wake_word
        self._wake_fn = wake_fn or (lambda: None)
        self._vad = vad
        self._vad_silence_seconds = vad_silence_seconds
        self._vad_max_record_seconds = vad_max_record_seconds
        self._vad_min_speech_seconds = vad_min_speech_seconds
        self._remote_url = remote_url.rstrip("/") if remote_url else None
        self._remote_timeout = remote_timeout
        self._session_id = str(uuid.uuid4())

    def _system_messages(self) -> list[dict]:
        return [{"role": "system", "content": _SYSTEM_PROMPT.format(name=self._name)}]

    def run_once(self) -> tuple[str, str]:
        """[Wake word] → record → [remote or local] transcribe/LLM/TTS → [speak].

        Returns (transcript, response). Returns ("", "") if no speech detected.
        """
        # Step 1: wait for trigger (Enter key or real wake word)
        if self._wake_word is not None:
            self._status(self._wake_word.trigger_prompt)
            self._wake_word.listen_until_wake_word()
            self._play_wake_chime()
            self._wake_fn()

        # Step 2: record — VAD stops at natural pause, fixed-duration otherwise
        tmp_path: str | None = None
        try:
            if self._vad is not None:
                self._status("Speak now...")
                result = self._recorder.record_with_vad(
                    vad=self._vad,
                    silence_seconds=self._vad_silence_seconds,
                    max_record_seconds=self._vad_max_record_seconds,
                    min_speech_seconds=self._vad_min_speech_seconds,
                )
                if result is None:
                    return "", ""
                tmp_path = str(result)
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                self._recorder.record(seconds=self._record_seconds, output_path=tmp_path)
                self._status("Recording complete.")

            # Step 3: remote (Mac Mini handles STT + LLM + TTS) or local
            if self._remote_url:
                return self._run_remote(tmp_path)

            self._status("Transcribing...")
            transcript = self._stt.transcribe(tmp_path)  # type: ignore[union-attr]
        finally:
            if tmp_path and not self._remote_url:
                Path(tmp_path).unlink(missing_ok=True)

        if not transcript.strip():
            return "", ""

        pending = {"role": "user", "content": transcript}
        messages = self._system_messages() + self._memory.get_messages() + [pending]  # type: ignore[union-attr]
        self._status("Sending to LLM...")
        response = self._brain.ask(messages)  # type: ignore[union-attr]
        self._memory.add("user", transcript)  # type: ignore[union-attr]
        self._memory.add("assistant", response)  # type: ignore[union-attr]

        if self._tts is not None and self._player is not None and response.strip():
            self._speak(response)

        return transcript, response

    def _run_remote(self, wav_path: str) -> tuple[str, str]:
        """POST wav to Mac Mini server; receive audio response + play it."""
        import httpx

        self._status("Sending to server...")
        try:
            with open(wav_path, "rb") as f:
                resp = httpx.post(
                    f"{self._remote_url}/v1/process",
                    files={"audio": ("audio.wav", f, "audio/wav")},
                    headers={"x-session-id": self._session_id},
                    timeout=self._remote_timeout,
                )
        except httpx.ConnectError as e:
            raise BrainError(f"Cannot reach CoreMind server at {self._remote_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise BrainError(f"CoreMind server timed out after {self._remote_timeout}s") from e
        finally:
            Path(wav_path).unlink(missing_ok=True)

        transcript = urllib.parse.unquote(resp.headers.get("x-transcript", ""))
        response_text = urllib.parse.unquote(resp.headers.get("x-response", ""))

        if resp.status_code == 502:
            error = urllib.parse.unquote(resp.headers.get("x-error", "LLM request failed"))
            raise BrainError(error)

        if resp.status_code != 200:
            raise BrainError(f"Server returned HTTP {resp.status_code}")

        if resp.content and self._player is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(resp.content)
                audio_path = f.name
            try:
                self._status("Speaking...")
                self._player.play(audio_path)
            except AudioOutputError as e:
                logger.warning("Audio playback failed: %s", e)
            finally:
                Path(audio_path).unlink(missing_ok=True)

        return transcript, response_text

    def _play_wake_chime(self) -> None:
        """Play a two-tone ascending chime to confirm wake word detection."""
        try:
            import numpy as np
            import sounddevice as sd

            sr = 22050
            device = self._player.device if self._player is not None else None

            def _tone(freq: float, dur: float) -> np.ndarray:
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                wave = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
                fade = max(1, int(sr * 0.015))
                wave[:fade] *= np.linspace(0, 1, fade)
                wave[-fade:] *= np.linspace(1, 0, fade)
                return wave

            chime = np.concatenate([_tone(660.0, 0.08), _tone(880.0, 0.10)])
            sd.play(chime, sr, device=device)
            sd.wait()
        except Exception:
            pass  # chime failure must never break the turn

    def _speak(self, text: str) -> None:
        self._status("Synthesizing speech...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tts_path = f.name
        try:
            self._tts.synthesize(text, tts_path)  # type: ignore[union-attr]
            self._status("Speaking...")
            self._player.play(tts_path)            # type: ignore[union-attr]
        except (TTSError, AudioOutputError) as e:
            logger.warning("TTS/playback failed — response shown as text only. Error: %s", e)
        finally:
            Path(tts_path).unlink(missing_ok=True)
