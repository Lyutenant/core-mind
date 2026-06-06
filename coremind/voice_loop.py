from __future__ import annotations

import logging
import socket
import tempfile
import threading
import time
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

# Transcript phrases that explicitly end the follow-up session and return to wake-word mode.
# Matched case-insensitively as whole words against the transcript.
_STOP_PHRASES: frozenset[str] = frozenset([
    "stop", "cancel", "goodbye", "bye", "good bye",
    "nevermind", "never mind", "that's all", "that is all",
    "enough", "quiet", "silence", "exit",
])


def _is_stop_phrase(transcript: str) -> bool:
    """Return True if the transcript is (or contains) a stop/exit phrase."""
    lowered = transcript.strip().lower()
    if lowered in _STOP_PHRASES:
        return True
    # also match if entire transcript is just the phrase plus punctuation
    words = lowered.split()
    for phrase in _STOP_PHRASES:
        phrase_words = phrase.split()
        n = len(phrase_words)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == phrase_words:
                return True
    return False


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
        follow_up_seconds: float = 0.0,
        follow_up_min_words: int = 2,
        post_response_cooldown_seconds: float = 1.0,
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
        self._follow_up_seconds = follow_up_seconds
        self._follow_up_min_words = follow_up_min_words
        self._post_response_cooldown = post_response_cooldown_seconds
        self._session_id = str(uuid.uuid4())

        # Node registration — only active when running in remote (node) mode
        if self._remote_url:
            from coremind.node_id import get_node_id
            self._node_id = get_node_id()
            t = threading.Thread(target=self._hub_sync_loop, daemon=True)
            t.start()
        else:
            self._node_id = None

    def _system_messages(self) -> list[dict]:
        return [{"role": "system", "content": _SYSTEM_PROMPT.format(name=self._name)}]

    # ------------------------------------------------------------------
    # Node ↔ Hub sync (registration, heartbeat, config hot-reload)
    # ------------------------------------------------------------------

    def _hub_sync_loop(self) -> None:
        """Background daemon: register with Hub, then heartbeat + config poll every 30s."""
        import httpx

        base = self._remote_url
        node_id = self._node_id
        name = self._name
        hostname = socket.gethostname()

        # Register immediately
        self._hub_register(httpx, base, node_id, name, hostname)

        while True:
            time.sleep(30)
            try:
                httpx.post(
                    f"{base}/v1/nodes/{node_id}/heartbeat",
                    timeout=5.0,
                )
            except Exception as e:
                logger.debug("Heartbeat failed: %s", e)

            try:
                r = httpx.get(f"{base}/v1/nodes/{node_id}/config", timeout=5.0)
                if r.status_code == 200:
                    self._apply_node_config(r.json())
            except Exception as e:
                logger.debug("Config poll failed: %s", e)

    def _hub_register(self, httpx, base: str, node_id: str, name: str, hostname: str) -> None:
        try:
            r = httpx.post(
                f"{base}/v1/nodes/register",
                json={"node_id": node_id, "name": name, "hostname": hostname},
                timeout=10.0,
            )
            if r.status_code == 200:
                data = r.json()
                if "config" in data:
                    self._apply_node_config(data["config"])
                logger.info("Registered with Hub as Node %s (%s)", name, node_id[:8])
            else:
                logger.warning("Hub registration returned HTTP %s", r.status_code)
        except Exception as e:
            logger.warning("Hub registration failed (will retry on next heartbeat): %s", e)

    def _apply_node_config(self, cfg: dict) -> None:
        """Hot-reload soft settings received from Hub without restarting."""
        if not cfg:
            return
        if "wake_word_threshold" in cfg and self._wake_word is not None:
            if hasattr(self._wake_word, "_threshold"):
                self._wake_word._threshold = float(cfg["wake_word_threshold"])
        if "vad_energy_threshold" in cfg and self._vad is not None:
            if hasattr(self._vad, "threshold"):
                self._vad.threshold = float(cfg["vad_energy_threshold"])
        if "vad_silence_seconds" in cfg:
            self._vad_silence_seconds = float(cfg["vad_silence_seconds"])
        if "vad_max_record_seconds" in cfg:
            self._vad_max_record_seconds = float(cfg["vad_max_record_seconds"])
        if "vad_min_speech_seconds" in cfg:
            self._vad_min_speech_seconds = float(cfg["vad_min_speech_seconds"])
        if "follow_up_seconds" in cfg:
            self._follow_up_seconds = float(cfg["follow_up_seconds"])
        if "follow_up_min_words" in cfg:
            self._follow_up_min_words = int(cfg["follow_up_min_words"])
        if "post_response_cooldown_seconds" in cfg:
            self._post_response_cooldown = float(cfg["post_response_cooldown_seconds"])
        logger.debug("Applied node config overrides: %s", list(cfg.keys()))

    def run_once(self) -> tuple[str, str]:
        """[Wake word] → record → [remote or local] transcribe/LLM/TTS → [speak].

        Returns (transcript, response). Returns ("", "") if no speech detected.
        After speaking, listens for follow-up turns until silence (if configured).
        """
        # Step 1: wait for trigger (Enter key or real wake word)
        if self._wake_word is not None:
            self._status(self._wake_word.trigger_prompt)
            self._wake_word.listen_until_wake_word()
            self._play_wake_chime()
            self._wake_fn()

        # Step 2: record — VAD stops at natural pause, fixed-duration otherwise
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
            transcript, response = self._run_remote(tmp_path)
        else:
            transcript, response = self._process_local_wav(tmp_path)

        if not transcript:
            return "", ""

        # Step 4: follow-up window — keep conversing without re-triggering wake word
        return self._follow_up_loop(transcript, response)

    def _process_local_wav(self, wav_path: str) -> tuple[str, str]:
        """Transcribe + LLM + optional TTS. Deletes wav_path when done."""
        try:
            self._status("Transcribing...")
            transcript = self._stt.transcribe(wav_path)  # type: ignore[union-attr]
        finally:
            Path(wav_path).unlink(missing_ok=True)

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

    def _follow_up_loop(self, last_t: str, last_r: str) -> tuple[str, str]:
        """After a response, listen for follow-up turns until silence or no VAD.

        Exits early when:
        - no speech detected within follow_up_seconds (natural silence)
        - transcript is a stop phrase ("stop", "goodbye", etc.)
        - transcript is shorter than follow_up_min_words (likely background noise)
        """
        if self._vad is None or self._follow_up_seconds <= 0:
            return last_t, last_r

        while True:
            # Brief cooldown so the speaker's output doesn't re-trigger the mic
            if self._post_response_cooldown > 0:
                time.sleep(self._post_response_cooldown)

            self._status("Listening for follow-up...")
            follow_wav = self._recorder.record_with_vad(
                vad=self._vad,
                silence_seconds=self._vad_silence_seconds,
                max_record_seconds=self._vad_max_record_seconds,
                min_speech_seconds=self._vad_min_speech_seconds,
                onset_timeout=self._follow_up_seconds,
            )
            if follow_wav is None:
                # Silence during follow-up window — return to idle / wake word
                return last_t, last_r

            try:
                if self._remote_url:
                    last_t, last_r = self._run_remote(str(follow_wav))
                else:
                    last_t, last_r = self._process_local_wav(str(follow_wav))
            except Exception as e:
                logger.error("Follow-up turn failed: %s", e)
                Path(follow_wav).unlink(missing_ok=True)
                return last_t, last_r

            if not last_t:
                return last_t, last_r

            # Stop phrase — user explicitly ended the session
            if _is_stop_phrase(last_t):
                logger.info("Stop phrase detected (%r) — returning to wake word", last_t)
                self._status("Returning to wake word...")
                return last_t, last_r

            # Transcript too short — likely background noise, not a real command
            if self._follow_up_min_words > 0 and len(last_t.split()) < self._follow_up_min_words:
                logger.info(
                    "Follow-up transcript too short (%r) — discarding, returning to wake word",
                    last_t,
                )
                self._status("Returning to wake word...")
                return last_t, last_r

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
