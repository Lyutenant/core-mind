from __future__ import annotations

import asyncio
import datetime
import json
import logging
import tempfile
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_config_path: Path = Path("config.yaml")
_settings = None
_stt = None
_tts = None
_brain = None
_sessions: dict = {}
_conversation_log: list[dict] = []
_event_listeners: list[asyncio.Queue] = []
_turn_count: int = 0

_SYSTEM_PROMPT = (
    "You are {name}, a voice assistant running on a Raspberry Pi. "
    "Keep responses concise and conversational. "
    "Avoid using markdown — your response will be spoken aloud."
)


def configure(config_path: str | None = None) -> None:
    global _config_path
    if config_path:
        _config_path = Path(config_path)


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

def _get_settings():
    global _settings
    if _settings is None:
        from coremind.config import load_settings
        _settings = load_settings(str(_config_path))
    return _settings


def _get_stt():
    global _stt
    if _stt is None:
        from coremind.stt.whisper_local import WhisperLocalSTT
        s = _get_settings()
        _stt = WhisperLocalSTT(model=s.stt.model, language=s.stt.language)
        logger.info("STT loaded: %s", s.stt.model)
    return _stt


def _get_brain():
    global _brain
    if _brain is None:
        from coremind.brain.ollama_client import OllamaClient
        s = _get_settings()
        _brain = OllamaClient(
            base_url=s.ollama.base_url,
            model=s.ollama.model,
            timeout=s.brain.timeout_seconds,
            no_think=s.ollama.no_think,
            options=s.ollama.options,
        )
    return _brain


def _get_tts():
    global _tts
    if _tts is None:
        s = _get_settings()
        provider = s.tts.provider
        if provider == "piper_local" and s.tts.model_path:
            from coremind.tts.piper_local import PiperLocalTTS
            _tts = PiperLocalTTS(model_path=s.tts.model_path)
            logger.info("TTS loaded: piper_local")
        elif provider == "espeak":
            from coremind.tts.piper_local import EspeakTTS
            _tts = EspeakTTS(voice=s.tts.voice or "en")
            logger.info("TTS loaded: espeak")
        elif provider == "mock":
            from coremind.tts.piper_local import MockTTS
            _tts = MockTTS()
        else:
            logger.warning("TTS unavailable — piper_local requires model_path")
    return _tts


def _reset_singletons() -> None:
    global _settings, _stt, _tts, _brain
    _settings = None
    _stt = None
    _tts = None
    _brain = None


def _get_session(session_id: str):
    from coremind.memory.session_memory import SessionMemory
    if session_id not in _sessions:
        _sessions[session_id] = SessionMemory(max_turns=_get_settings().memory.max_turns)
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# SSE broadcast
# ---------------------------------------------------------------------------

def _broadcast(event: dict) -> None:
    data = json.dumps(event)
    dead = []
    for q in _event_listeners:
        try:
            q.put_nowait(data)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            _event_listeners.remove(q)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, Response, StreamingResponse

    app = FastAPI(title="CoreMind Hub", version="0.1.0")

    _STATIC = Path(__file__).parent / "static"

    # -----------------------------------------------------------------------
    # Static / UI
    # -----------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(_STATIC / "index.html")

    # -----------------------------------------------------------------------
    # Health & status
    # -----------------------------------------------------------------------

    @app.get("/api/health")
    async def api_health():
        s = _get_settings()
        ollama_ok = False
        try:
            import httpx
            r = httpx.get(f"{s.ollama.base_url}/api/tags", timeout=3.0)
            ollama_ok = r.status_code == 200
        except Exception:
            pass
        return {
            "status": "ok",
            "name": s.app.name,
            "model": s.ollama.model,
            "ollama_url": s.ollama.base_url,
            "ollama_reachable": ollama_ok,
            "stt_provider": s.stt.provider,
            "stt_model": s.stt.model,
            "stt_loaded": _stt is not None,
            "tts_provider": s.tts.provider,
            "tts_loaded": _tts is not None,
            "session_count": len(_sessions),
            "turn_count": _turn_count,
        }

    # -----------------------------------------------------------------------
    # Config CRUD
    # -----------------------------------------------------------------------

    @app.get("/api/config")
    async def get_config():
        return _get_settings().model_dump()

    @app.post("/api/config")
    async def save_config(request: Request):
        data = await request.json()
        # Validate against Settings model before writing
        try:
            from coremind.config.settings import Settings
            Settings.model_validate(data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        try:
            _config_path.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")
        _reset_singletons()
        _broadcast({"type": "config_saved"})
        return {"status": "saved"}

    # -----------------------------------------------------------------------
    # Conversation history
    # -----------------------------------------------------------------------

    @app.get("/api/conversation")
    async def get_conversation():
        return _conversation_log

    @app.delete("/api/conversation")
    async def clear_conversation():
        _conversation_log.clear()
        _sessions.clear()
        _broadcast({"type": "conversation_cleared"})
        return {"status": "cleared"}

    # -----------------------------------------------------------------------
    # SSE events
    # -----------------------------------------------------------------------

    @app.get("/api/events")
    async def sse_events(request: Request):
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        _event_listeners.append(q)

        async def generate():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        data = await asyncio.wait_for(q.get(), timeout=25.0)
                        yield f"data: {data}\n\n"
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            finally:
                try:
                    _event_listeners.remove(q)
                except ValueError:
                    pass

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -----------------------------------------------------------------------
    # Core audio processing endpoint (called by CoreMind Node / Pi)
    # -----------------------------------------------------------------------

    @app.post("/v1/process")
    async def process_audio(
        audio: UploadFile = File(...),
        x_session_id: Optional[str] = Header(default=None),
    ) -> Response:
        global _turn_count
        s = _get_settings()
        session_id = x_session_id or str(uuid.uuid4())

        _broadcast({"type": "status", "text": "Audio received — transcribing..."})

        wav_bytes = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        try:
            transcript = _get_stt().transcribe(wav_path)
        except Exception as e:
            logger.error("STT failed: %s", e)
            _broadcast({"type": "status", "text": "STT failed."})
            return Response(content=b"", headers={"X-Transcript": "", "X-Response": ""})
        finally:
            Path(wav_path).unlink(missing_ok=True)

        if not transcript.strip():
            _broadcast({"type": "status", "text": "Idle"})
            return Response(content=b"", headers={"X-Transcript": "", "X-Response": ""})

        _broadcast({"type": "transcript", "text": transcript})
        _broadcast({"type": "status", "text": "Sending to LLM..."})

        memory = _get_session(session_id)
        messages = (
            [{"role": "system", "content": _SYSTEM_PROMPT.format(name=s.app.name)}]
            + memory.get_messages()
            + [{"role": "user", "content": transcript}]
        )

        try:
            response_text = _get_brain().ask(messages)
        except Exception as e:
            logger.error("LLM failed: %s", e)
            _broadcast({"type": "status", "text": f"LLM error: {e}"})
            return Response(
                content=b"",
                status_code=502,
                headers={
                    "X-Transcript": urllib.parse.quote(transcript),
                    "X-Response": "",
                    "X-Error": urllib.parse.quote(str(e)),
                },
            )

        memory.add("user", transcript)
        memory.add("assistant", response_text)

        _broadcast({"type": "response", "text": response_text})
        _broadcast({"type": "status", "text": "Synthesizing speech..."})

        audio_bytes = b""
        if response_text.strip():
            tts = _get_tts()
            if tts is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tts_path = f.name
                try:
                    tts.synthesize(response_text, tts_path)
                    audio_bytes = Path(tts_path).read_bytes()
                except Exception as e:
                    logger.warning("TTS failed: %s", e)
                finally:
                    Path(tts_path).unlink(missing_ok=True)

        _turn_count += 1
        turn = {
            "turn": _turn_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "transcript": transcript,
            "response": response_text,
            "session_id": session_id,
        }
        _conversation_log.append(turn)
        if len(_conversation_log) > 200:
            _conversation_log.pop(0)

        _broadcast({"type": "turn", **turn})
        _broadcast({"type": "status", "text": "Idle"})

        return Response(
            content=audio_bytes,
            media_type="audio/wav" if audio_bytes else "application/octet-stream",
            headers={
                "X-Transcript": urllib.parse.quote(transcript),
                "X-Response": urllib.parse.quote(response_text),
            },
        )

except ImportError:
    app = None  # type: ignore[assignment]
