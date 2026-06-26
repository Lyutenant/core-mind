from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
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
_dispatcher = None
_sessions: dict = {}                    # session_id → live SessionMemory (LLM context cache)
_sessions_registry: dict[str, dict] = {}   # session_id → Session (durable rounds, persisted)
_active_session_id: str | None = None   # the one session new voice/chat turns land in
_event_listeners: list[asyncio.Queue] = []
_turn_count: int = 0
_node_registry: dict[str, dict] = {}   # node_id → {name, hostname, online, last_seen, config_overrides}

# ---------------------------------------------------------------------------
# Node override persistence  (~/.coremind/node-overrides.json on the Hub)
# ---------------------------------------------------------------------------

_OVERRIDES_PATH = Path.home() / ".coremind" / "node-overrides.json"


def _load_node_overrides() -> dict:
    try:
        if _OVERRIDES_PATH.exists():
            return json.loads(_OVERRIDES_PATH.read_text())
    except Exception as exc:
        logger.warning("Could not load node overrides: %s", exc)
    return {}


def _save_node_overrides(data: dict) -> None:
    try:
        _OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _OVERRIDES_PATH.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        logger.warning("Could not save node overrides: %s", exc)


_persisted_overrides: dict = _load_node_overrides()

# ---------------------------------------------------------------------------
# Sessions  (~/.coremind/sessions.json on the Hub)
#
# A Session groups a conversation's rounds under a hash id. The server keeps many
# sessions and marks exactly one *active* (new voice/chat turns land there); each
# dashboard device chooses which session it *views*. Persisted across restarts so
# the conversation history survives a Hub restart and stays shared across devices.
# ---------------------------------------------------------------------------

_SESSIONS_PATH = Path.home() / ".coremind" / "sessions.json"
_MAX_SESSIONS = 50
_MAX_ROUNDS = 200


def _new_session() -> dict:
    now = datetime.datetime.now().isoformat()
    return {
        "id": uuid.uuid4().hex,
        "title": "New session",
        "created_at": now,
        "updated_at": now,
        "rounds": [],
    }


def _most_recent_session_id() -> str:
    return max(_sessions_registry.values(), key=lambda s: s.get("updated_at", ""))["id"]


def _save_sessions() -> None:
    """Atomically persist the session registry (tempfile + os.replace)."""
    try:
        _SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"active": _active_session_id, "sessions": _sessions_registry}, indent=2
        )
        fd, tmp = tempfile.mkstemp(dir=str(_SESSIONS_PATH.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp, _SESSIONS_PATH)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as exc:
        logger.warning("Could not save sessions: %s", exc)


def _load_sessions() -> None:
    global _sessions_registry, _active_session_id
    try:
        if _SESSIONS_PATH.exists():
            data = json.loads(_SESSIONS_PATH.read_text())
            sessions = data.get("sessions")
            if isinstance(sessions, dict):
                _sessions_registry = sessions
            _active_session_id = data.get("active")
    except Exception as exc:
        logger.warning("Could not load sessions: %s", exc)
    # Enforce invariants: a non-empty registry with a valid active pointer.
    if not _sessions_registry:
        s = _new_session()
        _sessions_registry[s["id"]] = s
        _active_session_id = s["id"]
    elif _active_session_id not in _sessions_registry:
        _active_session_id = _most_recent_session_id()


def _prune_sessions() -> None:
    """Keep the most-recently-updated _MAX_SESSIONS (never drop the active one)."""
    if len(_sessions_registry) <= _MAX_SESSIONS:
        return
    ordered = sorted(
        _sessions_registry.values(), key=lambda s: s.get("updated_at", ""), reverse=True
    )
    keep = {s["id"] for s in ordered[:_MAX_SESSIONS]}
    keep.add(_active_session_id)
    for sid in list(_sessions_registry.keys()):
        if sid not in keep:
            del _sessions_registry[sid]
            _sessions.pop(sid, None)


def _get_active_session() -> dict:
    """Return the active session, repairing the pointer / creating one if needed."""
    global _active_session_id
    if _active_session_id not in _sessions_registry:
        if _sessions_registry:
            _active_session_id = _most_recent_session_id()
        else:
            s = _new_session()
            _sessions_registry[s["id"]] = s
            _active_session_id = s["id"]
    return _sessions_registry[_active_session_id]


def _set_active(session_id: str) -> dict:
    """Make a session active and broadcast it (so follow-mode devices switch)."""
    global _active_session_id
    sess = _sessions_registry.get(session_id) or _get_active_session()
    _active_session_id = sess["id"]
    _save_sessions()
    _broadcast(
        {"type": "session_activated", "session_id": sess["id"], "title": sess["title"]}
    )
    return sess


def _append_round(session_id: str, turn: dict) -> None:
    """Append a completed turn to a session, trim, title, persist."""
    sess = _sessions_registry.get(session_id) or _get_active_session()
    sess["rounds"].append(turn)
    if len(sess["rounds"]) > _MAX_ROUNDS:
        sess["rounds"] = sess["rounds"][-_MAX_ROUNDS:]
    if sess.get("title", "New session") == "New session":
        first = (turn.get("transcript") or "").strip()
        if first:
            sess["title"] = first[:60]
    sess["updated_at"] = datetime.datetime.now().isoformat()
    _save_sessions()


_load_sessions()


def _build_system_prompt(s) -> str:
    parts = [
        f"You are {s.app.name}, a voice assistant running on a Raspberry Pi.",
        "Keep responses concise and conversational.",
        "Avoid using markdown — your response will be spoken aloud.",
    ]
    if s.app.personality:
        parts.append(
            "Adopt the following persona and tone in all responses, while staying "
            f"accurate and concise: {s.app.personality}"
        )
    if s.app.user_location:
        location_uses = "weather"
        if s.app.user_timezone:
            location_uses = "weather and time"
        parts.append(
            f"The user is located in {s.app.user_location}. "
            f"When they ask about {location_uses} without specifying a place, use their location."
        )
    if s.app.user_timezone:
        parts.append(f"The user's local timezone is {s.app.user_timezone}.")
    if s.app.home_airport:
        parts.append(
            f"The user's home airport is {s.app.home_airport}. "
            "Use it when they ask about aviation weather without specifying an airport."
        )
    return " ".join(parts)


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
        _stt = WhisperLocalSTT(
            model=s.stt.model,
            language=s.stt.language,
            compute_type=s.stt.compute_type,
            beam_size=s.stt.beam_size,
            vad_filter=s.stt.vad_filter,
            initial_prompt=s.stt.initial_prompt,
            hotwords=s.stt.hotwords,
        )
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


def _get_dispatcher():
    global _dispatcher
    if _dispatcher is None:
        from coremind.tools.dispatcher import ToolDispatcher
        s = _get_settings()
        _dispatcher = ToolDispatcher()
        if s.tools.enabled and s.tools.built_in:
            # The 'look' tool needs a vision-capable Ollama client (separate model).
            vision_client = None
            if s.ollama.vision_model:
                from coremind.brain.ollama_client import OllamaClient
                vision_client = OllamaClient(
                    base_url=s.ollama.base_url,
                    model=s.ollama.vision_model,
                    timeout=s.brain.timeout_seconds,
                )
            _dispatcher.register_built_ins(
                s.tools.built_in,
                default_timezone=s.app.user_timezone,
                home_airport=s.app.home_airport,
                taf_airport=s.app.taf_airport,
                vision_client=vision_client,
            )
            logger.info("Tools loaded: %s", s.tools.built_in)
        # Re-attach the MCP manager if one is already running (survives config reloads).
        if _mcp_manager is not None:
            _dispatcher.set_mcp_manager(_mcp_manager)
    return _dispatcher


def _reset_singletons() -> None:
    global _settings, _stt, _tts, _brain, _dispatcher
    _settings = None
    _stt = None
    _tts = None
    _brain = None
    _dispatcher = None


def _get_session(session_id: str):
    from coremind.memory.session_memory import SessionMemory
    if session_id not in _sessions:
        mem = SessionMemory(max_turns=_get_settings().memory.max_turns)
        # Seed the live context from the session's persisted rounds so a
        # restart-resumed (or device-switched) conversation keeps its memory.
        sess = _sessions_registry.get(session_id)
        if sess:
            for rnd in sess["rounds"][-mem.max_turns:]:
                if rnd.get("transcript"):
                    mem.add("user", rnd["transcript"])
                if rnd.get("response"):
                    mem.add("assistant", rnd["response"])
        _sessions[session_id] = mem
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
# LLM + tool loop
# ---------------------------------------------------------------------------

async def _run_llm_with_tools(messages: list[dict]) -> tuple[str, list[dict]]:
    """Call the LLM, execute any tool calls it requests, return (response_text, tool_calls_used)."""
    import asyncio

    brain = _get_brain()
    dispatcher = _get_dispatcher()
    tool_defs = dispatcher.get_tool_definitions()

    logger.debug("Active tools (%d): %s", len(tool_defs), [t["function"]["name"] for t in tool_defs])

    if not tool_defs:
        return brain.ask(messages), []

    conv = list(messages)
    _MAX_ROUNDS = 5
    result = await asyncio.to_thread(brain.ask_with_tools, conv, tool_defs)
    tool_calls_used: list[dict] = []

    for _ in range(_MAX_ROUNDS):
        if not result.tool_calls:
            break

        # Append the assistant turn (with tool_calls) before adding tool results.
        conv.append({
            "role": "assistant",
            "content": result.content,
            "tool_calls": result.tool_calls,
        })

        for call in result.tool_calls:
            fn = call.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments") or {}
            _broadcast({"type": "status", "text": f"Running tool: {fn_name}…"})
            tool_result = await dispatcher.execute_async(fn_name, fn_args)
            _broadcast({"type": "tool_call", "name": fn_name, "args": fn_args, "result": tool_result})
            logger.info("Tool %r → %r", fn_name, tool_result[:120] if tool_result else "")
            conv.append({"role": "tool", "tool_name": fn_name, "content": tool_result})
            tool_calls_used.append({"name": fn_name, "args": fn_args, "result": tool_result})

        _broadcast({"type": "status", "text": "Sending to LLM…"})
        result = await asyncio.to_thread(brain.ask_with_tools, conv, tool_defs)
    else:
        logger.warning("Tool loop hit max rounds (%d) — returning last content", _MAX_ROUNDS)

    return result.content, tool_calls_used


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

    app = FastAPI(title="CoreMind Hub", version="0.1.0")

    _STATIC = Path(__file__).parent / "static"

    # -----------------------------------------------------------------------
    # Background: mark nodes offline after 90s of missed heartbeats
    # -----------------------------------------------------------------------

    async def _node_offline_watcher():
        while True:
            await asyncio.sleep(30)
            now = datetime.datetime.utcnow()
            for node_id, info in list(_node_registry.items()):
                if info.get("online") and (now - info["last_seen"]).total_seconds() > 90:
                    _node_registry[node_id]["online"] = False
                    _broadcast({"type": "node_offline", "node_id": node_id})
                    logger.info("Node %s (%s) marked offline", info["name"], node_id[:8])

    _mcp_manager = None

    @app.on_event("startup")
    async def _startup():
        global _mcp_manager
        asyncio.create_task(_node_offline_watcher())
        s = _get_settings()
        if s.tools.enabled and s.tools.mcp_servers:
            from coremind.tools.mcp_manager import MCPManager
            _mcp_manager = MCPManager()
            await _mcp_manager.start(s.tools.mcp_servers)
            _get_dispatcher().set_mcp_manager(_mcp_manager)
            mcp_tool_count = len(_mcp_manager._schemas)
            if mcp_tool_count:
                logger.info(
                    "MCP tools ready: %d tool(s) from %d server(s). "
                    "Total tools available: %d",
                    mcp_tool_count,
                    len(s.tools.mcp_servers),
                    len(_get_dispatcher().get_tool_definitions()),
                )
            else:
                logger.warning(
                    "MCP servers configured but no tools registered — "
                    "Node may not be reachable yet. Will retry in background. "
                    "Check GET /api/tools to monitor status."
                )

    @app.on_event("shutdown")
    async def _shutdown():
        if _mcp_manager is not None:
            await _mcp_manager.stop()

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
            "mode": s.mode,
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
        from coremind.config.settings import Settings, merge_config_text

        # The dashboard only sends the sections it renders. Merge onto the
        # existing file so unmanaged sections (tools.mcp_servers, node_mcp, …)
        # and comments are preserved instead of being wiped to defaults.
        existing = _config_path.read_text() if _config_path.exists() else ""
        try:
            merged_text = merge_config_text(existing, data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to merge config: {e}")

        # Validate the MERGED result — that is what gets written and reloaded.
        try:
            Settings.model_validate(yaml.safe_load(merged_text) or {})
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        try:
            _config_path.write_text(merged_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")
        _reset_singletons()
        _broadcast({"type": "config_saved"})
        return {"status": "saved"}

    # -----------------------------------------------------------------------
    # Tool registry inspection
    # -----------------------------------------------------------------------

    @app.get("/api/tools")
    async def list_tools():
        """Return all currently registered tools (built-in + MCP).

        Useful for verifying MCP connectivity: if MCP tools are missing here,
        the Hub has not yet connected to the Node MCP server.
        """
        dispatcher = _get_dispatcher()
        tool_defs = dispatcher.get_tool_definitions()
        built_in_names = list(dispatcher._tools.keys())
        mcp_names = [
            t["function"]["name"] for t in tool_defs
            if t["function"]["name"] not in built_in_names
        ]
        schemas = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "source": "built_in" if t["function"]["name"] in built_in_names else "mcp",
                "parameters": t["function"].get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for t in tool_defs
        ]
        return {
            "total": len(tool_defs),
            "built_in": built_in_names,
            "mcp": mcp_names,
            "mcp_connected": _mcp_manager is not None and bool(mcp_names),
            "schemas": schemas,
        }

    @app.post("/api/tools/invoke")
    async def invoke_tool(request: Request):
        """Directly invoke a built-in tool by name with the given arguments.

        Restricted to built-in tools only — MCP tools (filesystem, music, etc.)
        can only be triggered through the voice loop / LLM tool calls.
        Built-in tools marked requires_confirmation=True are also blocked.
        """
        data = await request.json()
        name = (data.get("name") or "").strip()
        args = data.get("args") or {}
        if not name:
            return JSONResponse(status_code=400, content={"error": "name is required"})
        dispatcher = _get_dispatcher()
        tool = dispatcher._tools.get(name)
        if tool is None:
            return JSONResponse(
                status_code=403,
                content={"error": "Only built-in tools can be invoked directly. MCP tools run through the voice loop."},
            )
        if tool.requires_confirmation:
            return JSONResponse(
                status_code=403,
                content={"error": f"Tool '{name}' requires explicit confirmation and cannot be invoked directly."},
            )
        try:
            result = await dispatcher.execute_async(name, args)
            return {"result": result}
        except Exception as e:
            logger.error("Tool invoke error for %r: %s", name, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/api/tools/refresh")
    async def refresh_tools():
        """Force an immediate re-sync of every connected MCP server's tool list.

        Lets a capability just enabled on a Node (e.g. the camera's `capture_image`)
        show up without waiting for the periodic re-sync or restarting the Hub.
        """
        if _mcp_manager is None:
            return JSONResponse(
                status_code=409,
                content={"error": "No MCP servers configured — nothing to refresh."},
            )
        refreshed = await _mcp_manager.refresh()
        logger.info("Manual MCP tool refresh: %s", refreshed)
        return {"refreshed": refreshed}

    # -----------------------------------------------------------------------
    # Vision: manual snapshot to the dashboard
    # -----------------------------------------------------------------------

    @app.post("/api/vision/capture")
    async def vision_capture():
        """Capture one still frame from the Node camera and return it as a data URL.

        Calls the Node's `capture_image` MCP tool directly (it is intentionally blocked
        from /api/tools/invoke). Needs only the Node's `vision.enabled` — no vision model
        required. The JPEG is held in memory and handed straight to the browser; it is
        never written to disk and never logged (metadata only).
        """
        if _mcp_manager is None or "capture_image" not in _mcp_manager.tool_to_server:
            return JSONResponse(
                status_code=409,
                content={"error": "Node camera not available — enable vision on the Node and check the node MCP connection."},
            )
        from coremind.node_mcp.tools.camera_capture import ERROR_PREFIX
        from coremind.tools.built_in.vision_tool import looks_like_jpeg_base64
        try:
            frame_b64 = await _get_dispatcher().execute_async("capture_image", {})
        except Exception as e:
            logger.error("Vision capture error: %s", e)
            return JSONResponse(status_code=502, content={"error": str(e)})
        if frame_b64.startswith(ERROR_PREFIX):
            msg = frame_b64[len(ERROR_PREFIX):].strip()
            logger.warning("Vision capture: camera error — %s", msg)
            return JSONResponse(status_code=502, content={"error": f"Camera error: {msg}"})
        if not looks_like_jpeg_base64(frame_b64):
            logger.warning(
                "Vision capture: unreadable frame — %d chars, prefix=%r",
                len(frame_b64), frame_b64[:60],
            )
            return JSONResponse(status_code=502, content={"error": "Camera returned an unreadable frame."})
        logger.info("Vision capture: %d chars (base64)", len(frame_b64))
        return {
            "image": "data:image/jpeg;base64," + frame_b64,
            "captured_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

    @app.post("/api/vision/describe")
    async def vision_describe(request: Request):
        """Run the Hub vision model on an already-captured frame; return a text description.

        Stateless: the dashboard posts back the base64 it is displaying, so the description
        matches the frame on screen (no re-capture, no server-side frame cache). Requires
        `ollama.vision_model`.
        """
        s = _get_settings()
        if not s.ollama.vision_model:
            return JSONResponse(
                status_code=409,
                content={"error": "Vision model not configured (set ollama.vision_model)."},
            )
        data = await request.json()
        image = (data.get("image") or "").strip()
        prefix = "data:image/jpeg;base64,"
        if image.startswith(prefix):
            image = image[len(prefix):]
        from coremind.tools.built_in.vision_tool import LookAtSceneTool, looks_like_jpeg_base64
        if not looks_like_jpeg_base64(image):
            return JSONResponse(status_code=400, content={"error": "No valid image provided."})
        prompt = (data.get("prompt") or "").strip() or LookAtSceneTool._DEFAULT_PROMPT
        try:
            from coremind.brain.ollama_client import OllamaClient
            client = OllamaClient(
                base_url=s.ollama.base_url,
                model=s.ollama.vision_model,
                timeout=s.brain.timeout_seconds,
            )
            # Vision inference can take several seconds — keep it off the event loop.
            text = await asyncio.to_thread(client.describe_image, image, prompt)
        except Exception as e:
            logger.error("Vision describe error: %s", e)
            return JSONResponse(status_code=502, content={"error": str(e)})
        return {"description": text}

    # -----------------------------------------------------------------------
    # Conversation history
    # -----------------------------------------------------------------------

    @app.get("/api/conversation")
    async def get_conversation():
        # Back-compat: the active session's rounds.
        return _get_active_session()["rounds"]

    @app.delete("/api/conversation")
    async def clear_conversation():
        # "Clear" empties the active session (and its live memory); next turn re-titles.
        sess = _get_active_session()
        sess["rounds"] = []
        sess["title"] = "New session"
        sess["updated_at"] = datetime.datetime.now().isoformat()
        _sessions.pop(sess["id"], None)
        _save_sessions()
        _broadcast({"type": "conversation_cleared", "session_id": sess["id"]})
        _broadcast({"type": "session_list_changed"})
        return {"status": "cleared"}

    # -----------------------------------------------------------------------
    # Sessions (cross-device conversation sync)
    # -----------------------------------------------------------------------

    @app.get("/api/sessions")
    async def list_sessions():
        _get_active_session()  # guarantee a valid active session for first load
        items = [
            {
                "id": s["id"],
                "title": s["title"],
                "created_at": s["created_at"],
                "updated_at": s["updated_at"],
                "round_count": len(s["rounds"]),
                "active": s["id"] == _active_session_id,
            }
            for s in sorted(
                _sessions_registry.values(),
                key=lambda s: s.get("updated_at", ""),
                reverse=True,
            )
        ]
        return {"active": _active_session_id, "sessions": items}

    @app.post("/api/sessions")
    async def create_session():
        global _active_session_id
        sess = _new_session()
        _sessions_registry[sess["id"]] = sess
        _active_session_id = sess["id"]
        _prune_sessions()
        _save_sessions()
        _broadcast(
            {"type": "session_activated", "session_id": sess["id"], "title": sess["title"]}
        )
        _broadcast({"type": "session_list_changed"})
        return sess

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        sess = _sessions_registry.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="session not found")
        return sess

    @app.post("/api/sessions/{session_id}/activate")
    async def activate_session(session_id: str):
        if session_id not in _sessions_registry:
            raise HTTPException(status_code=404, detail="session not found")
        sess = _set_active(session_id)
        return {"status": "activated", "active": sess["id"]}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        global _active_session_id
        if session_id not in _sessions_registry:
            raise HTTPException(status_code=404, detail="session not found")
        del _sessions_registry[session_id]
        _sessions.pop(session_id, None)
        active_changed = _active_session_id == session_id
        if active_changed:
            _active_session_id = None
            _get_active_session()  # pick a survivor or create a fresh one
        _save_sessions()
        if active_changed:
            sess = _sessions_registry[_active_session_id]
            _broadcast(
                {"type": "session_activated", "session_id": sess["id"], "title": sess["title"]}
            )
        _broadcast({"type": "session_list_changed"})
        return {"status": "deleted", "active": _active_session_id}

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
    # Node registration & remote config
    # -----------------------------------------------------------------------

    @app.post("/v1/nodes/register")
    async def node_register(request: Request):
        data = await request.json()
        node_id = data.get("node_id", "")
        if not node_id:
            raise HTTPException(status_code=400, detail="node_id required")
        name = data.get("name", "Node")
        hostname = data.get("hostname", "unknown")
        node_config_ts_str: str | None = data.get("config_timestamp")
        config_snapshot: dict = data.get("config_snapshot", {})

        now = datetime.datetime.utcnow()
        existing = _node_registry.get(node_id, {})

        # Determine active overrides, seeding from disk if this is a fresh start.
        hub_record = _persisted_overrides.get(node_id, {})
        active_overrides: dict = existing.get("config_overrides") or hub_record.get("overrides", {})

        # Timestamp contest: whichever side was updated more recently wins.
        if node_config_ts_str and config_snapshot:
            try:
                node_ts = datetime.datetime.fromisoformat(node_config_ts_str)
                hub_ts_str = hub_record.get("updated_at")
                if hub_ts_str:
                    hub_ts = datetime.datetime.fromisoformat(hub_ts_str)
                    # Normalise both to UTC-aware for a safe comparison.
                    if node_ts.tzinfo is None:
                        node_ts = node_ts.replace(tzinfo=datetime.timezone.utc)
                    if hub_ts.tzinfo is None:
                        hub_ts = hub_ts.replace(tzinfo=datetime.timezone.utc)
                    node_wins = node_ts > hub_ts
                else:
                    node_wins = True  # no Hub record yet → Node's config.yaml is the authority
                if node_wins:
                    active_overrides = config_snapshot
                    _persisted_overrides[node_id] = {
                        "updated_at": node_config_ts_str,
                        "overrides": config_snapshot,
                    }
                    _save_node_overrides(_persisted_overrides)
                    logger.info(
                        "Node %s config.yaml (%s) is newer than Hub override (%s) — adopted",
                        node_id[:8], node_config_ts_str, hub_ts_str,
                    )
                else:
                    logger.debug(
                        "Hub override (%s) is newer than Node config.yaml (%s) — keeping Hub values",
                        hub_ts_str, node_config_ts_str,
                    )
            except Exception as exc:
                logger.debug("Timestamp comparison failed during registration: %s", exc)

        _node_registry[node_id] = {
            "node_id": node_id,
            "name": name,
            "hostname": hostname,
            "online": True,
            "registered_at": existing.get("registered_at", now),
            "last_seen": now,
            "config_overrides": active_overrides,
        }
        _broadcast({"type": "node_connected", "node_id": node_id, "name": name, "hostname": hostname})
        logger.info("Node registered: %s (%s @ %s)", name, node_id[:8], hostname)
        return {"status": "registered", "config": active_overrides}

    @app.post("/v1/nodes/{node_id}/heartbeat")
    async def node_heartbeat(node_id: str):
        if node_id not in _node_registry:
            raise HTTPException(status_code=404, detail="Unknown node — register first")
        now = datetime.datetime.utcnow()
        was_offline = not _node_registry[node_id].get("online")
        _node_registry[node_id]["last_seen"] = now
        _node_registry[node_id]["online"] = True
        if was_offline:
            info = _node_registry[node_id]
            _broadcast({"type": "node_connected", "node_id": node_id,
                        "name": info["name"], "hostname": info["hostname"]})
        return {"status": "ok"}

    @app.get("/v1/nodes")
    async def list_nodes():
        now = datetime.datetime.utcnow()
        result = []
        for info in _node_registry.values():
            age = (now - info["last_seen"]).total_seconds()
            result.append({
                "node_id": info["node_id"],
                "name": info["name"],
                "hostname": info["hostname"],
                "online": info["online"],
                "last_seen_seconds": round(age, 1),
                "config_overrides": info["config_overrides"],
            })
        return {"nodes": result}

    @app.get("/v1/nodes/{node_id}/config")
    async def get_node_config(node_id: str):
        if node_id not in _node_registry:
            raise HTTPException(status_code=404, detail="Unknown node — register first")
        return _node_registry[node_id]["config_overrides"]

    @app.put("/v1/nodes/{node_id}/config")
    async def set_node_config(node_id: str, request: Request):
        if node_id not in _node_registry:
            raise HTTPException(status_code=404, detail="Unknown node — register first")
        data = await request.json()
        # Only accept known tunable keys
        allowed = {
            "wake_word_threshold", "wake_word_vad_threshold",
            "vad_energy_threshold", "vad_silence_seconds",
            "vad_max_record_seconds", "vad_min_speech_seconds",
            "follow_up_seconds", "follow_up_min_words", "post_response_cooldown_seconds",
        }
        overrides = {k: v for k, v in data.items() if k in allowed}
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _node_registry[node_id]["config_overrides"] = overrides
        _persisted_overrides[node_id] = {"updated_at": now_iso, "overrides": overrides}
        _save_node_overrides(_persisted_overrides)
        _broadcast({"type": "node_config_changed", "node_id": node_id, "config": overrides})
        return {"status": "saved", "config": overrides}

    # -----------------------------------------------------------------------
    # Core audio processing endpoint (called by CoreMind Node / Pi)
    # -----------------------------------------------------------------------

    @app.post("/v1/process")
    async def process_audio(
        audio: UploadFile = File(...),
        x_session_id: Optional[str] = Header(default=None),
        x_confirm_gate: Optional[str] = Header(default=None),
    ) -> Response:
        global _turn_count
        s = _get_settings()
        # Voice turns flow into the active session (shared with dashboard chat),
        # so the transcript and LLM memory are unified across voice and typing.
        session_id = _get_active_session()["id"]

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

        # Wake-confirmation gate: on the first post-wake utterance (signalled by the Node via
        # X-Confirm-Gate), the transcript must end with a configured terminator word (e.g.
        # "over") or it's discarded as a false wake — no LLM, no TTS, no spoken reply.
        if x_confirm_gate and s.runtime.wake_confirm_words:
            from coremind.text_match import match_and_strip_terminator

            matched, stripped = match_and_strip_terminator(
                transcript, s.runtime.wake_confirm_words
            )
            if not matched or not stripped.strip():
                logger.info(
                    "Discarding turn as false wake (no confirmation word): %r", transcript
                )
                _broadcast({"type": "status", "text": "Idle"})
                return Response(
                    content=b"",
                    headers={
                        "X-Transcript": urllib.parse.quote(transcript),
                        "X-Response": "",
                        "X-Rejected": "terminator",
                    },
                )
            transcript = stripped

        _broadcast({"type": "transcript", "text": transcript})
        _broadcast({"type": "status", "text": "Sending to LLM..."})

        memory = _get_session(session_id)
        messages = (
            [{"role": "system", "content": _build_system_prompt(s)}]
            + memory.get_messages()
            + [{"role": "user", "content": transcript}]
        )

        try:
            response_text, tool_calls_used = await _run_llm_with_tools(messages)
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
            "tool_calls": tool_calls_used,
            "session_id": session_id,
            "source": "voice",
        }
        _append_round(session_id, turn)

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

    @app.post("/v1/chat")
    async def chat_text(request: Request):
        global _turn_count
        data = await request.json()
        text = data.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")

        # Typing into a viewed session continues it (and brings it to front as the
        # active session); unknown/missing ids fall back to the active session.
        requested = data.get("session_id")
        if requested and requested in _sessions_registry:
            if requested != _active_session_id:
                _set_active(requested)
            session_id = requested
        else:
            session_id = _get_active_session()["id"]
        s = _get_settings()

        _broadcast({"type": "transcript", "text": text})
        _broadcast({"type": "status", "text": "Sending to LLM..."})

        memory = _get_session(session_id)
        messages = (
            [{"role": "system", "content": _build_system_prompt(s)}]
            + memory.get_messages()
            + [{"role": "user", "content": text}]
        )

        llm_error: str | None = None
        response_text = ""
        tool_calls_used: list = []
        try:
            response_text, tool_calls_used = await _run_llm_with_tools(messages)
        except Exception as e:
            logger.error("LLM failed (chat): %s", e)
            llm_error = str(e)
            response_text = f"[Error: {e}]"

        if llm_error is None:
            memory.add("user", text)
            memory.add("assistant", response_text)
            _broadcast({"type": "response", "text": response_text})

        # Always broadcast a turn event so the pending card is cleared regardless
        # of whether the LLM succeeded or failed.
        _turn_count += 1
        turn = {
            "turn": _turn_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "transcript": text,
            "response": response_text,
            "tool_calls": tool_calls_used,
            "session_id": session_id,
            "source": "chat",
        }
        _append_round(session_id, turn)

        _broadcast({"type": "turn", **turn})
        _broadcast({"type": "status", "text": "Idle"})

        if llm_error:
            return JSONResponse(status_code=502, content={"error": llm_error})
        return {"response": response_text}

except ImportError:
    app = None  # type: ignore[assignment]
