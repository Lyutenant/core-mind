from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from coremind import ConfigError

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import get_default_environment, stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
    try:
        # Separate guard: streamable-http shipped later (mcp>=1.8) than the rest,
        # so an older install keeps stdio/SSE working and only that transport
        # degrades with an actionable message.
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        streamablehttp_client = None
except ImportError:
    MCP_AVAILABLE = False
    streamablehttp_client = None

if TYPE_CHECKING:
    from coremind.config.settings import MCPServerConfig

# Prepended to the text of an MCP tool result that carried the `isError` flag.
TOOL_ERROR_PREFIX = "TOOL ERROR (the action was not performed): "

# Longest server-provided prompt (instructions) we'll inject into the system prompt.
# Home Assistant's Assist prompt with a few dozen exposed entities is ~1-3 KB; the
# cap keeps a chatty server from swamping a small model's context.
MAX_INSTRUCTIONS_CHARS = 6000


def _is_fastmcp_result_wrapper(output_schema) -> bool:
    """True if an MCP tool's outputSchema is FastMCP's primitive ``result`` wrapper.

    FastMCP wraps non-object returns (e.g. a tool typed ``-> str``) in a generated model
    with a single ``result`` field, whose advertised outputSchema is an object whose
    ``properties`` are exactly ``{"result"}``. A tool that *genuinely* returns a dict
    (``-> dict[str, str]``) advertises a different schema (``additionalProperties`` / no
    ``result`` property), so this lets us unwrap only the real wrapper and never strip a
    field name off a legitimate single-key structured result.

    Known, accepted limitation: a tool that genuinely returns a typed object/Pydantic
    model whose *only* field is literally named ``result`` advertises a schema identical
    to the wrapper here (``properties == {"result"}``, same ``structuredContent``), so it
    will be unwrapped too. On the wire these differ only by ``title`` (the wrapper is
    ``{func_name}Output`` vs the model's class name); we deliberately do **not** gate on
    ``title`` because doing so would couple ``capture_image``'s correctness to a
    FastMCP-internal naming detail and misfire on custom-named tools. The mis-unwrap is
    contrived (needs an *external* FastMCP server — our Node tools are all ``-> str``,
    i.e. the real wrapper) and only cosmetic (the consumer is an LLM that gets the inner
    value instead of a one-key JSON object), so we accept it.
    """
    if not isinstance(output_schema, dict):
        return False
    props = output_schema.get("properties")
    return isinstance(props, dict) and set(props) == {"result"}


def _extract_tool_result(result, output_schema=None) -> str:
    """Reduce an MCP ``CallToolResult`` to a single string.

    Modern MCP servers (FastMCP on mcp>=1.10) derive an output schema from a tool's
    return annotation and return the value as *structured output*. For a non-object
    return (e.g. a tool typed ``-> str``) FastMCP wraps the value under a single
    ``"result"`` key, and — for backward compat — the legacy ``content`` block carries
    the JSON-serialized form (``{"result": "..."}``), not the raw value. Reading
    ``content`` alone therefore hands callers a JSON wrapper instead of the string the
    tool returned (which breaks e.g. the base64 image from ``capture_image``).

    So: when the tool advertises FastMCP's ``result`` wrapper schema *and*
    ``structuredContent`` is exactly ``{"result": <value>}``, return that value.
    Otherwise fall back to joining the ``content`` text blocks — which preserves the
    faithful JSON for genuine structured results and covers plain-text servers.
    """
    structured = _attr(result, "structuredContent", "structured_content")
    if (
        _is_fastmcp_result_wrapper(output_schema)
        and isinstance(structured, dict)
        and set(structured) == {"result"}
    ):
        value = structured["result"]
        text = value if isinstance(value, str) else str(value)
    else:
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(str(content))
        text = "\n".join(parts) if parts else "(empty response)"

    # An MCP error result (`isError`) means the server did NOT perform the action —
    # e.g. Home Assistant's "Error calling tool: MatchFailedError" or the SDK's
    # "Input validation error: 'light' is not one of [...]". The text alone reads
    # like any other result to a small model, which then happily reports success.
    # Make the failure explicit so the LLM (and the dashboard) can't miss it.
    if _attr(result, "isError", "is_error"):
        return f"{TOOL_ERROR_PREFIX}{text}"
    return text


def _attr(obj, *names, default=None):
    """First present attribute among ``names`` (mcp 1.x camelCase vs 2.x snake_case)."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _prompt_text(get_prompt_result) -> str:
    """Join the text content of an MCP ``GetPromptResult``'s messages."""
    parts: list[str] = []
    for msg in getattr(get_prompt_result, "messages", None) or []:
        content = getattr(msg, "content", None)
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts)


class MCPManager:
    """Connects to external MCP servers, fetches their tool schemas, and routes tool calls."""

    def __init__(
        self,
        *,
        resync_interval: float = 45.0,
        resync_max_failures: int = 3,
    ) -> None:
        self._sessions: dict[str, object] = {}           # server_name → ClientSession
        self._tasks: dict[str, asyncio.Task] = {}
        self._tool_to_server: dict[str, str] = {}        # tool_name → server_name
        self._schemas: list[dict] = []
        self._output_schemas: dict[str, dict] = {}       # tool_name → outputSchema (for unwrap)
        # server_name → text of the server's argument-free MCP prompt(s). Servers use
        # prompts to tell the LLM *how* to use their tools (Home Assistant: "when
        # controlling an area, pass area + domain" plus the exposed-entity list);
        # the Hub appends this to the system prompt. Refreshed with every re-sync.
        self._instructions: dict[str, str] = {}
        self._use_prompts: dict[str, bool] = {}          # server_name → MCPServerConfig.use_prompts
        # While a connection is held open, re-fetch its tool list every
        # `resync_interval` seconds so capabilities added on the server (e.g. a
        # newly enabled camera tool) are picked up without a reconnect or a Hub
        # restart. Doubles as a liveness probe: after `resync_max_failures`
        # consecutive failed re-syncs we drop the (likely dead) session and let
        # the reconnect loop take over.
        self._resync_interval = resync_interval
        self._resync_max_failures = resync_max_failures

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, servers: list[MCPServerConfig]) -> None:
        if not MCP_AVAILABLE:
            logger.warning(
                "mcp package is not installed — MCP servers will not connect. "
                "Install with: pip install 'coremind[tools]'"
            )
            return

        ready_events: dict[str, asyncio.Event] = {}
        for srv in servers:
            ev = asyncio.Event()
            ready_events[srv.name] = ev
            task = asyncio.create_task(
                self._hold_connection(srv, ev), name=f"mcp-{srv.name}"
            )
            self._tasks[srv.name] = task

        # Wait for all servers to signal ready (or fail on first attempt).
        # Schema registration happens inside _hold_connection once connected.
        await asyncio.gather(*[ev.wait() for ev in ready_events.values()])

    async def get_tool_definitions(self) -> list[dict]:
        return list(self._schemas)

    async def refresh(self) -> dict[str, int | None]:
        """Force an immediate tool-list re-sync of every connected server.

        Mirrors the periodic re-sync but on demand (e.g. a dashboard button), so a
        capability just enabled on a Node is picked up without waiting for the next
        poll. Returns {server_name: tool_count} per server; a value of ``None`` means
        that server's re-sync failed (its previously-registered tools are kept).
        """
        results: dict[str, int | None] = {}
        for name, session in list(self._sessions.items()):
            try:
                await self._register_schemas(name, session)
                results[name] = sum(1 for v in self._tool_to_server.values() if v == name)
            except Exception as exc:  # noqa: BLE001 — report per-server, don't abort the rest
                logger.warning("MCP server %r: manual refresh failed (%s)", name, exc)
                results[name] = None
        return results

    @property
    def tool_to_server(self) -> dict[str, str]:
        return self._tool_to_server

    @property
    def instructions(self) -> dict[str, str]:
        """server_name → prompt text the server published for using its tools."""
        return dict(self._instructions)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return f"No MCP server found for tool: {tool_name}"
        session = self._sessions.get(server_name)
        if session is None:
            return f"MCP server {server_name!r} is not connected"
        try:
            result = await session.call_tool(tool_name, arguments)
            return _extract_tool_result(result, self._output_schemas.get(tool_name))
        except Exception as exc:
            logger.error("MCP tool call %r on %r failed: %s", tool_name, server_name, exc)
            return f"Tool '{tool_name}' failed: {exc}"

    async def stop(self) -> None:
        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        self._sessions.clear()
        logger.debug("MCPManager stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _hold_connection(self, srv: MCPServerConfig, ready: asyncio.Event) -> None:
        """Connect to one MCP server and hold the session open.

        Retries with backoff if the initial connection fails (e.g. Hub starts before
        the Node MCP server is up) or if the connection drops later.  Tool schemas are
        re-registered on every successful (re)connect.
        """
        # Validate config once up front — no point retrying a misconfigured server.
        if srv.transport == "stdio":
            if not srv.command:
                logger.warning("MCP server %r: stdio transport requires 'command'", srv.name)
                ready.set()
                return
        elif srv.transport in ("http", "streamable-http"):
            if not srv.url:
                logger.warning(
                    "MCP server %r: %s transport requires 'url'", srv.name, srv.transport
                )
                ready.set()
                return
            if srv.transport == "streamable-http" and streamablehttp_client is None:
                logger.warning(
                    "MCP server %r: streamable-http transport needs mcp>=1.8 — "
                    "upgrade with: pip install -U 'coremind[tools]'",
                    srv.name,
                )
                ready.set()
                return
        else:
            logger.warning("MCP server %r: unknown transport %r", srv.name, srv.transport)
            ready.set()
            return

        # Resolve ${VAR} secrets once up front too: the process environment is fixed
        # for the Hub's lifetime, so a missing variable can never heal by retrying.
        try:
            headers = srv.resolved_headers()
            env = srv.resolved_env()
        except ConfigError as exc:
            logger.warning("MCP server %r: %s — server disabled", srv.name, exc)
            ready.set()
            return

        self._use_prompts[srv.name] = getattr(srv, "use_prompts", True)
        first_attempt = True
        retry_delay = 10.0

        while True:
            try:
                if srv.transport == "stdio":
                    spawn_env = {**get_default_environment(), **env} if env else None
                    params = StdioServerParameters(
                        command=srv.command[0], args=srv.command[1:], env=spawn_env
                    )
                    transport_cm = stdio_client(params)
                elif srv.transport == "streamable-http":
                    # URL is used as-is (e.g. Home Assistant's /api/mcp) — no
                    # path suffix convention like SSE's /sse.
                    transport_cm = streamablehttp_client(srv.url, headers=headers)
                else:
                    sse_url = srv.url.rstrip("/") + "/sse"
                    transport_cm = sse_client(sse_url, headers=headers)

                # stdio/SSE yield (read, write); streamable-http yields
                # (read, write, get_session_id) — index instead of unpacking.
                async with transport_cm as streams:
                    read_stream, write_stream = streams[0], streams[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        self._sessions[srv.name] = session
                        await self._register_schemas(srv.name, session)
                        if first_attempt:
                            logger.info("Connected to MCP server: %s (%s)", srv.name, srv.transport)
                            first_attempt = False
                            ready.set()
                        else:
                            logger.info("Reconnected to MCP server: %s", srv.name)
                        # Healthy (re)connect — reset the backoff for the next drop.
                        retry_delay = 10.0
                        # Hold the connection open, periodically re-syncing the tool
                        # list. Returns only when the session looks dead, dropping us
                        # into the reconnect path below.
                        await self._hold_and_resync(srv.name, session)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                if first_attempt:
                    logger.warning(
                        "MCP server %r: initial connection failed (%s) — retrying every %.0fs",
                        srv.name, exc, retry_delay,
                    )
                    first_attempt = False
                    ready.set()  # unblock start() — Hub proceeds without this server for now
                else:
                    logger.debug("MCP server %r: connection lost (%s) — retrying", srv.name, exc)
            finally:
                self._sessions.pop(srv.name, None)

            # Exponential backoff, capped at 60 s
            try:
                await asyncio.sleep(retry_delay)
            except asyncio.CancelledError:
                return
            retry_delay = min(retry_delay * 1.5, 60.0)

    async def _hold_and_resync(self, server_name: str, session) -> None:
        """Hold a connected session open, periodically re-fetching its tool list.

        Returns (rather than blocking forever) once `_resync_max_failures` re-syncs
        fail in a row, signalling the caller to tear down and reconnect. A single
        transient failure is tolerated so a healthy session isn't dropped on a blip.
        """
        consecutive_failures = 0
        while True:
            await asyncio.sleep(self._resync_interval)
            try:
                await self._register_schemas(server_name, session)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a poll failure must not crash the task
                consecutive_failures += 1
                logger.debug(
                    "MCP server %r: tool re-sync failed (%s) [%d/%d]",
                    server_name, exc, consecutive_failures, self._resync_max_failures,
                )
                if consecutive_failures >= self._resync_max_failures:
                    logger.info(
                        "MCP server %r: %d consecutive re-syncs failed — reconnecting",
                        server_name, consecutive_failures,
                    )
                    return

    async def _register_schemas(self, server_name: str, session) -> None:
        """Fetch tool list from a session and update _schemas / _tool_to_server.

        The list is fetched *before* any existing entries are dropped, so a failed
        ``list_tools`` (e.g. a dead connection during a periodic re-sync) leaves the
        previously-registered tools intact rather than wiping them. Logs at INFO only
        when the set of tool names changes, so the periodic re-sync stays quiet.
        """
        previous = {
            name for name, owner in self._tool_to_server.items() if owner == server_name
        }
        result = await session.list_tools()
        # Replace this server's entries with the freshly fetched list.
        self._schemas = [
            s for s in self._schemas
            if self._tool_to_server.get(s["function"]["name"]) != server_name
        ]
        self._tool_to_server = {
            k: v for k, v in self._tool_to_server.items() if v != server_name
        }
        # Drop this server's output schemas (its tools were just removed above); other
        # servers' entries still have a tool_to_server owner and are kept.
        self._output_schemas = {
            name: schema
            for name, schema in self._output_schemas.items()
            if name in self._tool_to_server
        }
        for tool in result.tools:
            self._schemas.append(self._to_ollama_schema(tool))
            self._tool_to_server[tool.name] = server_name
            output_schema = _attr(tool, "outputSchema", "output_schema")
            if output_schema:
                self._output_schemas[tool.name] = output_schema
        current = {tool.name for tool in result.tools}
        if current != previous:
            logger.info(
                "MCP server %r: %d tool(s) registered (was %d)",
                server_name, len(current), len(previous),
            )
        else:
            logger.debug(
                "MCP server %r: re-synced, %d tool(s) unchanged", server_name, len(current)
            )
        # Instructions are best-effort: a server without prompt support (the Node)
        # or a failed fetch must never affect the tool registration above.
        if self._use_prompts.get(server_name, True):
            try:
                await self._register_instructions(server_name, session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("MCP server %r: prompt fetch failed (%s)", server_name, exc)

    async def _register_instructions(self, server_name: str, session) -> None:
        """Fetch the server's argument-free prompts and store their text.

        MCP prompts are how a server tells the client's LLM how to use its tools.
        Only prompts that need no arguments are usable unattended; those are fetched
        and concatenated (most servers publish exactly one). An empty result clears
        any previous text; a failure (caught by the caller) keeps it.
        """
        listed = await session.list_prompts()
        texts: list[str] = []
        for prompt in getattr(listed, "prompts", None) or []:
            if any(getattr(arg, "required", False) for arg in (prompt.arguments or [])):
                continue
            fetched = await session.get_prompt(prompt.name)
            text = _prompt_text(fetched)
            if text:
                texts.append(text)
        combined = "\n\n".join(texts)
        if len(combined) > MAX_INSTRUCTIONS_CHARS:
            logger.warning(
                "MCP server %r: prompt text is %d chars — truncated to %d",
                server_name, len(combined), MAX_INSTRUCTIONS_CHARS,
            )
            combined = combined[:MAX_INSTRUCTIONS_CHARS]
        previous = self._instructions.get(server_name, "")
        if combined:
            self._instructions[server_name] = combined
        else:
            self._instructions.pop(server_name, None)
        if combined != previous:
            logger.info(
                "MCP server %r: %d chars of tool instructions %s",
                server_name, len(combined), "registered" if combined else "cleared",
            )

    @staticmethod
    def _to_ollama_schema(tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": _attr(tool, "inputSchema", "input_schema")
                or {"type": "object", "properties": {}},
            },
        }
