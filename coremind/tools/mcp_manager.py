from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

if TYPE_CHECKING:
    from coremind.config.settings import MCPServerConfig


class MCPManager:
    """Connects to external MCP servers, fetches their tool schemas, and routes tool calls."""

    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}           # server_name → ClientSession
        self._tasks: dict[str, asyncio.Task] = {}
        self._tool_to_server: dict[str, str] = {}        # tool_name → server_name
        self._schemas: list[dict] = []

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

    @property
    def tool_to_server(self) -> dict[str, str]:
        return self._tool_to_server

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return f"No MCP server found for tool: {tool_name}"
        session = self._sessions.get(server_name)
        if session is None:
            return f"MCP server {server_name!r} is not connected"
        try:
            result = await session.call_tool(tool_name, arguments)
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts) if parts else "(empty response)"
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
        elif srv.transport == "http":
            if not srv.url:
                logger.warning("MCP server %r: http transport requires 'url'", srv.name)
                ready.set()
                return
        else:
            logger.warning("MCP server %r: unknown transport %r", srv.name, srv.transport)
            ready.set()
            return

        first_attempt = True
        retry_delay = 10.0

        while True:
            try:
                if srv.transport == "stdio":
                    params = StdioServerParameters(command=srv.command[0], args=srv.command[1:])
                    transport_cm = stdio_client(params)
                else:
                    sse_url = srv.url.rstrip("/") + "/sse"
                    transport_cm = sse_client(sse_url)

                async with transport_cm as (read_stream, write_stream):
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
                        # Hold connection open until cancelled or broken.
                        await asyncio.get_event_loop().create_future()

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

    async def _register_schemas(self, server_name: str, session) -> None:
        """Fetch tool list from a session and update _schemas / _tool_to_server."""
        # Remove stale entries for this server before re-adding.
        self._schemas = [
            s for s in self._schemas
            if self._tool_to_server.get(s["function"]["name"]) != server_name
        ]
        self._tool_to_server = {
            k: v for k, v in self._tool_to_server.items() if v != server_name
        }
        result = await session.list_tools()
        for tool in result.tools:
            self._schemas.append(self._to_ollama_schema(tool))
            self._tool_to_server[tool.name] = server_name
        logger.info("MCP server %r: %d tool(s) registered", server_name, len(result.tools))

    @staticmethod
    def _to_ollama_schema(tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
