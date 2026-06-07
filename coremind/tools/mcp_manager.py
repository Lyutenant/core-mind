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

        # Wait for all servers to signal ready (or fail)
        await asyncio.gather(*[ev.wait() for ev in ready_events.values()])

        # Fetch tool schemas from every connected session
        for name, session in list(self._sessions.items()):
            try:
                result = await session.list_tools()
                for tool in result.tools:
                    self._schemas.append(self._to_ollama_schema(tool))
                    self._tool_to_server[tool.name] = name
                logger.info("MCP server %r: %d tool(s) registered", name, len(result.tools))
            except Exception as exc:
                logger.warning("Failed to list tools from MCP server %r: %s", name, exc)

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
        try:
            if srv.transport == "stdio":
                if not srv.command:
                    logger.warning("MCP server %r: stdio transport requires 'command'", srv.name)
                    ready.set()
                    return
                params = StdioServerParameters(command=srv.command[0], args=srv.command[1:])
                transport_cm = stdio_client(params)
            elif srv.transport == "http":
                if not srv.url:
                    logger.warning("MCP server %r: http transport requires 'url'", srv.name)
                    ready.set()
                    return
                sse_url = srv.url.rstrip("/") + "/sse"
                transport_cm = sse_client(sse_url)
            else:
                logger.warning("MCP server %r: unknown transport %r", srv.name, srv.transport)
                ready.set()
                return

            async with transport_cm as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._sessions[srv.name] = session
                    logger.info("Connected to MCP server: %s (%s)", srv.name, srv.transport)
                    ready.set()
                    # Hold the connection open until the task is cancelled.
                    await asyncio.get_event_loop().create_future()

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("MCP server %r connection failed: %s", srv.name, exc)
        finally:
            ready.set()  # always unblock start() even on failure
            self._sessions.pop(srv.name, None)

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
