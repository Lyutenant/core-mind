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
        for tool in result.tools:
            self._schemas.append(self._to_ollama_schema(tool))
            self._tool_to_server[tool.name] = server_name
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
