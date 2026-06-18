"""Unit tests for MCPManager schema syncing (no real MCP servers).

These drive the async internals via ``asyncio.run`` so they need no pytest-asyncio.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from coremind.tools.mcp_manager import MCPManager


def _tool(name: str):
    """Minimal stand-in for an mcp tool object (name/description/inputSchema)."""
    return SimpleNamespace(name=name, description=f"{name} tool", inputSchema=None)


class FakeSession:
    """Fake ClientSession whose list_tools() returns a scripted sequence of tool sets.

    Each call returns the next list; the final list repeats once exhausted. A list
    value of ``RAISE`` makes that call raise, simulating a dead connection.
    """

    RAISE = object()

    def __init__(self, *tool_lists):
        self._lists = list(tool_lists)
        self.calls = 0

    async def list_tools(self):
        idx = min(self.calls, len(self._lists) - 1)
        self.calls += 1
        value = self._lists[idx]
        if value is FakeSession.RAISE:
            raise RuntimeError("connection lost")
        return SimpleNamespace(tools=[_tool(n) for n in value])


def _names(mgr: MCPManager) -> set[str]:
    return {s["function"]["name"] for s in mgr._schemas}


def test_register_schemas_picks_up_new_tool():
    """A re-sync after a tool is added on the server registers the new tool."""
    async def scenario():
        mgr = MCPManager()
        session = FakeSession(
            ["play_music", "set_volume"],
            ["play_music", "set_volume", "capture_image"],
        )

        await mgr._register_schemas("node", session)
        assert _names(mgr) == {"play_music", "set_volume"}
        assert "capture_image" not in mgr.tool_to_server

        await mgr._register_schemas("node", session)
        assert _names(mgr) == {"play_music", "set_volume", "capture_image"}
        assert mgr.tool_to_server["capture_image"] == "node"

    asyncio.run(scenario())


def test_register_schemas_failure_keeps_existing_tools():
    """A failed list_tools during re-sync must not wipe already-registered tools."""
    async def scenario():
        mgr = MCPManager()
        session = FakeSession(["play_music", "capture_image"], FakeSession.RAISE)

        await mgr._register_schemas("node", session)
        assert _names(mgr) == {"play_music", "capture_image"}

        with pytest.raises(RuntimeError):
            await mgr._register_schemas("node", session)
        # Tools survive the failed poll.
        assert _names(mgr) == {"play_music", "capture_image"}
        assert mgr.tool_to_server["capture_image"] == "node"

    asyncio.run(scenario())


def test_register_schemas_scopes_removal_to_one_server():
    """Re-syncing one server leaves another server's tools untouched."""
    async def scenario():
        mgr = MCPManager()
        await mgr._register_schemas("node", FakeSession(["capture_image"]))
        await mgr._register_schemas("fs", FakeSession(["read_file"]))

        # Node loses its tool; fs must be unaffected.
        await mgr._register_schemas("node", FakeSession([]))
        assert _names(mgr) == {"read_file"}
        assert mgr.tool_to_server["read_file"] == "fs"

    asyncio.run(scenario())


def test_refresh_resyncs_connected_servers():
    """refresh() re-fetches each connected server and reports per-server counts."""
    async def scenario():
        mgr = MCPManager()
        node = FakeSession(["play_music"], ["play_music", "capture_image"])
        fs = FakeSession(["read_file"])
        mgr._sessions = {"node": node, "fs": fs}
        await mgr._register_schemas("node", node)  # initial state: 1 tool
        await mgr._register_schemas("fs", fs)

        result = await mgr.refresh()
        assert result == {"node": 2, "fs": 1}
        assert "capture_image" in mgr.tool_to_server
        assert mgr.tool_to_server["capture_image"] == "node"

    asyncio.run(scenario())


def test_refresh_reports_failed_server_as_none():
    """A server that fails to re-sync is reported as None and keeps its old tools."""
    async def scenario():
        mgr = MCPManager()
        node = FakeSession(["capture_image"], FakeSession.RAISE)
        mgr._sessions = {"node": node}
        await mgr._register_schemas("node", node)

        result = await mgr.refresh()
        assert result == {"node": None}
        assert "capture_image" in mgr.tool_to_server  # old tools preserved

    asyncio.run(scenario())


def test_hold_and_resync_picks_up_new_tool():
    """The periodic loop registers a tool added between polls without a reconnect."""
    async def scenario():
        mgr = MCPManager(resync_interval=0.01)
        session = FakeSession(["play_music"], ["play_music", "capture_image"])
        await mgr._register_schemas("node", session)  # initial connect state

        task = asyncio.create_task(mgr._hold_and_resync("node", session))
        try:
            for _ in range(100):
                await asyncio.sleep(0.01)
                if "capture_image" in mgr.tool_to_server:
                    break
            assert "capture_image" in mgr.tool_to_server
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_hold_and_resync_returns_after_repeated_failures():
    """A persistently failing session causes the loop to return (→ reconnect)."""
    async def scenario():
        mgr = MCPManager(resync_interval=0.01, resync_max_failures=3)
        session = FakeSession(FakeSession.RAISE)

        await asyncio.wait_for(mgr._hold_and_resync("node", session), timeout=2.0)
        assert session.calls >= 3

    asyncio.run(scenario())


def test_hold_and_resync_tolerates_single_failure():
    """One transient failure between good polls does not drop the session."""
    async def scenario():
        mgr = MCPManager(resync_interval=0.01, resync_max_failures=3)
        # good, fail, then good forever — should never hit the failure threshold.
        session = FakeSession(["a"], FakeSession.RAISE, ["a", "b"])

        task = asyncio.create_task(mgr._hold_and_resync("node", session))
        try:
            for _ in range(100):
                await asyncio.sleep(0.01)
                if "b" in mgr.tool_to_server:
                    break
            assert "b" in mgr.tool_to_server
            assert not task.done()  # survived the transient failure
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
