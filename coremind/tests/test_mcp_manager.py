"""Unit tests for MCPManager schema syncing (no real MCP servers).

These drive the async internals via ``asyncio.run`` so they need no pytest-asyncio.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from coremind.tools.mcp_manager import MCPManager, _extract_tool_result


def _result(*, content=None, structured=None):
    """Minimal stand-in for an mcp CallToolResult (content + structuredContent)."""
    blocks = [SimpleNamespace(text=t) for t in (content or [])]
    return SimpleNamespace(content=blocks, structuredContent=structured)


# FastMCP's primitive-wrapper outputSchema: an object with exactly a `result` property.
_WRAP_SCHEMA = {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]}


def test_extract_unwraps_fastmcp_result_key():
    """FastMCP wraps a ``-> str`` return as {"result": value} in structuredContent.

    The legacy content block carries the JSON-serialized wrapper, so reading content
    alone would hand back '{"result": "..."}'. With the wrapper outputSchema we recover
    the raw string — this is exactly what broke the base64 image from capture_image.
    """
    payload = "/9j/4AAQSkZJRgABAQ" + "A" * 80  # looks like base64 JPEG
    result = _result(content=[f'{{"result": "{payload}"}}'], structured={"result": payload})
    assert _extract_tool_result(result, _WRAP_SCHEMA) == payload


def test_extract_stringifies_non_string_result():
    schema = {"type": "object", "properties": {"result": {"type": "integer"}}}
    result = _result(content=["{\"result\": 42}"], structured={"result": 42})
    assert _extract_tool_result(result, schema) == "42"


def test_extract_falls_back_to_content_text():
    """Plain-text results (no structured output) come through the content blocks."""
    result = _result(content=["hello", "world"], structured=None)
    assert _extract_tool_result(result, None) == "hello\nworld"


def test_extract_preserves_genuine_single_result_field():
    """A tool whose real schema is NOT the FastMCP wrapper keeps its JSON object.

    e.g. a ``-> dict[str, str]`` tool that genuinely returns {"result": "ok"} advertises
    an additionalProperties schema (no `result` property), so we must NOT unwrap it —
    the field name is preserved via the faithful content JSON (Codex P2).
    """
    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    result = _result(content=['{"result": "ok"}'], structured={"result": "ok"})
    assert _extract_tool_result(result, schema) == '{"result": "ok"}'


def test_extract_does_not_unwrap_without_schema():
    """Conservative: structuredContent {"result": ...} with no outputSchema is not unwrapped.

    FastMCP always advertises the wrapper schema alongside structured content, so real
    wrappers (capture_image) are unaffected; an unknown server's payload stays faithful.
    """
    result = _result(content=['{"result": "ok"}'], structured={"result": "ok"})
    assert _extract_tool_result(result, None) == '{"result": "ok"}'


def test_extract_ignores_multi_key_structured_content():
    """A genuine multi-field structured result is not mistaken for the FastMCP wrapper."""
    schema = {"type": "object", "properties": {"a": {}, "b": {}}}
    result = _result(content=['{"a": 1, "b": 2}'], structured={"a": 1, "b": 2})
    assert _extract_tool_result(result, schema) == '{"a": 1, "b": 2}'


def test_extract_empty_response():
    assert _extract_tool_result(_result(content=[], structured=None), None) == "(empty response)"


def _tool(name: str, output_schema=None):
    """Minimal stand-in for an mcp tool object (name/description/in+outputSchema)."""
    return SimpleNamespace(
        name=name, description=f"{name} tool", inputSchema=None, outputSchema=output_schema
    )


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


def test_register_schemas_captures_and_scopes_output_schemas():
    """outputSchema is stored per tool and dropped only with its own server's tools."""
    async def scenario():
        mgr = MCPManager()
        wrap = {"type": "object", "properties": {"result": {"type": "string"}}}

        class _Session:
            def __init__(self, tools):
                self._tools = tools
            async def list_tools(self):
                return SimpleNamespace(tools=self._tools)

        await mgr._register_schemas("node", _Session([_tool("capture_image", wrap)]))
        await mgr._register_schemas("fs", _Session([_tool("read_file")]))
        assert mgr._output_schemas == {"capture_image": wrap}

        # Re-syncing node without the tool drops its schema; fs is untouched.
        await mgr._register_schemas("node", _Session([_tool("play")]))
        assert "capture_image" not in mgr._output_schemas

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


# --- _hold_connection up-front validation (no mcp package / network needed) ----
#
# Each invalid config must set the ready event and return without retrying, so a
# misconfigured server degrades to a log line instead of wedging Hub startup.


def _srv(**kwargs):
    from coremind.config.settings import MCPServerConfig

    return MCPServerConfig(**kwargs)


def _run_hold(srv) -> bool:
    """Run _hold_connection until it returns; True if it set the ready event."""
    async def scenario():
        mgr = MCPManager()
        ready = asyncio.Event()
        await asyncio.wait_for(mgr._hold_connection(srv, ready), timeout=2.0)
        return ready.is_set()

    return asyncio.run(scenario())


def test_hold_connection_rejects_unknown_transport():
    assert _run_hold(_srv(name="x", transport="carrier-pigeon")) is True


def test_hold_connection_streamable_http_requires_url():
    assert _run_hold(_srv(name="ha", transport="streamable-http")) is True


def test_hold_connection_missing_env_var_disables_server():
    """An undefined ${VAR} in headers fails closed at startup (no retry loop)."""
    srv = _srv(
        name="ha",
        transport="http",
        url="http://ha:8123",
        headers={"Authorization": "Bearer ${CM_TEST_UNDEFINED_TOKEN}"},
    )
    assert _run_hold(srv) is True


def test_hold_connection_streamable_http_needs_new_enough_mcp(monkeypatch):
    """With streamablehttp_client unavailable (old mcp), the server is skipped."""
    import coremind.tools.mcp_manager as mm

    monkeypatch.setattr(mm, "streamablehttp_client", None)
    srv = _srv(name="ha", transport="streamable-http", url="http://ha:8123/api/mcp")
    assert _run_hold(srv) is True
