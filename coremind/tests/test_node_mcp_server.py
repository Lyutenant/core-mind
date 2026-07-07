"""Node MCP server tool registration — atc_enabled gating.

Skipped where the optional mcp package isn't installed (e.g. the dev MacBook);
runs on the Pi/Mac Mini environments that have 'coremind[tools]'.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp.server.fastmcp")

from coremind.node_mcp.server import create_node_mcp_server

_ATC_TOOLS = {"play_atc", "list_atc_airports", "list_atc_channels", "stop_atc"}


def _tool_names(tmp_path, **kwargs) -> set[str]:
    server = create_node_mcp_server(
        music_dir=str(tmp_path / "music"),
        catalog_path=str(tmp_path / "music-catalog.json"),
        atc_catalog_path=str(tmp_path / "atc-catalog.json"),
        **kwargs,
    )
    return {t.name for t in asyncio.run(server.list_tools())}


def test_atc_tools_registered_by_default(tmp_path):
    names = _tool_names(tmp_path)
    assert _ATC_TOOLS <= names
    assert "play_stream" in names


def test_atc_disabled_hides_catalog_tools_but_keeps_play_stream(tmp_path):
    names = _tool_names(tmp_path, atc_enabled=False)
    assert not (_ATC_TOOLS & names)
    assert "play_stream" in names
    assert "stop_playback" in names  # the stop path for streams
