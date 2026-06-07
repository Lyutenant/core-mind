from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_node_mcp_server(music_dir: str = "~/Music"):
    """Build and return a FastMCP server exposing Node-local capabilities."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "mcp package is required for the Node MCP server. "
            "Install with: pip install 'coremind[tools]'"
        ) from exc

    from coremind.node_mcp.tools import music_player, volume_control

    music_player._music_dir = Path(music_dir).expanduser()

    mcp = FastMCP("CoreMind Node")

    @mcp.tool()
    def search_music(query: str) -> str:
        """Search ~/Music for audio files matching query. Returns a list of file paths."""
        return music_player.search_music(query)

    @mcp.tool()
    def play_music(path: str) -> str:
        """Play a local audio file at the given absolute path using mpv."""
        return music_player.play_music(path)

    @mcp.tool()
    def stop_playback() -> str:
        """Stop currently playing music."""
        return music_player.stop_playback()

    @mcp.tool()
    def set_volume(percent: int) -> str:
        """Set system audio volume to a percentage between 0 and 100."""
        return volume_control.set_volume(percent)

    return mcp


async def run_node_mcp_server(music_dir: str = "~/Music", port: int = 8767) -> None:
    """Start the Node MCP SSE server. Blocks until cancelled."""
    mcp = create_node_mcp_server(music_dir)
    logger.info("Node MCP server listening on port %d", port)
    await mcp.run_sse_async(host="0.0.0.0", port=port)
