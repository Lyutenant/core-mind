from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_node_mcp_server(
    music_dir: str = "~/Music",
    catalog_path: str = "~/.coremind/music-catalog.json",
    atc_catalog_path: str = "~/.coremind/atc-catalog.json",
    host: str = "127.0.0.1",
    port: int = 8767,
):
    """Build and return a FastMCP server exposing Node-local capabilities."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "mcp package is required for the Node MCP server. "
            "Install with: pip install 'coremind[tools]'"
        ) from exc

    from coremind.node_mcp.tools import atc_player, music_player, volume_control

    _music_dir = Path(music_dir).expanduser()
    _catalog_path = Path(catalog_path).expanduser()
    _atc_catalog_path = Path(atc_catalog_path).expanduser()

    music_player._music_dir = _music_dir
    music_player.init_catalog(_music_dir, _catalog_path)
    atc_player.init_atc_catalog(_atc_catalog_path)

    mcp = FastMCP("CoreMind Node", host=host, port=port)

    # --- Search ---

    @mcp.tool()
    def search_music(query: str) -> str:
        """Search the music catalog by artist, album, playlist name, or file path fragment."""
        return music_player.search_catalog(query)

    # --- Single-track playback ---

    @mcp.tool()
    def play_track(path: str) -> str:
        """Play a single audio file at the given absolute path."""
        return music_player.play_track(path)

    # --- Multi-track playback ---

    @mcp.tool()
    def play_artist(artist: str, shuffle: bool = False) -> str:
        """Play all tracks by an artist. Set shuffle=true to randomize order."""
        return music_player.play_artist(artist, shuffle=shuffle)

    @mcp.tool()
    def play_album(album: str, shuffle: bool = False) -> str:
        """Play an album in track order. Set shuffle=true to randomize."""
        return music_player.play_album(album, shuffle=shuffle)

    @mcp.tool()
    def play_playlist(name: str, shuffle: bool = False) -> str:
        """Play a saved playlist by name. Set shuffle=true to randomize order."""
        return music_player.play_playlist_by_name(name, shuffle=shuffle)

    # --- Stop ---

    @mcp.tool()
    def stop_playback() -> str:
        """Stop currently playing music."""
        return music_player.stop_playback()

    # --- Volume ---

    @mcp.tool()
    def set_volume(percent: int) -> str:
        """Set system audio volume (0–100)."""
        return volume_control.set_volume(percent)

    # --- Listing ---

    @mcp.tool()
    def list_artists() -> str:
        """List all artists known in the music catalog."""
        return music_player.list_artists()

    @mcp.tool()
    def list_albums(artist: str = "") -> str:
        """List all albums in the catalog, or albums by a specific artist if artist is provided."""
        return music_player.list_albums(artist=artist or None)

    @mcp.tool()
    def list_playlists() -> str:
        """List all saved playlist names."""
        return music_player.list_playlists()

    # --- Playlist management ---

    @mcp.tool()
    def create_playlist(name: str, tracks: list[str]) -> str:
        """Create a new playlist with the given name and list of absolute track paths."""
        return music_player.create_playlist(name, tracks)

    @mcp.tool()
    def add_to_playlist(playlist: str, tracks: list[str]) -> str:
        """Add one or more track paths to an existing playlist."""
        return music_player.add_to_playlist(playlist, tracks)

    @mcp.tool()
    def remove_from_playlist(playlist: str, tracks: list[str]) -> str:
        """Remove one or more track paths from a playlist."""
        return music_player.remove_from_playlist(playlist, tracks)

    # --- ATC tools ---

    @mcp.tool()
    def play_atc(query: str) -> str:
        """Stream a live ATC feed. Say the airport and channel type, e.g. 'KEWR tower' or 'Newark approach'."""
        return atc_player.play_atc(query)

    @mcp.tool()
    def list_atc_airports() -> str:
        """List all airports with ATC streams available in the catalog."""
        return atc_player.list_atc_airports()

    @mcp.tool()
    def list_atc_channels(airport: str) -> str:
        """List all ATC channels for a specific airport (e.g. 'KEWR' or 'Newark')."""
        return atc_player.list_atc_channels(airport)

    @mcp.tool()
    def stop_atc() -> str:
        """Stop the currently streaming ATC audio."""
        return atc_player.stop_atc()

    return mcp


async def run_node_mcp_server(
    music_dir: str = "~/Music",
    catalog_path: str = "~/.coremind/music-catalog.json",
    atc_catalog_path: str = "~/.coremind/atc-catalog.json",
    port: int = 8767,
    host: str = "127.0.0.1",
) -> None:
    """Start the Node MCP SSE server. Blocks until cancelled."""
    mcp = create_node_mcp_server(music_dir, catalog_path, atc_catalog_path, host=host, port=port)
    logger.info("Node MCP server listening on port %d", port)
    await mcp.run_sse_async()
