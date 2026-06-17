from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from coremind.node_mcp import playback

if TYPE_CHECKING:
    from coremind.node_mcp.catalog import MusicCatalog

_music_dir: Path = Path.home() / "Music"
_catalog: MusicCatalog | None = None
_catalog_path: Path = Path.home() / ".coremind" / "music-catalog.json"


def init_catalog(music_dir: Path, catalog_path: Path) -> None:
    global _catalog, _catalog_path
    from coremind.node_mcp.catalog import MusicCatalog, ensure_catalog
    _catalog_path = catalog_path
    _catalog = MusicCatalog(ensure_catalog(music_dir, catalog_path))


# ---------------------------------------------------------------------------
# Playback primitives
# ---------------------------------------------------------------------------

def play_track(path: str) -> str:
    """Stop any current playback and play a single file via mpv."""
    try:
        playback.start(["mpv", "--no-terminal", "--quiet", path])
    except FileNotFoundError:
        return "mpv is not installed. Run: sudo apt install mpv"
    return f"Now playing: {Path(path).name}"


def play_queue(paths: list[str], shuffle: bool = False) -> str:
    """Write a temp M3U and start mpv with --playlist for multi-track playback."""
    if not paths:
        return "No tracks to play."
    with tempfile.NamedTemporaryFile(
        suffix=".m3u", delete=False, mode="w", prefix="coremind_"
    ) as f:
        f.write("\n".join(paths))
        m3u = f.name
    cmd = ["mpv", "--no-terminal", "--quiet", f"--playlist={m3u}"]
    if shuffle:
        cmd.append("--shuffle")
    try:
        playback.start(cmd)
    except FileNotFoundError:
        return "mpv is not installed. Run: sudo apt install mpv"
    label = f"{len(paths)} track(s)"
    return f"Playing {label}{' (shuffled)' if shuffle else ''}."


def stop_playback() -> str:
    """Stop currently playing audio."""
    return playback.stop_current()


def pause_mpv() -> None:
    """Free the speaker/mic for a voice turn by stopping playback."""
    playback.pause()


def resume_mpv() -> None:
    """Relaunch the playback that was stopped for the voice turn."""
    playback.resume()


# ---------------------------------------------------------------------------
# Catalog-aware high-level playback
# ---------------------------------------------------------------------------

def play_artist(artist: str, shuffle: bool = False) -> str:
    if _catalog is None:
        return "Music catalog not loaded. Run 'coremind music scan' first."
    name = _catalog.find_artist(artist)
    if name is None:
        return f"Artist '{artist}' not found in catalog."
    tracks = _catalog.get_artist_tracks(name)
    if not tracks:
        return f"No tracks found for artist '{name}'."
    return play_queue(tracks, shuffle=shuffle)


def play_album(album: str, shuffle: bool = False) -> str:
    if _catalog is None:
        return "Music catalog not loaded. Run 'coremind music scan' first."
    name = _catalog.find_album(album)
    if name is None:
        return f"Album '{album}' not found in catalog."
    tracks = _catalog.get_album_tracks(name)
    if not tracks:
        return f"No tracks found for album '{name}'."
    return play_queue(tracks, shuffle=shuffle)


def play_playlist_by_name(name: str, shuffle: bool = False) -> str:
    if _catalog is None:
        return "Music catalog not loaded. Run 'coremind music scan' first."
    found = _catalog.find_playlist(name)
    if found is None:
        return f"Playlist '{name}' not found."
    tracks = _catalog.get_playlist_tracks(found)
    if not tracks:
        return f"Playlist '{found}' is empty."
    return play_queue(tracks, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Catalog search + listing
# ---------------------------------------------------------------------------

def search_catalog(query: str) -> str:
    if _catalog is None:
        return "Music catalog not loaded. Run 'coremind music scan' first."
    results = _catalog.search(query)
    if not results:
        return f"No music matching '{query}' found in catalog."
    lines = [Path(p).name for p in results]
    return "\n".join(lines)


def list_artists() -> str:
    if _catalog is None:
        return "Music catalog not loaded. Run 'coremind music scan' first."
    artists = _catalog.list_artists()
    if not artists:
        return "No artists found in catalog."
    return ", ".join(artists)


def list_albums(artist: str | None = None) -> str:
    if _catalog is None:
        return "Music catalog not loaded. Run 'coremind music scan' first."
    if artist:
        name = _catalog.find_artist(artist)
        albums = _catalog.list_albums(artist=name) if name else []
        label = f"albums by {artist}"
    else:
        albums = _catalog.list_albums()
        label = "albums"
    if not albums:
        return f"No {label} found in catalog."
    return ", ".join(albums)


def list_playlists() -> str:
    if _catalog is None:
        return "Music catalog not loaded. Run 'coremind music scan' first."
    playlists = _catalog.list_playlists()
    if not playlists:
        return "No playlists saved yet."
    return ", ".join(playlists)


# ---------------------------------------------------------------------------
# Playlist CRUD
# ---------------------------------------------------------------------------

def create_playlist(name: str, tracks: list[str]) -> str:
    if _catalog is None:
        return "Music catalog not loaded."
    _catalog.create_playlist(name, tracks, _catalog_path)
    return f"Created playlist '{name}' with {len(tracks)} track(s)."


def add_to_playlist(playlist: str, tracks: list[str]) -> str:
    if _catalog is None:
        return "Music catalog not loaded."
    found = _catalog.find_playlist(playlist)
    name = found if found else playlist
    _catalog.add_to_playlist(name, tracks, _catalog_path)
    return f"Added {len(tracks)} track(s) to playlist '{name}'."


def remove_from_playlist(playlist: str, tracks: list[str]) -> str:
    if _catalog is None:
        return "Music catalog not loaded."
    found = _catalog.find_playlist(playlist)
    if found is None:
        return f"Playlist '{playlist}' not found."
    _catalog.remove_from_playlist(found, tracks, _catalog_path)
    return f"Removed {len(tracks)} track(s) from playlist '{found}'."
