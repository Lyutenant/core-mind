from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wav", ".opus"}
_CATALOG_VERSION = 1


# ---------------------------------------------------------------------------
# Scan + persistence
# ---------------------------------------------------------------------------

def scan_library(music_dir: Path) -> dict:
    """Walk music_dir and infer artist/album from folder depth."""
    artists: dict[str, list[str]] = {}
    albums: dict[str, list[str]] = {}
    tracks: list[str] = []

    for path in sorted(music_dir.rglob("*")):
        if path.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue

        track_str = str(path)
        tracks.append(track_str)

        try:
            rel = path.relative_to(music_dir)
        except ValueError:
            continue

        parts = rel.parts  # e.g. ("Miles Davis", "Kind of Blue", "01 - So What.mp3")
        if len(parts) >= 3:
            artist = parts[0]
            album = parts[1]
            artists.setdefault(artist, []).append(track_str)
            albums.setdefault(album, []).append(track_str)
        elif len(parts) == 2:
            artist = parts[0]
            artists.setdefault(artist, []).append(track_str)

    return {
        "version": _CATALOG_VERSION,
        "scanned_at": datetime.now(tz=timezone.utc).isoformat(),
        "music_dir": str(music_dir),
        "artists": artists,
        "albums": albums,
        "tracks": tracks,
        "playlists": {},
    }


def load_catalog(catalog_path: Path) -> dict | None:
    try:
        if catalog_path.exists():
            return json.loads(catalog_path.read_text())
    except Exception as exc:
        logger.warning("Could not load music catalog: %s", exc)
    return None


def save_catalog(catalog: dict, catalog_path: Path) -> None:
    try:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(catalog, indent=2))
    except Exception as exc:
        logger.warning("Could not save music catalog: %s", exc)


def ensure_catalog(music_dir: Path, catalog_path: Path) -> dict:
    """Return existing catalog, scanning and saving if it doesn't exist."""
    existing = load_catalog(catalog_path)
    if existing is not None:
        # Warn if music_dir itself has been modified since last scan
        try:
            dir_mtime = datetime.fromtimestamp(music_dir.stat().st_mtime, tz=timezone.utc)
            scanned_at = datetime.fromisoformat(existing.get("scanned_at", "1970-01-01T00:00:00+00:00"))
            if dir_mtime > scanned_at:
                logger.warning(
                    "Music library may have changed since last scan. "
                    "Run 'coremind music scan' to update the catalog."
                )
        except Exception:
            pass
        return existing

    logger.info("No music catalog found — scanning %s", music_dir)
    data = scan_library(music_dir)
    save_catalog(data, catalog_path)
    logger.info(
        "Catalog built: %d tracks, %d artists, %d albums",
        len(data["tracks"]), len(data["artists"]), len(data["albums"]),
    )
    return data


# ---------------------------------------------------------------------------
# MusicCatalog — query + playlist mutation
# ---------------------------------------------------------------------------

class MusicCatalog:
    def __init__(self, data: dict) -> None:
        self._data = data
        self._music_dir = Path(data.get("music_dir", ""))

    # --- Query ---

    def search(self, query: str) -> list[str]:
        """Return up to 10 track paths matching query against artists, albums, playlists, paths."""
        q = query.lower()
        seen: set[str] = set()
        results: list[str] = []

        def _add(paths: list[str], limit: int = 3) -> None:
            for p in paths:
                if p not in seen and len(results) < 10:
                    seen.add(p)
                    results.append(p)
                if len(results) - (len(results) - len(seen)) >= limit:
                    break

        for artist, paths in self._data.get("artists", {}).items():
            if q in artist.lower():
                _add(sorted(paths))

        for album, paths in self._data.get("albums", {}).items():
            if q in album.lower():
                _add(sorted(paths))

        for name, paths in self._data.get("playlists", {}).items():
            if q in name.lower():
                _add(paths)

        # Full relative path substring fallback
        for track in self._data.get("tracks", []):
            if len(results) >= 10:
                break
            try:
                rel = str(Path(track).relative_to(self._music_dir)).lower()
            except ValueError:
                rel = track.lower()
            if q in rel and track not in seen:
                seen.add(track)
                results.append(track)

        return results

    def get_artist_tracks(self, artist: str) -> list[str]:
        return sorted(self._data.get("artists", {}).get(artist, []))

    def get_album_tracks(self, album: str) -> list[str]:
        return sorted(self._data.get("albums", {}).get(album, []),
                      key=lambda p: Path(p).name)

    def get_playlist_tracks(self, name: str) -> list[str]:
        return list(self._data.get("playlists", {}).get(name, []))

    def list_artists(self) -> list[str]:
        return sorted(self._data.get("artists", {}).keys())

    def list_albums(self, artist: str | None = None) -> list[str]:
        if artist is None:
            return sorted(self._data.get("albums", {}).keys())
        artist_tracks = set(self._data.get("artists", {}).get(artist, []))
        return sorted(
            album for album, paths in self._data.get("albums", {}).items()
            if artist_tracks & set(paths)
        )

    def list_playlists(self) -> list[str]:
        return sorted(self._data.get("playlists", {}).keys())

    def find_artist(self, query: str) -> str | None:
        """Case-insensitive artist name lookup."""
        q = query.lower()
        for artist in self._data.get("artists", {}):
            if artist.lower() == q:
                return artist
        # Partial match fallback
        for artist in self._data.get("artists", {}):
            if q in artist.lower():
                return artist
        return None

    def find_album(self, query: str) -> str | None:
        """Case-insensitive album name lookup."""
        q = query.lower()
        for album in self._data.get("albums", {}):
            if album.lower() == q:
                return album
        for album in self._data.get("albums", {}):
            if q in album.lower():
                return album
        return None

    def find_playlist(self, query: str) -> str | None:
        """Case-insensitive playlist name lookup."""
        q = query.lower()
        for name in self._data.get("playlists", {}):
            if name.lower() == q:
                return name
        for name in self._data.get("playlists", {}):
            if q in name.lower():
                return name
        return None

    # --- Playlist mutation ---

    def create_playlist(self, name: str, paths: list[str], catalog_path: Path) -> None:
        self._data.setdefault("playlists", {})[name] = list(paths)
        save_catalog(self._data, catalog_path)

    def add_to_playlist(self, name: str, paths: list[str], catalog_path: Path) -> None:
        pl = self._data.setdefault("playlists", {}).setdefault(name, [])
        for p in paths:
            if p not in pl:
                pl.append(p)
        save_catalog(self._data, catalog_path)

    def remove_from_playlist(self, name: str, paths: list[str], catalog_path: Path) -> None:
        pl = self._data.get("playlists", {}).get(name, [])
        to_remove = set(paths)
        self._data["playlists"][name] = [p for p in pl if p not in to_remove]
        save_catalog(self._data, catalog_path)
