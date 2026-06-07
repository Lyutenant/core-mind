"""Unit tests for MusicCatalog — no audio hardware required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coremind.node_mcp.catalog import (
    MusicCatalog,
    load_catalog,
    save_catalog,
    scan_library,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_library(tmp_path: Path) -> Path:
    """
    Create a minimal fake music library:
      tmp_path/Music/
        Miles Davis/
          Kind of Blue/
            01 - So What.mp3
            02 - Freddie Freeloader.mp3
        John Coltrane/
          A Love Supreme/
            01 - Acknowledgement.mp3
        loose_track.mp3
    """
    music = tmp_path / "Music"
    (music / "Miles Davis" / "Kind of Blue").mkdir(parents=True)
    (music / "John Coltrane" / "A Love Supreme").mkdir(parents=True)

    tracks = [
        music / "Miles Davis" / "Kind of Blue" / "01 - So What.mp3",
        music / "Miles Davis" / "Kind of Blue" / "02 - Freddie Freeloader.mp3",
        music / "John Coltrane" / "A Love Supreme" / "01 - Acknowledgement.mp3",
        music / "loose_track.mp3",
    ]
    for t in tracks:
        t.touch()

    return music


# ---------------------------------------------------------------------------
# scan_library
# ---------------------------------------------------------------------------

class TestScanLibrary:
    def test_track_count(self, tmp_path):
        music = _make_library(tmp_path)
        data = scan_library(music)
        assert len(data["tracks"]) == 4

    def test_artist_inference(self, tmp_path):
        music = _make_library(tmp_path)
        data = scan_library(music)
        assert "Miles Davis" in data["artists"]
        assert "John Coltrane" in data["artists"]
        assert len(data["artists"]["Miles Davis"]) == 2

    def test_album_inference(self, tmp_path):
        music = _make_library(tmp_path)
        data = scan_library(music)
        assert "Kind of Blue" in data["albums"]
        assert "A Love Supreme" in data["albums"]

    def test_loose_track_not_in_artists(self, tmp_path):
        music = _make_library(tmp_path)
        data = scan_library(music)
        # loose_track.mp3 is at depth-1 (one part), so it's in tracks but not in artists
        artist_track_names = {
            Path(p).name
            for paths in data["artists"].values()
            for p in paths
        }
        assert "loose_track.mp3" not in artist_track_names

    def test_version_field(self, tmp_path):
        music = _make_library(tmp_path)
        data = scan_library(music)
        assert data["version"] == 1

    def test_empty_playlists(self, tmp_path):
        music = _make_library(tmp_path)
        data = scan_library(music)
        assert data["playlists"] == {}


# ---------------------------------------------------------------------------
# save/load round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        music = _make_library(tmp_path)
        data = scan_library(music)
        catalog_path = tmp_path / "catalog.json"
        save_catalog(data, catalog_path)
        loaded = load_catalog(catalog_path)
        assert loaded is not None
        assert loaded["tracks"] == data["tracks"]
        assert loaded["artists"] == data["artists"]

    def test_load_missing_returns_none(self, tmp_path):
        assert load_catalog(tmp_path / "nonexistent.json") is None


# ---------------------------------------------------------------------------
# MusicCatalog — query methods
# ---------------------------------------------------------------------------

@pytest.fixture()
def catalog(tmp_path) -> MusicCatalog:
    music = _make_library(tmp_path)
    data = scan_library(music)
    return MusicCatalog(data)


class TestMusicCatalog:
    def test_list_artists(self, catalog):
        artists = catalog.list_artists()
        assert "Miles Davis" in artists
        assert "John Coltrane" in artists
        assert artists == sorted(artists)

    def test_list_albums(self, catalog):
        albums = catalog.list_albums()
        assert "Kind of Blue" in albums
        assert "A Love Supreme" in albums

    def test_list_albums_by_artist(self, catalog):
        albums = catalog.list_albums(artist="Miles Davis")
        assert "Kind of Blue" in albums
        assert "A Love Supreme" not in albums

    def test_get_artist_tracks_sorted(self, catalog):
        tracks = catalog.get_artist_tracks("Miles Davis")
        assert len(tracks) == 2
        assert tracks == sorted(tracks)

    def test_get_album_tracks(self, catalog):
        tracks = catalog.get_album_tracks("Kind of Blue")
        assert len(tracks) == 2
        names = [Path(t).name for t in tracks]
        assert "01 - So What.mp3" in names

    def test_find_artist_exact(self, catalog):
        assert catalog.find_artist("Miles Davis") == "Miles Davis"

    def test_find_artist_case_insensitive(self, catalog):
        assert catalog.find_artist("miles davis") == "Miles Davis"

    def test_find_artist_partial(self, catalog):
        assert catalog.find_artist("Coltrane") == "John Coltrane"

    def test_find_artist_missing(self, catalog):
        assert catalog.find_artist("Beethoven") is None

    def test_find_album_partial(self, catalog):
        assert catalog.find_album("kind of") == "Kind of Blue"

    def test_search_by_artist_name(self, catalog):
        results = catalog.search("Miles Davis")
        assert len(results) > 0
        assert all("Miles Davis" in r for r in results)

    def test_search_by_album_name(self, catalog):
        results = catalog.search("Kind of Blue")
        assert len(results) > 0

    def test_search_no_results(self, catalog):
        results = catalog.search("zzz_nonexistent_zzz")
        assert results == []

    def test_list_playlists_empty(self, catalog):
        assert catalog.list_playlists() == []


# ---------------------------------------------------------------------------
# MusicCatalog — playlist CRUD
# ---------------------------------------------------------------------------

class TestPlaylistCRUD:
    def test_create_playlist(self, tmp_path, catalog):
        catalog_path = tmp_path / "catalog.json"
        tracks = catalog.get_artist_tracks("Miles Davis")
        catalog.create_playlist("morning jazz", tracks, catalog_path)

        assert "morning jazz" in catalog.list_playlists()
        assert catalog_path.exists()

    def test_create_playlist_persists(self, tmp_path, catalog):
        catalog_path = tmp_path / "catalog.json"
        tracks = catalog.get_artist_tracks("Miles Davis")
        catalog.create_playlist("morning jazz", tracks, catalog_path)

        reloaded_data = load_catalog(catalog_path)
        assert reloaded_data is not None
        assert "morning jazz" in reloaded_data["playlists"]

    def test_add_to_playlist(self, tmp_path, catalog):
        catalog_path = tmp_path / "catalog.json"
        t1 = catalog.get_artist_tracks("Miles Davis")[:1]
        t2 = catalog.get_artist_tracks("John Coltrane")

        catalog.create_playlist("mix", t1, catalog_path)
        catalog.add_to_playlist("mix", t2, catalog_path)

        result = catalog.get_playlist_tracks("mix")
        assert len(result) == len(t1) + len(t2)

    def test_add_deduplicates(self, tmp_path, catalog):
        catalog_path = tmp_path / "catalog.json"
        tracks = catalog.get_artist_tracks("Miles Davis")
        catalog.create_playlist("dupe", tracks, catalog_path)
        catalog.add_to_playlist("dupe", tracks, catalog_path)

        result = catalog.get_playlist_tracks("dupe")
        assert len(result) == len(tracks)

    def test_remove_from_playlist(self, tmp_path, catalog):
        catalog_path = tmp_path / "catalog.json"
        tracks = catalog.get_artist_tracks("Miles Davis")
        catalog.create_playlist("shrink", tracks, catalog_path)
        catalog.remove_from_playlist("shrink", tracks[:1], catalog_path)

        result = catalog.get_playlist_tracks("shrink")
        assert len(result) == len(tracks) - 1

    def test_find_playlist_case_insensitive(self, tmp_path, catalog):
        catalog_path = tmp_path / "catalog.json"
        catalog.create_playlist("Morning Jazz", [], catalog_path)
        assert catalog.find_playlist("morning jazz") == "Morning Jazz"
