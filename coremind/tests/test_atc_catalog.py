"""Unit tests for coremind.node_mcp.atc_catalog — no network required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coremind.node_mcp.atc_catalog import (
    ATCCatalog,
    empty_catalog,
    load_atc_catalog,
    save_atc_catalog,
    stream_url,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CHANNELS = [
    {
        "airport": "KEWR",
        "airport_name": "Newark Liberty",
        "name": "Tower",
        "mount": "kewr_twr",
        "freq": "118.30",
    },
    {
        "airport": "KEWR",
        "airport_name": "Newark Liberty",
        "name": "Ground",
        "mount": "kewr_gnd",
        "freq": "121.80",
    },
    {
        "airport": "KEWR",
        "airport_name": "Newark Liberty",
        "name": "Approach",
        "mount": "kewr_app",
        "freq": "124.70",
    },
    {
        "airport": "KJFK",
        "airport_name": "John F Kennedy",
        "name": "Tower",
        "mount": "kjfk_twr",
        "freq": "119.10",
    },
    {
        "airport": "KIAD",
        "airport_name": "Washington Dulles",
        "name": "Tower",
        "mount": "kiad1_twr_1c19c_120250",
        "freq": "120.250",
    },
]


@pytest.fixture
def catalog() -> ATCCatalog:
    return ATCCatalog({"channels": list(_SAMPLE_CHANNELS)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestStreamUrl:
    def test_basic_mount(self):
        assert stream_url("kewr_twr") == "http://d.liveatc.net/kewr_twr"

    def test_obfuscated_mount(self):
        url = stream_url("kiad1_twr_1c19c_120250")
        assert url == "http://d.liveatc.net/kiad1_twr_1c19c_120250"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_load_missing_returns_none(self, tmp_path):
        result = load_atc_catalog(tmp_path / "nonexistent.json")
        assert result is None

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "atc.json"
        data = empty_catalog()
        data["channels"] = _SAMPLE_CHANNELS
        save_atc_catalog(data, path)
        loaded = load_atc_catalog(path)
        assert loaded is not None
        assert len(loaded["channels"]) == len(_SAMPLE_CHANNELS)

    def test_empty_catalog_has_required_keys(self):
        ec = empty_catalog()
        assert "version" in ec
        assert "scanned_at" in ec
        assert ec["channels"] == []

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "atc.json"
        save_atc_catalog(empty_catalog(), path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing:
    def test_list_airports_unique_and_sorted(self, catalog):
        airports = catalog.list_airports()
        # 3 unique airports
        assert len(airports) == 3
        # Each entry contains ICAO and name
        assert any("KEWR" in a for a in airports)
        assert any("KJFK" in a for a in airports)
        assert any("KIAD" in a for a in airports)
        # Sorted
        assert airports == sorted(airports)

    def test_list_channels_by_icao(self, catalog):
        channels = catalog.list_channels("KEWR")
        assert len(channels) == 3
        mounts = {ch["mount"] for ch in channels}
        assert "kewr_twr" in mounts
        assert "kewr_gnd" in mounts
        assert "kewr_app" in mounts

    def test_list_channels_empty_for_unknown(self, catalog):
        assert catalog.list_channels("ZZZZ") == []


# ---------------------------------------------------------------------------
# Finding / scoring
# ---------------------------------------------------------------------------


class TestFindChannel:
    def test_find_by_icao_and_type(self, catalog):
        ch = catalog.find_channel("KEWR tower")
        assert ch is not None
        assert ch["mount"] == "kewr_twr"

    def test_find_by_airport_name(self, catalog):
        ch = catalog.find_channel("Newark approach")
        assert ch is not None
        assert ch["mount"] == "kewr_app"

    def test_find_by_city_name_partial(self, catalog):
        ch = catalog.find_channel("Dulles tower")
        assert ch is not None
        assert ch["airport"] == "KIAD"

    def test_find_ground_keyword(self, catalog):
        ch = catalog.find_channel("KEWR ground")
        assert ch is not None
        assert ch["mount"] == "kewr_gnd"

    def test_find_unknown_returns_none(self, catalog):
        ch = catalog.find_channel("Timbuktu ATIS")
        assert ch is None

    def test_find_prefers_higher_score(self, catalog):
        # "KJFK" should beat KEWR even though both have "Tower"
        ch = catalog.find_channel("KJFK tower")
        assert ch is not None
        assert ch["airport"] == "KJFK"


# ---------------------------------------------------------------------------
# find_candidates (disambiguation support)
# ---------------------------------------------------------------------------


class TestFindCandidates:
    def test_single_best_match_returns_one(self, catalog):
        # "KEWR tower" — only one tower at KEWR, should return exactly one
        candidates = catalog.find_candidates("KEWR tower")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kewr_twr"

    def test_tied_channels_returned(self):
        # Two KIAD tower channels with identical names — both should be candidates
        multi_catalog = ATCCatalog({
            "channels": [
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower Runway 1C/19C", "mount": "kiad1_twr_1c19c", "freq": "120.250"},
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower Runway 1R/19L", "mount": "kiad1_twr_1r19l", "freq": "119.850"},
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Ground", "mount": "kiad_gnd", "freq": "121.900"},
            ]
        })
        # "Dulles tower" — both tower channels tie; ground does not
        candidates = multi_catalog.find_candidates("Dulles tower")
        mounts = {ch["mount"] for ch in candidates}
        assert "kiad1_twr_1c19c" in mounts
        assert "kiad1_twr_1r19l" in mounts
        assert "kiad_gnd" not in mounts

    def test_channel_name_breaks_tie_on_second_query(self):
        # After disambiguation, user's more specific query picks the right channel
        multi_catalog = ATCCatalog({
            "channels": [
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower Runway 1C/19C", "mount": "kiad1_twr_1c19c", "freq": "120.250"},
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower Runway 1R/19L", "mount": "kiad1_twr_1r19l", "freq": "119.850"},
            ]
        })
        # LLM re-queries with "KIAD Tower Runway 1C/19C"
        candidates = multi_catalog.find_candidates("KIAD Tower Runway 1C 19C")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kiad1_twr_1c19c"

    def test_no_match_returns_empty(self, catalog):
        candidates = catalog.find_candidates("Timbuktu ATIS")
        assert candidates == []

    def test_zero_score_returns_empty(self, catalog):
        candidates = catalog.find_candidates("xyz xyz xyz")
        assert candidates == []


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_ranked_results(self, catalog):
        results = catalog.search("KEWR")
        assert len(results) >= 1
        assert all(ch["airport"] == "KEWR" for ch in results)

    def test_search_empty_query_returns_nothing(self, catalog):
        results = catalog.search("xyzzy")
        assert results == []


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_upsert_new_channel(self, catalog):
        new_ch = {"airport": "KLGA", "name": "Tower", "mount": "klga_twr"}
        is_new = catalog.upsert_channel(new_ch)
        assert is_new is True
        assert any(ch["mount"] == "klga_twr" for ch in catalog._channels)

    def test_upsert_update_existing(self, catalog):
        updated = dict(_SAMPLE_CHANNELS[0], freq="999.99")
        is_new = catalog.upsert_channel(updated)
        assert is_new is False
        ch = next(c for c in catalog._channels if c["mount"] == "kewr_twr")
        assert ch["freq"] == "999.99"

    def test_to_dict_preserves_channels(self, catalog):
        d = catalog.to_dict()
        assert len(d["channels"]) == len(_SAMPLE_CHANNELS)
        assert "version" in d
        assert "scanned_at" in d
