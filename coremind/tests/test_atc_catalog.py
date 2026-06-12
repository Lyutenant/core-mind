"""Unit tests for coremind.node_mcp.atc_catalog — no network required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coremind.node_mcp.atc_catalog import (
    ATCCatalog,
    empty_catalog,
    load_atc_catalog,
    matches_query_frequency,
    pick_tower_candidate,
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
        url = stream_url("kewr_twr")
        assert url.startswith("http")
        assert url.endswith("/kewr_twr")

    def test_obfuscated_mount(self):
        url = stream_url("kiad1_twr_1c19c_120250")
        assert url.startswith("http")
        assert url.endswith("/kiad1_twr_1c19c_120250")


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
# Tower default + random tower pick
# ---------------------------------------------------------------------------


_KIAD_TOWERS = [
    {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (1C/19C)", "mount": "kiad1_twr_1c19c", "freq": "120.250"},
    {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (1R/19L)", "mount": "kiad1_twr_1r19l", "freq": "120.100"},
    {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (Emergency)", "mount": "kiad1_twr_emergency"},
]


class TestTowerDefault:
    def test_airport_only_query_defaults_to_tower(self, catalog):
        # "Newark" names no channel type — tower should win over ground/approach
        candidates = catalog.find_candidates("Newark")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kewr_twr"

    def test_icao_only_query_defaults_to_tower(self, catalog):
        candidates = catalog.find_candidates("KEWR")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kewr_twr"

    def test_explicit_type_overrides_default(self, catalog):
        candidates = catalog.find_candidates("Newark approach")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kewr_app"

    def test_no_airport_match_still_returns_nothing(self, catalog):
        # The tower default must not make garbage queries match every tower
        assert catalog.find_candidates("xyz xyz xyz") == []

    def test_airport_only_with_multiple_towers_ties_on_towers(self):
        cat = ATCCatalog({
            "channels": _KIAD_TOWERS + [
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Ground (East)", "mount": "kiad1_gnd_east"},
            ]
        })
        candidates = cat.find_candidates("Dulles")
        mounts = {ch["mount"] for ch in candidates}
        assert mounts == {"kiad1_twr_1c19c", "kiad1_twr_1r19l", "kiad1_twr_emergency"}


class TestTypeAliases:
    def test_operations_resolves_to_ops_channel(self):
        # "operations" must be recognized as a type, not fall through to the
        # tower default
        cat = ATCCatalog({
            "channels": _KIAD_TOWERS + [
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Dulles Ops", "mount": "kiad1_ops"},
            ]
        })
        candidates = cat.find_candidates("Dulles operations")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kiad1_ops"

    def test_final_approach_prefers_final_feeds(self):
        # "final approach" carries both type keywords; the final feed must
        # outrank generic approach feeds rather than tie with them
        cat = ATCCatalog({
            "channels": [
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Potomac Approach (Final East)", "mount": "kiad1_app_final_east"},
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Potomac Approach (LUCKE)", "mount": "kiad1_app_lucke"},
            ]
        })
        candidates = cat.find_candidates("Dulles final approach")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kiad1_app_final_east"

    def test_abbreviated_type_overrides_tower_default(self):
        # "app"/"dep"/"gnd" abbreviations must carry the channel type — not
        # fall through to the tower default
        cat = ATCCatalog({
            "channels": [
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower", "mount": "kewr_twr"},
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Approach", "mount": "kewr_app"},
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Departure", "mount": "kewr_dep"},
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Ground", "mount": "kewr_gnd"},
            ]
        })
        assert cat.find_candidates("KEWR app")[0]["mount"] == "kewr_app"
        assert cat.find_candidates("Newark dep")[0]["mount"] == "kewr_dep"
        assert cat.find_candidates("KEWR gnd")[0]["mount"] == "kewr_gnd"
        for q in ("KEWR app", "Newark dep", "KEWR gnd"):
            assert len(cat.find_candidates(q)) == 1

    def test_company_alias_overrides_tower_default(self):
        # "company"/"co" must carry the ramp type — not default to tower
        cat = ATCCatalog({
            "channels": [
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower", "mount": "kewr_twr"},
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Ramp/Company/Misc", "mount": "kewr_ramp"},
            ]
        })
        for q in ("KEWR company", "KEWR co"):
            candidates = cat.find_candidates(q)
            assert len(candidates) == 1, q
            assert candidates[0]["mount"] == "kewr_ramp", q

    def test_short_keys_do_not_fire_inside_words(self):
        # "app" inside "apple" must not be parsed as approach
        cat = ATCCatalog({
            "channels": [
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower", "mount": "kewr_twr"},
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Approach", "mount": "kewr_app"},
            ]
        })
        # No recognizable type word → tower default
        candidates = cat.find_candidates("Newark apple")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kewr_twr"

    def test_ops_does_not_fire_inside_other_words(self):
        # Word-boundary matching: "stops" must not be parsed as the "ops" type
        cat = ATCCatalog({
            "channels": _KIAD_TOWERS + [
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Dulles Ops", "mount": "kiad1_ops"},
            ]
        })
        # No type word → tower default, not the Ops channel
        candidates = cat.find_candidates("Dulles nonstops")
        assert all("Tower" in ch["name"] for ch in candidates)


class TestFrequencyMatch:
    def test_frequency_breaks_tower_tie_via_freq_field(self):
        cat = ATCCatalog({"channels": list(_KIAD_TOWERS)})
        candidates = cat.find_candidates("Dulles tower 120.250")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kiad1_twr_1c19c"

    def test_frequency_matches_numerically(self):
        # "120.25" must match the catalog's "120.250"
        cat = ATCCatalog({"channels": list(_KIAD_TOWERS)})
        candidates = cat.find_candidates("Dulles tower 120.25")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kiad1_twr_1c19c"

    def test_frequency_matches_mount_encoding(self):
        # No freq field — the dot-stripped mount suffix identifies the channel
        cat = ATCCatalog({
            "channels": [
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (1C/19C)", "mount": "kiad1_twr_1c19c_120250"},
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (1L/19R, 12/30)", "mount": "kiad1_twr_1l19r_134425"},
            ]
        })
        candidates = cat.find_candidates("Dulles tower 134.425")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kiad1_twr_1l19r_134425"

    def test_trailing_zero_frequency_does_not_prefix_match(self):
        # "120.100" must not also match *_120150 via zero-stripped prefix
        cat = ATCCatalog({
            "channels": [
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (1R/19L)", "mount": "kiad1_twr_1r19l_120100"},
                {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (1C/19C)", "mount": "kiad1_twr_1c19c_120150"},
            ]
        })
        candidates = cat.find_candidates("Dulles tower 120.100")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kiad1_twr_1r19l_120100"
        # Shorthand "120.1" decodes to the same MHz value
        candidates = cat.find_candidates("Dulles tower 120.1")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kiad1_twr_1r19l_120100"

    def test_frequency_stays_scoped_to_named_airport(self):
        # "KEWR tower 119.1" must not stream KJFK's Tower 119.1 — the
        # frequency bonus only applies to channels at the requested airport
        cat = ATCCatalog({
            "channels": [
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower #1", "mount": "kewr_twr"},
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower #2", "mount": "kewr_twr_sf1"},
                {"airport": "KJFK", "airport_name": "John F Kennedy", "name": "Tower 119.1", "mount": "kjfk_twr_1191", "freq": "119.10"},
            ]
        })
        candidates = cat.find_candidates("KEWR tower 119.1")
        assert candidates
        assert all(ch["airport"] == "KEWR" for ch in candidates)
        # And the freq-present query must not random-pick among them
        assert pick_tower_candidate(candidates, "KEWR tower 119.1") is None

    def test_frequency_without_airport_matches_anywhere(self):
        cat = ATCCatalog({
            "channels": [
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower #1", "mount": "kewr_twr"},
                {"airport": "KJFK", "airport_name": "John F Kennedy", "name": "Tower 119.1", "mount": "kjfk_twr_1191", "freq": "119.10"},
            ]
        })
        candidates = cat.find_candidates("tower 119.1")
        assert len(candidates) == 1
        assert candidates[0]["mount"] == "kjfk_twr_1191"

    def test_multi_frequency_mount_decodes_all_segments(self):
        # foo_121350_128325 encodes two frequencies — both must be matchable
        cat = ATCCatalog({
            "channels": [
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Approach (North)", "mount": "kewr_app_121350_128325"},
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Approach (South)", "mount": "kewr_app_127600"},
            ]
        })
        for freq in ("121.350", "128.325"):
            candidates = cat.find_candidates(f"Newark approach {freq}")
            assert len(candidates) == 1, freq
            assert candidates[0]["mount"] == "kewr_app_121350_128325"

    def test_matches_query_frequency(self):
        tower = {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower", "mount": "kewr_twr", "freq": "118.30"}
        no_freq = {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower #2", "mount": "kewr_twr_sf1"}
        assert matches_query_frequency(tower, "KEWR tower") is None
        assert matches_query_frequency(tower, "KEWR tower 118.3") is True
        assert matches_query_frequency(tower, "KEWR tower 119.1") is False
        # Unknown channel frequency cannot confirm the request
        assert matches_query_frequency(no_freq, "KEWR tower 118.3") is False

    def test_play_atc_refuses_mismatched_frequency(self):
        # Codex scenario: a single airport tower must not stream when the
        # requested frequency doesn't match it
        from coremind.node_mcp.tools import atc_player
        old_catalog = atc_player._catalog
        atc_player._catalog = ATCCatalog({
            "channels": [
                {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower", "mount": "kewr_twr", "freq": "118.30"},
            ]
        })
        try:
            result = atc_player.play_atc("KEWR tower 119.1")
            assert "No channel matches that frequency" in result
            assert "Streaming" not in result
            # Without the frequency, the same channel streams (or at least
            # is not refused for frequency reasons)
            assert matches_query_frequency(
                atc_player._catalog.find_candidates("KEWR tower")[0], "KEWR tower"
            ) is None
        finally:
            atc_player._catalog = old_catalog

    def test_query_with_frequency_never_random_picks(self):
        # Frequency matched nothing — surface the ambiguity, don't guess
        assert pick_tower_candidate(list(_KIAD_TOWERS), "Dulles tower 123.450") is None

    def test_query_without_frequency_still_picks(self):
        assert pick_tower_candidate(list(_KIAD_TOWERS), "Dulles tower") is not None


class TestPickTowerCandidate:
    def test_picks_one_of_tied_towers(self):
        picked = pick_tower_candidate(list(_KIAD_TOWERS))
        assert picked is not None
        assert picked["mount"].startswith("kiad1_twr_")

    def test_never_picks_emergency_when_regular_towers_exist(self):
        for _ in range(50):
            picked = pick_tower_candidate(list(_KIAD_TOWERS))
            assert picked["mount"] != "kiad1_twr_emergency"

    def test_never_picks_non_primary_feeds(self):
        # KEWR-style pool: backup/temp/TCA feeds must not be auto-picked
        kewr = [
            {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower #1", "mount": "kewr_twr"},
            {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower #2", "mount": "kewr_twr_sf1"},
            {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower (TCA)", "mount": "kewr_twr_tca"},
            {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower/TCA/Backup", "mount": "kewr_twr_bu_tca"},
            {"airport": "KEWR", "airport_name": "Newark Liberty", "name": "Tower (Temp Feed)", "mount": "kewr_twr2"},
        ]
        for _ in range(50):
            picked = pick_tower_candidate(list(kewr))
            assert picked["mount"] in {"kewr_twr", "kewr_twr_sf1"}

    def test_never_picks_helicopter_feed(self):
        klax = [
            {"airport": "KLAX", "airport_name": "Los Angeles Intl", "name": "Tower (Helicopters) #1", "mount": "klax3n_heli"},
            {"airport": "KLAX", "airport_name": "Los Angeles Intl", "name": "Tower (North) #1", "mount": "klax3"},
        ]
        for _ in range(20):
            assert pick_tower_candidate(list(klax))["mount"] == "klax3"

    def test_twr_abbreviation_in_name_only_counts_as_tower(self):
        # KJFK-style: "Gnd/Twr" has the abbreviation in the name but not the
        # mount — the tie must still resolve to a random pick, not None
        kjfk = [
            {"airport": "KJFK", "airport_name": "John F Kennedy", "name": "Gnd/Twr", "mount": "kjfk9_s"},
            {"airport": "KJFK", "airport_name": "John F Kennedy", "name": "Tower", "mount": "kjfk_twr"},
            {"airport": "KJFK", "airport_name": "John F Kennedy", "name": "Tower #2", "mount": "kjfk_twr3"},
        ]
        picked = pick_tower_candidate(kjfk)
        assert picked is not None
        assert picked["airport"] == "KJFK"

    def test_marker_only_in_mount_is_not_primary(self):
        # name says "Tower" but the mount marks it as backup — never pick it
        towers = [
            {"airport": "KXYZ", "airport_name": "Example Field", "name": "Tower", "mount": "kxyz_twr"},
            {"airport": "KXYZ", "airport_name": "Example Field", "name": "Tower", "mount": "kxyz_twr_backup"},
        ]
        for _ in range(20):
            assert pick_tower_candidate(list(towers))["mount"] == "kxyz_twr"

    def test_no_primary_feed_returns_none(self):
        # All non-primary → leave the tie ambiguous (disambiguation list)
        non_primary = [
            {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (Emergency)", "mount": "kiad1_twr_emergency"},
            {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower (Backup)", "mount": "kiad1_twr_backup"},
        ]
        assert pick_tower_candidate(non_primary) is None

    def test_mixed_types_returns_none(self):
        mixed = _KIAD_TOWERS + [
            {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Ground (East)", "mount": "kiad1_gnd_east"},
        ]
        assert pick_tower_candidate(mixed) is None

    def test_multiple_airports_returns_none(self):
        cross = [
            {"airport": "KIAD", "airport_name": "Washington Dulles", "name": "Tower", "mount": "kiad1_twr"},
            {"airport": "KJFK", "airport_name": "John F Kennedy", "name": "Tower", "mount": "kjfk_twr"},
        ]
        assert pick_tower_candidate(cross) is None

    def test_single_candidate_returns_none(self):
        assert pick_tower_candidate(_KIAD_TOWERS[:1]) is None


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
