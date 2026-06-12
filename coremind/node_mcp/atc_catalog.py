from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CATALOG_VERSION = 1

# Spoken-word → keywords that appear in stream channel names.
# Abbreviations ("app", "dep", "twr"…) are listed as keys too, so queries
# like "KJFK app" carry the channel type instead of defaulting to tower.
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "tower":     ["tower", "twr"],
    "twr":       ["tower", "twr"],
    "ground":    ["ground", "gnd"],
    "gnd":       ["ground", "gnd"],
    "approach":  ["approach", "app"],
    "app":       ["approach", "app"],
    "departure": ["departure", "dep"],
    "dep":       ["departure", "dep"],
    "atis":      ["atis"],
    "delivery":  ["delivery", "del", "clearance"],
    "clearance": ["delivery", "del", "clearance"],
    "del":       ["delivery", "del", "clearance"],
    "ramp":      ["ramp", "co", "company"],
    "company":   ["ramp", "co", "company"],
    "co":        ["ramp", "co", "company"],
    "center":    ["center", "ctr"],
    "ctr":       ["center", "ctr"],
    "unicom":    ["unicom", "ctaf"],
    "ctaf":      ["unicom", "ctaf"],
    "ops":        ["ops", "operations"],
    "operations": ["ops", "operations"],
    "fbo":       ["fbo"],
    "emergency": ["emergency"],
    "final":     ["final"],
    "de-ice":    ["de-ice", "ice"],
    "deice":     ["de-ice", "ice"],
}

# When the query names an airport but no channel type, default to tower.
_DEFAULT_TYPE_KEYWORDS = _TYPE_KEYWORDS["tower"]

# Stream provider endpoints — keep these literals here and nowhere else.
_STREAM_BASE = "http://d.liveatc.net"
_PLS_BASE = "https://www.liveatc.net/play"


def stream_url(mount: str) -> str:
    return f"{_STREAM_BASE}/{mount}"


# VHF airband frequency in a spoken query, e.g. "120.250" or "119.1"
_FREQ_RE = re.compile(r"\b1\d{2}\.\d{1,3}\b")

# 4-6 digit mount segment encoding MHz with the dot stripped:
# 1191 → 119.1, 12575 → 125.75, 120250 → 120.250. Segments of 1-3 digits
# are receiver IDs, not frequencies (same rule as the scan command).
# The trailing separator is a lookahead so consecutive segments all match
# (foo_121350_128325 encodes two frequencies).
_MOUNT_FREQ_RE = re.compile(r"_(\d{4,6})(?=_|$)")


def query_frequencies(q: str) -> list[str]:
    """Extract explicit VHF frequencies like '120.250' from a query."""
    return _FREQ_RE.findall(q)


def _mount_frequencies(mount: str) -> list[float]:
    """Decode the MHz values encoded in a mount name."""
    return [float(f"{d[:3]}.{d[3:]}") for d in _MOUNT_FREQ_RE.findall(mount)]


def matches_query_frequency(ch: dict, query: str) -> bool | None:
    """Whether a channel matches the explicit frequency in a query.

    Returns None when the query names no frequency. False means the
    requested frequency is stale, mistyped, or unknown for this channel —
    callers must not stream it silently.
    """
    freqs = query_frequencies(query.lower())
    if not freqs:
        return None
    return any(_freq_matches(ch, f) for f in freqs)


def _freq_matches(ch: dict, freq: str) -> bool:
    """True if a spoken frequency identifies this channel.

    Checks the catalog freq field numerically (so '120.25' matches
    '120.250'), the channel name (e.g. 'Tower 119.1'), and the decoded
    mount encoding (kiad1_twr_1c19c_120250 → 120.250). Mount values are
    compared numerically — substring matching would let '120.100' (zero-
    stripped to '1201') prefix-match unrelated mounts like *_120150.
    """
    try:
        qf = float(freq)
    except ValueError:
        return False
    ch_freq = ch.get("freq", "")
    try:
        if ch_freq and abs(float(ch_freq) - qf) < 0.0005:
            return True
    except ValueError:
        pass
    if freq in ch.get("name", ""):
        return True
    return any(abs(m - qf) < 0.0005 for m in _mount_frequencies(ch.get("mount", "")))


def pls_url(mount: str) -> str:
    """Playlist endpoint used to probe whether a mount is live."""
    return f"{_PLS_BASE}/{mount}.pls"


def _is_tower(ch: dict) -> bool:
    # Must match every channel the scorer treats as a tower (_TYPE_KEYWORDS
    # ["tower", "twr"] against both name and mount), or airport-only queries
    # tie on a channel this rejects and the random pick never happens.
    text = f"{ch.get('name', '')} {ch.get('mount', '')}".lower()
    return "tower" in text or "twr" in text


# Name tokens that mark a tower feed as non-primary (backup/special-purpose).
# Random tie-breaking must never start one of these instead of the real tower.
_NON_PRIMARY_TOWER_TOKENS = (
    "emergency", "backup", "temp", "secondary", "heli", "tca", "spare",
    "alternate", "(alt)",
)


def _is_primary_tower(ch: dict) -> bool:
    # Scan name + mount, same text _is_tower uses — a marker encoded only
    # in the mount (e.g. name "Tower", mount kxyz_twr_backup) still
    # disqualifies the feed from random picking.
    text = f"{ch.get('name', '')} {ch.get('mount', '')}".lower()
    return not any(tok in text for tok in _NON_PRIMARY_TOWER_TOKENS)


def pick_tower_candidate(candidates: list[dict], query: str = "") -> dict | None:
    """Resolve a score tie by picking a primary tower channel at random.

    Applies only when every tied candidate is a tower at the same airport
    (e.g. KIAD's per-runway towers) — runway designators like "1C/19C" are
    not resolvable by voice, so asking the user to pick is a dead end.
    Only primary feeds are eligible: backup/temp/helicopter/emergency
    feeds are filtered out first. Returns None when the tie is not
    tower-only, spans airports, or has no primary feed left — callers then
    fall back to the disambiguation list.

    A query naming an explicit frequency is never resolved randomly: if it
    still ties (the frequency matched nothing, or several channels), the
    caller must surface the options instead of guessing.
    """
    if len(candidates) < 2:
        return None
    if query and query_frequencies(query):
        return None
    airports = {ch.get("airport", "").upper() for ch in candidates}
    if len(airports) != 1 or not all(_is_tower(ch) for ch in candidates):
        return None
    pool = [ch for ch in candidates if _is_primary_tower(ch)]
    return random.choice(pool) if pool else None


def load_atc_catalog(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not load ATC catalog: %s", exc)
    return None


def save_atc_catalog(catalog: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(catalog, indent=2))
    except Exception as exc:
        logger.warning("Could not save ATC catalog: %s", exc)


def empty_catalog() -> dict:
    return {
        "version": _CATALOG_VERSION,
        "scanned_at": datetime.now(tz=timezone.utc).isoformat(),
        "channels": [],
    }


class ATCCatalog:
    def __init__(self, data: dict) -> None:
        self._channels: list[dict] = data.get("channels", [])

    # --- Listing ---

    def list_airports(self) -> list[str]:
        """Return sorted unique 'ICAO — Airport Name' strings."""
        seen: dict[str, str] = {}
        for ch in self._channels:
            icao = ch.get("airport", "").upper()
            name = ch.get("airport_name", "")
            if icao and icao not in seen:
                seen[icao] = name
        return sorted(f"{icao} — {name}" if name else icao for icao, name in seen.items())

    def list_channels(self, airport: str) -> list[dict]:
        """Return all channels for a given airport (ICAO prefix match)."""
        q = airport.upper().strip()
        return [ch for ch in self._channels if ch.get("airport", "").upper() == q]

    # --- Search / matching ---

    def find_channel(self, query: str) -> dict | None:
        """Return the best-matching channel for a natural-language query, or None."""
        candidates = self.find_candidates(query)
        return candidates[0] if len(candidates) == 1 else (candidates[0] if candidates else None)

    def find_candidates(self, query: str) -> list[dict]:
        """Return all channels tied at the top score (used for disambiguation).

        Returns an empty list if no channel scores > 0.
        If multiple channels share the highest score, all are returned so the
        caller can ask the user to pick one.
        """
        results = self._score_all(query)
        if not results:
            return []
        results.sort(key=lambda x: x[1], reverse=True)
        best_score = results[0][1]
        if best_score == 0:
            return []
        return [ch for ch, s in results if s == best_score]

    def search(self, query: str) -> list[dict]:
        """Return all channels with a positive score, ranked best-first."""
        results = [(ch, s) for ch, s in self._score_all(query) if s > 0]
        results.sort(key=lambda x: x[1], reverse=True)
        return [ch for ch, _ in results[:10]]

    def _score_all(self, query: str) -> list[tuple[dict, int]]:
        q = query.lower()
        airport_icao, type_keywords = self._parse_query(q)
        freqs = query_frequencies(q)
        airport_scores = [self._airport_score(ch, q, airport_icao) for ch in self._channels]
        # The query names an airport if it contains an ICAO code or matched
        # any channel's airport — used to keep frequency matches from
        # escaping to another airport.
        names_airport = airport_icao is not None or any(s > 0 for s in airport_scores)
        scored = []
        for ch, airport_score in zip(self._channels, airport_scores):
            score = self._score(ch, q, airport_score, type_keywords, freqs, names_airport)
            scored.append((ch, score))
        return scored

    def _airport_score(self, ch: dict, q: str, airport_icao: str | None) -> int:
        """How strongly this channel's airport matches the query (0 = not at all)."""
        ch_airport = ch.get("airport", "").upper()
        ch_name = ch.get("airport_name", "").lower()
        if airport_icao and ch_airport == airport_icao:
            return 3
        if ch_name and ch_name in q:
            return 2
        if ch_airport.lower() in q:
            return 2
        # Try matching any word from airport name against query
        for word in ch_name.split():
            if len(word) >= 4 and word in q:
                return 1
        return 0

    def _parse_query(self, q: str) -> tuple[str | None, list[str]]:
        """Extract ICAO code (or None) and channel-type keywords from the query."""
        # Look for a 4-letter ICAO code (K/C/E/Y + 3 letters)
        icao_match = re.search(r'\b([kceyKCEY][a-zA-Z]{3})\b', q)
        airport_icao = icao_match.group(1).upper() if icao_match else None

        # Collect type keywords from every spoken form present in the query —
        # "final approach" must carry both, not stop at the first key hit.
        # Short keys (abbreviations like "app", "dep") match as exact words
        # so they don't fire inside "apple" or "stops"; longer words keep
        # suffix tolerance so "towers" and "operations" match.
        type_kws: list[str] = []
        for spoken, kws in _TYPE_KEYWORDS.items():
            pattern = rf"\b{re.escape(spoken)}\b" if len(spoken) <= 4 else rf"\b{re.escape(spoken)}"
            if re.search(pattern, q):
                for kw in kws:
                    if kw not in type_kws:
                        type_kws.append(kw)

        return airport_icao, type_kws

    def _score(
        self,
        ch: dict,
        q: str,
        airport_score: int,
        type_keywords: list[str],
        freqs: list[str],
        query_names_airport: bool,
    ) -> int:
        ch_name = ch.get("airport_name", "").lower()
        ch_channel = ch.get("name", "").lower()
        ch_mount = ch.get("mount", "").lower()

        score = airport_score

        # Channel type match; no spoken type defaults to tower, but only for
        # channels that already matched the airport — otherwise queries with
        # no airport at all would score every tower in the catalog.
        if not type_keywords and airport_score > 0:
            type_keywords = _DEFAULT_TYPE_KEYWORDS
        for kw in type_keywords:
            if kw in ch_channel or kw in ch_mount:
                score += 2
                break

        # Explicit frequency is the strongest channel signal — it breaks
        # ties between same-type channels ("Dulles tower 120.250"). Gated to
        # the requested airport when the query names one, so a reused or
        # mistyped frequency can never pull in a different airport's channel
        # ("KEWR tower 119.1" must not stream KJFK's Tower 119.1).
        if airport_score > 0 or not query_names_airport:
            for f in freqs:
                if _freq_matches(ch, f):
                    score += 3
                    break

        # Generic substring fallback in channel name. Skip words that belong
        # to the airport name (already credited above) — KATL feeds named
        # "Atlanta Approach (…)" must not outrank "Final Approach …" just
        # because they repeat the airport in the channel name. Gated like
        # the frequency bonus: channel-name tokens (e.g. the "119.1" in
        # KJFK's "Tower 119.1") must not pull in a different airport.
        if airport_score > 0 or not query_names_airport:
            for word in q.split():
                if len(word) >= 3 and word in ch_channel and word not in ch_name:
                    score += 1

        return score

    # --- Mutation ---

    def upsert_channel(self, channel: dict) -> bool:
        """Add or update a channel by mount name. Returns True if it was new."""
        mount = channel.get("mount", "")
        for i, ch in enumerate(self._channels):
            if ch.get("mount") == mount:
                self._channels[i] = channel
                return False
        self._channels.append(channel)
        return True

    def to_dict(self) -> dict:
        return {
            "version": _CATALOG_VERSION,
            "scanned_at": datetime.now(tz=timezone.utc).isoformat(),
            "channels": self._channels,
        }
