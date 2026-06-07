from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CATALOG_VERSION = 1

# Spoken-word → keywords that appear in LiveATC channel names
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "tower":     ["tower", "twr"],
    "ground":    ["ground", "gnd"],
    "approach":  ["approach", "app"],
    "departure": ["departure", "dep"],
    "atis":      ["atis"],
    "delivery":  ["delivery", "del", "clearance"],
    "clearance": ["delivery", "del", "clearance"],
    "ramp":      ["ramp", "co", "company"],
    "center":    ["center", "ctr"],
    "unicom":    ["unicom", "ctaf"],
}


def stream_url(mount: str) -> str:
    return f"http://d.liveatc.net/{mount}"


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
        results = self._score_all(query)
        if not results:
            return None
        results.sort(key=lambda x: x[1], reverse=True)
        best_score = results[0][1]
        if best_score == 0:
            return None
        return results[0][0]

    def search(self, query: str) -> list[dict]:
        """Return all channels with a positive score, ranked best-first."""
        results = [(ch, s) for ch, s in self._score_all(query) if s > 0]
        results.sort(key=lambda x: x[1], reverse=True)
        return [ch for ch, _ in results[:10]]

    def _score_all(self, query: str) -> list[tuple[dict, int]]:
        q = query.lower()
        airport_icao, type_keywords = self._parse_query(q)
        scored = []
        for ch in self._channels:
            score = self._score(ch, q, airport_icao, type_keywords)
            scored.append((ch, score))
        return scored

    def _parse_query(self, q: str) -> tuple[str | None, list[str]]:
        """Extract ICAO code (or None) and channel-type keywords from the query."""
        # Look for a 4-letter ICAO code (K/C/E/Y + 3 letters)
        icao_match = re.search(r'\b([kceyKCEY][a-zA-Z]{3})\b', q)
        airport_icao = icao_match.group(1).upper() if icao_match else None

        # Collect type keywords from the spoken-word map
        type_kws: list[str] = []
        for spoken, kws in _TYPE_KEYWORDS.items():
            if spoken in q:
                type_kws.extend(kws)
                break  # take the first match only

        return airport_icao, type_kws

    def _score(
        self,
        ch: dict,
        q: str,
        airport_icao: str | None,
        type_keywords: list[str],
    ) -> int:
        score = 0
        ch_airport = ch.get("airport", "").upper()
        ch_name = ch.get("airport_name", "").lower()
        ch_channel = ch.get("name", "").lower()
        ch_mount = ch.get("mount", "").lower()

        # Airport match
        if airport_icao and ch_airport == airport_icao:
            score += 3
        elif ch_name and ch_name in q:
            score += 2
        elif ch_airport.lower() in q:
            score += 2
        else:
            # Try matching any word from airport name against query
            for word in ch_name.split():
                if len(word) >= 4 and word in q:
                    score += 1
                    break

        # Channel type match
        for kw in type_keywords:
            if kw in ch_channel or kw in ch_mount:
                score += 2
                break

        # Generic substring fallback in channel name
        for word in q.split():
            if len(word) >= 3 and word in ch_channel:
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
