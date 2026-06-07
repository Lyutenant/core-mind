"""Bundled ICAO airport database (OurAirports, public domain).

Source: https://ourairports.com/data/
Covers ~19K airports (large, medium, small) with 4-letter ICAO codes.
Loaded once and cached; the JSON is ~1.5 MB.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict


class AirportInfo(TypedDict, total=False):
    name: str
    city: str
    country: str
    iata: str


_DATA_FILE = Path(__file__).parent / "data" / "airports.json"


@lru_cache(maxsize=1)
def _db() -> dict[str, AirportInfo]:
    if not _DATA_FILE.exists():
        return {}
    return json.loads(_DATA_FILE.read_text())


def lookup_icao(icao: str) -> AirportInfo | None:
    """Return airport info for a 4-letter ICAO code, or None."""
    return _db().get(icao.strip().upper())


def display_name(icao: str, info: AirportInfo) -> str:
    """Format a single-line airport description."""
    parts = [icao, "—", info["name"]]
    if info.get("city") and info["city"].lower() not in info["name"].lower():
        parts.append(f"({info['city']})")
    if info.get("iata"):
        parts.append(f"[{info['iata']}]")
    return " ".join(parts)


def search_airports(query: str, *, max_results: int = 8) -> list[tuple[str, AirportInfo]]:
    """Find airports by ICAO, IATA, name, or city. Returns (icao, info) pairs."""
    q = query.strip().upper()
    q_lower = query.strip().lower()
    # 3–4 letter all-alpha queries are treated as codes (ICAO/IATA only).
    # Longer or mixed queries get full name/city matching.
    is_code_query = q.isalpha() and len(q) in (3, 4)
    results: list[tuple[str, AirportInfo, int]] = []

    for icao, info in _db().items():
        score = 0
        if icao == q:
            score = 100
        elif info.get("iata") == q:
            score = 90
        elif not is_code_query:
            if q_lower in info["name"].lower():
                score = 50 + (20 if info["name"].lower().startswith(q_lower) else 0)
            elif q_lower in info.get("city", "").lower():
                score = 40
        if score:
            results.append((icao, info, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return [(icao, info) for icao, info, _ in results[:max_results]]
