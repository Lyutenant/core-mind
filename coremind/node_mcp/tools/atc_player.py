from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from coremind.node_mcp import playback

if TYPE_CHECKING:
    from coremind.node_mcp.atc_catalog import ATCCatalog

logger = logging.getLogger(__name__)

_catalog: ATCCatalog | None = None
_catalog_path: Path = Path.home() / ".coremind" / "atc-catalog.json"


_DEFAULT_CATALOG = Path(__file__).parent.parent.parent / "data" / "atc-catalog-default.json"


def init_atc_catalog(catalog_path: Path) -> None:
    global _catalog, _catalog_path
    from coremind.node_mcp.atc_catalog import ATCCatalog, load_atc_catalog
    _catalog_path = catalog_path
    data = load_atc_catalog(catalog_path)
    if data is None:
        # Fall back to the bundled default catalog
        data = load_atc_catalog(_DEFAULT_CATALOG)
        if data is not None:
            logger.info(
                "No user ATC catalog at %s — loaded bundled default (%d channels). "
                "Run 'coremind atc scan' to build your own.",
                catalog_path,
                len(data.get("channels", [])),
            )
        else:
            logger.warning(
                "ATC catalog not found at %s — run 'coremind atc scan' or 'coremind atc add'.",
                catalog_path,
            )
    else:
        logger.info("ATC catalog loaded: %d channel(s)", len(data.get("channels", [])))
    _catalog = ATCCatalog(data or {"channels": []})


def _no_catalog_msg() -> str:
    return (
        "ATC catalog is empty. Build it first:\n"
        "  coremind atc scan KEWR KJFK KIAD   (auto-discover)\n"
        "  coremind atc add KIAD 'Tower' kiad1_twr_1c19c_120250  (manual)"
    )


def play_atc(query: str) -> str:
    """Find the best-matching ATC channel and stream it.

    Queries that name only an airport default to its tower. When multiple
    tower channels at one airport tie (per-runway towers), one is picked at
    random. Other ties return a disambiguation list so the LLM can ask the
    user to pick one and call play_atc again with a more specific query.
    """
    if _catalog is None or not _catalog._channels:
        return _no_catalog_msg()

    from coremind.node_mcp.atc_catalog import (
        matches_query_frequency,
        pick_tower_candidate,
        stream_url,
    )
    candidates = _catalog.find_candidates(query)

    if len(candidates) > 1:
        picked = pick_tower_candidate(candidates, query)
        if picked is not None:
            candidates = [picked]

    if not candidates:
        airports = _catalog.list_airports()
        airport_list = ", ".join(airports[:8]) or "none"
        return (
            f"No ATC channel matched '{query}'. "
            f"Available airports: {airport_list}. "
            "Try 'list ATC airports' for the full list."
        )

    if len(candidates) > 5:
        # Query too vague — give a hint rather than a wall of options
        airport_hint = candidates[0].get("airport_name") or candidates[0].get("airport", "")
        return (
            f"'{query}' matched {len(candidates)} channels for {airport_hint}. "
            "Try adding a channel type — for example 'tower', 'ground', or 'approach'."
        )

    if len(candidates) > 1:
        # Disambiguate: list the tied options
        airport_name = (
            candidates[0].get("airport_name")
            or candidates[0].get("airport", "")
        )
        lines = [f"Multiple channels found for {airport_name}:"]
        for ch in candidates:
            name = ch.get("name", ch["mount"])
            freq = f" ({ch['freq']} MHz)" if ch.get("freq") else ""
            lines.append(f"- {name}{freq}")
        lines.append("Which would you like?")
        return "\n".join(lines)

    # Single unambiguous match — but never stream a channel whose frequency
    # doesn't match one the user explicitly asked for (stale or mistyped
    # frequencies must surface, not silently tune the best-effort channel)
    ch = candidates[0]
    if matches_query_frequency(ch, query) is False:
        name = ch.get("name", ch["mount"])
        freq = f" ({ch['freq']} MHz)" if ch.get("freq") else ""
        return (
            f"No channel matches that frequency. Closest match: "
            f"{ch.get('airport', '')} {name}{freq}. "
            "Ask for it without the frequency to play it, or use "
            "'list ATC channels' to see what's available."
        )

    url = stream_url(ch["mount"])
    label = f"{ch.get('airport', '')} {ch.get('name', ch['mount'])}"
    if ch.get("freq"):
        label += f" ({ch['freq']} MHz)"

    try:
        playback.start(["mpv", "--no-terminal", "--quiet", url])
    except FileNotFoundError:
        return "mpv is not installed. Run: sudo apt install mpv"

    return f"Streaming {label}."


def list_atc_airports() -> str:
    """List all airports with ATC streams in the catalog."""
    if _catalog is None or not _catalog._channels:
        return _no_catalog_msg()
    airports = _catalog.list_airports()
    if not airports:
        return "No airports in ATC catalog."
    return "\n".join(airports)


def list_atc_channels(airport: str) -> str:
    """List all ATC channels available for a specific airport."""
    if _catalog is None or not _catalog._channels:
        return _no_catalog_msg()
    channels = _catalog.list_channels(airport.upper().strip())
    if not channels:
        # Try partial name match
        q = airport.lower()
        channels = [
            ch for ch in _catalog._channels
            if q in ch.get("airport_name", "").lower() or q in ch.get("airport", "").lower()
        ]
    if not channels:
        return f"No ATC channels found for '{airport}'."
    lines = []
    for ch in channels:
        freq = f"  {ch['freq']} MHz" if ch.get("freq") else ""
        lines.append(f"{ch.get('name', ch['mount'])}{freq}")
    return "\n".join(lines)


def stop_atc() -> str:
    """Stop the current ATC audio stream."""
    return playback.stop_current()
