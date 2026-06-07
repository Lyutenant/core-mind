from __future__ import annotations

from coremind.tools.registry import Tool


class AirportTool(Tool):
    name = "lookup_airport"
    description = (
        "Look up an airport by ICAO code, IATA code, name, or city. "
        "Use this to find the ICAO code for an airport the user mentions by name, "
        "or to confirm what airport an ICAO code refers to."
    )
    requires_confirmation = False
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "ICAO code (e.g. 'KIAD'), IATA code (e.g. 'IAD'), "
                    "airport name (e.g. 'Dulles'), or city name (e.g. 'Newark')"
                ),
            }
        },
        "required": ["query"],
    }

    def run(self, query: str) -> str:
        from coremind.airports import display_name, lookup_icao, search_airports

        q = query.strip()

        # Fast path: exact ICAO lookup
        info = lookup_icao(q)
        if info:
            return display_name(q.upper(), info)

        # Name / city / IATA search
        results = search_airports(q)
        if not results:
            return f"No airport found for '{q}'."
        if len(results) == 1:
            icao, info = results[0]
            return display_name(icao, info)
        lines = [f"Airports matching '{q}':"]
        for icao, info in results:
            lines.append(f"  {display_name(icao, info)}")
        return "\n".join(lines)
