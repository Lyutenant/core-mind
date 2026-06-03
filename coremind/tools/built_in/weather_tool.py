from __future__ import annotations

import logging
import urllib.parse

import httpx

from coremind.tools.registry import Tool

logger = logging.getLogger(__name__)

_WTTR_URL = "https://wttr.in/{location}?format=%l:+%C.+Temperature:+%t.+Humidity:+%h."


class WeatherTool(Tool):
    name = "get_weather"
    description = (
        "Get the current weather conditions for a city or location. "
        "Returns conditions, temperature, and humidity."
    )
    requires_confirmation = False
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location, e.g. 'London' or 'New York'.",
            }
        },
        "required": ["location"],
    }

    def run(self, location: str = "", **kwargs) -> str:
        if not location:
            return "Please specify a location to get the weather for."
        url = _WTTR_URL.format(location=urllib.parse.quote(location))
        try:
            resp = httpx.get(url, timeout=8.0, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return f"Could not retrieve weather for {location}."
            return text
        except httpx.TimeoutException:
            return f"Weather request timed out for {location}. Try again in a moment."
        except Exception as e:
            logger.warning("WeatherTool error for %r: %s", location, e)
            return f"Could not get weather for {location}: {e}"
