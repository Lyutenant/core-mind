from __future__ import annotations

import datetime
import logging
import urllib.parse

import httpx

from coremind.tools.registry import Tool

logger = logging.getLogger(__name__)

_WTTR_JSON_URL = "https://wttr.in/{location}?format=j1"


def _c_to_f(c: str | int) -> int:
    return round(int(c) * 9 / 5 + 32)


def _day_label(date_str: str, index: int) -> str:
    if index == 0:
        return "Today"
    if index == 1:
        return "Tomorrow"
    try:
        return datetime.date.fromisoformat(date_str).strftime("%A")
    except Exception:
        return f"Day {index + 1}"


def _hourly_desc(day: dict, target_time: str = "1200") -> str:
    """Return condition description for the closest hourly slot (prefers noon)."""
    hourly = day.get("hourly", [])
    for h in hourly:
        if h.get("time") == target_time:
            return h.get("weatherDesc", [{}])[0].get("value", "")
    # fallback: first entry
    return hourly[0].get("weatherDesc", [{}])[0].get("value", "") if hourly else ""


def _rain_note(day: dict) -> str:
    """Return a rain/snow note if chance is notable (≥20%)."""
    hourly = day.get("hourly", [])
    if not hourly:
        return ""
    max_rain = max(int(h.get("chanceofrain", 0)) for h in hourly)
    max_snow = max(int(h.get("chanceofsnow", 0)) for h in hourly)
    if max_snow >= 20:
        return f" Chance of snow up to {max_snow}%."
    if max_rain >= 20:
        return f" Chance of rain up to {max_rain}%."
    return ""


def _format_current(current: dict, location: str, today: dict | None) -> str:
    desc = current.get("weatherDesc", [{}])[0].get("value", "unknown")
    temp_c = current.get("temp_C", "?")
    feels_c = current.get("FeelsLikeC", temp_c)
    temp_f = current.get("temp_F", str(_c_to_f(temp_c)))
    feels_f = current.get("FeelsLikeF", str(_c_to_f(feels_c)))
    humidity = current.get("humidity", "?")
    wind_kmph = current.get("windspeedKmph", "?")
    wind_dir = current.get("winddir16Point", "")
    uv = current.get("uvIndex", "")

    sentences = [
        f"Currently in {location}: {desc}.",
        f"Temperature {temp_c}°C ({temp_f}°F)",
    ]
    # only mention feels-like when it differs by ≥2°C
    try:
        if abs(int(temp_c) - int(feels_c)) >= 2:
            sentences[-1] += f", feels like {feels_c}°C ({feels_f}°F)"
    except (ValueError, TypeError):
        pass
    sentences[-1] += "."
    sentences.append(f"Humidity {humidity}%, wind {wind_kmph} km/h {wind_dir}.")
    if uv and int(uv) >= 6:
        sentences.append(f"UV index is high at {uv} — sun protection recommended.")

    if today:
        max_c = today.get("maxtempC", "?")
        min_c = today.get("mintempC", "?")
        max_f = today.get("maxtempF", str(_c_to_f(max_c)))
        min_f = today.get("mintempF", str(_c_to_f(min_c)))
        astronomy = today.get("astronomy", [{}])[0]
        sunrise = astronomy.get("sunrise", "")
        sunset = astronomy.get("sunset", "")
        sentences.append(
            f"Today's high {max_c}°C ({max_f}°F), low {min_c}°C ({min_f}°F)."
        )
        if sunrise and sunset:
            sentences.append(f"Sunrise {sunrise}, sunset {sunset}.")
        rain = _rain_note(today)
        if rain:
            sentences.append(rain.strip())

    return " ".join(sentences)


def _format_forecast_day(day: dict, index: int) -> str:
    label = _day_label(day.get("date", ""), index)
    desc = _hourly_desc(day)
    max_c = day.get("maxtempC", "?")
    min_c = day.get("mintempC", "?")
    max_f = day.get("maxtempF", str(_c_to_f(max_c)))
    min_f = day.get("mintempF", str(_c_to_f(min_c)))
    astronomy = day.get("astronomy", [{}])[0]
    sunrise = astronomy.get("sunrise", "")
    sunset = astronomy.get("sunset", "")

    text = f"{label}: {desc}, high {max_c}°C ({max_f}°F), low {min_c}°C ({min_f}°F)."
    if sunrise and sunset:
        text += f" Sunrise {sunrise}, sunset {sunset}."
    rain = _rain_note(day)
    if rain:
        text += rain
    return text


class WeatherTool(Tool):
    name = "get_weather"
    description = (
        "Get the weather for a location. "
        "Use days=1 for current conditions and today's forecast (default), "
        "days=2 to include tomorrow, or days=3 for a full 3-day outlook."
    )
    requires_confirmation = False
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location, e.g. 'London' or 'New York'.",
            },
            "days": {
                "type": "integer",
                "description": (
                    "How many days to cover: 1 = current + today (default), "
                    "2 = + tomorrow, 3 = 3-day forecast."
                ),
            },
        },
        "required": ["location"],
    }

    def run(self, location: str = "", days: int = 1, **kwargs) -> str:
        if not location:
            return "Please specify a location to get the weather for."

        days = max(1, min(3, int(days)))
        url = _WTTR_JSON_URL.format(location=urllib.parse.quote(location))

        try:
            resp = httpx.get(url, timeout=10.0, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            return f"Weather request timed out for {location}. Try again in a moment."
        except Exception as e:
            logger.warning("WeatherTool error for %r: %s", location, e)
            return f"Could not get weather for {location}: {e}"

        current = (data.get("current_condition") or [{}])[0]
        forecast = data.get("weather") or []

        today = forecast[0] if forecast else None
        parts = [_format_current(current, location, today)]

        # Add tomorrow and/or day-after forecasts when requested
        for i in range(1, days):
            if i < len(forecast):
                parts.append(_format_forecast_day(forecast[i], i))

        return " ".join(parts)
