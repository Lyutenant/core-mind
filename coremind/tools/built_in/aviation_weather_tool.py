from __future__ import annotations

import datetime
import logging
import urllib.parse
from typing import Any

import httpx

from coremind.tools.registry import Tool

logger = logging.getLogger(__name__)

_BASE = "https://aviationweather.gov/api/data"
_METAR_URL = _BASE + "/metar?ids={icao}&format=json&hours=1"
_TAF_URL = _BASE + "/taf?ids={icao}&format=json"
_PIREP_URL = _BASE + "/pirep?id={icao}&distance=75&age=3&format=json"

_SKY_COVER = {
    "SKC": "sky clear",
    "CLR": "sky clear",
    "CAVOK": "sky clear",
    "NSC": "sky clear",
    "NCD": "sky clear",
    "FEW": "few clouds",
    "SCT": "scattered clouds",
    "BKN": "broken ceiling",
    "OVC": "overcast",
    "VV": "vertical visibility",
    "OVX": "obscured",
}

_TURB_INT = {
    "NEG": "smooth",
    "SMTH-LGT": "smooth to light turbulence",
    "LGT": "light turbulence",
    "LGT-MOD": "light to moderate turbulence",
    "MOD": "moderate turbulence",
    "MOD-SEV": "moderate to severe turbulence",
    "SEV": "severe turbulence",
    "EXTM": "extreme turbulence",
}

_ICG_INT = {
    "NEG": "no icing",
    "TRC": "trace icing",
    "LGT": "light icing",
    "MOD": "moderate icing",
    "SEV": "severe icing",
    "TRC-LGT": "trace to light icing",
    "LGT-MOD": "light to moderate icing",
    "MOD-SEV": "moderate to severe icing",
}


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def _wind(wdir: Any, wspd: Any, wgst: Any = None) -> str:
    try:
        spd = int(wspd)
        dr = int(wdir)
    except (TypeError, ValueError):
        return "wind unknown"
    if spd == 0:
        return "calm winds"
    text = f"wind {dr:03d} at {spd} knots"
    if wgst:
        try:
            text += f" gusting {int(wgst)} knots"
        except (TypeError, ValueError):
            pass
    return text


def _visibility(vis: Any) -> str:
    if vis is None:
        return ""
    s = str(vis).strip()
    if s in ("10+", "P6SM", "6+"):
        return "visibility greater than 10 miles"
    try:
        n = float(s)
        if n >= 10:
            return "visibility greater than 10 miles"
        if n < 0.25:
            return "visibility less than a quarter mile"
        if n < 1:
            return f"visibility {n} miles"
        return f"visibility {int(n) if n == int(n) else n} miles"
    except ValueError:
        return f"visibility {s}"


def _sky(clouds: list[dict]) -> str:
    if not clouds:
        return "sky clear"
    layers = []
    for c in clouds:
        cover = (c.get("cover") or "").upper()
        base = c.get("base")
        label = _SKY_COVER.get(cover, cover.lower())
        if cover in ("SKC", "CLR", "CAVOK", "NSC", "NCD"):
            return "sky clear"
        if base is not None:
            layers.append(f"{label} at {int(base):,} feet")
        else:
            layers.append(label)
    return "; ".join(layers) if layers else "sky clear"


def _altimeter_inhg(altim_hpa: Any) -> str:
    try:
        return f"{float(altim_hpa) * 0.02953:.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _c_to_f(c: Any) -> str:
    try:
        return str(round(float(c) * 9 / 5 + 32))
    except (TypeError, ValueError):
        return "?"


def _fmt_c(c: Any) -> str:
    """Format Celsius value: integer if whole number, else 1 decimal."""
    try:
        v = float(c)
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    except (TypeError, ValueError):
        return "?"


def _timestamp_to_label(ts: int) -> str:
    """Convert Unix timestamp to spoken time like '2 PM', '11 PM tonight'."""
    try:
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        hour = dt.strftime("%-I %p").lstrip("0")  # "2 PM"
        delta_days = (dt.date() - now.date()).days
        if delta_days == 0:
            return f"{hour} today"
        if delta_days == 1:
            return f"{hour} tomorrow"
        if delta_days == -1:
            return f"{hour} yesterday"
        return f"{hour} {dt.strftime('%A')}"
    except Exception:
        return "later"


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_metar(icao: str) -> dict | None:
    url = _METAR_URL.format(icao=urllib.parse.quote(icao.upper()))
    try:
        r = httpx.get(url, timeout=8.0, follow_redirects=True)
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None
    except Exception as e:
        logger.warning("METAR fetch failed for %r: %s", icao, e)
        return None


def _fetch_taf(icao: str) -> dict | None:
    url = _TAF_URL.format(icao=urllib.parse.quote(icao.upper()))
    try:
        r = httpx.get(url, timeout=8.0, follow_redirects=True)
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None
    except Exception as e:
        logger.warning("TAF fetch failed for %r: %s", icao, e)
        return None


def _fetch_pireps(icao: str) -> list[dict]:
    url = _PIREP_URL.format(icao=urllib.parse.quote(icao.upper()))
    try:
        r = httpx.get(url, timeout=8.0, follow_redirects=True)
        if r.status_code == 204:
            return []
        r.raise_for_status()
        return r.json() or []
    except Exception as e:
        logger.warning("PIREP fetch failed for %r: %s", icao, e)
        return []


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_metar(obs: dict) -> str:
    name = obs.get("name", obs.get("icaoId", "unknown"))
    flt_cat = obs.get("fltCat", "")
    wind = _wind(obs.get("wdir"), obs.get("wspd"), obs.get("wgst"))
    vis = _visibility(obs.get("visib"))
    sky = _sky(obs.get("clouds") or [])
    temp_c = obs.get("temp")
    dewp_c = obs.get("dewp")
    altim = _altimeter_inhg(obs.get("altim"))
    wx = obs.get("wxString") or ""

    parts = [f"{name}:"]
    if flt_cat:
        parts[0] += f" {flt_cat}."
    parts.append(f"{wind.capitalize()}.")
    if vis:
        parts.append(f"{vis.capitalize()}.")
    parts.append(f"{sky.capitalize()}.")
    if wx.strip():
        parts.append(f"{wx.strip()}.")
    if temp_c is not None and dewp_c is not None:
        parts.append(
            f"Temperature {_fmt_c(temp_c)}°C ({_c_to_f(temp_c)}°F), "
            f"dewpoint {_fmt_c(dewp_c)}°C ({_c_to_f(dewp_c)}°F)."
        )
    parts.append(f"Altimeter {altim}.")
    return " ".join(parts)


def _format_taf(taf: dict) -> str:
    name = taf.get("name", taf.get("icaoId", "unknown"))
    fcsts = taf.get("fcsts") or []

    parts = [f"{name} forecast:"]
    shown = 0
    for fcst in fcsts:
        if shown >= 4:
            break
        change = fcst.get("fcstChange")
        prob = fcst.get("probability")
        time_from = fcst.get("timeFrom")

        wind = _wind(fcst.get("wdir"), fcst.get("wspd"), fcst.get("wgst"))
        vis = _visibility(fcst.get("visib"))
        sky = _sky(fcst.get("clouds") or [])
        wx = fcst.get("wxString") or ""

        conditions = []
        conditions.append(wind)
        if vis:
            conditions.append(vis)
        conditions.append(sky)
        if wx.strip():
            conditions.append(wx.strip())
        cond_str = ", ".join(conditions) + "."

        if change is None:
            # Initial forecast period
            parts.append(cond_str.capitalize())
        elif change == "FM" and time_from:
            label = _timestamp_to_label(time_from)
            parts.append(f"From {label}: {cond_str}")
        elif change in ("BECMG", "TEMPO") and time_from:
            label = _timestamp_to_label(time_from)
            verb = "Becoming" if change == "BECMG" else "Temporarily"
            parts.append(f"{verb} after {label}: {cond_str}")
        elif prob and time_from:
            label = _timestamp_to_label(time_from)
            parts.append(f"{prob}% chance after {label}: {cond_str}")
        else:
            parts.append(cond_str.capitalize())
        shown += 1

    return " ".join(parts)


def _format_pireps(reports: list[dict], airport_name: str = "") -> str:
    if not reports:
        near = f" near {airport_name}" if airport_name else ""
        return f"No recent pilot reports{near} within 75 miles."

    # Filter to most informative (has turbulence, icing, or weather info)
    # Sort by most recent
    useful = [r for r in reports if r.get("tbInt1") or r.get("icgInt1") or r.get("wxString")]
    display = (useful or reports)[:3]

    near = f" near {airport_name}" if airport_name else ""
    count = len(reports)
    label = f"{count} pilot report{'s' if count != 1 else ''}{near} in the last 3 hours"
    summaries = []

    for r in display:
        flt_lvl = r.get("fltLvl")
        flt_type = r.get("fltLvlType", "")
        tb = r.get("tbInt1", "")
        icg = r.get("icgInt1", "")
        wx = r.get("wxString", "")

        if flt_type == "DURD" or flt_lvl == 0:
            alt = "during approach or departure"
        elif flt_lvl:
            alt = f"at {int(flt_lvl) * 100:,} feet"
        else:
            alt = ""

        notes = []
        if tb:
            notes.append(_TURB_INT.get(tb.upper(), tb.lower()))
        if icg and icg.upper() not in ("NEG", ""):
            notes.append(_ICG_INT.get(icg.upper(), icg.lower()))
        if wx and wx.strip():
            notes.append(wx.strip().lower())
        if not notes:
            notes.append("conditions reported")

        line = (f"{alt}: " if alt else "") + ", ".join(notes)
        summaries.append(line.capitalize())

    return f"{label}: {'. '.join(summaries)}."


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class AviationWeatherTool(Tool):
    name = "get_aviation_weather"
    description = (
        "Get aviation weather for an airport. "
        "report_type options: 'metar' for current conditions (default), "
        "'taf' for terminal forecast, 'pirep' for recent pilot reports, "
        "'full' for all three. Defaults to home airport when none specified."
    )
    requires_confirmation = False
    parameters = {
        "type": "object",
        "properties": {
            "airport": {
                "type": "string",
                "description": "ICAO airport code, e.g. 'KJYO', 'KIAD', 'EGLL'.",
            },
            "report_type": {
                "type": "string",
                "enum": ["metar", "taf", "pirep", "full"],
                "description": "metar=current obs, taf=forecast, pirep=pilot reports, full=all three.",
            },
        },
    }

    def __init__(
        self,
        home_airport: str | None = None,
        taf_airport: str | None = None,
    ) -> None:
        self._home = (home_airport or "").upper() or None
        self._taf = (taf_airport or "").upper() or self._home

    def run(self, airport: str = "", report_type: str = "metar", **kwargs) -> str:
        icao = (airport or "").strip().upper() or self._home
        if not icao:
            return "Please specify an airport ICAO code (e.g. 'KJYO')."

        rtype = (report_type or "metar").lower()

        if rtype == "metar":
            return self._do_metar(icao)
        if rtype == "taf":
            return self._do_taf(icao)
        if rtype == "pirep":
            return self._do_pirep(icao)
        if rtype == "full":
            parts = [self._do_metar(icao), self._do_taf(icao), self._do_pirep(icao)]
            return " ".join(p for p in parts if p)
        return f"Unknown report type '{report_type}'. Use metar, taf, pirep, or full."

    def _do_metar(self, icao: str) -> str:
        obs = _fetch_metar(icao)
        if obs is None:
            return f"No METAR available for {icao}."
        return _format_metar(obs)

    def _do_taf(self, icao: str) -> str:
        taf = _fetch_taf(icao)
        if taf is None and self._taf and self._taf != icao:
            # Fall back to configured TAF airport (e.g. KIAD for KJYO)
            logger.info("No TAF for %s — falling back to %s", icao, self._taf)
            taf = _fetch_taf(self._taf)
            if taf:
                return f"No TAF for {icao}. Using nearest TAF airport: " + _format_taf(taf)
        if taf is None:
            return f"No TAF available for {icao}."
        return _format_taf(taf)

    def _do_pirep(self, icao: str) -> str:
        reports = _fetch_pireps(icao)
        # Get a human-friendly station name from a cached METAR if possible
        obs = _fetch_metar(icao)
        name = obs.get("name", icao) if obs else icao
        return _format_pireps(reports, airport_name=name)
