"""Unit tests for built-in tools (no network calls)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from coremind.tools.built_in.aviation_weather_tool import (
    AviationWeatherTool,
    _format_metar,
    _format_pireps,
    _format_taf,
    _wind,
    _visibility,
    _sky,
    _altimeter_inhg,
)

# ---------------------------------------------------------------------------
# Sample data captured from live API
# ---------------------------------------------------------------------------

SAMPLE_METAR = {
    "icaoId": "KJYO",
    "name": "Leesburg Exec, VA, US",
    "fltCat": "VFR",
    "wdir": 320,
    "wspd": 7,
    "wgst": None,
    "visib": "10+",
    "clouds": [],
    "wxString": None,
    "temp": 28,
    "dewp": 13,
    "altim": 1022.4,
    "rawOb": "METAR KJYO 041735Z AUTO 32007KT 10SM CLR 28/13 A3019 RMK AO2",
}

SAMPLE_METAR_IFR = {
    "icaoId": "KDCA",
    "name": "Reagan National, VA, US",
    "fltCat": "IFR",
    "wdir": 0,
    "wspd": 0,
    "wgst": None,
    "visib": "1.5",
    "clouds": [
        {"cover": "OVC", "base": 400},
    ],
    "wxString": "Fog",
    "temp": 12,
    "dewp": 12,
    "altim": 1013.2,
}

SAMPLE_TAF = {
    "icaoId": "KIAD",
    "name": "Washington/Dulles Intl",
    "rawTAF": "TAF KIAD 041720Z 0418/0524 33008KT P6SM SKC",
    "fcsts": [
        {
            "timeFrom": 1780596000,
            "timeTo": 1780614000,
            "fcstChange": None,
            "probability": None,
            "wdir": 330,
            "wspd": 8,
            "wgst": None,
            "visib": "6+",
            "clouds": [{"cover": "SKC", "base": None}],
            "wxString": None,
        },
        {
            "timeFrom": 1780614000,
            "timeTo": 1780668000,
            "fcstChange": "FM",
            "probability": None,
            "wdir": 0,
            "wspd": 0,
            "wgst": None,
            "visib": "6+",
            "clouds": [{"cover": "SKC", "base": None}],
            "wxString": None,
        },
    ],
}

SAMPLE_PIREPS = [
    {
        "fltLvl": 350,
        "fltLvlType": "OTHER",
        "acType": "B738",
        "tbInt1": "LGT",
        "icgInt1": "",
        "wxString": "",
        "rawOb": "SMQ UA /OV SBJ135010/TM 1646/FL350/TP B738/TB CONS LGT CHOP",
    },
    {
        "fltLvl": 120,
        "fltLvlType": "OTHER",
        "acType": "E55P",
        "tbInt1": "NEG",
        "icgInt1": "NEGclr",
        "wxString": "",
        "rawOb": "AVP UA /OV AVP/TM 1629/FL120/TP E55P/SK SKC",
    },
]

SAMPLE_PIREPS_DURD = [
    {
        "fltLvl": 0,
        "fltLvlType": "DURD",
        "acType": "SR22",
        "tbInt1": "",
        "icgInt1": "",
        "wxString": "10KT tailwind",
        "rawOb": "CDW UA /OV CDW/TM 1602/FLDURD/TP SR22/RM 10KT TAILWIND",
    },
]


# ---------------------------------------------------------------------------
# Decoder unit tests
# ---------------------------------------------------------------------------

def test_wind_calm():
    assert _wind(0, 0) == "calm winds"


def test_wind_normal():
    result = _wind(320, 7)
    assert "320" in result
    assert "7" in result
    assert "knots" in result


def test_wind_gusts():
    result = _wind(280, 15, 25)
    assert "gusting 25 knots" in result


def test_visibility_plus():
    assert "greater than 10" in _visibility("10+")


def test_visibility_numeric():
    assert "3 miles" in _visibility("3")


def test_visibility_low():
    result = _visibility("0.5")
    assert "0.5 miles" in result or "miles" in result


def test_sky_clear_skc():
    assert "sky clear" in _sky([{"cover": "SKC", "base": None}])


def test_sky_clear_empty():
    assert "sky clear" in _sky([])


def test_sky_broken_ceiling():
    result = _sky([{"cover": "BKN", "base": 1500}])
    assert "broken ceiling" in result
    assert "1,500 feet" in result


def test_sky_overcast():
    result = _sky([{"cover": "OVC", "base": 800}])
    assert "overcast" in result
    assert "800 feet" in result


def test_altimeter_conversion():
    result = _altimeter_inhg(1022.4)
    assert result == "30.19"


def test_altimeter_standard():
    result = _altimeter_inhg(1013.25)
    assert result.startswith("29.9")


# ---------------------------------------------------------------------------
# METAR formatting
# ---------------------------------------------------------------------------

def test_format_metar_vfr():
    text = _format_metar(SAMPLE_METAR)
    assert "Leesburg Exec" in text
    assert "VFR" in text
    assert "320" in text
    assert "7 knots" in text
    assert "greater than 10" in text
    assert "sky clear" in text.lower()
    assert "28" in text
    assert "30.19" in text


def test_format_metar_ifr():
    text = _format_metar(SAMPLE_METAR_IFR)
    assert "IFR" in text
    assert "calm winds" in text.lower()
    assert "overcast" in text.lower()
    assert "400 feet" in text
    assert "Fog" in text


# ---------------------------------------------------------------------------
# TAF formatting
# ---------------------------------------------------------------------------

def test_format_taf_contains_airport():
    text = _format_taf(SAMPLE_TAF)
    assert "Dulles" in text or "KIAD" in text


def test_format_taf_initial_conditions():
    text = _format_taf(SAMPLE_TAF)
    assert "330" in text
    assert "8 knots" in text
    assert "sky clear" in text.lower()


def test_format_taf_from_group():
    text = _format_taf(SAMPLE_TAF)
    assert "From" in text
    assert "calm winds" in text.lower()


# ---------------------------------------------------------------------------
# PIREP formatting
# ---------------------------------------------------------------------------

def test_format_pireps_empty():
    text = _format_pireps([], airport_name="Leesburg")
    assert "No recent pilot reports" in text
    assert "75 miles" in text


def test_format_pireps_with_data():
    text = _format_pireps(SAMPLE_PIREPS, airport_name="Leesburg")
    assert "35,000 feet" in text
    assert "light turbulence" in text


def test_format_pireps_durd():
    text = _format_pireps(SAMPLE_PIREPS_DURD)
    assert "approach or departure" in text


def test_format_pireps_smooth():
    text = _format_pireps(SAMPLE_PIREPS)
    assert "smooth" in text


# ---------------------------------------------------------------------------
# AviationWeatherTool — mocked HTTP
# ---------------------------------------------------------------------------

def _mock_get_factory(metar=None, taf=None, pireps=None):
    """Return a mock httpx.get that serves pre-canned responses."""
    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "metar" in url:
            resp.status_code = 200 if metar is not None else 204
            resp.json.return_value = [metar] if metar else []
        elif "taf" in url:
            resp.status_code = 200 if taf is not None else 204
            resp.json.return_value = [taf] if taf else []
        elif "pirep" in url:
            if pireps is None:
                resp.status_code = 204
                resp.json.return_value = []
            else:
                resp.status_code = 200
                resp.json.return_value = pireps
        else:
            resp.status_code = 200
            resp.json.return_value = []
        return resp
    return mock_get


@patch("coremind.tools.built_in.aviation_weather_tool.httpx.get")
def test_tool_metar(mock_get):
    mock_get.side_effect = _mock_get_factory(metar=SAMPLE_METAR)
    tool = AviationWeatherTool(home_airport="KJYO", taf_airport="KIAD")
    result = tool.run(report_type="metar")
    assert "Leesburg" in result
    assert "VFR" in result


@patch("coremind.tools.built_in.aviation_weather_tool.httpx.get")
def test_tool_taf(mock_get):
    mock_get.side_effect = _mock_get_factory(taf=SAMPLE_TAF)
    tool = AviationWeatherTool(home_airport="KJYO", taf_airport="KIAD")
    result = tool.run(report_type="taf")
    assert "Dulles" in result or "KIAD" in result


@patch("coremind.tools.built_in.aviation_weather_tool.httpx.get")
def test_tool_pirep_empty(mock_get):
    mock_get.side_effect = _mock_get_factory(metar=SAMPLE_METAR, pireps=None)
    tool = AviationWeatherTool(home_airport="KJYO", taf_airport="KIAD")
    result = tool.run(report_type="pirep")
    assert "No recent pilot reports" in result


@patch("coremind.tools.built_in.aviation_weather_tool.httpx.get")
def test_tool_pirep_with_data(mock_get):
    mock_get.side_effect = _mock_get_factory(metar=SAMPLE_METAR, pireps=SAMPLE_PIREPS)
    tool = AviationWeatherTool(home_airport="KJYO", taf_airport="KIAD")
    result = tool.run(report_type="pirep")
    assert "35,000 feet" in result
    assert "light turbulence" in result


@patch("coremind.tools.built_in.aviation_weather_tool.httpx.get")
def test_tool_full(mock_get):
    mock_get.side_effect = _mock_get_factory(
        metar=SAMPLE_METAR, taf=SAMPLE_TAF, pireps=SAMPLE_PIREPS
    )
    tool = AviationWeatherTool(home_airport="KJYO", taf_airport="KIAD")
    result = tool.run(report_type="full")
    assert "VFR" in result
    assert "Dulles" in result or "forecast" in result
    assert "pilot report" in result.lower()


@patch("coremind.tools.built_in.aviation_weather_tool.httpx.get")
def test_tool_taf_fallback(mock_get):
    """When home airport has no TAF, fall back to taf_airport."""
    def selective_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "metar" in url:
            resp.status_code = 200
            resp.json.return_value = [SAMPLE_METAR]
        elif "taf" in url and "KJYO" in url.upper():
            resp.status_code = 200
            resp.json.return_value = []  # KJYO has no TAF
        elif "taf" in url and "KIAD" in url.upper():
            resp.status_code = 200
            resp.json.return_value = [SAMPLE_TAF]
        else:
            resp.status_code = 204
            resp.json.return_value = []
        return resp

    mock_get.side_effect = selective_get
    tool = AviationWeatherTool(home_airport="KJYO", taf_airport="KIAD")
    result = tool.run(airport="KJYO", report_type="taf")
    assert "KIAD" in result or "Dulles" in result or "nearest TAF" in result.lower()


def test_tool_no_airport_no_home():
    tool = AviationWeatherTool()
    result = tool.run(report_type="metar")
    assert "Please specify" in result


def test_tool_uses_home_airport():
    tool = AviationWeatherTool(home_airport="KJYO", taf_airport="KIAD")
    # default airport should be KJYO when none passed
    assert tool._home == "KJYO"
    assert tool._taf == "KIAD"


def test_tool_unknown_report_type():
    with patch("coremind.tools.built_in.aviation_weather_tool.httpx.get") as mock_get:
        mock_get.side_effect = _mock_get_factory(metar=SAMPLE_METAR)
        tool = AviationWeatherTool(home_airport="KJYO")
        result = tool.run(report_type="sigmet")
        assert "Unknown report type" in result


# ---------------------------------------------------------------------------
# WeatherTool — default location + unknown-location coaching (mocked HTTP)
# ---------------------------------------------------------------------------

from coremind.tools.built_in.weather_tool import WeatherTool  # noqa: E402

SAMPLE_WTTR = {
    "current_condition": [
        {
            "weatherDesc": [{"value": "Sunny"}],
            "temp_C": "22",
            "temp_F": "72",
            "FeelsLikeC": "22",
            "FeelsLikeF": "72",
            "humidity": "40",
            "windspeedKmph": "10",
            "winddir16Point": "NW",
            "uvIndex": "4",
        }
    ],
    "weather": [],
}


def _mock_wttr_ok():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = SAMPLE_WTTR
    return resp


def _mock_wttr_error(status_code: int, body: str = ""):
    import httpx

    resp = MagicMock()
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=MagicMock(),
        response=MagicMock(status_code=status_code, text=body),
    )
    return resp


def _mock_wttr_not_found():
    # wttr.in answers unknown names with 500 + "location not found" body.
    return _mock_wttr_error(500, "location not found: location not found")


def test_weather_schema_location_optional_with_default():
    # With a configured default, the LLM may omit `location` and the schema says so.
    tool = WeatherTool(default_location="Newark, NJ")
    assert "location" not in tool.parameters.get("required", [])
    assert "Omit" in tool.parameters["properties"]["location"]["description"]
    assert "Defaults to the user's location" in tool.description


def test_weather_schema_location_required_without_default():
    # Without a default the tool can't fulfill an omitted location — the schema
    # must not advertise one, or the model makes unusable calls.
    tool = WeatherTool()
    assert tool.parameters["required"] == ["location"]
    assert "Omit" not in tool.parameters["properties"]["location"]["description"]
    assert "Defaults to the user's location" not in tool.description


@patch("coremind.tools.built_in.weather_tool.httpx.get")
def test_weather_uses_default_location_when_omitted(mock_get):
    mock_get.return_value = _mock_wttr_ok()
    tool = WeatherTool(default_location="Newark, NJ")
    result = tool.run()
    assert "Newark, NJ" in result
    assert "Sunny" in result
    assert "Newark%2C%20NJ" in mock_get.call_args[0][0]


@patch("coremind.tools.built_in.weather_tool.httpx.get")
def test_weather_explicit_location_wins_over_default(mock_get):
    mock_get.return_value = _mock_wttr_ok()
    tool = WeatherTool(default_location="Newark, NJ")
    tool.run(location="London")
    assert "London" in mock_get.call_args[0][0]


def test_weather_no_location_no_default():
    tool = WeatherTool()
    assert "Please specify" in tool.run()


@patch("coremind.tools.built_in.weather_tool.httpx.get")
def test_weather_unknown_location_coaches_retry_with_default(mock_get):
    # A garbled STT token ("Wazard") must coach the model back into the tool
    # loop, not dead-end the turn.
    mock_get.return_value = _mock_wttr_not_found()
    tool = WeatherTool(default_location="Newark, NJ")
    result = tool.run(location="Wazard")
    assert "doesn't recognize 'Wazard'" in result
    assert "without the 'location' parameter" in result


@patch("coremind.tools.built_in.weather_tool.httpx.get")
def test_weather_unknown_location_no_default_asks_user(mock_get):
    mock_get.return_value = _mock_wttr_not_found()
    tool = WeatherTool()
    result = tool.run(location="Wazard")
    assert "doesn't recognize 'Wazard'" in result
    assert "which city" in result


@patch("coremind.tools.built_in.weather_tool.httpx.get")
def test_weather_unknown_default_location_reports_config_problem(mock_get):
    # If the *configured* location itself is rejected, retrying would loop —
    # point at the config instead.
    mock_get.return_value = _mock_wttr_not_found()
    tool = WeatherTool(default_location="Atlantis")
    result = tool.run()
    assert "user_location" in result
    assert "without the 'location' parameter" not in result


def test_dispatcher_wires_user_location_into_weather_tool():
    from coremind.tools.dispatcher import ToolDispatcher

    d = ToolDispatcher()
    d.register_built_ins(["weather"], user_location="Newark, NJ")
    assert d._tools["get_weather"]._default == "Newark, NJ"


@patch("coremind.tools.built_in.weather_tool.httpx.get")
def test_weather_404_also_coaches(mock_get):
    mock_get.return_value = _mock_wttr_error(404, "unknown location")
    tool = WeatherTool(default_location="Newark, NJ")
    result = tool.run(location="Wazard")
    assert "doesn't recognize 'Wazard'" in result


@patch("coremind.tools.built_in.weather_tool.httpx.get")
def test_weather_transient_failure_is_not_a_location_problem(mock_get):
    # Rate limits / outages must never trigger the location-coaching path —
    # a retry-then-"fix your config" answer would be confidently wrong.
    for status in (429, 502, 503):
        mock_get.return_value = _mock_wttr_error(status, "service unavailable")
        tool = WeatherTool(default_location="Newark, NJ")
        result = tool.run(location="London")
        assert "temporarily unavailable" in result
        assert str(status) in result
        assert "doesn't recognize" not in result
        assert "user_location" not in result


@patch("coremind.tools.built_in.weather_tool.httpx.get")
def test_weather_500_without_not_found_body_is_service_error(mock_get):
    mock_get.return_value = _mock_wttr_error(500, "internal server error")
    tool = WeatherTool(default_location="Newark, NJ")
    result = tool.run(location="London")
    assert "temporarily unavailable" in result
    assert "doesn't recognize" not in result
