from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coremind.tools.registry import Tool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_BUILT_IN_FACTORIES: dict[str, type[Tool]] = {}


def _load_built_in_factories() -> dict[str, type[Tool]]:
    from coremind.tools.built_in.aviation_weather_tool import AviationWeatherTool
    from coremind.tools.built_in.time_tool import TimeTool
    from coremind.tools.built_in.weather_tool import WeatherTool
    return {"time": TimeTool, "weather": WeatherTool, "aviation_weather": AviationWeatherTool}


class ToolDispatcher:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_built_ins(
        self,
        names: list[str],
        *,
        default_timezone: str | None = None,
        home_airport: str | None = None,
        taf_airport: str | None = None,
    ) -> None:
        factories = _load_built_in_factories()
        for name in names:
            if name not in factories:
                logger.warning("Unknown built-in tool: %r (available: %s)", name, list(factories))
                continue
            if name == "time":
                from coremind.tools.built_in.time_tool import TimeTool
                self.register(TimeTool(timezone=default_timezone))
            elif name == "aviation_weather":
                from coremind.tools.built_in.aviation_weather_tool import AviationWeatherTool
                self.register(AviationWeatherTool(home_airport=home_airport, taf_airport=taf_airport))
            else:
                self.register(factories[name]())

    def get_tool_definitions(self) -> list[dict]:
        return [t.to_ollama_schema() for t in self._tools.values()]

    def execute(self, tool_name: str, arguments: dict) -> str:
        tool = self._tools.get(tool_name)
        if tool is None:
            logger.warning("Tool not found: %r", tool_name)
            return f"Unknown tool: {tool_name}"
        if tool.requires_confirmation:
            logger.warning("Tool %r requires confirmation — blocked", tool_name)
            return f"Tool '{tool_name}' requires explicit confirmation before it can run."
        try:
            result = tool.run(**arguments)
            logger.debug("Tool %r(%s) → %r", tool_name, arguments, result[:80] if result else "")
            return result
        except TypeError as e:
            logger.warning("Tool %r called with bad arguments %s: %s", tool_name, arguments, e)
            return f"Tool '{tool_name}' received unexpected arguments: {e}"
        except Exception as e:
            logger.error("Tool %r raised: %s", tool_name, e)
            return f"Tool '{tool_name}' failed: {e}"
