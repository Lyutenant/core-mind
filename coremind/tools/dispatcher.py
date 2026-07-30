from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coremind.tools.registry import Tool

if TYPE_CHECKING:
    from coremind.tools.mcp_manager import MCPManager

logger = logging.getLogger(__name__)

_BUILT_IN_FACTORIES: dict[str, type[Tool]] = {}

# MCP tools that are internal plumbing for a Hub-side built-in and must NEVER be
# advertised to the LLM directly — even if the wrapping built-in isn't registered
# (e.g. the Node camera is enabled but `look` is off / has no vision model). The LLM
# calling these raw would dump large binary payloads (base64 images) into the context.
_ALWAYS_HIDDEN_MCP_TOOLS: frozenset[str] = frozenset({"capture_image"})


def _load_built_in_factories() -> dict[str, type[Tool]]:
    from coremind.tools.built_in.airport_tool import AirportTool
    from coremind.tools.built_in.aviation_weather_tool import AviationWeatherTool
    from coremind.tools.built_in.time_tool import TimeTool
    from coremind.tools.built_in.vision_tool import LookAtSceneTool
    from coremind.tools.built_in.weather_tool import WeatherTool
    return {
        "time": TimeTool,
        "weather": WeatherTool,
        "aviation_weather": AviationWeatherTool,
        "airport": AirportTool,
        # 'look' needs constructor args (dispatcher + vision client) — see register_built_ins.
        "look": LookAtSceneTool,
    }


class ToolDispatcher:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._mcp_manager: MCPManager | None = None

    def set_mcp_manager(self, mcp: MCPManager) -> None:
        self._mcp_manager = mcp

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_built_ins(
        self,
        names: list[str],
        *,
        default_timezone: str | None = None,
        user_location: str | None = None,
        home_airport: str | None = None,
        taf_airport: str | None = None,
        vision_client=None,
    ) -> None:
        factories = _load_built_in_factories()
        for name in names:
            if name not in factories:
                logger.warning("Unknown built-in tool: %r (available: %s)", name, list(factories))
                continue
            if name == "time":
                from coremind.tools.built_in.time_tool import TimeTool
                self.register(TimeTool(timezone=default_timezone))
            elif name == "weather":
                from coremind.tools.built_in.weather_tool import WeatherTool
                self.register(WeatherTool(default_location=user_location))
            elif name == "aviation_weather":
                from coremind.tools.built_in.aviation_weather_tool import AviationWeatherTool
                self.register(AviationWeatherTool(home_airport=home_airport, taf_airport=taf_airport))
            elif name == "look":
                if vision_client is None:
                    logger.warning(
                        "Built-in tool 'look' requires ollama.vision_model to be set — skipping."
                    )
                    continue
                from coremind.tools.built_in.vision_tool import LookAtSceneTool
                self.register(LookAtSceneTool(dispatcher=self, vision_client=vision_client))
            else:
                self.register(factories[name]())

    def _hidden_mcp_tools(self) -> set[str]:
        """MCP tool names to hide from the LLM.

        Two sources: always-hidden internal tools (e.g. the Node's `capture_image`,
        hidden even when its `look` wrapper isn't registered), plus any MCP tools a
        registered built-in declares it wraps. The LLM should never call these raw —
        the `look` tool fetches `capture_image` itself and returns text, so a direct
        call would only dump base64 into the conversation.
        """
        hidden: set[str] = set(_ALWAYS_HIDDEN_MCP_TOOLS)
        for t in self._tools.values():
            hidden.update(getattr(t, "wraps_mcp_tools", ()) or ())
        return hidden

    def get_tool_definitions(self) -> list[dict]:
        schemas = [t.to_ollama_schema() for t in self._tools.values()]
        if self._mcp_manager:
            hidden = self._hidden_mcp_tools()
            # MCP schemas are already fetched at startup; retrieve synchronously.
            schemas.extend(
                s for s in self._mcp_manager._schemas  # noqa: SLF001
                if s["function"]["name"] not in hidden
            )
        return schemas

    async def execute_async(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call, routing to built-in or MCP as needed."""
        tool = self._tools.get(tool_name)
        if tool is not None:
            return await self._run_tool_async(tool, arguments)
        if self._mcp_manager and tool_name in self._mcp_manager.tool_to_server:
            return await self._mcp_manager.call_tool(tool_name, arguments)
        logger.warning("Tool not found: %r", tool_name)
        return f"Unknown tool: {tool_name}"

    async def _run_tool_async(self, tool: Tool, arguments: dict) -> str:
        if tool.requires_confirmation:
            logger.warning("Tool %r requires confirmation — blocked", tool.name)
            return f"Tool '{tool.name}' requires explicit confirmation before it can run."
        try:
            result = await tool.run_async(**arguments)
            logger.debug("Tool %r(%s) → %r", tool.name, arguments, result[:80] if result else "")
            return result
        except TypeError as e:
            logger.warning("Tool %r called with bad arguments %s: %s", tool.name, arguments, e)
            return f"Tool '{tool.name}' received unexpected arguments: {e}"
        except Exception as e:
            logger.error("Tool %r raised: %s", tool.name, e)
            return f"Tool '{tool.name}' failed: {e}"

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
