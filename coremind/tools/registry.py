from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Tool(ABC):
    name: str
    description: str
    requires_confirmation: bool = False
    # Full JSON Schema object for the tool's parameters.
    # Example: {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
    # Leave as empty dict for tools that take no arguments.
    parameters: dict = {}

    @abstractmethod
    def run(self, **kwargs) -> str: ...

    async def run_async(self, **kwargs) -> str:
        """Async entry point. Defaults to the synchronous ``run``.

        Tools that need to await I/O (e.g. fetching a frame from the Node over MCP)
        override this instead of ``run``.
        """
        return self.run(**kwargs)

    def to_ollama_schema(self) -> dict:
        params = dict(self.parameters)
        if "type" not in params:
            params["type"] = "object"
        if "properties" not in params:
            params["properties"] = {}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())
