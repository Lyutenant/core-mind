from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Tool(ABC):
    name: str
    description: str
    requires_confirmation: bool = True

    @abstractmethod
    def run(self, **kwargs) -> str: ...


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())
