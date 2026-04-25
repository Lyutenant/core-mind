from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionMemory:
    max_turns: int = 10
    _turns: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})
        if len(self._turns) > self.max_turns * 2:
            self._turns = self._turns[-(self.max_turns * 2):]

    def get_messages(self) -> list[dict]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()
