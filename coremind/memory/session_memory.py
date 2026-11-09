from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionMemory:
    """Last-N-turns conversation memory in Ollama chat-message form.

    A *turn* starts at a user message and runs until the next one. A turn that used
    tools is stored as the full exchange — the assistant message carrying its
    ``tool_calls``, one ``tool`` message per result, then the spoken reply — not just
    the reply text. If the history showed only "turn off the light" → "done", with the
    tool call stripped out, a small model learns that pattern and starts answering
    device commands from memory without calling anything (observed with qwen3:8b +
    Home Assistant). Keeping the calls in view keeps the model calling.
    """

    max_turns: int = 10
    # Tool results are kept short in memory: enough to show what happened, not the
    # whole payload (a forecast or an HA JSON blob can run to hundreds of chars).
    max_tool_result_chars: int = 400
    _turns: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})
        self._truncate()

    def add_turn(
        self,
        user_text: str,
        response_text: str,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Record one complete turn, including any tool calls it made.

        ``tool_calls`` is the Hub's flat per-turn list of
        ``{"name", "args", "result"}`` dicts (the same shape persisted in a session
        round). They are replayed as a single assistant tool-call message followed by
        one ``tool`` message per result, which is valid Ollama history even when the
        live turn took several rounds.
        """
        self._turns.append({"role": "user", "content": user_text})
        if tool_calls:
            self._turns.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": tc.get("name", ""), "arguments": tc.get("args") or {}}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                self._turns.append({
                    "role": "tool",
                    "tool_name": tc.get("name", ""),
                    "content": self._clip(str(tc.get("result") or "")),
                })
        self._turns.append({"role": "assistant", "content": response_text})
        self._truncate()

    def get_messages(self) -> list[dict]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    # ------------------------------------------------------------------

    def _clip(self, text: str) -> str:
        limit = self.max_tool_result_chars
        if limit and len(text) > limit:
            return text[:limit] + "…"
        return text

    def _truncate(self) -> None:
        """Keep the last ``max_turns`` turns, cutting only at a user message so a
        tool exchange is never left headless."""
        starts = [i for i, m in enumerate(self._turns) if m.get("role") == "user"]
        if len(starts) > self.max_turns:
            self._turns = self._turns[starts[-self.max_turns]:]
        elif not starts and len(self._turns) > self.max_turns * 2:
            # No user messages at all (odd caller) — fall back to a message cap.
            self._turns = self._turns[-(self.max_turns * 2):]
