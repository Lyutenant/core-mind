from __future__ import annotations

from coremind.wake_word.base import WakeWordDetector


class DummyWakeWordDetector(WakeWordDetector):
    """Push-to-talk trigger: blocks until the user presses Enter."""

    @property
    def trigger_prompt(self) -> str:
        return "Press Enter to speak..."

    def listen_until_wake_word(self) -> None:
        input()
