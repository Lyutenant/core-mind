from __future__ import annotations

from coremind.wake_word.base import WakeWordDetector


class DummyWakeWordDetector(WakeWordDetector):
    def listen_until_wake_word(self) -> None:
        return
