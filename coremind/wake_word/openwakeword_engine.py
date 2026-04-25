from __future__ import annotations

from coremind.wake_word.base import WakeWordDetector


class OpenWakeWordDetector(WakeWordDetector):
    def listen_until_wake_word(self) -> None:
        raise NotImplementedError("Install openwakeword: pip install openwakeword")
