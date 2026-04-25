from __future__ import annotations

from coremind.tts.base import TextToSpeech


class OpenAITTS(TextToSpeech):
    def __init__(self, api_key: str, voice: str = "alloy") -> None:
        self.api_key = api_key
        self.voice = voice

    def synthesize(self, text: str, output_path: str) -> str:
        raise NotImplementedError("Install openai: pip install openai")
