from __future__ import annotations

from typing import TYPE_CHECKING

from coremind import ConfigError

if TYPE_CHECKING:
    from coremind.config.settings import STTConfig
    from coremind.stt.base import SpeechToText

SUPPORTED_PROVIDERS = ("whisper_local", "whisper_cpp", "mock")


def make_stt(cfg: "STTConfig") -> "SpeechToText":
    """Build the STT backend selected by ``stt.provider``.

    Raises STTError when the chosen backend's optional dependency is missing,
    and ConfigError for an unknown provider.
    """
    if cfg.provider == "whisper_local":
        from coremind.stt.whisper_local import WhisperLocalSTT

        return WhisperLocalSTT(
            model=cfg.model,
            language=cfg.language,
            compute_type=cfg.compute_type,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
            initial_prompt=cfg.initial_prompt,
            hotwords=cfg.hotwords,
        )
    if cfg.provider == "whisper_cpp":
        from coremind.stt.whisper_cpp import WhisperCppSTT

        wc = cfg.whisper_cpp
        return WhisperCppSTT(
            model=wc.model,
            language=cfg.language,
            model_path=wc.model_path,
            n_threads=wc.n_threads,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
            vad_model_path=wc.vad_model_path,
            initial_prompt=cfg.initial_prompt,
            hotwords=cfg.hotwords,
        )
    if cfg.provider == "mock":
        from coremind.stt.whisper_local import MockSTT

        return MockSTT()
    raise ConfigError(
        f"Unsupported stt.provider: {cfg.provider!r}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def stt_model_label(cfg: "STTConfig") -> str:
    """The model the selected provider actually uses, for status/log display."""
    if cfg.provider == "whisper_cpp":
        wc = cfg.whisper_cpp
        return wc.model_path or wc.model
    if cfg.provider == "mock":
        return "mock"
    return cfg.model
