from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

from coremind import AudioInputError, device_cache

logger = logging.getLogger(__name__)

# Names that should never be auto-picked as the mic: a UVC webcam usually also
# exposes an audio capture endpoint, which is rarely the mic the user wants.
_WEBCAM_NAME_TOKENS = ("cam", "camera", "webcam")

# Names that should never be auto-picked as the speaker: a USB microphone or
# mic array (e.g. the reSpeaker) often also exposes a playback endpoint, which
# is not the speaker the user wants — sending the chime/TTS there is silent.
_NON_SPEAKER_NAME_TOKENS = ("mic array", "4-mic", "microphone")


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float

    def is_input(self) -> bool:
        return self.max_input_channels > 0

    def is_output(self) -> bool:
        return self.max_output_channels > 0


def _query_all() -> list[AudioDevice]:
    try:
        import sounddevice as sd
    except ImportError as e:
        raise AudioInputError("sounddevice is not installed: pip install sounddevice") from e

    try:
        raw = sd.query_devices()
    except Exception as e:
        raise AudioInputError(f"Failed to query audio devices: {e}") from e

    if isinstance(raw, dict):
        raw = [raw]

    return [
        AudioDevice(
            index=i,
            name=d["name"],
            max_input_channels=int(d["max_input_channels"]),
            max_output_channels=int(d["max_output_channels"]),
            default_sample_rate=float(d["default_samplerate"]),
        )
        for i, d in enumerate(raw)
    ]


def list_input_devices() -> list[AudioDevice]:
    return [d for d in _query_all() if d.is_input()]


def list_output_devices() -> list[AudioDevice]:
    return [d for d in _query_all() if d.is_output()]


def get_default_input_device() -> Optional[AudioDevice]:
    try:
        import sounddevice as sd
        idx = sd.default.device[0]
        if idx < 0:
            return None
        return next((d for d in _query_all() if d.index == idx and d.is_input()), None)
    except Exception:
        return None


def get_default_output_device() -> Optional[AudioDevice]:
    try:
        import sounddevice as sd
        idx = sd.default.device[1]
        if idx < 0:
            return None
        return next((d for d in _query_all() if d.index == idx and d.is_output()), None)
    except Exception:
        return None


def coerce_device(value: Optional[Union[int, str]]) -> Optional[Union[int, str]]:
    """Normalize a device value: an all-digit string becomes an int index.

    Config from YAML preserves types, but env-var overrides and pydantic's
    smart-union can leave a numeric value as a string — which sounddevice would
    then treat as a *name* substring rather than an index. This collapses
    ``"1"`` → ``1`` while leaving genuine name substrings (``"USB Audio"``)
    untouched.
    """
    if isinstance(value, str):
        s = value.strip()
        return int(s) if s.isdigit() else s
    return value


def _name_present(name: str, devices: list[AudioDevice]) -> bool:
    needle = name.lower()
    return any(needle in d.name.lower() for d in devices)


def _has_token(name: str, tokens: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(t in n for t in tokens)


def _auto_select_name(
    devices: list[AudioDevice],
    cached: Optional[str],
    default: Optional[AudioDevice],
    avoid_tokens: tuple[str, ...],
) -> Optional[str]:
    if not devices:
        return None
    # Reuse a cached pick only if it's still present *and* isn't a device the
    # heuristic would now reject — so a stale cache pointing at a webcam/mic-array
    # (e.g. written before this avoidance existed) self-heals on the next boot.
    if cached and _name_present(cached, devices) and not _has_token(cached, avoid_tokens):
        return cached
    usb = [d for d in devices if "usb" in d.name.lower()]
    preferred = [d for d in usb if not _has_token(d.name, avoid_tokens)]
    if preferred:
        return preferred[0].name
    # No acceptable USB device — prefer a real default speaker over a rejected USB
    # endpoint (e.g. a mic array's silent playback output).
    if default is not None and not _has_token(default.name, avoid_tokens):
        return default.name
    if usb:
        return usb[0].name      # last resort: a rejected USB device beats nothing
    if default is not None:
        return default.name
    return devices[0].name


def auto_select_input_name() -> Optional[str]:
    """Pick a sensible default microphone, pinned by name. None = let PortAudio decide."""
    return _auto_select_name(
        list_input_devices(),
        device_cache.get(device_cache.INPUT_KEY),
        get_default_input_device(),
        avoid_tokens=_WEBCAM_NAME_TOKENS,
    )


def auto_select_output_name() -> Optional[str]:
    """Pick a sensible default speaker, pinned by name. None = let PortAudio decide."""
    return _auto_select_name(
        list_output_devices(),
        device_cache.get(device_cache.OUTPUT_KEY),
        get_default_output_device(),
        avoid_tokens=_NON_SPEAKER_NAME_TOKENS,
    )


def resolve_input_device(
    configured: Optional[Union[int, str]],
) -> Optional[Union[int, str]]:
    """Resolve the configured mic value to a device, auto-selecting if unset.

    Explicit int (index) or name substring is returned as-is. ``None`` or
    ``"auto"`` triggers name-based auto-selection (logged once, persisted).
    """
    if configured not in (None, "auto"):
        return coerce_device(configured)
    name = auto_select_input_name()
    if name is not None:
        logger.info("Auto-selected microphone: '%s'", name)
        device_cache.remember(device_cache.INPUT_KEY, name)
    return name


def resolve_output_device(
    configured: Optional[Union[int, str]],
) -> Optional[Union[int, str]]:
    """Resolve the configured speaker value to a device, auto-selecting if unset."""
    if configured not in (None, "auto"):
        return coerce_device(configured)
    name = auto_select_output_name()
    if name is not None:
        logger.info("Auto-selected speaker: '%s'", name)
        device_cache.remember(device_cache.OUTPUT_KEY, name)
    return name
