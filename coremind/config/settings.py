from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from coremind import ConfigError

logger = logging.getLogger(__name__)

_ENV_PREFIX = "COREMIND_"
_ENV_DELIMITER = "__"


class AppConfig(BaseModel):
    name: str = "CoreMind"
    log_level: str = "INFO"
    user_location: Optional[str] = None   # e.g. "San Francisco, CA"
    user_timezone: Optional[str] = None   # IANA tz name e.g. "America/Los_Angeles"
    home_airport: Optional[str] = None    # ICAO code e.g. "KJYO"
    taf_airport: Optional[str] = None     # nearest airport with TAF e.g. "KIAD"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


class RuntimeConfig(BaseModel):
    mode: str = "push_to_talk"
    require_confirmation_for_actions: bool = True
    follow_up_seconds: float = 5.0          # seconds to wait for speech onset before returning to wake word; 0.0 to disable
    follow_up_min_words: int = 2             # discard follow-up if transcript is shorter; helps filter background noise
    post_response_cooldown_seconds: float = 1.0  # silence after playback before mic reopens (suppresses echo)


class AudioConfig(BaseModel):
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 16000
    channels: int = 1
    record_seconds: int = 6


class STTConfig(BaseModel):
    provider: str = "whisper_local"
    model: str = "small"
    language: str = "en"
    # Accuracy/quality knobs (whisper_local only).
    compute_type: str = "int8"          # int8 | int8_float32 | float32 — higher = more accurate, slower
    beam_size: int = 5                  # higher = more accurate, slower
    vad_filter: bool = False            # Silero VAD: strip silence/noise before decoding
    # Decoder biasing toward your vocabulary/jargon — the practical substitute
    # for fine-tuning on your own voice. Both are optional free text.
    initial_prompt: Optional[str] = None  # context sentence(s) fed to the decoder
    hotwords: Optional[str] = None        # comma/space-separated words to bias toward


class TTSConfig(BaseModel):
    provider: str = "piper_local"
    model_path: Optional[str] = None  # piper_local: path to .onnx voice model file
    voice: Optional[str] = None       # espeak: voice variant (e.g. "en", "en+f3")

    @field_validator("model_path")
    @classmethod
    def expand_model_path(cls, v: Optional[str]) -> Optional[str]:
        return str(Path(v).expanduser()) if v is not None else None


class BrainConfig(BaseModel):
    provider: str = "ollama"
    timeout_seconds: int = 60
    allow_mock_fallback: bool = False


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    no_think: bool = False
    options: dict = {}


class OpenClawConfig(BaseModel):
    base_url: Optional[str] = None
    agent: str = "coremind"


class VADConfig(BaseModel):
    enabled: bool = True
    energy_threshold: float = 0.01
    silence_seconds: float = 1.2
    max_record_seconds: float = 20.0
    min_speech_seconds: float = 0.3


class WakeWordConfig(BaseModel):
    enabled: bool = False
    provider: str = "dummy"  # dummy or openwakeword
    model: str = "hey_jarvis_v0.1"  # built-in name or path to .onnx
    threshold: float = 0.5
    inference_framework: str = "onnx"  # "onnx" (Pi-compatible) or "tflite"


class MemoryConfig(BaseModel):
    enabled: bool = True
    max_turns: int = 10


class RemoteBrainConfig(BaseModel):
    enabled: bool = False
    url: str = ""           # e.g. http://100.x.x.x:8765
    timeout_seconds: float = 90.0


class MCPServerConfig(BaseModel):
    name: str
    transport: str          # "stdio" | "http"
    command: Optional[list[str]] = None   # stdio only: command + args to spawn
    url: Optional[str] = None             # http only: base URL e.g. http://pi:8767


class ToolsConfig(BaseModel):
    enabled: bool = True
    # Which built-in tools to register. Available: "time", "weather", "aviation_weather".
    built_in: list[str] = ["time", "weather", "airport"]
    # External MCP servers to connect to at Hub startup.
    mcp_servers: list[MCPServerConfig] = []


class NodeMCPConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8767
    music_dir: str = "~/Music"
    catalog_path: str = "~/.coremind/music-catalog.json"
    atc_catalog_path: str = "~/.coremind/atc-catalog.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_nested_delimiter=_ENV_DELIMITER,
        case_sensitive=False,
    )

    # Device role — determines which features are active on this machine.
    #   hub:        Mac Mini: runs STT + LLM + TTS for Nodes
    #   node:       Raspberry Pi: audio terminal, sends audio to Hub
    #   standalone: all-in-one, no Hub needed
    mode: Literal["hub", "node", "standalone"] = "hub"

    app: AppConfig = AppConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    audio: AudioConfig = AudioConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    brain: BrainConfig = BrainConfig()
    ollama: OllamaConfig = OllamaConfig()
    openclaw: OpenClawConfig = OpenClawConfig()
    memory: MemoryConfig = MemoryConfig()
    vad: VADConfig = VADConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    remote_brain: RemoteBrainConfig = RemoteBrainConfig()
    tools: ToolsConfig = ToolsConfig()
    node_mcp: NodeMCPConfig = NodeMCPConfig()


def _coerce(val: str) -> object:
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    if val.lower() == "null":
        return None
    try:
        return int(val)
    except ValueError:
        pass
    return val


def _env_overrides() -> dict:
    """Parse COREMIND_ env vars into a nested dict that can override YAML."""
    result: dict = {}
    for raw_key, val in os.environ.items():
        if not raw_key.upper().startswith(_ENV_PREFIX):
            continue
        parts = raw_key[len(_ENV_PREFIX):].lower().split(_ENV_DELIMITER.lower())
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(val)
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_merge_inplace(base, override: dict) -> None:
    """Merge ``override`` into ``base`` in place.

    Mutating in place (rather than rebuilding) is what lets a ruamel.yaml
    ``CommentedMap`` keep the comment tokens attached to its existing keys:
    reassigning a value preserves the comment, recursing preserves nested
    comments, and a null value overwrites (so clearing a field still works).
    Works equally on a plain ``dict``.
    """
    for k, v in override.items():
        existing = base.get(k) if hasattr(base, "get") else None
        if isinstance(existing, dict) and isinstance(v, dict):
            _deep_merge_inplace(existing, v)
        else:
            base[k] = v


def merge_config_text(existing_text: str, incoming: dict) -> str:
    """Merge a partial config payload onto the existing config YAML text.

    The Hub dashboard only POSTs the sections it renders; writing that partial
    object straight to disk would wipe every unmanaged section (``tools``,
    ``node_mcp``, …) and unmanaged keys (``app.home_airport`` …) back to their
    model defaults. Merging onto the on-disk text instead preserves them.

    Uses ruamel.yaml round-trip mode so comments and key order survive the
    rewrite. Falls back to PyYAML when ruamel is not installed — that loses
    comments but still preserves all sections (the critical fix), so a missing
    optional dependency never breaks saving.
    """
    try:
        from io import StringIO

        from ruamel.yaml import YAML
    except ImportError:
        base = yaml.safe_load(existing_text) if existing_text.strip() else {}
        merged = _deep_merge(base or {}, incoming)
        return yaml.dump(
            merged, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    ruamel = YAML()  # round-trip: preserves comments + order
    ruamel.preserve_quotes = True
    base = ruamel.load(existing_text) if existing_text.strip() else None
    if base is None:
        base = ruamel.load("{}")
    _deep_merge_inplace(base, incoming)
    buf = StringIO()
    ruamel.dump(base, buf)
    return buf.getvalue()


def load_settings(config_path: str = "config.yaml") -> Settings:
    path = Path(config_path)
    yaml_data: dict = {}

    if not path.exists():
        logger.info(
            "No config file at %s — creating with defaults. "
            "Run 'coremind setup' to configure via the web UI.",
            config_path,
        )
        defaults = Settings()
        try:
            path.write_text(
                yaml.dump(
                    defaults.model_dump(),
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            )
        except OSError as e:
            logger.warning("Could not write default config to %s: %s", config_path, e)
        return defaults
    else:
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse {config_path}: {e}") from e

        if raw is not None:
            if not isinstance(raw, dict):
                raise ConfigError(
                    f"{config_path} must be a YAML mapping at the top level, got {type(raw).__name__}"
                )
            yaml_data = raw

    merged = _deep_merge(yaml_data, _env_overrides())

    try:
        return Settings.model_validate(merged)
    except Exception as e:
        raise ConfigError(f"Invalid config in {config_path}: {e}") from e
