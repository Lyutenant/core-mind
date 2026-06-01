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
