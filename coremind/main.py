from __future__ import annotations

import logging
import sys
from typing import Optional

import typer
from rich.console import Console

from coremind.config import load_settings

console = Console()
app = typer.Typer(
    name="coremind",
    help="CoreMind voice assistant framework — Raspberry Pi edition.",
    no_args_is_help=True,
)
audio_app = typer.Typer(help="Audio device diagnostics and testing.")
chat_app = typer.Typer(help="Voice interaction commands.")

app.add_typer(audio_app, name="audio")
app.add_typer(chat_app, name="chat")

_settings = None
_config_path: str = "config.yaml"


def _get_settings():
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # httpx logs every HTTP request at INFO — suppress routine heartbeat noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.callback()
def main(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    global _settings, _config_path
    _config_path = config
    _settings = load_settings(config)
    level = "DEBUG" if verbose else _settings.app.log_level
    _setup_logging(level)


# ---------------------------------------------------------------------------
# Audio commands
# ---------------------------------------------------------------------------

@audio_app.command("list-devices")
def audio_list_devices() -> None:
    """List available audio input and output devices."""
    from rich.table import Table
    from coremind.audio_input.devices import list_input_devices, list_output_devices
    from coremind import AudioInputError

    try:
        inputs = list_input_devices()
        outputs = list_output_devices()
    except AudioInputError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    table = Table(title="Audio Input Devices", show_lines=True)
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Name")
    table.add_column("Channels", width=9)
    table.add_column("Sample Rate", width=12)
    for d in inputs:
        table.add_row(str(d.index), d.name, str(d.max_input_channels), f"{d.default_sample_rate:.0f} Hz")
    console.print(table)

    table2 = Table(title="Audio Output Devices", show_lines=True)
    table2.add_column("Index", style="cyan", width=6)
    table2.add_column("Name")
    table2.add_column("Channels", width=9)
    table2.add_column("Sample Rate", width=12)
    for d in outputs:
        table2.add_row(str(d.index), d.name, str(d.max_output_channels), f"{d.default_sample_rate:.0f} Hz")
    console.print(table2)


@audio_app.command("record-test")
def audio_record_test(
    seconds: int = typer.Option(5, "--seconds", "-s", help="Recording duration in seconds."),
    output: str = typer.Option("test.wav", "--output", "-o", help="Output WAV file path."),
    device: Optional[int] = typer.Option(None, "--device", "-d", help="Input device index (see list-devices)."),
) -> None:
    """Record a short WAV file from the microphone."""
    from coremind.audio_input.recorder import Recorder
    from coremind import AudioInputError

    cfg = _get_settings().audio
    dev = device if device is not None else cfg.input_device

    console.print(f"Recording [bold]{seconds}s[/bold] → [cyan]{output}[/cyan]  (device: {dev if dev is not None else 'default'})")
    console.print("🎙  Speak now...", highlight=False)

    try:
        recorder = Recorder(device=dev, sample_rate=cfg.sample_rate, channels=cfg.channels)
        path = recorder.record(seconds=seconds, output_path=output)
        console.print(f"[green]Saved:[/green] {path}")
    except AudioInputError as e:
        console.print(f"[red]Recording failed:[/red] {e}")
        raise typer.Exit(code=1)


@audio_app.command("play-test")
def audio_play_test(
    file: str = typer.Option(..., "--file", "-f", help="WAV file to play."),
    device: Optional[int] = typer.Option(None, "--device", "-d", help="Output device index (see list-devices)."),
) -> None:
    """Play a WAV file through the speaker."""
    from coremind.audio_output.player import Player
    from coremind import AudioOutputError

    cfg = _get_settings().audio
    dev = device if device is not None else cfg.output_device

    console.print(f"Playing [cyan]{file}[/cyan]  (device: {dev if dev is not None else 'default'})")

    try:
        player = Player(device=dev)
        player.play(file)
        console.print("[green]Done.[/green]")
    except AudioOutputError as e:
        console.print(f"[red]Playback failed:[/red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Chat commands
# ---------------------------------------------------------------------------

def _build_voice_loop(settings, *, enable_wake_word: bool = False, enable_vad: bool = False):
    from coremind import ConfigError, STTError, TTSError, WakeWordError
    from coremind.audio_input.recorder import Recorder
    from coremind.audio_output.player import Player
    from coremind.voice_loop import VoiceLoop

    acfg = settings.audio
    recorder = Recorder(
        device=acfg.input_device,
        sample_rate=acfg.sample_rate,
        channels=acfg.channels,
    )

    # Determine mode: "node" sends audio to Hub; "hub"/"standalone" processes locally.
    # Legacy: also honour remote_brain.enabled for backward compatibility.
    remote_cfg = settings.remote_brain
    use_remote = (
        settings.mode == "node"
        or (settings.mode not in ("hub", "node", "standalone") and remote_cfg.enabled)
        or (settings.mode == "hub" and remote_cfg.enabled)
    )
    if use_remote:
        if not remote_cfg.url:
            raise ConfigError(
                "mode is 'node' but remote_brain.url is not set. "
                "Set the Hub URL in config.yaml or via 'coremind setup'."
            )
        console.print(f"[dim]Node mode — Hub: {remote_cfg.url}[/dim]")
        stt = None
        brain = None
        memory = None
        tts = None
        player = Player(device=acfg.output_device)
    else:
        from coremind.brain.ollama_client import MockBrainClient, OllamaClient
        from coremind.brain.router import BrainRouter
        from coremind.memory.session_memory import SessionMemory
        from coremind.stt.whisper_local import MockSTT, WhisperLocalSTT

        stt_provider = settings.stt.provider
        if stt_provider == "whisper_local":
            try:
                stt = WhisperLocalSTT(model=settings.stt.model, language=settings.stt.language)
            except STTError:
                console.print(
                    "[yellow]Warning:[/yellow] faster-whisper not installed — using MockSTT. "
                    r"Run: pip install 'coremind\[stt]'"
                )
                stt = MockSTT()
        elif stt_provider == "mock":
            stt = MockSTT()
        else:
            raise ConfigError(
                f"Unsupported stt.provider: {stt_provider!r}. "
                "Supported: whisper_local, mock."
            )

        provider = settings.brain.provider
        if provider == "ollama":
            primary = OllamaClient(
                base_url=settings.ollama.base_url,
                model=settings.ollama.model,
                timeout=settings.brain.timeout_seconds,
                no_think=settings.ollama.no_think,
                options=settings.ollama.options,
            )
        elif provider == "mock":
            primary = MockBrainClient()
        else:
            raise ConfigError(
                f"Unsupported brain.provider: {provider!r}. "
                "Supported: ollama, mock."
            )
        fallback = MockBrainClient() if settings.brain.allow_mock_fallback else None
        brain = BrainRouter(primary=primary, fallback=fallback)

        tts = None
        player = None
        tts_provider = settings.tts.provider
        if tts_provider == "piper_local":
            from coremind.tts.piper_local import PiperLocalTTS
            model_path = settings.tts.model_path
            if not model_path:
                console.print(
                    "[yellow]TTS disabled:[/yellow] tts.model_path not set. "
                    "Set it to your Piper .onnx model file path, or use provider: espeak."
                )
            else:
                try:
                    tts = PiperLocalTTS(model_path=model_path)
                    player = Player(device=acfg.output_device)
                except TTSError as e:
                    console.print(f"[yellow]TTS disabled:[/yellow] {e}")
        elif tts_provider == "espeak":
            from coremind.tts.piper_local import EspeakTTS
            tts = EspeakTTS(voice=settings.tts.voice or "en")
            player = Player(device=acfg.output_device)
        elif tts_provider == "mock":
            from coremind.tts.piper_local import MockTTS
            tts = MockTTS()
        else:
            raise ConfigError(
                f"Unsupported tts.provider: {tts_provider!r}. "
                "Supported: piper_local, espeak, mock."
            )

        memory = SessionMemory(max_turns=settings.memory.max_turns)

    # VAD
    vad = None
    if enable_vad and settings.vad.enabled:
        from coremind.vad.simple_energy import SimpleEnergyVAD
        vad = SimpleEnergyVAD(threshold=settings.vad.energy_threshold)
        console.print(
            f"[dim]VAD enabled (threshold={settings.vad.energy_threshold}, "
            f"silence={settings.vad.silence_seconds}s, "
            f"max={settings.vad.max_record_seconds}s)[/dim]"
        )

    # Wake word
    wake_word = None
    if enable_wake_word:
        wwcfg = settings.wake_word
        if not wwcfg.enabled or wwcfg.provider == "dummy":
            from coremind.wake_word.dummy import DummyWakeWordDetector
            wake_word = DummyWakeWordDetector()
        elif wwcfg.provider == "openwakeword":
            from coremind.wake_word.openwakeword_engine import OpenWakeWordDetector
            try:
                wake_word = OpenWakeWordDetector(
                    model=wwcfg.model,
                    threshold=wwcfg.threshold,
                    device=acfg.input_device,
                    sample_rate=16000,  # openwakeword always requires 16 kHz
                    inference_framework=wwcfg.inference_framework,
                )
                console.print(
                    f"[dim]Wake word: {wwcfg.model} (threshold={wwcfg.threshold})[/dim]"
                )
            except WakeWordError as e:
                console.print(
                    f"[yellow]Wake word disabled:[/yellow] {e} — falling back to push-to-talk."
                )
                from coremind.wake_word.dummy import DummyWakeWordDetector
                wake_word = DummyWakeWordDetector()
        else:
            raise ConfigError(
                f"Unsupported wake_word.provider: {wwcfg.provider!r}. "
                "Supported: dummy, openwakeword."
            )

    return VoiceLoop(
        name=settings.app.name,
        recorder=recorder,
        stt=stt,
        brain=brain,
        memory=memory,
        record_seconds=acfg.record_seconds,
        status_fn=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        tts=tts,
        player=player,
        wake_word=wake_word,
        wake_fn=lambda: console.print("[bold green]Wake word detected![/bold green] Speak your question...") if wake_word is not None else None,
        vad=vad,
        vad_silence_seconds=settings.vad.silence_seconds,
        vad_max_record_seconds=settings.vad.max_record_seconds,
        vad_min_speech_seconds=settings.vad.min_speech_seconds,
        remote_url=remote_cfg.url if use_remote else None,
        remote_timeout=remote_cfg.timeout_seconds,
        follow_up_seconds=settings.runtime.follow_up_seconds,
        follow_up_min_words=settings.runtime.follow_up_min_words,
        post_response_cooldown_seconds=settings.runtime.post_response_cooldown_seconds,
    )


@chat_app.command("once")
def chat_once() -> None:
    """Record one utterance, transcribe it, and get an LLM response."""
    from coremind import AudioInputError, BrainError, STTError

    settings = _get_settings()
    loop = _build_voice_loop(settings)

    console.print(
        f"[bold]Press Enter to record ({settings.audio.record_seconds}s)...[/bold]",
        end="",
    )
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(code=0)
    console.print("[dim]Recording...[/dim]")

    try:
        transcript, response = loop.run_once()
    except AudioInputError as e:
        console.print(f"[red]Recording failed:[/red] {e}")
        raise typer.Exit(code=1)
    except STTError as e:
        console.print(f"[red]Transcription failed:[/red] {e}")
        raise typer.Exit(code=1)
    except BrainError as e:
        console.print(f"[red]LLM error:[/red] {e}")
        raise typer.Exit(code=1)

    if not transcript.strip():
        console.print("[yellow]No speech detected.[/yellow]")
        return

    console.print(f"\n[bold]You:[/bold] {transcript}")
    console.print(f"\n[bold cyan]{settings.app.name}:[/bold cyan] {response}")


@chat_app.command("loop")
def chat_loop() -> None:
    """Start a continuous push-to-talk voice loop. Press Ctrl+C to quit."""
    from coremind import AudioInputError, BrainError, STTError

    settings = _get_settings()
    loop = _build_voice_loop(settings)

    console.print(
        f"[bold green]{settings.app.name} voice loop started.[/bold green] "
        "Press Ctrl+C to quit.\n"
    )

    turn = 0
    while True:
        turn += 1
        console.print(
            f"[dim]Turn {turn} —[/dim] "
            f"[bold]Press Enter to record ({settings.audio.record_seconds}s)...[/bold]",
            end="",
        )
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye.[/yellow]")
            break

        console.print("[dim]Recording...[/dim]")

        try:
            transcript, response = loop.run_once()
        except AudioInputError as e:
            console.print(f"[red]Recording failed:[/red] {e}")
            continue
        except STTError as e:
            console.print(f"[red]Transcription failed:[/red] {e}")
            continue
        except BrainError as e:
            console.print(f"[red]LLM error:[/red] {e}")
            if not settings.brain.allow_mock_fallback:
                console.print(
                    "[yellow]Tip:[/yellow] Set brain.allow_mock_fallback: true in config.yaml "
                    "to continue with mock responses when Ollama is unreachable."
                )
            continue
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye.[/yellow]")
            break

        if not transcript.strip():
            console.print("[yellow]No speech detected. Try again.[/yellow]")
            continue

        console.print(f"\n[bold]You:[/bold] {transcript}")
        console.print(f"\n[bold cyan]{settings.app.name}:[/bold cyan] {response}\n")


# ---------------------------------------------------------------------------
# Run command (full MVP)
# ---------------------------------------------------------------------------

@app.command("run")
def run() -> None:
    """Start CoreMind in full voice assistant mode (wake word + VAD + TTS)."""
    from coremind import AudioInputError, BrainError, STTError, WakeWordError

    settings = _get_settings()
    if settings.mode == "hub":
        console.print(
            "[yellow]Warning:[/yellow] mode is 'hub' — this device is configured as a Hub.\n"
            "  To start the Hub server: [bold]coremind server[/bold]\n"
            "  To run as a standalone assistant: set [bold]mode: standalone[/bold] in config.yaml\n"
            "  Continuing anyway…\n"
        )
    loop = _build_voice_loop(settings, enable_wake_word=True, enable_vad=True)

    console.print(
        f"[bold green]{settings.app.name} ready.[/bold green] "
        "Press Ctrl+C to quit.\n"
    )

    turn = 0
    while True:
        turn += 1
        console.print(f"[dim]--- Turn {turn} ---[/dim]")
        try:
            transcript, response = loop.run_once()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye.[/yellow]")
            break
        except AudioInputError as e:
            console.print(f"[red]Recording failed:[/red] {e}")
            continue
        except STTError as e:
            console.print(f"[red]Transcription failed:[/red] {e}")
            continue
        except BrainError as e:
            console.print(f"[red]LLM error:[/red] {e}")
            continue
        except WakeWordError as e:
            console.print(f"[red]Wake word error:[/red] {e}")
            continue

        if not transcript.strip():
            console.print("[yellow]No speech detected.[/yellow]")
            continue

        console.print(f"\n[bold]You:[/bold] {transcript}")
        console.print(f"\n[bold cyan]{settings.app.name}:[/bold cyan] {response}\n")


# ---------------------------------------------------------------------------
# Server command (run on Mac Mini)
# ---------------------------------------------------------------------------

@app.command("server")
def server_cmd(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind address."),
    port: int = typer.Option(8765, "--port", "-p", help="Port to listen on."),
) -> None:
    """Start the CoreMind HTTP server (run this on Mac Mini, not Pi)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red] Run: pip install 'coremind[server]'")
        raise typer.Exit(code=1)

    try:
        from coremind.server.app import app as _server_app, configure as _srv_configure
        if _server_app is None:
            raise ImportError("fastapi not available")
    except ImportError:
        console.print("[red]fastapi not installed.[/red] Run: pip install 'coremind[server]'")
        raise typer.Exit(code=1)

    _srv_configure(config_path=_config_path)
    console.print(
        f"[bold green]CoreMind Hub starting[/bold green] — "
        f"listening on {host}:{port}\n"
        f"  Dashboard: http://localhost:{port}\n"
        "  Node config: set remote_brain.enabled: true and "
        f"remote_brain.url: http://<this-host>:{port}"
    )
    uvicorn.run(_server_app, host=host, port=port)


@app.command("setup")
def setup_cmd(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind address."),
    port: int = typer.Option(8766, "--port", "-p", help="Port to listen on."),
) -> None:
    """Start a web UI to configure this device (Hub or Node). Runs on port 8766 by default.

    Use this on any device to set up config.yaml via a browser — no manual file editing needed.
    After saving, stop this server (Ctrl+C) and run 'coremind server' (Hub) or 'coremind run' (Node).
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red] Run: pip install 'coremind[server]'")
        raise typer.Exit(code=1)

    try:
        from coremind.server.app import app as _server_app, configure as _srv_configure
        if _server_app is None:
            raise ImportError("fastapi not available")
    except ImportError:
        console.print("[red]fastapi not installed.[/red] Run: pip install 'coremind[server]'")
        raise typer.Exit(code=1)

    _srv_configure(config_path=_config_path)
    mode = _get_settings().mode
    mode_label = {"hub": "Hub", "node": "Node", "standalone": "Standalone"}.get(mode, mode)
    console.print(
        f"[bold green]CoreMind Setup UI[/bold green] — "
        f"current mode: [bold]{mode_label}[/bold]\n"
        f"  Open [bold]http://localhost:{port}[/bold] in your browser\n"
        "  Go to Settings → App → Mode to set this device's role\n"
        "  Save, then Ctrl+C and run:\n"
        "    Hub:        [bold]coremind server[/bold]\n"
        "    Node:       [bold]coremind run[/bold]\n"
        "    Standalone: [bold]coremind run[/bold]"
    )
    uvicorn.run(_server_app, host=host, port=port)


# ---------------------------------------------------------------------------
# Doctor command
# ---------------------------------------------------------------------------

@app.command("doctor")
def doctor_cmd() -> None:
    """Run system diagnostics: Python, config, audio, LLM, STT, TTS, disk."""
    import shutil
    import sys
    import tempfile
    from dataclasses import dataclass, field
    from pathlib import Path

    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    @dataclass
    class Check:
        name: str
        status: str   # "ok" | "warn" | "fail" | "skip"
        detail: str
        hint: str = field(default="")

    checks: list[Check] = []
    settings = _get_settings()

    # 1. Python version
    vi = sys.version_info
    version_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if vi >= (3, 11):
        checks.append(Check("Python version", "ok", version_str))
    else:
        checks.append(Check("Python version", "fail", version_str,
                            "CoreMind requires Python 3.11+. Upgrade Python."))

    # 2. Config file
    config_path = Path(_config_path)
    if config_path.exists():
        checks.append(Check("Config file", "ok",
                            f"{_config_path} loaded  (mode: {settings.mode})"))
    else:
        checks.append(Check("Config file", "warn",
                            f"{_config_path} not found — using defaults",
                            "Run 'coremind setup' to create a config file via the web UI."))

    # 3. Audio input
    try:
        from coremind.audio_input.devices import list_input_devices
        devs = list_input_devices()
        if devs:
            checks.append(Check("Audio input", "ok", f"{len(devs)} device(s) found"))
        else:
            checks.append(Check("Audio input", "warn", "No input devices found",
                                "Check microphone. Run 'coremind audio list-devices'."))
    except Exception as e:
        checks.append(Check("Audio input", "fail", str(e),
                            "sounddevice may be missing. Run: pip install sounddevice"))

    # 4. Audio output
    try:
        from coremind.audio_input.devices import list_output_devices
        devs = list_output_devices()
        if devs:
            checks.append(Check("Audio output", "ok", f"{len(devs)} device(s) found"))
        else:
            checks.append(Check("Audio output", "warn", "No output devices found",
                                "Check speaker. Run 'coremind audio list-devices'."))
    except Exception as e:
        checks.append(Check("Audio output", "fail", str(e),
                            "sounddevice may be missing. Run: pip install sounddevice"))

    # 5. LLM / Hub reachability
    if settings.mode == "node":
        remote_url = settings.remote_brain.url
        if not remote_url:
            checks.append(Check("Hub connection", "fail", "remote_brain.url not set",
                                "Set the Hub URL in config.yaml or run 'coremind setup'."))
        else:
            try:
                import httpx
                httpx.get(remote_url, timeout=5.0, follow_redirects=True)
                checks.append(Check("Hub connection", "ok",
                                    f"Hub reachable at {remote_url}"))
            except Exception as e:
                checks.append(Check("Hub connection", "fail",
                                    f"{remote_url} — {type(e).__name__}",
                                    "Is the Hub running? Run 'coremind server' on the Mac Mini."))
    else:
        ollama_url = settings.ollama.base_url
        configured_model = settings.ollama.model
        try:
            import httpx
            r = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
            r.raise_for_status()
            model_names = [m["name"] for m in r.json().get("models", [])]
            available = configured_model in model_names
            if available:
                checks.append(Check("Ollama", "ok",
                                    f"Reachable — model {configured_model!r} available"))
            else:
                short_list = ", ".join(model_names[:4]) or "none"
                checks.append(Check("Ollama", "warn",
                                    f"Reachable — model {configured_model!r} not found "
                                    f"(available: {short_list})",
                                    f"Run: ollama pull {configured_model}"))
        except Exception as e:
            checks.append(Check("Ollama", "fail",
                                f"{ollama_url} unreachable — {type(e).__name__}",
                                "Is Ollama running? Try: ollama serve"))

    # 6. STT backend
    if settings.mode == "node":
        checks.append(Check("STT", "skip", "Not used on Node — Hub handles transcription"))
    else:
        stt_provider = settings.stt.provider
        if stt_provider == "mock":
            checks.append(Check("STT", "warn", "Using MockSTT — no real transcription",
                                "Set stt.provider: whisper_local for real speech recognition."))
        elif stt_provider == "whisper_local":
            try:
                import faster_whisper  # noqa: F401
                checks.append(Check("STT", "ok",
                                    f"faster-whisper available (model: {settings.stt.model})"))
            except ImportError:
                checks.append(Check("STT", "warn",
                                    "faster-whisper not installed — will fall back to MockSTT",
                                    r"Install: pip install 'coremind\[stt]'"))
        else:
            checks.append(Check("STT", "skip", f"Provider: {stt_provider!r}"))

    # 7. TTS backend
    if settings.mode == "node":
        checks.append(Check("TTS", "skip", "Not used on Node — Hub sends audio WAV"))
    else:
        tts_provider = settings.tts.provider
        if tts_provider == "mock":
            checks.append(Check("TTS", "warn", "Using MockTTS — no audio playback",
                                "Set tts.provider: espeak or piper_local."))
        elif tts_provider == "espeak":
            espeak_ng = shutil.which("espeak-ng")
            if espeak_ng:
                checks.append(Check("TTS", "ok", f"espeak-ng found ({espeak_ng})"))
            elif shutil.which("espeak"):
                checks.append(Check("TTS", "fail",
                                    "only legacy espeak found — runtime requires espeak-ng",
                                    "Install: sudo apt install espeak-ng"))
            else:
                checks.append(Check("TTS", "fail", "espeak-ng not on PATH",
                                    "Install: sudo apt install espeak-ng"))
        elif tts_provider == "piper_local":
            model_path = settings.tts.model_path
            if not model_path:
                checks.append(Check("TTS", "fail", "tts.model_path not set",
                                    "Set the path to your Piper .onnx voice model."))
            elif not Path(model_path).exists():
                checks.append(Check("TTS", "fail", f"model file not found: {model_path}",
                                    "Download a voice at github.com/rhasspy/piper/releases"))
            else:
                try:
                    from piper.voice import PiperVoice  # noqa: F401
                    checks.append(Check("TTS", "ok",
                                        f"piper-tts available, model: {model_path}"))
                except ImportError:
                    checks.append(Check("TTS", "warn",
                                        "piper-tts not installed (model file found)",
                                        r"Install: pip install 'coremind\[tts]'"))
        else:
            checks.append(Check("TTS", "skip", f"Provider: {tts_provider!r}"))

    # 8. Disk write permission
    write_dir = config_path.parent.resolve()
    try:
        with tempfile.NamedTemporaryFile(dir=write_dir, delete=True):
            pass
        checks.append(Check("Disk write", "ok", f"{write_dir} is writable"))
    except Exception as e:
        checks.append(Check("Disk write", "fail", str(e),
                            "Check filesystem permissions for the config directory."))

    # Render
    _STATUS = {
        "ok":   "[bold green]  OK  [/bold green]",
        "warn": "[bold yellow] WARN [/bold yellow]",
        "fail": "[bold red] FAIL [/bold red]",
        "skip": "[dim] SKIP [/dim]",
    }

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(width=7, no_wrap=True)
    table.add_column(style="bold", width=18, no_wrap=True)
    table.add_column()
    table.add_column(style="dim italic")

    for c in checks:
        table.add_row(_STATUS[c.status], c.name, c.detail, c.hint)

    fails = sum(1 for c in checks if c.status == "fail")
    warns = sum(1 for c in checks if c.status == "warn")
    if fails == 0 and warns == 0:
        summary = "[bold green]All checks passed.[/bold green]"
    else:
        parts = []
        if fails:
            parts.append(f"[bold red]{fails} failure{'s' if fails != 1 else ''}[/bold red]")
        if warns:
            parts.append(f"[bold yellow]{warns} warning{'s' if warns != 1 else ''}[/bold yellow]")
        summary = "  " + ",  ".join(parts) + "  "

    console.print()
    console.print(Panel(
        table,
        title=f"[bold]CoreMind Doctor[/bold]  —  mode: {settings.mode}",
        subtitle=summary,
        border_style="blue",
    ))
    console.print()

    if fails:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
