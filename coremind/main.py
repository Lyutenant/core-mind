from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from coremind.config import load_settings

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(
    name="coremind",
    help="CoreMind voice assistant framework — Raspberry Pi edition.",
    no_args_is_help=True,
)
audio_app = typer.Typer(help="Audio device diagnostics and testing.")
chat_app = typer.Typer(help="Voice interaction commands.")
music_app = typer.Typer(help="Music library management.")
atc_app = typer.Typer(help="ATC stream catalog management.")
vision_app = typer.Typer(help="Camera / vision diagnostics.")

app.add_typer(audio_app, name="audio")
app.add_typer(chat_app, name="chat")
app.add_typer(music_app, name="music")
app.add_typer(atc_app, name="atc")
app.add_typer(vision_app, name="vision")

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
    # mcp.client.sse logs at ERROR when the Node restarts and drops the SSE
    # connection mid-stream ("peer closed connection without sending complete
    # message body"). This is expected on Node restart — our reconnect loop
    # handles it. Suppress to avoid alarming noise in Hub logs.
    logging.getLogger("mcp.client.sse").setLevel(logging.CRITICAL)


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
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Input device index or name substring (see list-devices). Default: auto-select."),
) -> None:
    """Record a short WAV file from the microphone."""
    from coremind.audio_input.devices import coerce_device, resolve_input_device
    from coremind.audio_input.recorder import Recorder
    from coremind import AudioInputError

    cfg = _get_settings().audio
    dev = coerce_device(device) if device is not None else resolve_input_device(cfg.input_device)

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
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Output device index or name substring (see list-devices). Default: auto-select."),
) -> None:
    """Play a WAV file through the speaker."""
    from coremind.audio_input.devices import coerce_device, resolve_output_device
    from coremind.audio_output.player import Player
    from coremind import AudioOutputError

    cfg = _get_settings().audio
    dev = coerce_device(device) if device is not None else resolve_output_device(cfg.output_device)

    console.print(f"Playing [cyan]{file}[/cyan]  (device: {dev if dev is not None else 'default'})")

    try:
        player = Player(device=dev)
        player.play(file)
        console.print("[green]Done.[/green]")
    except AudioOutputError as e:
        console.print(f"[red]Playback failed:[/red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Vision commands
# ---------------------------------------------------------------------------

@vision_app.command("test")
def vision_test(
    output: str = typer.Option("frame.jpg", "--output", "-o", help="Output JPEG file path."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Camera index or /dev path. Default: auto-select (vision.camera_device / camera_index)."),
) -> None:
    """Capture one frame from the camera and save it. Run this on the Node (Pi) to test the webcam."""
    from coremind import VisionError
    from coremind.vision.camera_select import resolve_camera_source
    from coremind.vision.opencv_camera import OpenCVCamera

    cfg = _get_settings().vision
    if device is not None:
        source = int(device) if device.strip().isdigit() else device
    else:
        source = resolve_camera_source(cfg.camera_device, cfg.camera_index)

    console.print(f"Capturing from camera [bold]{source!r}[/bold] → [cyan]{output}[/cyan]")
    try:
        cam = OpenCVCamera(
            camera_index=source if isinstance(source, int) else 0,
            camera_device=source if isinstance(source, str) else None,
            width=cfg.resolution_width,
            height=cfg.resolution_height,
        )
        jpeg = cam.capture_jpeg()
    except VisionError as e:
        console.print(f"[red]Capture failed:[/red] {e}")
        raise typer.Exit(code=1)

    Path(output).write_bytes(jpeg)
    console.print(f"[green]Saved:[/green] {output}  ({len(jpeg)} bytes)")


@vision_app.command("describe")
def vision_describe(
    image: str = typer.Option("", "--image", "-i", help="Image file to describe. If omitted, captures one from the camera."),
    prompt: str = typer.Option("", "--prompt", "-p", help="What to look for. Defaults to a general description."),
) -> None:
    """Run the Hub vision model on an image (or a fresh capture). Run this on the Mac (Hub)."""
    import base64

    from coremind import BrainError, VisionError
    from coremind.brain.ollama_client import OllamaClient

    s = _get_settings()
    if not s.ollama.vision_model:
        console.print(
            "[red]ollama.vision_model is not set.[/red] "
            "Set it to e.g. 'llava:7b' in config.yaml, then run: ollama pull llava:7b"
        )
        raise typer.Exit(code=1)

    try:
        if image:
            jpeg = Path(image).read_bytes()
        else:
            from coremind.vision.camera_select import resolve_camera_source
            from coremind.vision.opencv_camera import OpenCVCamera
            console.print("[dim]No --image given — capturing from the local camera…[/dim]")
            source = resolve_camera_source(s.vision.camera_device, s.vision.camera_index)
            cam = OpenCVCamera(
                camera_index=source if isinstance(source, int) else 0,
                camera_device=source if isinstance(source, str) else None,
                width=s.vision.resolution_width,
                height=s.vision.resolution_height,
            )
            jpeg = cam.capture_jpeg()
    except VisionError as e:
        console.print(f"[red]Capture failed:[/red] {e}")
        raise typer.Exit(code=1)
    except OSError as e:
        console.print(f"[red]Could not read image:[/red] {e}")
        raise typer.Exit(code=1)

    b64 = base64.b64encode(jpeg).decode("ascii")
    question = prompt.strip() or "Describe what you see in this image in one or two concise sentences."
    console.print(f"[dim]Asking {s.ollama.vision_model}…[/dim]")
    client = OllamaClient(
        base_url=s.ollama.base_url,
        model=s.ollama.vision_model,
        timeout=s.brain.timeout_seconds,
    )
    try:
        description = client.describe_image(b64, question)
    except BrainError as e:
        console.print(f"[red]Vision model error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]Vision:[/bold cyan] {description}")


# ---------------------------------------------------------------------------
# Chat commands
# ---------------------------------------------------------------------------

def _build_voice_loop(
    settings,
    *,
    enable_wake_word: bool = False,
    enable_vad: bool = False,
    on_listening_start=None,
    on_turn_complete=None,
):
    from coremind import ConfigError, STTError, TTSError, WakeWordError
    from coremind.audio_input.devices import resolve_input_device, resolve_output_device
    from coremind.audio_input.recorder import Recorder
    from coremind.audio_output.player import Player
    from coremind.voice_loop import VoiceLoop

    acfg = settings.audio
    # Resolve once (auto-selecting by stable name when unset) so the recorder,
    # player, wake-word detector, and chime all share the same physical devices.
    input_device = resolve_input_device(acfg.input_device)
    output_device = resolve_output_device(acfg.output_device)
    recorder = Recorder(
        device=input_device,
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
        player = Player(device=output_device)
    else:
        from coremind.brain.ollama_client import MockBrainClient, OllamaClient
        from coremind.brain.router import BrainRouter
        from coremind.memory.session_memory import SessionMemory
        from coremind.stt.whisper_local import MockSTT, WhisperLocalSTT

        stt_provider = settings.stt.provider
        if stt_provider == "whisper_local":
            try:
                stt = WhisperLocalSTT(
                    model=settings.stt.model,
                    language=settings.stt.language,
                    compute_type=settings.stt.compute_type,
                    beam_size=settings.stt.beam_size,
                    vad_filter=settings.stt.vad_filter,
                    initial_prompt=settings.stt.initial_prompt,
                    hotwords=settings.stt.hotwords,
                )
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
                    player = Player(device=output_device)
                except TTSError as e:
                    console.print(f"[yellow]TTS disabled:[/yellow] {e}")
        elif tts_provider == "espeak":
            from coremind.tts.piper_local import EspeakTTS
            tts = EspeakTTS(voice=settings.tts.voice or "en")
            player = Player(device=output_device)
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
                    device=input_device,
                    sample_rate=16000,  # openwakeword always requires 16 kHz
                    inference_framework=wwcfg.inference_framework,
                    vad_threshold=wwcfg.vad_threshold,
                )
                console.print(
                    f"[dim]Wake word: {wwcfg.model} "
                    f"(threshold={wwcfg.threshold}, vad_gate={wwcfg.vad_threshold})[/dim]"
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
        personality=settings.app.personality,
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
        wake_confirm_words=settings.runtime.wake_confirm_words,
        config_mtime=Path(_config_path).stat().st_mtime if Path(_config_path).exists() else 0.0,
        on_listening_start=on_listening_start,
        on_turn_complete=on_turn_complete,
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
    # Start Node MCP server before building the voice loop so callbacks are ready
    _on_listening_start = None
    _on_turn_complete = None
    if settings.node_mcp.enabled:
        import asyncio as _asyncio
        import threading as _threading
        from coremind.node_mcp.server import run_node_mcp_server
        from coremind.node_mcp.tools import music_player as _music_player

        def _start_node_mcp() -> None:
            _asyncio.run(
                run_node_mcp_server(
                    music_dir=settings.node_mcp.music_dir,
                    catalog_path=settings.node_mcp.catalog_path,
                    atc_catalog_path=settings.node_mcp.atc_catalog_path,
                    port=settings.node_mcp.port,
                    host=settings.node_mcp.host,
                    camera_enabled=settings.vision.enabled,
                    camera_provider=settings.vision.provider,
                    camera_index=settings.vision.camera_index,
                    camera_device=settings.vision.camera_device,
                    camera_width=settings.vision.resolution_width,
                    camera_height=settings.vision.resolution_height,
                )
            )

        _t = _threading.Thread(target=_start_node_mcp, daemon=True, name="node-mcp")
        _t.start()
        logger.info("Node MCP server started on port %d", settings.node_mcp.port)
        _on_listening_start = _music_player.pause_mpv
        _on_turn_complete = _music_player.resume_mpv

    loop = _build_voice_loop(
        settings,
        enable_wake_word=True,
        enable_vad=True,
        on_listening_start=_on_listening_start,
        on_turn_complete=_on_turn_complete,
    )

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
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Bind address. Default 127.0.0.1 is correct when using Caddy. Use 0.0.0.0 to expose directly without a reverse proxy."),
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
        "    Hub:        [bold]coremind server --host 0.0.0.0[/bold]   [dim]# or set up Caddy (see docs/setup-hub.md)[/dim]\n"
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
        from coremind.audio_input.devices import list_input_devices, resolve_input_device
        devs = list_input_devices()
        if devs:
            chosen = resolve_input_device(settings.audio.input_device)
            label = f"using {chosen!r}" if chosen is not None else "using system default"
            checks.append(Check("Audio input", "ok", f"{len(devs)} device(s) found — {label}"))
        else:
            checks.append(Check("Audio input", "warn", "No input devices found",
                                "Check microphone. Run 'coremind audio list-devices'."))
    except Exception as e:
        checks.append(Check("Audio input", "fail", str(e),
                            "sounddevice may be missing. Run: pip install sounddevice"))

    # 4. Audio output
    try:
        from coremind.audio_input.devices import list_output_devices, resolve_output_device
        devs = list_output_devices()
        if devs:
            chosen = resolve_output_device(settings.audio.output_device)
            label = f"using {chosen!r}" if chosen is not None else "using system default"
            checks.append(Check("Audio output", "ok", f"{len(devs)} device(s) found — {label}"))
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


# ---------------------------------------------------------------------------
# Music commands
# ---------------------------------------------------------------------------

@music_app.command("scan")
def music_scan() -> None:
    """Scan the music library and rebuild the catalog from folder structure."""
    settings = _get_settings()
    from coremind.node_mcp.catalog import scan_library, save_catalog

    music_dir = Path(settings.node_mcp.music_dir).expanduser()
    catalog_path = Path(settings.node_mcp.catalog_path).expanduser()

    if not music_dir.exists():
        console.print(f"[red]Music directory not found:[/red] {music_dir}")
        raise typer.Exit(code=1)

    console.print(f"Scanning [cyan]{music_dir}[/cyan] …")
    data = scan_library(music_dir)
    save_catalog(data, catalog_path)
    console.print(
        f"[green]Catalog saved:[/green] {catalog_path}\n"
        f"  Tracks:  {len(data['tracks'])}\n"
        f"  Artists: {len(data['artists'])}\n"
        f"  Albums:  {len(data['albums'])}"
    )


# ---------------------------------------------------------------------------
# ATC commands
# ---------------------------------------------------------------------------

_ATC_JS_SNIPPET = Path(__file__).parent / "data" / "extract-liveatc-mounts.js"

# Bare mounts — probed as {icao}{variant} (no underscore).
# Small GA airports (unicom/ctaf) often use just the ICAO code or a number.
_ATC_BARE_VARIANTS = ["", "1", "2", "3"]

# Suffixed mounts — probed as {icao}_{suffix}.
# Mirrors GUESSED_SUFFIXES from atc-hub/scripts/scrape-liveatc.py.
_ATC_SUFFIXES = [
    "twr", "twr2",
    "gnd", "gnd_twr", "gnd2",
    "app", "app_n", "app_s", "app_e", "app_w", "app_final", "app_dep", "app2",
    "dep", "dep_n", "dep_s", "dep2",
    "del", "clnc_del",
    "atis",
    "ramp",
    "unicom", "ctaf", "ptd",
]

_DEFAULT_ATC_CATALOG = Path(__file__).parent / "data" / "atc-catalog-default.json"

import re as _re


def _freq_from_mount(mount: str) -> str:
    """Decode frequency from trailing numeric segment in mount name.

    LiveATC encodes MHz by stripping the decimal point:
      4 digits 1191   → 119.1 MHz
      5 digits 12575  → 125.75 MHz
      6 digits 120250 → 120.250 MHz
    Segments of 1–3 digits are receiver IDs, not frequencies.
    """
    m = _re.search(r'_(\d{4,6})(?:_|$)', mount)
    if m:
        d = m.group(1)
        return f"{d[:3]}.{d[3:]}"
    return ""


@atc_app.command("scan")
def atc_scan(
    icaos: list[str] = typer.Argument(..., help="ICAO codes to scan, e.g. KEWR KJFK KIAD"),
    browser_mounts: str = typer.Option(
        "", "--browser-mounts", "-b",
        help="Path to browser-extracted mounts JSON (output of 'coremind atc js')."
    ),
    airport_names: str = typer.Option("", "--names", help="Override airport names (comma-separated, matched by order)"),
    rate: float = typer.Option(1.0, "--rate", "-r", help="Requests per second (default: 1.0)"),
) -> None:
    """Probe LiveATC for available streams and add them to the catalog.

    Probes two sets of candidates per airport:
      1. Browser-extracted mounts (--browser-mounts FILE) — authoritative for airports
         with obfuscated mount names (e.g. KIAD). See 'coremind atc js' for instructions.
      2. Guessed suffixes (twr, gnd, app …) — catches standard simple-name airports.

    Airport names are looked up automatically from the bundled ICAO database.
    Use 'coremind atc add' to manually add a single channel.
    """
    import time

    import httpx

    from coremind.airports import lookup_icao
    from coremind.node_mcp.atc_catalog import (
        ATCCatalog,
        load_atc_catalog,
        pls_url,
        save_atc_catalog,
    )

    settings = _get_settings()
    catalog_path = Path(settings.node_mcp.atc_catalog_path).expanduser()

    # Seed from user catalog, then bundled default, then empty
    data = (
        load_atc_catalog(catalog_path)
        or load_atc_catalog(_DEFAULT_ATC_CATALOG)
        or {"version": 1, "channels": []}
    )
    catalog = ATCCatalog(data)

    # Load browser-extracted mounts: {ICAO_UPPER: [mount, ...]}
    browser: dict[str, list[str]] = {}
    if browser_mounts.strip():
        bm_path = Path(browser_mounts).expanduser()
        try:
            raw: list[dict] = __import__("json").loads(bm_path.read_text())
            for entry in raw:
                mount = entry.get("mount", "").strip()
                icao_key = entry.get("icao", "").strip().upper()
                if mount and icao_key:
                    browser.setdefault(icao_key, [])
                    if mount not in browser[icao_key]:
                        browser[icao_key].append(mount)
            total_bm = sum(len(v) for v in browser.values())
            console.print(f"[dim]Browser mounts loaded: {total_bm} across {len(browser)} airports[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] could not load {bm_path}: {exc}")

    name_overrides = [n.strip() for n in airport_names.split(",") if n.strip()]
    delay = 1.0 / max(rate, 0.1)

    def probe(mount: str) -> tuple[bool, str]:
        """Return (valid, title). Rate-limited."""
        try:
            r = httpx.get(pls_url(mount), timeout=8.0, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"})
            body = r.text.strip()
            if r.status_code == 200 and body.startswith("[playlist]"):
                title = ""
                for line in body.splitlines():
                    if line.startswith("Title1="):
                        title = line.split("=", 1)[1].strip()
                        break
                return True, title
        except Exception:
            pass
        time.sleep(delay)
        return False, ""

    total_new = 0
    for i, icao in enumerate(icaos):
        icao = icao.upper().strip()
        if i < len(name_overrides):
            airport_name = name_overrides[i]
        else:
            db_info = lookup_icao(icao)
            airport_name = db_info["name"] if db_info else ""
        console.print(f"\n[bold]{icao}[/bold]{' — ' + airport_name if airport_name else ''}")

        icao_lower = icao.lower()
        seen: set[str] = set()

        # Build ordered candidate list: browser mounts first, then bare variants, then suffixes
        candidates: list[str] = []
        for m in browser.get(icao, []):
            if m not in seen:
                seen.add(m)
                candidates.append(m)
        for v in _ATC_BARE_VARIANTS:
            m = f"{icao_lower}{v}"
            if m not in seen:
                seen.add(m)
                candidates.append(m)
        for suffix in _ATC_SUFFIXES:
            m = f"{icao_lower}_{suffix}"
            if m not in seen:
                seen.add(m)
                candidates.append(m)

        for mount in candidates:
            valid, title = probe(mount)
            if valid:
                name = title or mount
                freq = _freq_from_mount(mount)
                channel: dict = {
                    "airport": icao,
                    "airport_name": airport_name,
                    "name": name,
                    "mount": mount,
                }
                if freq:
                    channel["freq"] = freq
                is_new = catalog.upsert_channel(channel)
                if is_new:
                    total_new += 1
                freq_str = f"  [dim]{freq} MHz[/dim]" if freq else ""
                console.print(f"  [green]✓[/green]  {mount}  {name}{freq_str}")
            else:
                console.print(f"  [dim]✗  {mount}[/dim]")

    save_atc_catalog(catalog.to_dict(), catalog_path)
    console.print(
        f"\n[green]Catalog saved:[/green] {catalog_path}\n"
        f"  {total_new} new channel(s) added."
    )


@atc_app.command("js")
def atc_js() -> None:
    """Show instructions for extracting LiveATC mounts from your browser.

    For airports with obfuscated mount names (e.g. KIAD, KDCA), LiveATC's
    search pages must be opened in a browser. A JavaScript snippet extracts
    all stream links from the page.
    """
    js_path = _ATC_JS_SNIPPET
    console.print("\n[bold]Step 1 — Open LiveATC in your browser[/bold]")
    console.print("  Go to the LiveATC airport search page for your airport")
    console.print("  (the page whose address ends in /search/?icao=KIAD —")
    console.print("  replace KIAD with the airport you want)\n")
    console.print("[bold]Step 2 — Run the JavaScript snippet[/bold]")
    console.print("  Open DevTools → Console, paste the contents of:")
    console.print(f"  [cyan]{js_path}[/cyan]\n")
    console.print("[bold]Step 3 — Save the output[/bold]")
    console.print("  Copy the JSON printed in the console and save it to a file,")
    console.print("  e.g. ~/browser-mounts.json. Repeat for each airport, appending")
    console.print("  to the same file (it's a JSON array).\n")
    console.print("[bold]Step 4 — Run atc scan with the file[/bold]")
    console.print("  [green]coremind atc scan KIAD KDCA --browser-mounts ~/browser-mounts.json[/green]\n")
    if js_path.exists():
        console.print(f"[dim]JS snippet ({js_path.stat().st_size} bytes):[/dim]")
        console.print(js_path.read_text())


@atc_app.command("add")
def atc_add(
    icao: str = typer.Argument(..., help="Airport ICAO code, e.g. KIAD"),
    name: str = typer.Argument(..., help="Channel name, e.g. 'Tower Runway 1C/19C'"),
    mount: str = typer.Argument(..., help="Stream mount name, e.g. kiad1_twr_1c19c_120250"),
    freq: str = typer.Option("", "--freq", "-f", help="Frequency in MHz, e.g. 120.250"),
    airport_name: str = typer.Option("", "--airport-name", "-a", help="Override airport name (auto-looked up if omitted)"),
) -> None:
    """Manually add or update an ATC channel (for airports with non-standard mounts)."""
    from coremind.airports import lookup_icao
    from coremind.node_mcp.atc_catalog import (
        ATCCatalog,
        empty_catalog,
        load_atc_catalog,
        save_atc_catalog,
    )

    settings = _get_settings()
    catalog_path = Path(settings.node_mcp.atc_catalog_path).expanduser()

    data = load_atc_catalog(catalog_path) or empty_catalog()
    catalog = ATCCatalog(data)

    # Auto-fill airport name from database if not provided
    resolved_name = airport_name.strip()
    if not resolved_name:
        db_info = lookup_icao(icao)
        resolved_name = db_info["name"] if db_info else ""

    channel: dict = {
        "airport": icao.upper().strip(),
        "airport_name": resolved_name,
        "name": name.strip(),
        "mount": mount.strip(),
    }
    if freq.strip():
        channel["freq"] = freq.strip()

    is_new = catalog.upsert_channel(channel)
    save_atc_catalog(catalog.to_dict(), catalog_path)

    action = "Added" if is_new else "Updated"
    freq_str = f" ({freq} MHz)" if freq.strip() else ""
    console.print(
        f"[green]{action}:[/green] {channel['airport']} {name}{freq_str}  →  {mount}\n"
        f"  Catalog: {catalog_path}"
    )


@atc_app.command("test")
def atc_test(
    query: str = typer.Argument(..., help="Channel query, e.g. 'KJYO tower' or 'Newark approach'"),
    wait: float = typer.Option(3.0, "--wait", "-w", help="Seconds to wait before checking mpv status."),
    catalog: str = typer.Option("", "--catalog", help="Override catalog path."),
) -> None:
    """Test ATC streaming: resolve a query, start mpv, and confirm the stream stays alive."""
    import subprocess
    import time

    from coremind.node_mcp.atc_catalog import (
        ATCCatalog,
        load_atc_catalog,
        matches_query_frequency,
        pick_tower_candidate,
        pls_url,
        stream_url,
    )

    settings = _get_settings()
    catalog_path = Path(catalog).expanduser() if catalog else Path(settings.node_mcp.atc_catalog_path).expanduser()

    _DEFAULT_CATALOG = Path(__file__).parent / "data" / "atc-catalog-default.json"
    data = load_atc_catalog(catalog_path)
    if data is None:
        console.print(f"[yellow]No catalog at {catalog_path} — trying bundled default.[/yellow]")
        data = load_atc_catalog(_DEFAULT_CATALOG)
    if data is None:
        console.print("[red]No ATC catalog found.[/red]")
        raise typer.Exit(1)

    cat = ATCCatalog(data)
    console.print(f"Catalog loaded: [cyan]{len(cat._channels)}[/cyan] channel(s) from [dim]{catalog_path}[/dim]")

    candidates = cat.find_candidates(query)
    if not candidates:
        console.print(f"[red]No channel matched '{query}'.[/red]")
        airports = cat.list_airports()
        console.print("Available airports:", ", ".join(airports[:10]))
        raise typer.Exit(1)

    if len(candidates) > 1:
        picked = pick_tower_candidate(candidates, query)
        if picked is not None:
            console.print(
                f"[dim]{len(candidates)} tower channels tied — picked one at random "
                f"(same behavior as the voice loop).[/dim]"
            )
            candidates = [picked]
        else:
            console.print(f"[yellow]{len(candidates)} channels matched (ambiguous):[/yellow]")
            for ch in candidates:
                freq_str = f"  {ch['freq']} MHz" if ch.get("freq") else ""
                console.print(f"  {ch['airport']} | {ch['name']}{freq_str}  →  {ch['mount']}")
            console.print("Re-run with a more specific query.")
            raise typer.Exit(1)

    ch = candidates[0]
    if matches_query_frequency(ch, query) is False:
        console.print(
            f"[red]The requested frequency does not match this channel "
            f"(same behavior as the voice loop — it would not stream).[/red]\n"
            f"  Closest match: {ch.get('airport', '')} {ch.get('name', ch['mount'])}"
        )
        console.print("Re-run without the frequency to test this channel.")
        raise typer.Exit(1)

    url = stream_url(ch["mount"])
    label = f"{ch.get('airport', '')} {ch.get('name', ch['mount'])}"
    freq_str = f" ({ch['freq']} MHz)" if ch.get("freq") else ""
    console.print(f"\nMatch: [bold]{label}{freq_str}[/bold]")
    console.print(f"Mount: [cyan]{ch['mount']}[/cyan]")

    # Verify mount is reachable before spawning mpv
    try:
        import httpx
        console.print(f"\nChecking PLS endpoint for mount [cyan]{ch['mount']}[/cyan]…")
        r = httpx.get(pls_url(ch["mount"]), timeout=8.0, follow_redirects=True)
        if r.status_code == 200 and "[playlist]" in r.text:
            console.print("[green]✓ PLS responds — mount is valid[/green]")
        else:
            console.print(f"[red]✗ PLS returned HTTP {r.status_code} — mount may be stale or down[/red]")
            console.print(f"  Response: {r.text[:120]}")
    except Exception as e:
        console.print(f"[yellow]PLS check failed ({e}) — proceeding anyway[/yellow]")

    # Start mpv with stderr visible so errors are shown
    console.print(f"\nStarting mpv… (waiting {wait:.0f}s to confirm stream)")
    try:
        proc = subprocess.Popen(
            ["mpv", "--no-terminal", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        console.print("[red]mpv not found. Install with: sudo apt install mpv[/red]")
        raise typer.Exit(1)

    time.sleep(wait)
    exit_code = proc.poll()

    if exit_code is None:
        console.print(f"[green]✓ mpv is running — stream is alive[/green]  (PID {proc.pid})")
        console.print("Stopping test stream.")
        proc.terminate()
        proc.wait(timeout=3)
    else:
        stderr_out = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
        console.print(f"[red]✗ mpv exited after {wait:.0f}s (exit code {exit_code})[/red]")
        if stderr_out:
            console.print("[yellow]mpv stderr:[/yellow]")
            for line in stderr_out.splitlines():
                console.print(f"  {line}")
        else:
            console.print("[dim]No stderr output — stream may be unreachable or mount is invalid.[/dim]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
