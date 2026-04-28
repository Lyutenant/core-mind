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


@app.callback()
def main(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    global _settings
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

def _build_voice_loop(settings):
    from coremind import ConfigError, STTError
    from coremind.audio_input.recorder import Recorder
    from coremind.brain.ollama_client import MockBrainClient, OllamaClient
    from coremind.brain.router import BrainRouter
    from coremind.memory.session_memory import SessionMemory
    from coremind.stt.whisper_local import MockSTT, WhisperLocalSTT
    from coremind.voice_loop import VoiceLoop

    acfg = settings.audio
    recorder = Recorder(
        device=acfg.input_device,
        sample_rate=acfg.sample_rate,
        channels=acfg.channels,
    )

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
            "Supported values for Phase 2: whisper_local, mock."
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
            "Supported values for Phase 2: ollama, mock."
        )
    fallback = MockBrainClient() if settings.brain.allow_mock_fallback else None
    brain = BrainRouter(primary=primary, fallback=fallback)

    memory = SessionMemory(max_turns=settings.memory.max_turns)
    return VoiceLoop(
        name=settings.app.name,
        recorder=recorder,
        stt=stt,
        brain=brain,
        memory=memory,
        record_seconds=acfg.record_seconds,
        status_fn=lambda msg: console.print(f"[dim]{msg}[/dim]"),
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
    """Start CoreMind in full voice assistant mode."""
    console.print("[yellow]Full run mode not yet implemented (Phase 6).[/yellow]")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
