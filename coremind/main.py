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

@chat_app.command("once")
def chat_once() -> None:
    """Record one utterance, transcribe it, and get an LLM response."""
    console.print("[yellow]Voice chat not yet implemented (Phase 2).[/yellow]")
    raise typer.Exit(code=0)


@chat_app.command("loop")
def chat_loop() -> None:
    """Start a continuous push-to-talk voice loop."""
    console.print("[yellow]Voice loop not yet implemented (Phase 2).[/yellow]")
    raise typer.Exit(code=0)


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
