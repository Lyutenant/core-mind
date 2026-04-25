# CoreMind

A Raspberry Pi 5 voice assistant. Records speech, transcribes it locally, sends to an LLM on a remote machine, and speaks the response aloud.

**Status: Phase 1 — Audio Device Diagnostics**

## Hardware

- Raspberry Pi 5, 8 GB RAM
- USB microphone or ReSpeaker XVF3800
- Speaker (USB, 3.5 mm, or HDMI)
- Mac Mini running Ollama, reachable via Tailscale

## Setup

**Raspberry Pi (Linux) — install PortAudio first:**
```bash
sudo apt install libportaudio2
```

**All platforms:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml
# Edit config.yaml with your Ollama URL and preferred settings
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and update:

- `ollama.base_url` — Tailscale IP of your Mac Mini running Ollama
- `ollama.model` — model name to use (e.g. `qwen3:8b`)
- `audio.input_device` / `audio.output_device` — device index from `audio list-devices`, or leave null for system default

Sensitive values can be overridden with environment variables:

```bash
export COREMIND_OLLAMA__BASE_URL=http://100.x.x.x:11434
```

## Usage

```bash
# Show all commands
python -m coremind.main --help

# List audio devices
python -m coremind.main audio list-devices

# Record a test clip
python -m coremind.main audio record-test --seconds 5 --output test.wav

# Play it back
python -m coremind.main audio play-test --file test.wav

# One-shot voice interaction
python -m coremind.main chat once

# Continuous voice loop
python -m coremind.main chat loop

# Full run mode (MVP)
python -m coremind.main run
```

## Development

```bash
pytest          # run unit tests (no hardware required)
```

## Architecture

```
coremind/
  config/       Pydantic settings models
  audio_input/  Microphone recording
  audio_output/ Speaker playback
  stt/          Speech-to-text backends
  tts/          Text-to-speech backends
  brain/        LLM client and router
  wake_word/    Wake-word detection (Phase 7+)
  vad/          Voice activity detection (Phase 8+)
  memory/       Session memory (Phase 9+)
  tools/        Tool registry (Phase 10+)
```

## Roadmap

| Phase | Description |
|-------|-------------|
| 0 | Repository setup ✓ |
| 1 | Audio device diagnostics ✓ |
| 2 | Push-to-talk voice loop |
| 3 | LLM backend / brain |
| 4 | Speech-to-text |
| 5 | Text-to-speech |
| 6 | Full MVP v0.1 |
| 7+ | Wake word, VAD, memory, tools |
