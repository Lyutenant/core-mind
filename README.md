# CoreMind

**v0.4.7** · A two-component voice assistant. The Pi handles audio I/O and wake word; a Mac Mini runs all inference.

```
[CoreMind Node]  ──audio──▶  [CoreMind Hub]  ──▶  Ollama LLM
 Raspberry Pi                  Mac Mini             (local)
 wake word                     STT (Whisper)        tool calls?
 VAD record                    tool loop     ──▶  weather / time / MCP…
 play audio   ◀──audio──       TTS (Piper)
```

**CoreMind Hub** (Mac Mini) — web dashboard, STT, LLM, TTS, tool calling.  
**CoreMind Node** (Pi) — wake word, VAD recording, audio playback.

---

## Hardware

| Component | Role |
|-----------|------|
| Raspberry Pi 5, 8 GB | CoreMind Node |
| USB microphone (ReSpeaker XVF3800 works as a plain mic) | connected to Pi |
| Speaker (USB, 3.5 mm, or HDMI) | connected to Pi |
| Mac Mini | CoreMind Hub — runs Ollama + Hub server |
| Tailscale | private network between Pi and Mac Mini |

---

## Quick Start

**Hub (Mac Mini):**
```bash
brew install portaudio espeak-ng
git clone https://github.com/Lyutenant/core-mind.git && cd core-mind
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,stt,server,tools]"   # tools = MCP client for music/ATC
pip install piper-tts                       # optional: better TTS than espeak (see Hub Setup)
cp config.hub.example.yaml config.yaml      # starting point; tune the rest from the dashboard
coremind server --host 0.0.0.0
```

**Node (Raspberry Pi):**
```bash
sudo apt install -y git libportaudio2 portaudio19-dev python3.11 python3.11-venv
git clone https://github.com/Lyutenant/core-mind.git && cd core-mind
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Wake word (required for always-on "Hey Jarvis"; see Node Setup for details)
pip install 'coremind[wake_word]'           # onnxruntime inference backend
pip install openwakeword --no-deps          # tflite has no Pi ARM64 wheel
python -c "import openwakeword; openwakeword.utils.download_models()"

cp config.node.example.yaml config.yaml     # then edit Hub URL + audio devices
coremind run
```

Open **http://localhost:8765** on the Mac Mini for the dashboard — the primary place to
configure CoreMind (assistant name, personality, STT/TTS/LLM, tools). No YAML editing needed
after first start.

> **Exposing the Hub to the Pi over Tailscale:** `coremind server` binds to `127.0.0.1` by default. To let the Pi reach it, either set up [Caddy on the Mac Mini](docs/setup-hub.md#7-caddy-reverse-proxy) (recommended) or run `coremind server --host 0.0.0.0` for a simpler but less secure alternative.

---

## Documentation

| Guide | Contents |
|-------|----------|
| [Hub Setup](docs/setup-hub.md) | Full install, config, Caddy reverse proxy |
| [Node Setup](docs/setup-node.md) | Full install, config, audio, Caddy, systemd |
| [Tools](docs/tools.md) | Built-in tools, MCP servers, music player, ATC streaming |
| [Troubleshooting](docs/troubleshooting.md) | `coremind doctor`, logs, updating |

---

## Commands Reference

```bash
# Setup — any device
coremind setup                          # open config UI at http://localhost:8766

# Hub (Mac Mini)
coremind server                         # start Hub on 127.0.0.1:8765
coremind server --host 0.0.0.0         # bind all interfaces (use behind Caddy instead)

# Node (Pi) — audio diagnostics
coremind audio list-devices             # list input/output devices
coremind audio record-test -s 5 -o test.wav
coremind audio play-test -f test.wav

# Node (Pi) — voice interaction
coremind run                            # full mode: wake word + VAD + remote brain
coremind chat loop                      # push-to-talk loop (no wake word)
coremind chat once                      # single push-to-talk turn

# Music library (Node)
coremind music scan                     # scan music_dir and rebuild catalog

# ATC streams (Node)
coremind atc scan KBWI KPHL            # auto-discover standard airports
coremind atc js                         # instructions for obfuscated-mount airports
coremind atc scan KXXX --browser-mounts ~/browser-mounts.json
coremind atc add KIAD "Tower" kiad1_twr_1c19c_120250 --freq 120.250
coremind atc test "KIAD tower"          # test ATC stream without starting the voice loop
coremind atc test "KIAD tower 120.250"  # a frequency pins the exact channel

# Diagnostics — any device
coremind doctor                         # check Python, config, audio, Ollama, STT, TTS

# systemd service (Pi)
systemctl --user status coremind
systemctl --user restart coremind
journalctl --user -u coremind -f

# Development
pytest                                  # run unit tests (no hardware required)
```

---

## Package Architecture

```
coremind/
  config/         Pydantic settings (loaded from config.yaml)
  audio_input/    Microphone recording + VAD
  audio_output/   Speaker playback (auto-resampling for USB speakers)
  stt/            Speech-to-text (faster-whisper, mock)
  tts/            Text-to-speech (Piper, espeak, mock)
  brain/          LLM client (Ollama + tool calling, mock)
  memory/         Session memory
  wake_word/      Wake-word detection (openwakeword/onnx, dummy)
  vad/            Voice activity detection (energy-based)
  server/         CoreMind Hub — FastAPI + web dashboard
  tools/          Dispatcher, built-in tools, Hub MCP client
  node_mcp/       Node MCP server — 17 tools (music + ATC)
  airports.py     Bundled ICAO database (19K airports, offline)
```

---

## Status

Implemented and working (v0.4.7):

- Full voice pipeline on real hardware: wake word (openwakeword/onnx) → VAD → faster-whisper → Ollama → Piper/espeak
- Hub web dashboard: chat, Nodes panel, Settings editor, Tools panel — mobile-responsive
- Configurable assistant personality (free-text persona/tone, with presets)
- Session memory + follow-up listening window with safety mechanisms
- Built-in tools (time, weather, aviation weather, airport) + offline 19K-airport ICAO database
- Hub MCP client (stdio + HTTP/SSE, auto-reconnect) and Node MCP server (13 music + 4 ATC tools)
- Music catalog from folder structure; LiveATC streaming with scored channel matching and frequency pinning
- Node auto-registration with heartbeat; `coremind doctor` diagnostics; systemd user service on the Pi
- Loopback-by-default binding with Caddy reverse-proxy examples

## Planned

- **Porcupine wake word** — custom "Hey CoreMind" phrase via Picovoice `.ppn` model
- **WebRTC VAD** — optional upgrade from the energy-threshold VAD for noisy rooms
- **Long-term memory** — deferred until the voice loop is fully reliable; no vector DB yet
- **More MCP integrations** — home automation, calendars, etc. (all via MCP, not bespoke code)
- **Audio refinements** — Bluetooth speakers, ReSpeaker XVF3800 echo cancellation / DoA
