# CoreMind

A two-component voice assistant. The Pi handles audio I/O and wake word; a Mac Mini runs all inference.

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
| USB microphone or ReSpeaker XVF3800 | connected to Pi |
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
pip install -e ".[dev,stt,server]"
cp config.hub.example.yaml config.yaml   # then edit
coremind server --host 0.0.0.0
```

**Node (Raspberry Pi):**
```bash
sudo apt install -y git libportaudio2 portaudio19-dev python3.11 python3.11-venv
git clone https://github.com/Lyutenant/core-mind.git && cd core-mind
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp config.node.example.yaml config.yaml  # then edit Hub URL + audio devices
coremind run
```

Open **http://localhost:8765** on the Mac Mini for the dashboard.

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

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 0–6 | Repo, audio, STT, LLM, TTS, push-to-talk MVP | ✓ |
| 7 | Wake word (openwakeword) | ✓ |
| 8 | Voice activity detection | ✓ |
| 9 | Hub web server + dashboard | ✓ |
| 10a | Built-in tools (time, weather, aviation weather, airport) | ✓ |
| 10b | Hub MCP client (connect to any MCP server) | ✓ |
| 10c | Node MCP server + music player + mic isolation | ✓ |
| 12 | systemd user service | ✓ |
| 13 | `coremind doctor` diagnostics | ✓ |
| 14 | Node auto-registration + Nodes panel | ✓ |
| 15 | Mobile-responsive dashboard | ✓ |
| 16 | Follow-up loop safety | ✓ |
| 34 | Music catalog (artist/album/playlist + voice CRUD) | ✓ |
| 35 | Live ATC streaming (17 MCP tools) | ✓ |
| 36 | Bundled ICAO airport database (19K airports) | ✓ |
