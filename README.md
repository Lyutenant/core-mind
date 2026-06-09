# CoreMind

A two-component voice assistant system built around a Raspberry Pi and a Mac Mini.

```
[CoreMind Node]  ──audio──▶  [CoreMind Hub]  ──▶  Ollama LLM
 Raspberry Pi                  Mac Mini             (local)
 wake word                     STT (Whisper)        tool calls?
 VAD record                    tool loop     ──▶  weather / time / MCP…
 play audio   ◀──audio──       TTS (Piper)
```

**CoreMind Hub** runs on the Mac Mini. It serves a web dashboard, handles speech-to-text, calls the LLM, synthesizes the response, and returns audio to the Node.

**CoreMind Node** runs on the Raspberry Pi. It listens for a wake word, records audio with VAD, sends it to the Hub, and plays the returned audio.

---

## Table of Contents

- [Hardware](#hardware)
- [Architecture Decision](#architecture-decision)
- [Hub Setup — Mac Mini](#hub-setup--mac-mini)
  - [1.1 Install system dependencies](#11-install-system-dependencies)
  - [1.2 Clone the repository](#12-clone-the-repository)
  - [1.3 Create the Python virtual environment](#13-create-the-python-virtual-environment)
  - [1.4 Install dependencies](#14-install-dependencies)
  - [1.5 Create the Hub config](#15-create-the-hub-config)
  - [1.6 Start the Hub](#16-start-the-hub)
  - [1.7 Find your Tailscale IP](#17-find-your-tailscale-ip)
- [Node Setup — Raspberry Pi](#node-setup--raspberry-pi)
  - [2.1 Install system dependencies](#21-install-system-dependencies)
  - [2.2 Clone the repository](#22-clone-the-repository)
  - [2.3 Create the Python virtual environment](#23-create-the-python-virtual-environment)
  - [2.4 Install dependencies](#24-install-dependencies)
  - [2.5 Create the Node config](#25-create-the-node-config)
  - [2.6 Find audio device indexes](#26-find-audio-device-indexes)
  - [2.7 Test audio](#27-test-audio)
  - [2.8 Start the Node](#28-start-the-node)
- [Web Dashboard](#web-dashboard)
  - [Nodes Panel](#nodes-panel)
  - [Wake Word Threshold](#wake-word-threshold)
  - [Follow-up Conversation](#follow-up-conversation)
  - [Mobile Access](#mobile-access)
- [Tool Calling](#tool-calling)
  - [Built-in Tools](#built-in-tools)
  - [MCP Servers](#mcp-servers)
  - [Voice-Controlled Music Player](#voice-controlled-music-player)
  - [Live ATC Streaming](#live-atc-streaming)
- [Persistent Service — systemd](#persistent-service--systemd)
- [All-on-Pi Mode (No Hub)](#all-on-pi-mode-no-hub)
- [Updating](#updating)
- [Commands Reference](#commands-reference)
- [Package Architecture](#package-architecture)
- [Roadmap](#roadmap)

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

## Architecture Decision

In the default **remote brain** mode, the Pi only handles audio I/O and wake word. All heavy computation (STT, LLM, TTS) runs on the Mac Mini. This reduces per-turn latency from ~30 s to ~12–15 s (dominated by the LLM).

If you want to run everything on the Pi (no Mac Mini), see [All-on-Pi Mode](#all-on-pi-mode-no-hub).

---

## Hub Setup — Mac Mini

### 1.1 Install system dependencies

```bash
brew install portaudio espeak-ng
```

### 1.2 Clone the repository

```bash
git clone https://github.com/Lyutenant/core-mind.git
cd core-mind
```

### 1.3 Create the Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version   # should print Python 3.11.x
```

### 1.4 Install dependencies

```bash
pip install -e ".[dev,stt,server]"
```

This installs `coremind` (editable), `faster-whisper` (STT), and `fastapi`/`uvicorn` (Hub server).

**Optional — Piper TTS (neural voice, recommended):**
```bash
pip install piper-tts

mkdir -p ~/piper-voices
curl -L -o ~/piper-voices/en_US-amy-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
curl -L -o ~/piper-voices/en_US-amy-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
```

**Optional — espeak-ng (simpler, robotic voice, no download):**
```bash
brew install espeak-ng   # already done if you ran step 1.1
```

### 1.5 Create the Hub config

**Option A — browser UI (recommended):**
```bash
coremind setup        # opens config UI at http://localhost:8766
```
Set Mode to **Hub**, fill in the settings, click **Save Configuration**, then Ctrl+C.

**Option B — copy and edit manually:**
```bash
cp config.hub.example.yaml config.yaml
```

Key Hub settings:
```yaml
mode: hub

app:
  name: Jarvis
  user_location: "San Francisco, CA"   # fallback for weather/time (optional)
  user_timezone: "America/Los_Angeles" # IANA timezone (optional)
  home_airport: "KJYO"                 # default for aviation weather (optional)
  taf_airport: "KIAD"                  # nearest airport with TAF

runtime:
  follow_up_seconds: 5.0   # listen this long after a response; 0.0 to disable

stt:
  provider: whisper_local
  model: base           # tiny/base/small

tts:
  provider: piper_local
  model_path: /Users/yourname/piper-voices/en_US-amy-medium.onnx
  # or: provider: espeak

ollama:
  base_url: http://localhost:11434
  model: gemma4:e4b     # ollama pull gemma4:e4b

tools:
  enabled: true
  built_in: [time, weather, aviation_weather]

remote_brain:
  enabled: false        # Hub does not forward to another Hub
```

### 1.6 Start the Hub

```bash
coremind server
```

You should see:
```
CoreMind Hub starting — listening on 0.0.0.0:8765
  Dashboard: http://localhost:8765
```

Open **http://localhost:8765** in your browser.

### 1.7 Find your Tailscale IP

```bash
tailscale ip -4
# example: 100.x.x.x
```

You will need this IP when configuring the Node.

---

## Node Setup — Raspberry Pi

### 2.1 Install system dependencies

```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev python3.11 python3.11-venv git
python3.11 --version   # should print Python 3.11.x
```

If 3.11 is not in apt:
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

### 2.2 Clone the repository

```bash
git clone https://github.com/Lyutenant/core-mind.git
cd core-mind
```

### 2.3 Create the Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version   # Python 3.11.x
```

### 2.4 Install dependencies

```bash
pip install -e ".[dev]"
```

**Optional — wake word detection (required for always-on "Hey Jarvis" mode):**
```bash
# Step 1: install onnxruntime (Pi-compatible inference backend)
pip install 'coremind[wake_word]'

# Step 2: install openwakeword without its tflite dependency
# (tflite-runtime has no Pi ARM64 wheel for Python 3.11+)
pip install openwakeword --no-deps

# Step 3: download the built-in wake word models
python -c "import openwakeword; openwakeword.utils.download_models()"
```

> **Note:** Always use `inference_framework: onnx` in config — `tflite` is incompatible with NumPy 2.x on Raspberry Pi OS.

### 2.5 Create the Node config

**Option A — browser UI:**
```bash
coremind setup        # opens config UI at http://<pi-ip>:8766
```
Set Mode to **Node**, enter the Hub URL, set audio device indexes, save, Ctrl+C.

**Option B — copy and edit manually:**
```bash
cp config.node.example.yaml config.yaml
```

Key Node settings:
```yaml
mode: node

app:
  name: Jarvis   # must match the Hub's name

audio:
  input_device: null    # set to your USB mic index (see step 2.6)
  output_device: null   # set to your speaker index
  sample_rate: 16000

runtime:
  follow_up_seconds: 5.0

remote_brain:
  enabled: true
  url: http://100.x.x.x:8765    # Mac Mini Tailscale IP from step 1.7
  timeout_seconds: 90.0

wake_word:
  enabled: true
  provider: openwakeword
  model: hey_jarvis_v0.1
  threshold: 0.5
  inference_framework: onnx

vad:
  enabled: true
  energy_threshold: 0.01
  silence_seconds: 1.2
  max_record_seconds: 20.0
  min_speech_seconds: 0.3
```

### 2.6 Find audio device indexes

```bash
coremind audio list-devices
```

Update `audio.input_device` and `audio.output_device` in `config.yaml` with the correct indexes.

### 2.7 Test audio

```bash
coremind audio record-test --seconds 5 --output test.wav
coremind audio play-test --file test.wav
```

Verify that your voice is recorded and plays back before continuing.

### 2.8 Start the Node

```bash
coremind run
```

You should see:
```
Jarvis ready. Press Ctrl+C to quit.
--- Turn 1 ---
Listening for wake word (hey_jarvis_v0.1)...
```

Say **"Hey Jarvis"** — you will hear a chime and recording begins. Your question is sent to the Hub, processed, and the audio response plays on the Pi.

---

## Web Dashboard

Open **http://localhost:8765** (or `http://<mac-mini-tailscale-ip>:8765` from any Tailscale device). Works on iPhone — see [Mobile Access](#mobile-access).

| Section | What it shows |
|---------|--------------|
| Dashboard | Live conversation log with tool call badges, real-time status |
| Nodes | Connected Pi Nodes — online/offline status, per-node sliders |
| Settings → App | Mode, assistant name, log level, location/timezone |
| Settings → STT | Whisper model, language |
| Settings → TTS | Provider, model path, voice |
| Settings → Ollama | Server URL, model, no-think, options |
| Settings → Memory | Session memory max turns |
| Settings → Tools | Enable/disable built-in tools |
| Settings → VAD | Energy threshold, silence/max/min durations |
| Settings → Wake Word | Provider, model, detection threshold |
| Settings → Audio | Device indexes, sample rate, channels |
| Settings → Hub Connection | Hub URL (for Node remote brain) |

Settings are saved to `config.yaml` on the Hub when you click **Save Configuration**. Most changes take effect without a restart.

### Nodes Panel

When a Node starts it automatically appears in the **Nodes** view — no pairing step. Each Node card shows:

- Name, hostname, online/offline indicator (green = seen in last 90 s)
- Per-node sliders: wake word threshold, VAD energy, VAD silence, max record, min speech, follow-up window, min follow-up words, post-response cooldown

Slider changes reach the Node within 30 seconds via heartbeat poll — no restart required. **Reset to defaults** clears all overrides and the Node reverts to its `config.yaml` values on the next poll.

**Overrides survive restarts.** The Hub persists overrides to `~/.coremind/node-overrides.json` using a latest-wins timestamp: if you edited `config.yaml` on the Pi after the last dashboard save, the Pi's values win. If the dashboard was updated more recently, the Hub values win.

All connected browsers (Mac Mini, iPhone, etc.) receive SSE events and update the Nodes panel in real time.

### Wake Word Threshold

The `threshold` value (0.0–1.0) controls how confident openwakeword must be before firing.

| Threshold | Behavior |
|-----------|----------|
| 0.3–0.4 | Very sensitive — low miss rate, more false positives from TV/background speech |
| **0.5** | **Default** — balanced for quiet home environments |
| 0.6–0.7 | Conservative — fewer false positives; may miss soft triggers |
| 0.8+ | Very strict — only use if false positives are constant |

Adjust from the Nodes panel slider in real time — no Node restart needed.

### Follow-up Conversation

After responding, the Node listens for a follow-up without requiring the wake phrase again. Three mechanisms prevent the session from running indefinitely on background noise:

- **Stop phrases** — saying "stop", "goodbye", "cancel", "that's all", etc. ends the session and returns to wake-word mode.
- **Min word filter** — follow-up transcripts shorter than `follow_up_min_words` (default 2) are discarded as background noise.
- **Post-response cooldown** — a brief pause (`post_response_cooldown_seconds`, default 1.0 s) after TTS finishes before the mic reopens, preventing speaker echo from re-triggering.

All three are tunable from the Nodes panel or `config.yaml` under `runtime:`.

### Mobile Access

The dashboard layout is responsive. On a phone:

- The sidebar collapses and a **bottom tab bar** replaces it (Chat | Nodes | Settings).
- Settings forms reflow to a single column.
- All controls meet the 44 px minimum touch target.

Open `http://<mac-mini-tailscale-ip>:8765` in Safari and use **Add to Home Screen** to bookmark it as a web app.

---

## Tool Calling

CoreMind uses the Ollama native tool API — the LLM decides when to call tools and the Hub executes them transparently.

### Built-in Tools

| Tool | Trigger example | Notes |
|------|----------------|-------|
| `get_current_time` | "What time is it?" | Uses `app.user_timezone` if set |
| `get_weather` | "What's the weather?" / "Forecast for tomorrow?" | wttr.in, no API key; 1–3 day forecasts |
| `get_aviation_weather` | "What's the METAR at Leesburg?" / "Any PIREPs?" | NOAA aviationweather.gov, no API key; METAR/TAF/PIREP |
| `lookup_airport` | "What's the ICAO for Heathrow?" / "What airport is IAD?" | Bundled offline database, 19K airports, no API key |

Enable in Hub `config.yaml`:
```yaml
tools:
  enabled: true
  built_in: [time, weather, aviation_weather, airport]
```

**Weather and time defaults:**
```yaml
app:
  user_location: "San Francisco, CA"   # fallback when no location is mentioned
  user_timezone: "America/Los_Angeles" # IANA timezone name
```
You can still ask "What's the weather in Tokyo?" to override the default. The `get_weather` tool supports multi-day forecasts — ask "Will it rain tomorrow?" and the LLM passes `days=2` automatically.

**Aviation weather defaults:**
```yaml
app:
  home_airport: "KJYO"   # used when no airport is specified
  taf_airport: "KIAD"    # nearest airport with TAF (small airports often lack one)
```
Ask "What's the METAR?" → uses `home_airport`. Ask "Is there a TAF?" → tries `home_airport`, auto-falls back to `taf_airport`. Use `report_type="full"` for a complete pre-flight briefing (METAR + TAF + PIREPs).

When a tool fires, a chip badge appears on the turn card in the dashboard (e.g., `⚙ get_aviation_weather`).

### MCP Servers

Any MCP-compatible server can be wired in through config — no code changes needed. The Hub connects at startup and makes the server's tools available to the LLM alongside built-in tools.

**Startup order does not matter.** If the Node MCP server is not yet up when the Hub starts, the Hub retries the connection automatically with exponential backoff (10 s → 60 s max). Tools become available to the LLM as soon as the Node is reachable — no Hub restart needed.

```bash
pip install 'coremind[tools]'   # install the mcp SDK once
```

Add servers to Hub `config.yaml`:
```yaml
tools:
  mcp_servers:
    - name: filesystem        # stdio: spawn a local subprocess
      transport: stdio
      command: ["npx", "@modelcontextprotocol/server-filesystem", "/path/to/docs"]
    - name: node              # http: connect to the Pi's Node MCP server
      transport: http
      url: http://100.x.x.x:8767   # Pi's Tailscale IP
```

**Diagnosing MCP tool registration:**
```bash
curl http://localhost:8765/api/tools
# {"total":21,"built_in":["get_current_time",...],"mcp":["play_atc",...],"mcp_connected":true}
```

If `mcp` is an empty list, check Hub logs for `"initial connection failed"` or `"mcp package not installed"`. The Hub will keep retrying — wait ~10 s after the Node starts and query again.

### Voice-Controlled Music Player

The Pi runs a local MCP server (port 8767) that exposes 17 tools (13 music + 4 ATC). The Hub's LLM calls these over Tailscale just like any other tool.

**Pi setup:**
```bash
sudo apt install mpv          # audio player
pip install 'coremind[tools]' # mcp SDK
```

**Node `config.yaml`:**
```yaml
node_mcp:
  enabled: true
  port: 8767
  music_dir: ~/Music                           # root of your music library
  catalog_path: ~/.coremind/music-catalog.json # built by 'coremind music scan'
```

**Hub `config.yaml`:**
```yaml
tools:
  mcp_servers:
    - name: node
      transport: http
      url: http://100.x.x.x:8767
```

**Music library organization**

Organize music with Artist and Album subfolders — the catalog infers structure from folder depth:

```
~/Music/
  Miles Davis/
    Kind of Blue/
      01 - So What.mp3
      02 - Freddie Freeloader.mp3
  John Coltrane/
    A Love Supreme/
      01 - Acknowledgement.mp3
```

After adding music, build (or rebuild) the catalog:
```bash
coremind music scan
# Scanned 127 tracks — 12 artists, 18 albums.
```

The catalog is saved to `~/.coremind/music-catalog.json`. Re-run whenever you add new music. CoreMind warns in logs if the music directory is newer than the catalog.

**Available tools (13):**

| Tool | Voice example |
|------|--------------|
| `search_music` | "Find any jazz tracks" |
| `play_track` | "Play /home/pi/Music/..." |
| `play_artist` | "Play Miles Davis" / "Play Coltrane, shuffled" |
| `play_album` | "Play Kind of Blue" |
| `play_playlist` | "Play my morning playlist" |
| `stop_playback` | "Stop the music" |
| `set_volume` | "Set the volume to 60 percent" |
| `list_artists` | "What artists do I have?" |
| `list_albums` | "What albums by Miles Davis?" |
| `list_playlists` | "What playlists do I have?" |
| `create_playlist` | "Create a playlist called 'morning jazz' with these tracks" |
| `add_to_playlist` | "Add that album to my workout playlist" |
| `remove_from_playlist` | "Remove that track from morning jazz" |

Playlists are persisted in the catalog JSON immediately and survive Node restarts.

**Mic isolation during playback**

When music is playing and the wake word fires, CoreMind suspends mpv (`SIGSTOP`) before recording starts so only your voice reaches the mic. After the response is spoken, mpv resumes (`SIGCONT`) from exactly where it paused — no gap, no restart. A `try/finally` guarantees resume even on errors or timeouts. This covers both music and ATC streams since they share one mpv process slot.

---

### Live ATC Streaming

Stream live ATC audio by voice command. The same mic isolation that applies to music works here too — ATC is suspended during voice turns and resumed after the response.

**Disambiguation**

When multiple channels match a query (e.g. an airport with several tower frequencies), CoreMind asks you to pick:

```
You: Stream ATC from Dulles
CoreMind: Multiple tower frequencies found for Washington Dulles:
          - Tower Runway 1C/19C (120.250 MHz)
          - Tower Runway 1R/19L (119.850 MHz)
          Which would you like?
You: Runway 1C and 19C
CoreMind: Streaming KIAD Tower Runway 1C/19C (120.250 MHz).
```

**Setup**

```bash
# On the Pi:
sudo apt install mpv           # audio player (same as music)
pip install 'coremind[tools]'  # mcp SDK

# On the Hub (Mac Mini):
pip install 'coremind[tools]'  # mcp SDK
```

**Node `config.yaml`** (Pi):
```yaml
node_mcp:
  enabled: true          # must be true to expose ATC tools to the Hub
  port: 8767
  atc_catalog_path: ~/.coremind/atc-catalog.json
```

**Hub `config.yaml`** (Mac Mini):
```yaml
tools:
  mcp_servers:
    - name: node
      transport: http
      url: http://100.x.x.x:8767   # replace with Pi's Tailscale IP
```

Without the `mcp_servers` entry on the Hub, the LLM has no visibility into the ATC (or music) tools even if the Node MCP server is running.

**Default catalog — works out of the box**

CoreMind ships with a bundled catalog of 397 channels across 16 airports (KEWR, KJFK, KLGA, KTEB, KIAD, KDCA, KJYO, KAPA, KBOS, KORD, KATL, KLAX, KSFO, KSEA, KMIA, ZNY). If no user catalog exists at `~/.coremind/atc-catalog.json`, the Node MCP server loads this default automatically — no setup needed for these airports.

**Adding more airports (run on the Pi)**

For airports with standard mount names, scan discovers them automatically:

```bash
# SSH into the Pi, then:
coremind atc scan KBWI KPHL KPDK
# ✓  kbwi_twr  BWI Tower
# ✓  kbwi_gnd  BWI Ground
# ...
```

For airports with obfuscated mount names (KIAD-style), LiveATC's search page must be visited in a browser. `atc js` walks you through the process:

```bash
coremind atc js      # prints step-by-step instructions + JS snippet path
```

1. Open `https://www.liveatc.net/search/?icao=KXXX` in your browser
2. Open DevTools → Console, paste the JS snippet shown by `atc js`
3. Copy the JSON output and save it to a file, e.g. `~/browser-mounts.json`
4. Scan with the file — CoreMind validates each mount and adds the working ones:

```bash
coremind atc scan KXXX --browser-mounts ~/browser-mounts.json
```

The catalog is saved to `~/.coremind/atc-catalog.json`. The Node MCP server picks up changes automatically on the next tool call — no restart needed.

**Available tools (4):**

| Tool | Voice example |
|------|--------------|
| `play_atc` | "Play Newark tower" / "Stream KEWR approach" / "Put on Dulles ground" |
| `list_atc_airports` | "What airports do you have ATC for?" |
| `list_atc_channels` | "What ATC channels do you have for Newark?" |
| `stop_atc` | "Stop the ATC" |

---

## Persistent Service — systemd

Running `coremind run` in a tmux session works, but a systemd user service starts automatically at boot and restarts on crashes.

**Install the service:**
```bash
REPO=~/core-mind   # adjust if you cloned elsewhere

mkdir -p ~/.config/systemd/user
cp $REPO/coremind/service/coremind.service ~/.config/systemd/user/coremind.service
sed -i "s|%h/core-mind|$REPO|g" ~/.config/systemd/user/coremind.service
systemctl --user daemon-reload
```

**Enable and start:**
```bash
systemctl --user enable coremind   # auto-start at login
systemctl --user start coremind    # start now
```

**Start at boot without logging in first (run once):**
```bash
loginctl enable-linger $USER
```

**Status and logs:**
```bash
systemctl --user status coremind
journalctl --user -u coremind -f        # follow live logs
journalctl --user -u coremind -n 50     # last 50 lines
```

**Common commands:**
```bash
systemctl --user stop coremind          # stop
systemctl --user restart coremind       # restart after a config change
systemctl --user disable coremind       # remove from auto-start
```

> The service file uses `%h` for your home directory and waits 3 seconds after boot to let audio devices settle.

---

## All-on-Pi Mode (No Hub)

Set `remote_brain.enabled: false` and configure everything locally:

```yaml
remote_brain:
  enabled: false

stt:
  provider: whisper_local
  model: base    # tiny is fastest on Pi CPU

tts:
  provider: espeak    # zero-config; or piper_local if you install piper-tts

brain:
  provider: ollama

ollama:
  base_url: http://100.x.x.x:11434    # Mac Mini Ollama (recommended)
  # or: http://localhost:11434         # if Ollama runs on the Pi itself
```

Install STT and espeak on the Pi:
```bash
pip install 'coremind[stt]'
sudo apt install espeak-ng
```

---

## Updating

When you pull changes from the repo:

```bash
cd core-mind
git pull
# Editable install: source changes take effect immediately.
# Only reinstall if pyproject.toml changed:
pip install -e ".[dev]"
```

If running the systemd service:
```bash
systemctl --user restart coremind
```

---

## Commands Reference

```bash
# Setup — any device
coremind setup                          # open config UI at http://localhost:8766
coremind setup --port 9000             # custom port

# Hub (Mac Mini)
coremind server                         # start Hub on port 8765
coremind server --port 9000             # custom port

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

# ATC streams (Node) — default catalog ships with 397 channels across 16 airports
coremind atc scan KBWI KPHL            # auto-discover more airports
coremind atc js                         # instructions for obfuscated-mount airports
coremind atc scan KXXX --browser-mounts ~/browser-mounts.json  # scan with browser data
coremind atc add KIAD "Tower" kiad1_twr_1c19c_120250 --freq 120.250  # manual entry

# Diagnostics — any device
coremind doctor                         # check Python, config, audio, Ollama, STT, TTS, disk

# systemd service (Pi)
systemctl --user status coremind
systemctl --user restart coremind
systemctl --user stop coremind
journalctl --user -u coremind -f
journalctl --user -u coremind -n 50

# Development
pytest                                  # run unit tests (no hardware required)
```

### `coremind doctor`

Pre-flight check with a colour-coded summary. Exit code is 1 if any check fails.

```
┌──────── CoreMind Doctor ────────────────────────┐
│   OK   Python version    3.11.9                  │
│   OK   Config file       config.yaml (mode: hub) │
│   OK   Audio input       3 device(s) found       │
│   OK   Audio output      2 device(s) found       │
│   OK   Ollama            gemma4:e4b available     │
│   OK   STT               faster-whisper available │
│   OK   TTS               espeak-ng found          │
│   OK   Disk write        /tmp is writable         │
└─────────────────────────────────────────────────┘
```

---

## Package Architecture

```
coremind/
  airports.py     Bundled ICAO database (19K airports, offline lookup + search)
  data/
    airports.json OurAirports data — ICAO → name/city/country/IATA
  config/         Pydantic settings models (loaded from config.yaml)
  audio_input/    Microphone recording + VAD
  audio_output/   Speaker playback (with auto-resampling for USB speakers)
  stt/            Speech-to-text (faster-whisper, mock)
  tts/            Text-to-speech (Piper, espeak, mock)
  brain/          LLM client and router (Ollama + tool calling, mock)
  memory/         Session memory (short-term conversation context)
  wake_word/      Wake-word detection (openwakeword/onnx, dummy/Enter-key)
  vad/            Voice activity detection (energy-based)
  server/         CoreMind Hub — FastAPI server + web dashboard
  tools/          Tool dispatcher, built-in tools, Hub MCP client
    built_in/     get_current_time, get_weather, get_aviation_weather, lookup_airport
  node_mcp/       Node MCP server — exposes Pi capabilities to Hub (17 tools)
    playback.py   Shared mpv process (music + ATC mutual exclusion)
    catalog.py    Music catalog (scan, search, playlist CRUD)
    atc_catalog.py ATC stream catalog (scoring-based channel search)
    tools/        music_player, atc_player, volume_control
```

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repository setup | ✓ |
| 1 | Audio device diagnostics | ✓ |
| 2 | Push-to-talk voice loop | ✓ |
| 3 | LLM backend / brain | ✓ |
| 4 | Speech-to-text (Whisper) | ✓ |
| 5 | Text-to-speech (Piper, espeak) | ✓ |
| 6 | Full MVP v0.1 | ✓ |
| 7 | Wake word (openwakeword) | ✓ |
| 8 | Voice activity detection | ✓ |
| 9 | CoreMind Hub web server + dashboard | ✓ |
| 10a | Tool layer — built-in tools (time, weather, aviation weather) | ✓ |
| 10b | Tool layer — Hub MCP client (connect to any MCP server) | ✓ |
| 10c | Tool layer — Node MCP server + music player + mic isolation | ✓ |
| 12 | systemd user service (Pi auto-start at boot) | ✓ |
| 13 | `coremind doctor` diagnostics command | ✓ |
| 14 | Node auto-registration + Hub Nodes panel | ✓ |
| 15 | Mobile-responsive dashboard (iPhone + Tailscale) | ✓ |
| 16 | Follow-up loop safety (stop phrases, min-word filter, cooldown) | ✓ |
| 34 | Music catalog (artist/album/playlist browse + voice CRUD) | ✓ |
| 35 | Live ATC streaming (auto-discovery + manual catalog; disambiguation; 4 MCP tools) | ✓ |
| 36 | Bundled ICAO airport database (19K airports, offline `lookup_airport` tool; auto-fills ATC catalog) | ✓ |
