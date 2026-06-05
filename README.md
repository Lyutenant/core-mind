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

## Hardware

| Component | Details |
|-----------|---------|
| Raspberry Pi 5, 8 GB | CoreMind Node |
| USB microphone or ReSpeaker XVF3800 | connected to Pi |
| Speaker (USB, 3.5 mm, or HDMI) | connected to Pi |
| Mac Mini | CoreMind Hub — runs Ollama + Hub server |
| Tailscale | connects Pi and Mac Mini on a private network |

---

## Quick Architecture Decision

In the default **remote brain** mode, the Pi only handles audio I/O and wake word. All heavy computation (STT, LLM, TTS) runs on the Mac Mini. This reduces per-turn latency from ~30 s to ~12–15 s (dominated by the LLM).

If you want to run everything on the Pi (no Mac Mini), set `remote_brain.enabled: false` in the Pi config and configure STT, TTS, and Ollama to run locally.

---

## Part 1 — Mac Mini Setup (CoreMind Hub)

### 1.1 Install system dependencies

```bash
# macOS — Homebrew
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
```

Verify:
```bash
python --version   # should print Python 3.11.x
```

### 1.4 Install the package and server dependencies

```bash
pip install -e ".[dev,stt,server]"
```

This installs:
- `coremind` in editable mode (changes to source take effect immediately)
- `faster-whisper` for local speech-to-text
- `fastapi`, `uvicorn`, `python-multipart` for the Hub web server

**Optional — Piper TTS (neural voice, recommended):**
```bash
pip install piper-tts
```

Then download a voice model from [rhasspy/piper-voices](https://github.com/rhasspy/piper-voices). You need both the `.onnx` and `.onnx.json` files:
```bash
mkdir -p ~/piper-voices
curl -L -o ~/piper-voices/en_US-amy-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
curl -L -o ~/piper-voices/en_US-amy-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
```

**Optional — espeak-ng (simpler, robotic voice, no model download):**
```bash
brew install espeak-ng   # already done in 1.1 if you ran it
```

### 1.5 Create the Hub config

**Option A — browser UI (recommended):**
```bash
source .venv/bin/activate
coremind setup        # opens config UI at http://localhost:8766
```
Open the URL, set Mode to **Hub**, fill in the settings, click **Save Configuration**, then Ctrl+C and start the Hub normally.

**Option B — copy the example file and edit manually:**
```bash
cp config.hub.example.yaml config.yaml
```

Key Hub settings:

```yaml
mode: hub

app:
  name: Jarvis          # your assistant's name
  user_location: "San Francisco, CA"   # default location for weather/time questions (optional)
  user_timezone: "America/Los_Angeles" # IANA timezone — used by time tool (optional)
  home_airport: "KJYO"                 # default airport for aviation weather queries (optional)
  taf_airport: "KIAD"                  # nearest airport with TAF, used when home airport lacks one

runtime:
  follow_up_seconds: 5.0   # after a response, listen this long for a follow-up; 0.0 to disable

stt:
  provider: whisper_local
  model: base           # tiny/base/small — base is fastest reasonable choice on Mac

tts:
  provider: piper_local
  model_path: /Users/yourname/piper-voices/en_US-amy-medium.onnx
  # or: provider: espeak

brain:
  provider: ollama
  timeout_seconds: 90

ollama:
  base_url: http://localhost:11434    # Ollama runs locally on Mac Mini
  model: gemma4:e4b                   # must be pulled: ollama pull gemma4:e4b
  no_think: false

tools:
  enabled: true
  built_in: [time, weather, aviation_weather]   # add aviation_weather for METAR/TAF/PIREP

remote_brain:
  enabled: false    # Hub does not forward to another Hub
```

### 1.6 Start the Hub

```bash
source .venv/bin/activate
coremind server
# or: python -m coremind.main server
```

You should see:
```
CoreMind Hub starting — listening on 0.0.0.0:8765
  Dashboard: http://localhost:8765
  Node config: set remote_brain.enabled: true and remote_brain.url: http://<this-host>:8765
```

Open **http://localhost:8765** in your browser. The dashboard shows live conversation turns and lets you edit all settings without touching YAML.

### 1.7 Find your Mac Mini's Tailscale IP

```bash
tailscale ip -4
# example: 100.x.x.x
```

You will need this IP when configuring the Node.

---

## Part 2 — Raspberry Pi Setup (CoreMind Node)

### 2.1 Install system dependencies

```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev python3.11 python3.11-venv git
```

Check the Python version:
```bash
python3.11 --version    # should print Python 3.11.x
```

If 3.11 is not available from apt:
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
```

Verify:
```bash
python --version   # Python 3.11.x
```

### 2.4 Install the package and Node dependencies

```bash
pip install -e ".[dev]"
```

**Optional — wake word detection (required for always-on "Hey Jarvis" mode):**
```bash
# Step 1: install onnxruntime (the inference backend we use on Pi)
pip install 'coremind[wake_word]'

# Step 2: install openwakeword WITHOUT its dependencies
# (its tflite-runtime dependency has no Pi ARM64 wheel for Python 3.11+)
pip install openwakeword --no-deps

# Step 3: download the built-in wake word models
python -c "import openwakeword; openwakeword.utils.download_models()"
```

> **Note:** Always use `inference_framework: onnx` in config. The `tflite` backend requires `tflite-runtime` which is incompatible with NumPy 2.x on Raspberry Pi OS.

### 2.5 Create the Node config

**Option A — browser UI:**
```bash
source .venv/bin/activate
coremind setup        # opens config UI at http://<pi-ip>:8766
```
Set Mode to **Node**, enter the Hub URL, set audio device indexes, save, Ctrl+C.

**Option B — copy the example file:**
```bash
cp config.node.example.yaml config.yaml
```

Key Node settings:

```yaml
mode: node

app:
  name: Jarvis   # must match the Hub's name

audio:
  # Run: coremind audio list-devices  to find device indexes
  input_device: null    # or: 0, 1, 2 — your USB microphone index
  output_device: null   # or: 0, 1, 2 — your speaker index
  sample_rate: 16000    # match your mic's native rate

runtime:
  follow_up_seconds: 5.0   # after a response, listen this long for a follow-up; 0.0 to disable

remote_brain:
  enabled: true
  url: http://100.x.x.x:8765    # your Mac Mini's Tailscale IP from step 1.7
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

### 2.6 Find your audio device indexes

```bash
source .venv/bin/activate
coremind audio list-devices
```

Update `audio.input_device` and `audio.output_device` in `config.yaml` with the correct indexes.

### 2.7 Test audio recording and playback

```bash
coremind audio record-test --seconds 5 --output test.wav
coremind audio play-test --file test.wav
```

Verify that your voice was recorded and played back correctly before continuing.

### 2.8 Start the Node

```bash
source .venv/bin/activate
coremind run
```

You should see:
```
Jarvis ready. Press Ctrl+C to quit.
--- Turn 1 ---
Listening for wake word (hey_jarvis_v0.1)...
```

Say **"Hey Jarvis"** — you will hear a two-tone chime and see:
```
Wake word detected! Speak your question...
```

Speak your question. The audio is sent to the Hub, which transcribes, queries Ollama, synthesizes speech, and returns audio to play on the Pi.

---

## Part 3 — Hub Web Dashboard

Open **http://localhost:8765** (or `http://<mac-mini-ip>:8765` from another machine on Tailscale).

| Section | What it shows |
|---------|--------------|
| Dashboard | Live conversation log with tool call badges, real-time processing status |
| System status (sidebar) | Ollama reachability, STT/TTS loaded state |
| Settings → App | Mode (Hub/Node/Standalone), assistant name, log level, user location/timezone |
| Settings → STT | Whisper provider, model size, language |
| Settings → TTS | Provider (Piper/espeak), model path, voice |
| Settings → Ollama | Server URL, model, no-think, inference options |
| Settings → Memory | Session memory max turns |
| Settings → Tools | Enable/disable built-in tools (time, weather) |
| Settings → VAD | Energy threshold, silence/max/min durations |
| Settings → Wake Word | Provider, model, detection threshold |
| Settings → Audio | Device indexes, sample rate, channels |
| Settings → Hub Connection | Hub URL (for Node remote brain) |

All settings are saved to `config.yaml` on the Hub when you click **Save Configuration**. The Hub reloads automatically — no restart required for most changes (STT/TTS/LLM model changes take effect on the next request).

---

## Part 4 — Running Without Remote Brain (All-on-Pi Mode)

If you want to run everything on the Pi without a Hub:

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
  # or: http://localhost:11434         # if running Ollama on the Pi itself
```

Install STT on the Pi:
```bash
pip install 'coremind[stt]'
```

Install espeak:
```bash
sudo apt install espeak-ng
```

---

## Part 5 — Tool Calling

CoreMind uses the Ollama native tool API so the LLM can call real-world functions before answering. Tools are enabled on the Hub and run transparently — the LLM decides when to use them.

### Built-in tools (available now)

| Tool | Trigger example | Notes |
|------|----------------|-------|
| `get_current_time` | "What time is it?" | No external deps; uses `app.user_timezone` if set |
| `get_weather` | "What's the weather?" / "Forecast for tomorrow?" | wttr.in, no API key; supports 1–3 day forecasts |
| `get_aviation_weather` | "What's the METAR at Leesburg?" / "Any PIREPs nearby?" | NOAA aviationweather.gov, no API key; METAR/TAF/PIREP |

Enable in Hub `config.yaml`:
```yaml
tools:
  enabled: true
  built_in: [time, weather, aviation_weather]
```

**Weather and time defaults** — set a location and timezone so you can ask without specifying a place:
```yaml
app:
  user_location: "San Francisco, CA"   # fallback when no location is mentioned
  user_timezone: "America/Los_Angeles" # IANA name — time tool reports in this timezone
```
You can still ask "What's the weather in Tokyo?" to override the default. The timezone must be a valid [IANA timezone name](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones); an invalid value logs a warning and falls back to server local time.

The `get_weather` tool supports multi-day forecasts. Ask "What's the forecast for this week?" or "Will it rain tomorrow?" and the LLM passes `days=2` or `days=3` automatically.

**Aviation weather defaults** — set your home airport so you can ask without an ICAO code:
```yaml
app:
  home_airport: "KJYO"   # used when no airport is specified
  taf_airport: "KIAD"    # nearest airport with TAF (small airports often lack one)
```
Ask "What's the METAR?" → uses `home_airport`. Ask "Is there a TAF?" → tries `home_airport`, falls back to `taf_airport` automatically with a note. Ask "What's the METAR at KIAD?" to override. Use `report_type="full"` for a complete pre-flight briefing (METAR + TAF + PIREPs in one response).

When a tool fires, a chip badge appears on the turn card in the dashboard (e.g., `⚙ get_aviation_weather`).

### Adding MCP servers (Phase B — coming soon)

Any MCP-compatible server can be wired in through config — no code changes needed:

```yaml
tools:
  mcp_servers:
    - name: filesystem
      transport: stdio
      command: ["npx", "@modelcontextprotocol/server-filesystem", "/path/to/docs"]
```

### Node-side tools (Phase C — coming soon)

The Pi can run a local MCP server exposing its own capabilities (music playback, volume control, etc.). The Hub connects to it over Tailscale just like any other MCP server:

```yaml
# Hub config.yaml
tools:
  mcp_servers:
    - name: node
      transport: http
      url: http://100.x.x.x:8767   # Pi's Tailscale IP

# Node config.yaml
node_mcp:
  enabled: true
  port: 8767
  music_dir: ~/Music
```

---

## Part 6 — Running the Node as a Persistent Service (recommended)

Running `coremind run` in tmux works, but a systemd user service starts automatically at boot and restarts on crashes — no SSH session required.

### 6.1 Install the service file

```bash
# On the Pi — adjust the path if you cloned to a different location
REPO=~/core-mind          # change to e.g. ~/Sandbox/core-mind if needed

mkdir -p ~/.config/systemd/user
cp $REPO/coremind/service/coremind.service ~/.config/systemd/user/coremind.service

# Update the paths inside the installed file to match your repo location
sed -i "s|%h/core-mind|$REPO|g" ~/.config/systemd/user/coremind.service

systemctl --user daemon-reload
```

### 6.2 Enable and start

```bash
systemctl --user enable coremind   # start automatically at login
systemctl --user start coremind    # start right now
```

### 6.3 Start at boot (without needing to log in first)

```bash
loginctl enable-linger $USER
```

This tells systemd to start your user services at boot, even before you SSH in. Only needs to be run once.

### 6.4 Check status and logs

```bash
systemctl --user status coremind
journalctl --user -u coremind -f        # follow live logs
journalctl --user -u coremind -n 50     # last 50 lines
```

### 6.5 Common commands

```bash
systemctl --user stop coremind          # stop the service
systemctl --user restart coremind       # restart (e.g. after config change)
systemctl --user disable coremind       # remove from auto-start
```

> **Note:** The service file uses `%h` for your home directory, so it works regardless of your username. It waits 3 seconds after start to let audio devices settle after boot.

---

## Keeping the Node in Sync with the Hub

When you push changes to the repo:

```bash
# On Pi
cd core-mind
git pull
# No pip reinstall needed for source changes (editable install).
# Only reinstall if pyproject.toml changed:
pip install -e ".[dev]"
```

---

## Commands Reference

```bash
# Initial setup — any device (Hub or Node)
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
coremind chat loop                      # push-to-talk loop (local mode, no wake word)
coremind chat once                      # single push-to-talk interaction

# Development
pytest                                  # run unit tests (no hardware required)
```

---

## Architecture

```
coremind/
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
  tools/          Tool layer — dispatcher, built-in tools, MCP client (Phase B)
    built_in/     get_current_time, get_weather (multi-day), get_aviation_weather (METAR/TAF/PIREP)
  node_mcp/       Node MCP server — exposes Pi capabilities to Hub (Phase C)
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
| 10a | Tool layer — built-in tools (time, weather, aviation weather) + Ollama tool loop | ✓ |
| 10b | Tool layer — Hub MCP client (connect to any MCP server) | planned |
| 10c | Tool layer — Node MCP server (Pi exposes local capabilities) | planned |
| 11 | OpenClaw integration | planned |
| 12 | systemd user service (Pi auto-start at boot) | ✓ |
| 13 | Observability / doctor command | planned |
