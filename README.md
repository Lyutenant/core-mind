# CoreMind

A two-component voice assistant system built around a Raspberry Pi and a Mac Mini.

```
[CoreMind Node]  ──audio──▶  [CoreMind Hub]  ──▶  Ollama LLM
 Raspberry Pi                  Mac Mini             (local)
 wake word                     STT (Whisper)
 VAD record                    TTS (Piper)
 play audio   ◀──audio──       response
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

```bash
cp config.hub.example.yaml config.yaml
```

Edit `config.yaml`. Key Hub settings:

```yaml
app:
  name: Jarvis          # your assistant's name

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
  model: qwen3:8b                     # must be pulled: ollama pull qwen3:8b
  no_think: true                      # skip chain-of-thought for faster voice responses

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
pip install 'coremind[wake_word]'
# also install onnxruntime for Pi-compatible inference:
pip install onnxruntime
# download the built-in wake word models:
python -c "import openwakeword; openwakeword.utils.download_models()"
```

> **Note:** Use `inference_framework: onnx` in config (not tflite). The tflite runtime is incompatible with NumPy 2.x which ships with Raspberry Pi OS.

### 2.5 Create the Node config

```bash
cp config.node.example.yaml config.yaml
```

Edit `config.yaml`. Key Node settings:

```yaml
app:
  name: Jarvis   # must match the Hub's name

audio:
  # Run: coremind audio list-devices  to find device indexes
  input_device: null    # or: 0, 1, 2 — your USB microphone index
  output_device: null   # or: 0, 1, 2 — your speaker index
  sample_rate: 16000    # match your mic's native rate

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
| Dashboard | Live conversation log, real-time processing status |
| System status (sidebar) | Ollama reachability, STT/TTS loaded state |
| Settings → App | Assistant name, log level |
| Settings → STT | Whisper provider, model size, language |
| Settings → TTS | Provider (Piper/espeak), model path, voice |
| Settings → Ollama | Server URL, model, no-think, inference options |
| Settings → Memory | Session memory max turns |
| Settings → VAD | Energy threshold, silence/max/min durations |
| Settings → Wake Word | Provider, model, detection threshold |
| Settings → Audio | Device indexes, sample rate, channels |
| Settings → Remote Brain | Hub URL (for Node config reference) |

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
  config/       Pydantic settings models (loaded from config.yaml)
  audio_input/  Microphone recording + VAD
  audio_output/ Speaker playback
  stt/          Speech-to-text (faster-whisper, mock)
  tts/          Text-to-speech (Piper, espeak, mock)
  brain/        LLM client and router (Ollama, mock)
  memory/       Session memory (short-term conversation context)
  wake_word/    Wake-word detection (openwakeword, dummy/Enter-key)
  vad/          Voice activity detection (energy-based)
  server/       CoreMind Hub — FastAPI server + web dashboard
  tools/        Tool registry (Phase 10, not yet implemented)
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
| 10 | Tool layer | planned |
| 11 | OpenClaw integration | planned |
| 12 | systemd service | planned |
| 13 | Observability / doctor command | planned |
