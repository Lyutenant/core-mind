# Hub Setup — Mac Mini

The Hub runs on the Mac Mini. It serves the web dashboard, handles STT, calls Ollama, runs TTS, and returns audio to the Node.

---

## 1. Install system dependencies

```bash
brew install portaudio espeak-ng
```

`portaudio` is required for audio I/O. `espeak-ng` is the fallback TTS voice (optional if you use Piper instead).

---

## 2. Clone the repository

```bash
git clone https://github.com/Lyutenant/core-mind.git
cd core-mind
```

---

## 3. Create the Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version   # should print Python 3.11.x
```

---

## 4. Install dependencies

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

---

## 5. Create the Hub config

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
  model: base               # tiny / base / small

tts:
  provider: piper_local
  model_path: /Users/yourname/piper-voices/en_US-amy-medium.onnx
  # or: provider: espeak

ollama:
  base_url: http://localhost:11434
  model: gemma4:e4b          # ollama pull gemma4:e4b

tools:
  enabled: true
  built_in: [time, weather, aviation_weather, airport]
```

See [Tools](tools.md) for MCP server config (music, ATC, etc.).

---

## 6. Start the Hub

```bash
coremind server
```

By default the Hub binds to `127.0.0.1:8765` — only accessible locally. To reach it from other devices on your Tailscale network, set up Caddy (next section).

---

## 7. Caddy reverse proxy

Caddy listens on the Tailscale interface and proxies to the Hub running on loopback.

**Install Caddy:**
```bash
brew install caddy
```

**Get your Tailscale IP:**
```bash
tailscale ip -4
# e.g., 100.x.x.x
```

**Create a Caddyfile** (or copy the example):
```bash
cp Caddyfile.hub.example Caddyfile
# Edit and replace 100.x.x.x with your actual Tailscale IP
```

The example Caddyfile:
```
http://100.x.x.x:8765 {
    reverse_proxy localhost:8765
}
```

**Run Caddy:**
```bash
caddy run
```

The dashboard is now reachable at `http://<tailscale-ip>:8765` from any device on your Tailscale network, including iPhone.

**Run Caddy as a background service (optional):**
```bash
brew services start caddy
```
This starts Caddy automatically at login and loads `~/.config/caddy/Caddyfile` (or the default system Caddyfile). Copy your Caddyfile there if you use this option.

---

## 8. Verify

Run the pre-flight check:
```bash
coremind doctor
```

Expected output:
```
┌──────── CoreMind Doctor ─────────────────────────┐
│   OK   Python version    3.11.x                   │
│   OK   Config file       config.yaml (mode: hub)  │
│   OK   Ollama            gemma4:e4b available      │
│   OK   STT               faster-whisper available  │
│   OK   TTS               espeak-ng found           │
│   OK   Disk write        /tmp is writable          │
└──────────────────────────────────────────────────┘
```

Open the dashboard in your browser:
- **Local only:** `http://localhost:8765`
- **Via Tailscale (with Caddy):** `http://<tailscale-ip>:8765`
