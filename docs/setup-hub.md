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
  model: distil-large-v3    # tiny / base / small / medium / distil-large-v3 / large-v3
                            # distil-large-v3 recommended: small models (tiny/base)
                            # mishear accents and proper nouns
  compute_type: int8_float32  # int8 (fastest) / int8_float32 / float32 (most accurate)
  beam_size: 8                # higher = more accurate, slower (5–8 is a good range)
  vad_filter: true            # drop silence/noise before decoding
  hotwords: "CoreMind, KJYO, METAR, Ollama"   # bias decoding toward your vocabulary
  initial_prompt: "Aviation and home-assistant voice commands."

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

### Improving recognition accuracy

If distil-large-v3 still mishears your speech, tune these before reaching for a
bigger model (all editable in the dashboard **Settings → STT** tab):

- **`hotwords`** — a comma/space-separated list of names, jargon, and ICAO codes
  to bias the decoder toward. This is the practical substitute for fine-tuning on
  your own voice: add the specific words Whisper keeps getting wrong and grow the
  list over time. (Requires faster-whisper ≥ 1.0; older versions fold the list
  into `initial_prompt` automatically.)
- **`initial_prompt`** — a short context sentence describing your domain, which
  nudges spelling and word choice.
- **`vad_filter: true`** — strips silence/noise before decoding, which cuts the
  hallucinated text Whisper sometimes emits during pauses.
- **`beam_size: 8`** and **`compute_type: int8_float32`** — trade a little speed
  for accuracy. The Mac Mini can usually afford both.

True fine-tuning on recordings of your voice is possible but heavy (needs labeled
audio, a GPU run, and re-export to CTranslate2 format) — the knobs above solve
most accuracy problems without it.

---

## 6. Start the Hub

```bash
coremind server
```

By default the Hub binds to `127.0.0.1:8765` — only accessible locally. To reach it from other devices on your Tailscale network, set up Caddy (next section).

---

## 7. Caddy reverse proxy

Caddy listens on all interfaces and allows only loopback + Tailscale clients; everyone else gets 403. The Hub itself stays on `127.0.0.1`.

**Install Caddy:**
```bash
brew install caddy
```

**Create a Caddyfile** (or copy the example — no editing needed):
```bash
cp Caddyfile.hub.example Caddyfile
```

The example Caddyfile:
```
:8765 {
    @allowed remote_ip 127.0.0.1 ::1 100.64.0.0/10

    handle @allowed {
        reverse_proxy 127.0.0.1:8765
    }

    respond "Forbidden" 403
}
```

Why `:8765` instead of binding the Tailscale IP:
- **No boot race** — binding a specific Tailscale IP fails if Caddy starts before Tailscale is up; `:8765` always binds.
- **No IP to configure** — `100.64.0.0/10` is the entire Tailscale address range, so the file works as-is.
- On macOS, Caddy's wildcard bind coexists with the Hub's `127.0.0.1:8765` bind on the same port: local connections go straight to the Hub, remote connections go through Caddy. (This is macOS-specific — the Pi's Node Caddyfile uses an interface bind instead; see `docs/setup-node.md`.)

To restrict access to specific devices instead of the whole tailnet, list their Tailscale IPs:
```
@allowed remote_ip 127.0.0.1 ::1 100.x.x.x 100.y.y.y
```

**Run Caddy:**
```bash
caddy run
```

The dashboard is now reachable at `http://<mac-tailscale-ip>:8765` from any device on your Tailscale network, including iPhone.

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
