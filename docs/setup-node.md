# Node Setup — Raspberry Pi

The Node runs on the Raspberry Pi. It listens for a wake word, records audio with VAD, sends it to the Hub, and plays the returned audio. All inference (STT, LLM, TTS) runs on the Hub.

---

## 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev python3.11 python3.11-venv git
python3.11 --version   # should print Python 3.11.x
```

If Python 3.11 is not available in apt:
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

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
python --version   # Python 3.11.x
```

---

## 4. Install dependencies

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

**Optional — music and ATC streaming:**
```bash
sudo apt install mpv          # audio player (used by Node MCP)
pip install 'coremind[tools]' # mcp SDK
```

**Optional — camera / vision ("what do you see?"):**
```bash
pip install 'coremind[vision]'   # opencv for USB webcam capture
coremind vision test -o frame.jpg   # plug in a USB webcam, confirm it captures
```
Then set `vision.enabled: true` in the Node config (step 5) and `ollama.vision_model` on the Hub. Full walkthrough: [Tools → Vision](tools.md#vision-look).

---

## 5. Create the Node config

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
  input_device: null    # set to your USB mic index (see step 6)
  output_device: null   # set to your speaker index
  sample_rate: 16000

runtime:
  follow_up_seconds: 5.0

remote_brain:
  enabled: true
  url: http://100.x.x.x:8765    # Hub's Tailscale IP (see Hub Setup step 7)
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

node_mcp:
  enabled: false          # set to true if you want music/ATC tools
  port: 8767
  music_dir: ~/Music
  atc_catalog_path: ~/.coremind/atc-catalog.json
```

---

## 6. Find audio device indexes

```bash
coremind audio list-devices
```

Update `audio.input_device` and `audio.output_device` in `config.yaml` with the correct indexes.

---

## 7. Test audio

```bash
coremind audio record-test --seconds 5 --output test.wav
coremind audio play-test --file test.wav
```

Verify that your voice records and plays back before continuing.

---

## 8. Start the Node

```bash
coremind run
```

```
Jarvis ready. Press Ctrl+C to quit.
--- Turn 1 ---
Listening for wake word (hey_jarvis_v0.1)...
```

Say **"Hey Jarvis"** — you will hear a chime, recording begins, and the audio response plays on the Pi.

---

## 9. Caddy reverse proxy (required for Node MCP)

If `node_mcp.enabled: true`, the Node MCP server binds to `127.0.0.1:8767`. The Hub (Mac Mini) needs to reach this port over Tailscale — Caddy on the Pi exposes it on the Tailscale interface.

**Install Caddy on the Pi:**
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy
```

**Get both Tailscale IPs** (run `tailscale ip -4` on each machine):
```bash
tailscale ip -4    # on the Pi      → 100.x.x.x
tailscale ip -4    # on the Mac Mini → 100.y.y.y
```

**Install the Caddyfile** (Caddy's systemd service reads `/etc/caddy/Caddyfile`):
```bash
sudo cp Caddyfile.node.example /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile   # replace 100.x.x.x (Pi) and 100.y.y.y (Mac Mini)
```

The Caddyfile:
```
{
    auto_https off
}

:8767 {
    # Listen only on the Pi's Tailscale IP (avoids the 8767 port conflict)
    bind 100.x.x.x

    # Allow the Hub (Mac Mini) + the Pi itself for local testing
    @hub remote_ip 100.x.x.x 100.y.y.y

    handle @hub {
        reverse_proxy 127.0.0.1:8767 {
            header_up Host localhost:8767
        }
    }

    respond "Forbidden" 403
}
```

Three details that matter:

- **`header_up Host localhost:8767` is required** — FastMCP's DNS-rebinding guard only accepts `Host: localhost:*` values; without the port it rejects the request with 421.
- **`bind 100.x.x.x` (the Pi's Tailscale IP) is required** — without it, Caddy binds the wildcard `0.0.0.0:8767`, which on Linux conflicts with the MCP server's `127.0.0.1:8767` bind ("address already in use"). A site address like `http://100.x.x.x:8767` does **not** restrict the listener — it only matches the Host header — so the explicit `bind` directive is what prevents the conflict. The Hub's Caddyfile on macOS can use a wildcard bind; the Pi cannot.
- **The `remote_ip` allowlist must contain the Hub's IP** (`100.y.y.y`, the Mac Mini) — that's who connects through the proxy. The Pi's own IP (`100.x.x.x`) is also listed so you can `curl http://100.x.x.x:8767` from the Pi when testing — since Caddy only listens on the Tailscale IP, that local request arrives from the Pi's Tailscale address, not loopback. Other tailnet devices (phones, laptops) get 403.

**Start Caddy:**
```bash
sudo systemctl enable caddy
sudo systemctl start caddy
```

**Fix the boot ordering** (recommended): because Caddy binds the Tailscale IP, it fails at boot if it starts before Tailscale is up. Make Caddy wait:
```bash
sudo systemctl edit caddy
```
Add:
```ini
[Unit]
After=tailscaled.service
Wants=tailscaled.service
```
Then `sudo systemctl daemon-reload && sudo systemctl restart caddy`.

**Update Hub `config.yaml`** to point to the Pi's Tailscale IP:
```yaml
tools:
  mcp_servers:
    - name: node
      transport: http
      url: http://100.x.x.x:8767   # Pi's Tailscale IP
```

---

## 10. Persistent service — systemd

A systemd user service starts the Node automatically at boot and restarts on crashes.

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

**Restart after a config change:**
```bash
systemctl --user restart coremind
```

---

## All-on-Pi mode (no Hub)

Set `remote_brain.enabled: false` and configure STT/TTS locally:

```yaml
remote_brain:
  enabled: false

stt:
  provider: whisper_local
  model: base          # tiny is fastest on Pi CPU

tts:
  provider: espeak     # zero-config; or piper_local if you install piper-tts

ollama:
  base_url: http://100.x.x.x:11434    # Mac Mini Ollama (recommended)
  # or: http://localhost:11434          # if Ollama runs on Pi
```

Install STT and espeak:
```bash
pip install 'coremind[stt]'
sudo apt install espeak-ng
```
