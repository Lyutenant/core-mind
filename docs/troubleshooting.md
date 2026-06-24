# Troubleshooting

---

## `coremind doctor`

Run this first on any device. It checks Python version, config, audio devices, Ollama, STT, TTS, and disk write access, then prints a colour-coded summary.

```bash
coremind doctor
```

Example output:
```
┌──────── CoreMind Doctor ─────────────────────────┐
│   OK   Python version    3.11.9                   │
│   OK   Config file       config.yaml (mode: hub)  │
│   OK   Audio input       3 device(s) found        │
│   OK   Audio output      2 device(s) found        │
│   OK   Ollama            gemma4:e4b available      │
│   OK   STT               faster-whisper available  │
│   OK   TTS               espeak-ng found           │
│   OK   Disk write        /tmp is writable          │
└──────────────────────────────────────────────────┘
```

Exit code is 1 if any check fails. Each `FAIL` line includes a hint (e.g. the install command for a missing dependency).

---

## Logs

**Node (Pi) — systemd service:**
```bash
journalctl --user -u coremind -f        # follow live logs
journalctl --user -u coremind -n 50     # last 50 lines
systemctl --user status coremind
```

**Hub (Mac Mini) — foreground:**
Logs print directly to the terminal when you run `coremind server`.

**Log level** is configured in `config.yaml`:
```yaml
app:
  log_level: INFO   # DEBUG for verbose output
```

---

## MCP tools not showing up

```bash
curl http://localhost:8765/api/tools
# {"total":4,"built_in":[...],"mcp":[],"mcp_connected":false}
```

If `mcp` is empty:
1. Check that `pip install 'coremind[tools]'` was run on the Hub.
2. Check that the Node is running (`systemctl --user status coremind` on the Pi).
3. Check Hub logs for `"initial connection failed"` — the Hub retries automatically; wait ~10 s after the Node starts.
4. If using Caddy on the Pi, confirm Caddy is running (`systemctl status caddy`) and the Tailscale IP in the Hub config matches the Pi's IP (`tailscale ip -4`).

Caddy-on-the-Pi failure signatures (see `Caddyfile.node.example` for the working config):

| Symptom | Cause |
|---------|-------|
| Caddy fails to start: `address already in use` | Missing `bind 100.x.x.x` — without it Caddy binds the wildcard `0.0.0.0:8767`, which conflicts with the MCP server's `127.0.0.1:8767` |
| Caddy fails at boot but starts manually | Caddy started before Tailscale was up — add the systemd ordering override (see `docs/setup-node.md`) |
| Hub gets **403** | `remote_ip` allowlist doesn't include the Hub's Tailscale IP |
| Hub gets **421** | Missing `header_up Host localhost:8767` (the port suffix is required) |

---

## ATC stream not playing

Use the built-in test command to diagnose without the voice loop:
```bash
coremind atc test "KIAD tower"
```

This checks the LiveATC PLS endpoint for liveness and runs mpv with visible stderr — if mpv is failing silently during normal playback, the error will appear here. It resolves the query exactly like the voice loop, including the tower default and random tower pick.

---

## ATC plays an unexpected channel (or asks instead of playing)

This is usually the channel matcher, not a streaming problem (see [Tools — channel selection](tools.md#live-atc-streaming)):

- **Naming only an airport plays a tower** — that's the default. Say a channel type ("ground", "atis", "approach"…) for anything else.
- **A different tower plays each time** — airports with per-runway towers (KIAD, KATL, …) pick one primary tower at random. Say a frequency ("Dulles tower 120.250") to pin one.
- **It asks "which would you like?"** — several same-type channels matched (e.g. multiple ground frequencies). Answer with a distinguishing word from the list or a frequency.
- **It refuses with "No channel matches that frequency"** — the spoken frequency is stale/mistyped, or the catalog entry has no frequency data to confirm against. Re-ask without the frequency, or fix the entry with `coremind atc add ... --freq`.

---

## Audio device problems

```bash
coremind audio list-devices     # see available device indexes
coremind audio record-test -s 5 -o test.wav
coremind audio play-test -f test.wav
```

If recording works but playback fails with a sample rate error, the speaker may require resampling. CoreMind auto-resamples; if the issue persists, try specifying the output device index explicitly in `config.yaml`.

---

## Wake word not triggering

- Lower the threshold in the Nodes panel (try 0.35–0.45). Adjust in real time — no Node restart needed.
- Check that `inference_framework: onnx` is set (not `tflite`) — tflite has no Pi ARM64 wheel for Python 3.11+.
- Run `coremind doctor` and check for `WARN` on the wake word entry.

---

## Vision / "what do you see?" not working

The camera lives on the **Node (Pi)**; the vision model runs on the **Hub (Mac)**. Test each half on its own machine.

- **On the Pi:** `coremind vision test -o frame.jpg`.
  - `opencv is not installed` → `pip install 'coremind[vision]'`.
  - `Could not open camera index N` → wrong index. Try `--device 1` (or 2…); confirm the USB webcam is detected (`ls /dev/video*`).
- **On the Mac:** `coremind vision describe --image frame.jpg`.
  - `ollama.vision_model is not set` → set it in the Hub `config.yaml` (e.g. `llava:7b`).
  - Model errors / not found → `ollama pull llava:7b` (or `moondream` if low on RAM).
- **The assistant won't "look" at all:** the `look` tool is opt-in. Confirm all three: `ollama.vision_model` is set, `look` is in the Hub's `tools.built_in`, and the Node's `vision.enabled: true`. Then check `GET /api/tools` — `capture_image` should be registered on the Node side and `look` available on the Hub.
- **It says it can't get an image:** the Node is unreachable or its camera is off. Confirm `node_mcp.enabled: true` on the Pi, the `node` entry in the Hub's `tools.mcp_servers`, and `mcp_connected: true` in `GET /api/tools`.

---

## Random noise triggers the wake word

Don't just raise `threshold` — that also makes the real wake phrase harder to detect. Reach
for the **Wake VAD Gate** first:

- Raise **Wake VAD Gate** from 0 in the Nodes panel (try ~0.5). This is openWakeWord's Silero
  speech pre-gate: it suppresses wake predictions on frames with no speech, so pure room noise
  stops firing the wake — while the spoken phrase still triggers at the same `threshold`. Silero
  loads on demand the first time you enable the gate, so this works live, no Node restart needed.
- If false-wakes persist at a high gate, nudge `threshold` up a little as a secondary step.

---

## Updating

```bash
cd core-mind
git pull
# Editable install: Python source changes take effect immediately.
# Only reinstall if pyproject.toml changed:
pip install -e ".[dev]"
```

If running the systemd service on the Pi:
```bash
systemctl --user restart coremind
```
