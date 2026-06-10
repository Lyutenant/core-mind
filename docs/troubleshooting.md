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

---

## ATC stream not playing

Use the built-in test command to diagnose without the voice loop:
```bash
coremind atc test "KIAD tower"
```

This checks the LiveATC PLS endpoint for liveness and runs mpv with visible stderr — if mpv is failing silently during normal playback, the error will appear here.

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
