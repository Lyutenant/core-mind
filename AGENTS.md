# AGENTS.md — CoreMind Voice Assistant

The operating manual for anyone, human or AI coding agent (Claude Code, Codex), working on this
repo. It's committed, so it switches with the branch: it always describes the code that is
checked out. Keep it project-wide and durable. Feature-in-progress notes go in
`.ai-notes/<branch>.md`, and detailed subsystem design in
[`docs/architecture.md`](docs/architecture.md).

## Session Start (every session, before writing code)

1. **Read `AGENTS.local.md`** at the repo root. This gitignored, per-machine file declares your
   **role** (see Roles) and any machine-specific notes. If it's missing, or the role is not one of
   `maintainer` / `agent`, your role is **`agent`**.
2. **Find your branch note:** run `git branch --show-current`, replace every `/` with `__`, and
   read `.ai-notes/<that>.md` if it exists (`feat/login` → `.ai-notes/feat__login.md`). It is
   the source of truth for that branch's goals, decisions and TODOs. Keep it updated while you
   work. Ignore other branches' notes; they don't apply here.
3. **Run `git status`** so you know what's already changed.
4. **Stay on one branch per session.** Don't `git checkout` another branch mid-session, because
   your loaded context would describe the wrong code. Start a new session, or use a separate
   worktree.

## Roles

### `maintainer`: human-supervised (the maintainer's own machine)

- Work on `main` by default. Use a `feat/<topic>` / `fix/<topic>` branch + PR only when the
  maintainer asks for one.
- **Commit only when explicitly asked.** Push only when asked.
- May give deploy/operations guidance for the Hub and the Pi.
- Identity check: if `git config user.name` is the AI-agent account (`GLaDos-Unit-02`), stop and
  ask. The role file is probably on the wrong machine.

### `agent`: autonomous (the dedicated AI-agent account on the dev Mac Mini)

- Always work on your own branch: `claude/<topic>` (Claude Code) or `codex/<topic>` (Codex),
  created from the latest `origin/main`, in **your own git worktree**. Claude and Codex never
  share a checkout.
- Commit freely on your branch (no need to ask). Run the tests before every push.
- Push early and open a **draft PR** (`gh pr create --draft`), filling in the PR template.
- Before marking the PR ready:
  - rebase onto `origin/main` and push with `--force-with-lease` (your own branch only)
  - make sure the tests pass locally and CI is green
  - review your own diff
  - complete the PR description
- Review feedback: read it with `gh pr view <N> --comments`, push fixes to the same branch, and
  reply on the PR.
- **Never:**
  - push to `main`, merge or approve a PR, or force-push anything but your own branch
  - bump the version or create tags
  - deploy, install or modify system services, or touch `~/.coremind`
- **This checkout is never deployed.** The live Hub may run on the same machine under another
  user:
  - never bind `0.0.0.0`, and never use ports 8765/8767
  - test servers with FastAPI's `TestClient`, or on an alternate loopback port
- You have no hardware (mic, speaker, camera, Pi). Anything that needs a real-device check goes
  under **"Hardware verification needed"** in the PR, never "verified".

## Branches, PRs and Notes

- **Branch names:** `claude/<topic>`, `codex/<topic>` (agent); `feat/<topic>`, `fix/<topic>`
  (maintainer).
- **Merging:** the maintainer reviews and **squash-merges**, using the PR title as the commit
  message in this repo's style (`feat: …`, `fix: …`, `chore: …`, `refactor: …`, `docs: …`).
  Branches are deleted on merge. `main` is protected: a PR, an approval and a passing `tests`
  check are required, with a maintainer bypass.
- **Changes that touch hardware** (audio, wake word, camera, playback) are tested by the
  maintainer on the real Hub/Pi before merging.
- **`.ai-notes/<branch>.md`** (gitignored) is local working state while you work.
- **The PR description is the handoff** to the maintainer.
- **Before merge,** move anything durable into this file or `docs/`, in the same PR.
- **Update this file** (and `docs/architecture.md`) in the same change whenever behavior,
  architecture, commands or config change.

---

## What This Project Is

**CoreMind** is a two-component voice assistant. A Raspberry Pi handles audio I/O (wake word, recording, playback); a Mac Mini runs all inference (STT, LLM, TTS).

**CoreMind Hub** — Mac Mini
- FastAPI server + web dashboard at `http://<mac-mini>:8765` (the primary config UI — no YAML editing needed)
- STT (faster-whisper), LLM routing (Ollama), TTS (Piper/espeak), tool execution
- Start: `coremind server --host 0.0.0.0` (or behind Caddy with default `127.0.0.1` bind)

**CoreMind Node** — Raspberry Pi 5
- Always-on audio terminal: wake word (openwakeword), VAD recording, audio playback
- Also runs a local MCP server (port 8767) exposing music/ATC/volume tools to the Hub
- Start: `coremind run` (installed as a systemd user service for boot autostart)

### Data flow

```text
[Node — Pi]                          [Hub — Mac Mini]
wake word detected
→ record audio (VAD)
→ POST /v1/process (WAV) ──────────▶ STT (Whisper)
                                     → LLM (Ollama) ──▶ tool calls?
                                     →   execute tools (built-in + MCP)
                                     →   LLM final response
                                     → TTS (Piper)
← receive audio WAV   ◀──────────── return WAV + headers
→ play audio
→ follow-up listening window (no wake word needed)
```

### Hardware

- Raspberry Pi 5 (8 GB), Raspberry Pi OS 64-bit, USB mic + speaker
- Mac Mini running Ollama, reachable from the Pi over **Tailscale**
- Development happens on the maintainer's MacBook and on a dedicated dev Mac Mini (AI-agent account); code goes through GitHub and is pulled onto the deployment Hub (Mac Mini, a separate account) and the Pi
- Treat mic/speaker as plain ALSA/PipeWire devices — no special hardware features assumed

---

## Package Layout

```text
coremind/
  main.py           Typer CLI: run, server, setup, doctor, audio, chat, music, atc
  voice_loop.py     Core record→STT→LLM→TTS loop; hub-sync daemon; follow-up window;
                    stop phrases; on_listening_start/on_turn_complete callbacks
  node_id.py        Stable UUID for Node auto-registration (~/.coremind/node-id)
  airports.py       Offline ICAO airport DB (OurAirports, 19K airports)

  data/
    airports.json              19,085 airports; ~1.5 MB; public domain
    atc-catalog-default.json   397 LiveATC channels across 16 airports (fallback catalog)
    extract-liveatc-mounts.js  Browser snippet for harvesting LiveATC mount names

  audio_input/      devices.py, recorder.py
  audio_output/     player.py (auto-resampling for USB speakers)
  stt/              base.py, whisper_local.py (faster-whisper) + MockSTT,
                    whisper_cpp.py (whisper.cpp via pywhispercpp; Metal on Mac),
                    __init__.py make_stt() — the single provider-selection point
  tts/              base.py, piper_local.py, openai_tts.py + MockTTS
  brain/            base.py, ollama_client.py (ask + ask_with_tools), router.py
  wake_word/        base.py, dummy.py, openwakeword_engine.py (onnx for Pi)
  vad/              base.py, simple_energy.py
  vision/           base.py (Camera + MockCamera), opencv_camera.py (USB webcam capture, Node)
  memory/           session_memory.py (last-N-turns; no vector DB)
  config/           settings.py — Pydantic models, YAML + env var overrides

  tools/            Hub-side tool layer
    registry.py        Tool base class (name, description, requires_confirmation, parameters)
    dispatcher.py      ToolDispatcher — built-in registry + execute_async routing to MCP
    mcp_manager.py     MCP client: stdio + HTTP/SSE transports, auto-reconnect with backoff
    built_in/
      time_tool.py              get_current_time (configured timezone)
      weather_tool.py           get_weather via wttr.in; 1–3 day forecasts; no API key;
                                schema built per-instance: location optional + defaults to
                                app.user_location when configured, required otherwise; an
                                unknown-location response (404 / "not found" body) coaches
                                the LLM to retry with the default (STT-garble resilience);
                                other HTTP failures report a plain service error
      aviation_weather_tool.py  get_aviation_weather; METAR/TAF/PIREP from aviationweather.gov
      airport_tool.py           lookup_airport; offline ICAO/IATA/name/city lookup
      vision_tool.py            look; fetches a frame from the Node (capture_image over MCP),
                                runs the Hub's Ollama vision model, returns a text description

  node_mcp/         Node-side MCP server (Pi capabilities)
    server.py          FastMCP SSE server — host/port from config; 18 tools (+ capture_image when
                       vision.enabled; the 4 catalog ATC tools only when node_mcp.atc_enabled)
    playback.py        Shared mpv slot: start/stop/pause/resume (pause = terminate + relaunch)
    catalog.py         MusicCatalog: folder-structure scan, search, playlist CRUD
    atc_catalog.py     ATCCatalog: scored channel matching (tower default; random
                       pick among tied towers), stream/PLS URL builders
    tools/
      music_player.py  14 tools: play/search/list/playlist CRUD + play_stream (generic http(s)
                       stream playback — the primitive resolver MCPs feed; see Node MCP playback)
      atc_player.py    4 tools: play_atc, list_atc_airports/channels, stop_atc
      volume_control.py set_volume (pactl/amixer)
      camera_capture.py capture_image — USB webcam → base64 JPEG (only registered when
                        vision.enabled); image held in memory, never written to disk

  server/           CoreMind Hub
    app.py            FastAPI + SSE + config API + node registry + tool endpoints + vision snapshot
    static/index.html Web dashboard (single file, no build step; mobile-responsive)

  service/coremind.service   systemd user service for the Pi
  tests/                     unit tests (no audio hardware required)
```

Top-level extras: `AGENTS.md` (this file), `CLAUDE.md` (imports it), `.github/` (CI + PR template), `Caddyfile.hub.example`, `Caddyfile.node.example`, `docs/` (setup-hub, setup-node, tools, troubleshooting, architecture, roadmap), `config.hub.example.yaml`, `config.node.example.yaml`.

---

## Commands

```bash
coremind run                  # Node voice loop (or standalone mode)
coremind server               # Hub server; binds 127.0.0.1 (use --host 0.0.0.0 or Caddy)
coremind setup                # one-shot browser config wizard on port 8766
coremind doctor               # pre-flight check: Python, config, audio, Ollama, STT, TTS
coremind audio list-devices | record-test | play-test
coremind chat once | loop     # push-to-talk text loop (no wake word)
coremind music scan           # rebuild music catalog from folder structure
coremind atc scan KEWR KJFK   # probe LiveATC for channels; atc add / atc js / atc test
coremind vision test          # capture a webcam frame on the Node; vision describe (Hub)
```

---

## Configuration

`config.yaml` at project root (gitignored; examples are checked in). Loaded by Pydantic models in `config/settings.py`; env vars override. **Never hard-code Tailscale IPs in source.**

Key sections (see `config.*.example.yaml` for the full reference):

```yaml
mode: hub | node | standalone

app:        name, personality (free-text persona/tone), log_level, user_location, user_timezone, home_airport, taf_airport
audio:      input_device, output_device (null = auto-select by stable name; int index or name substring to pin), sample_rate, channels
stt:        provider (whisper_local | whisper_cpp | mock), model (faster-whisper: distil-large-v3 for Hub; tiny/base for Pi standalone), language,
            whisper_cpp: {model (large-v3-turbo), model_path, n_threads, vad_model_path}   # pip install 'coremind[stt-cpp]'
tts:        provider (piper_local | espeak | openai), model_path
brain:      provider (ollama), timeout_seconds
ollama:     base_url, model, vision_model (for the 'look' tool), no_think
runtime:    follow_up_seconds, follow_up_min_words, post_response_cooldown_seconds, wake_confirm_words
vad:        enabled, silence_seconds, max_record_seconds, energy_threshold
wake_word:  provider (openwakeword), threshold, vad_threshold (Silero pre-gate, 0=off), inference_framework: onnx
memory:     enabled, max_turns
remote_brain: enabled, url            # Node → Hub
tools:
  enabled: true
  built_in: [time, weather, aviation_weather, airport, look]   # 'look' needs ollama.vision_model
  mcp_servers:                        # requires pip install 'coremind[tools]'
    - {name: node, transport: http, url: "http://<pi-tailscale-ip>:8767"}      # http = SSE
    - {name: filesystem, transport: stdio, command: ["npx", "..."]}            # optional env: {...}
    - {name: homeassistant, transport: streamable-http,                        # URL used as-is
       url: "http://<ha-host>:8123/api/mcp",
       headers: {Authorization: "Bearer ${HA_TOKEN}"}}                         # ${VAR} from Hub env
vision:                              # camera capture on the Node (Pi)
  enabled: true                      # exposes the capture_image MCP tool; pip install 'coremind[vision]'
  provider: opencv                   # opencv (USB webcam) | mock
  camera_index: null                 # null = auto-select by stable /dev/v4l/by-id path; int pins an index
  camera_device: null                # explicit /dev path (wins over camera_index); hub/reboot-safe
node_mcp:                             # Node-side MCP server (Pi)
  enabled: true
  host: "127.0.0.1"                   # "0.0.0.0" for direct Hub access without Caddy
  port: 8767
  atc_enabled: true                   # false hides the 4 catalog ATC tools (external ATC
                                      # resolver MCP + play_stream owns ATC instead)
  music_dir: ~/Music
  catalog_path: ~/.coremind/music-catalog.json
  atc_catalog_path: ~/.coremind/atc-catalog.json
```

---

## Network & Security Model

- **Both servers bind `127.0.0.1` by default.** Remote access goes through a Caddy reverse proxy listening on the Tailscale IP (`Caddyfile.hub.example`, `Caddyfile.node.example`), or via explicit `--host 0.0.0.0` / `node_mcp.host: "0.0.0.0"` for the simple path.
- **FastMCP DNS-rebinding guard:** when the Node MCP server binds loopback, it only accepts `Host: localhost:*` / `127.0.0.1:*`. The Node Caddyfile must include `header_up Host localhost:8767` (port suffix required) or the Hub gets 421s.
- `POST /api/tools/invoke` executes **built-in tools only**; MCP tools return 403 and must go through the voice loop. Tools with `requires_confirmation: True` are also blocked.
- Dashboard rendering escapes all schema-derived values (`esc()`); parameter inputs are looked up via `data-param` attributes, never interpolated IDs.

Security rules (always):
1. Never hard-code API keys; never log secrets; keep `.env` out of git.
2. Wake-word processing stays local on the Pi; don't stream room audio to the cloud by default.
3. Every network request has a timeout. Validate config on startup. Fail closed for tools.
4. `requires_confirmation: True` on any tool that writes, deletes, or sends data — destructive actions are impossible without explicit confirmation.

---

## Key Subsystems (summary)

One line per subsystem. **Read the linked section of
[`docs/architecture.md`](docs/architecture.md) before changing that subsystem.**

- **[Device auto-selection](docs/architecture.md#device-auto-selection):**
  - mic/speaker/camera are picked by stable name/path, never enumeration index
  - `null` config auto-selects and caches the pick in `~/.coremind/device-cache.json`
  - explicit config pins a device
- **[System prompt and personality](docs/architecture.md#system-prompt-and-personality):**
  - `_build_system_prompt` (`server/app.py`) serves voice + chat
  - `app.personality` is the persona, and the Hub's value governs in remote mode
  - the location clause defends against STT-garbled place names
- **[Speech-to-text providers](docs/architecture.md#speech-to-text-providers):**
  - `make_stt()` is the one selection point (`whisper_local` default, `whisper_cpp`, `mock`)
  - whisper.cpp settings are nested under `stt.whisper_cpp`; decode knobs are shared
  - the Hub runs STT off the event loop and logs a `turn timing:` line per voice turn
- **[Voice loop](docs/architecture.md#voice-loop):**
  - wake → VAD → process → speak → follow-up window, with stop phrases, a minimum word count and
    an echo cooldown
  - mpv is terminated + relaunched around every turn
- **[Wake confirmation gate](docs/architecture.md#wake-confirmation-gate):**
  - the first post-wake utterance must end with a terminator ("over"), or it is silently dropped
  - in remote mode it runs on the Hub (`X-Confirm-Gate` / `X-Rejected`)
- **[Tool layer](docs/architecture.md#tool-layer):**
  - Ollama tool loop, max 5 rounds
  - built-in tools plus MCP over stdio, SSE (`http`) and `streamable-http`
  - `${VAR}` secrets are expanded at connect time only
  - reconnect with backoff, 45 s re-sync, `POST /api/tools/refresh`
  - FastMCP `{"result": …}` unwrap, gated on `outputSchema`
- **[Node registration](docs/architecture.md#node-registration):**
  - auto-register + 30 s heartbeat, re-register on a 404
  - hot-reloaded Node params
  - latest-wins overrides in `~/.coremind/node-overrides.json`
- **[Sessions](docs/architecture.md#sessions):**
  - per-session JSON files + `index.json`, with atomic writes; the directory is the source of
    truth
  - one server-global *active* session; each device picks which session it *views*, with
    follow/Resume-live
  - SSE sync + watchdog
- **[Vision on demand](docs/architecture.md#vision-on-demand):**
  - `look` (Hub) → `capture_image` (Node MCP) → local Ollama vision model → text
  - images stay in memory only, never go to the cloud; opt-in
  - dashboard Camera card
- **[Node MCP playback](docs/architecture.md#node-mcp-playback):**
  - one shared mpv slot; pause = terminate + relaunch, because the USB speaker is single-open
  - audio MCPs follow the **resolver pattern**: return a URL → `play_stream`; never spawn their
    own player
- **[ATC channel selection](docs/architecture.md#atc-channel-selection):**
  - defaults to tower, with a random pick among primary towers
  - a frequency in the query pins the channel
  - the stream provider's domain lives only in `atc_catalog.py`

**Non-negotiables that cut across subsystems:**
- New tool capabilities go through **MCP**, never a parallel tool protocol.
- New physical capabilities (sensors, actuators) attach to the **Node** as Node MCP tools; the
  **Hub** does the inference.

## Working Rules

1. **One thing at a time.** Make a feature work end-to-end before starting the next.
2. Run the tests after meaningful changes, and fix failures before continuing.
3. Update README/docs, `docs/architecture.md` and this file when behavior or architecture
   changes.
4. Prefer working code over theoretical architecture. Keep modules small and replaceable behind
   simple interfaces.
5. Do not silently remove safety checks. Do not commit secrets.
6. **Commits and pushes follow your role** (see Roles). The maintainer role commits only when
   asked; the agent role commits freely on its own branch and never touches `main`.
7. Keep the Pi lightweight: inference belongs on the Mac Mini.
8. Optional backends (faster-whisper, pywhispercpp, piper, mcp, openwakeword) are behind guarded imports. The
   app must not crash when one is missing; it explains how to install it instead.
9. No silent fallback to mock backends in production. Log loudly if Ollama is unreachable.
10. **Prefer auto-detection.** Unset config should auto-pick a sensible default (resolve it to a
    stable name/path, log it, persist it), with explicit config as an override. Don't make the
    user hand-fill values that can be discovered.

### Coding style

- Python 3.11+, type hints everywhere, Pydantic models for config/data shapes.
- Small functions, isolated side effects, no global mutable state, dependency injection for STT/TTS/brain clients.
- Exception hierarchy: `CoreMindError` → `AudioInputError`, `AudioOutputError`, `STTError`, `TTSError`, `BrainError`, `ConfigError`.

### Testing

- **Run the suite with the project virtualenv:** `.venv/bin/python -m pytest -q`. Install it with
  `pip install -e '.[dev,server,tools]'`.
  - A system/conda Python without the optional deps (`fastapi`, `soundfile`, `mcp`,
    `ruamel.yaml`) silently skips or breaks tests and gives misleading counts.
  - Report passed/skipped counts, and explain any skip.
- Unit tests must not require audio hardware; use `MockSTT`/`MockTTS`/`MockBrainClient`.
- Hardware checks are explicit CLI commands (`coremind audio record-test`, `coremind atc test`),
  never default pytest tests.
- Test logic and clients: config loading, routing, catalog matching, tool confirmation policy,
  request formatting.
- Dev machines don't have the deployment-only tooling (Caddy, openwakeword's Pi runtime) or the
  hardware, and may not be able to reach the live Hub/Pi. Simulate locally (e.g. an in-process
  FastMCP round-trip, `TestClient`), and leave real-device checks to the maintainer.
