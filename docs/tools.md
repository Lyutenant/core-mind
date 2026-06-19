# Tools

CoreMind uses the Ollama native tool API. The LLM decides when to call tools and the Hub executes them transparently. A chip badge appears on each turn card in the dashboard when a tool fires (e.g. `⚙ get_aviation_weather`).

---

## Built-in Tools

Enable in Hub `config.yaml`:
```yaml
tools:
  enabled: true
  built_in: [time, weather, aviation_weather, airport]
```

| Tool | Trigger example | Notes |
|------|----------------|-------|
| `get_current_time` | "What time is it?" | Uses `app.user_timezone` if set |
| `get_weather` | "What's the weather?" / "Will it rain tomorrow?" | wttr.in, no API key; 1–3 day forecasts with rain chance, UV index, sunrise/sunset |
| `get_aviation_weather` | "What's the METAR at Leesburg?" / "Any PIREPs?" | NOAA aviationweather.gov, no API key; METAR/TAF/PIREP with full briefing format |
| `lookup_airport` | "What's the ICAO for Heathrow?" / "What airport is IAD?" | Bundled offline database, 19K airports |
| `look` | "What do you see?" / "Is the window open?" | Captures from the Node's webcam, describes it with a local Ollama vision model — see [Vision](#vision-look) below |

**Weather and time defaults** (Hub `config.yaml`):
```yaml
app:
  user_location: "San Francisco, CA"   # fallback when no location is mentioned
  user_timezone: "America/Los_Angeles" # IANA timezone name
```

You can still ask "What's the weather in Tokyo?" to override the default.

**Aviation weather defaults:**
```yaml
app:
  home_airport: "KJYO"   # used when no airport is specified
  taf_airport: "KIAD"    # nearest airport with TAF (small airports often lack one)
```
Ask "What's the METAR?" → uses `home_airport`. Ask "Is there a TAF?" → tries `home_airport`, auto-falls back to `taf_airport`. Ask for `report_type="full"` for a complete pre-flight briefing (METAR + TAF + PIREPs).

---

## Vision (`look`)

The `look` tool gives the assistant eyes: a **USB webcam on the Node (Pi)** captures a still frame, and a **local vision model on the Hub (Mac)** describes it. The Pi does the cheap capture; the Mac runs the model. Room images never leave your network — inference is local via Ollama.

```text
"What do you see?"  →  Pi captures a frame (capture_image, MCP)
                    →  Mac runs ollama.vision_model on it
                    →  spoken description
```

**Setup**

1. Plug a USB webcam into the Pi and install the camera extra **on the Pi**:
   ```bash
   pip install 'coremind[vision]'
   ```
   Test it: `coremind vision test -o frame.jpg`

2. Pull a vision model **on the Mac** and set it in the Hub `config.yaml`:
   ```bash
   ollama pull llava:7b      # or 'moondream' (small/low-RAM), 'llama3.2-vision' (best quality)
   ```
   ```yaml
   ollama:
     vision_model: llava:7b
   tools:
     built_in: [time, weather, airport, look]   # add 'look'
   ```
   Test it on the Mac: `coremind vision describe --image frame.jpg`

3. Enable the camera in the Node `config.yaml`:
   ```yaml
   vision:
     enabled: true
     provider: opencv
     camera_index: 0     # try 1, 2… if you have multiple cameras
   ```
   Make sure `node_mcp.enabled: true` and the Hub's `tools.mcp_servers` has the `node` entry (same as for music/ATC).

Then ask **"what do you see?"** by voice or in the dashboard chat.

**Dashboard snapshot**

The dashboard's **Camera** card (right-hand panel of the main view) captures a still from the Node webcam and shows it in the browser — handy for aiming the camera or a quick glance without speaking. Click **Capture snapshot** to grab a frame; click **Describe this** to run the Hub vision model on that exact frame.

- A raw snapshot needs only the Node's `vision.enabled` — `ollama.vision_model` is **not** required (only **Describe** uses the model).
- Endpoints: `POST /api/vision/capture` → `{image: <data-url>, captured_at}`; `POST /api/vision/describe` (body `{image}`) → `{description}`. These are dedicated routes — `capture_image` stays blocked from `/api/tools/invoke`.
- Same privacy posture as `look`: the frame is held in memory and streamed to the browser, never written to disk; logs record metadata only.

**Privacy & notes**

- Images are held in memory only and never written to disk (except `coremind vision test`, which you explicitly ask for). Logs record metadata, not images.
- Inference is local-only — there is no cloud fallback.
- The model can't run on the Pi (too heavy); it only runs on the Mac. The Pi just grabs the frame.
- `look` is opt-in: it does nothing unless `ollama.vision_model` is set, `look` is in `tools.built_in`, and the Node's `vision.enabled` is true.

---

## MCP Servers

Any MCP-compatible server can be wired in — no code changes needed. The Hub connects at startup and exposes the server's tools to the LLM alongside built-in tools.

```bash
pip install 'coremind[tools]'   # install the mcp SDK once (Hub and Node)
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
      url: http://100.y.y.y:8767   # Pi's Tailscale IP
```

**Startup order does not matter.** If the Node is not yet up when the Hub starts, the Hub retries with exponential backoff (10 s → 60 s max). Tools become available as soon as the Node is reachable — no Hub restart needed.

**Diagnose MCP registration:**
```bash
curl http://localhost:8765/api/tools
# {"total":21,"built_in":["get_current_time",...],"mcp":["play_atc",...],"mcp_connected":true}
```

If `mcp` is empty, check Hub logs for `"initial connection failed"` or `"mcp package not installed"`.

---

## Testing Tools from the Dashboard

The **Tools** panel (🔧 in the sidebar) lists all registered tools and lets you invoke **built-in tools** directly from the browser — useful for verifying weather responses, checking the current time in your timezone, or confirming aviation weather for a given airport.

Select a built-in tool, fill in the parameters, and click **▶ Run**. The raw result is shown inline.

MCP tools (music, ATC, filesystem) are listed for reference — their parameters and descriptions are visible — but cannot be run from the dashboard. Trigger them through the voice loop or Chat.

---

## Voice-Controlled Music Player

The Pi runs a local MCP server (port 8767) that exposes 17 tools. The Hub's LLM calls these over Tailscale just like any built-in tool.

**Pi setup** (once):
```bash
sudo apt install mpv
pip install 'coremind[tools]'
```

**Node `config.yaml`:**
```yaml
node_mcp:
  enabled: true
  port: 8767
  music_dir: ~/Music
  catalog_path: ~/.coremind/music-catalog.json
```

**Hub `config.yaml`:**
```yaml
tools:
  mcp_servers:
    - name: node
      transport: http
      url: http://100.y.y.y:8767
```

**Organize your music library** — the catalog infers structure from folder depth:
```
~/Music/
  Miles Davis/
    Kind of Blue/
      01 - So What.mp3
  John Coltrane/
    A Love Supreme/
      01 - Acknowledgement.mp3
```

**Build the catalog:**
```bash
coremind music scan
# Scanned 127 tracks — 12 artists, 18 albums.
```

Re-run whenever you add new music. CoreMind warns in logs if the music directory is newer than the catalog.

**Music tools (13):**

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
| `create_playlist` | "Create a playlist called 'morning jazz'" |
| `add_to_playlist` | "Add that album to my workout playlist" |
| `remove_from_playlist` | "Remove that track from morning jazz" |

**Mic isolation during playback:** When music is playing and the wake word fires, CoreMind suspends mpv (`SIGSTOP`) before recording so only your voice reaches the mic. After the response, mpv resumes (`SIGCONT`) from exactly where it paused — no gap or restart. This covers both music and ATC since they share one mpv slot.

---

## Live ATC Streaming

Stream live ATC audio from LiveATC by voice command. Mic isolation works the same as with music.

**ATC tools (4):**

| Tool | Voice example |
|------|--------------|
| `play_atc` | "Play Newark tower" / "Stream KEWR approach" / "Put on Dulles ground" |
| `list_atc_airports` | "What airports do you have ATC for?" |
| `list_atc_channels` | "What ATC channels do you have for Newark?" |
| `stop_atc` | "Stop the ATC" |

**Channel selection:**

- Naming only an airport ("Stream ATC from Dulles") defaults to its **tower**.
- When several tower channels at one airport tie (per-runway towers like KIAD's `1C/19C` / `1R/19L`), CoreMind picks one at random — runway designators aren't practical to disambiguate by voice. Only **primary** tower feeds are eligible: backup, temp, secondary, TCA, helicopter, and emergency feeds are never auto-picked (if no primary feed exists, CoreMind asks instead).
- Saying an explicit frequency pins the channel: "Dulles tower 120.250" streams that exact tower. A query containing a frequency is never resolved randomly, and a channel whose frequency doesn't match (or can't be confirmed from the catalog) is never streamed silently — CoreMind names the closest match and asks instead.
- Other ambiguous matches (e.g. multiple ground frequencies) still ask you to pick:
```
You: Stream Dulles ground
CoreMind: Multiple frequencies found for Washington Dulles:
          - Ground (East) (121.900 MHz)
          - Ground (West 1L/19R, 12/30) (121.625 MHz)
          Which would you like?
```

**Default catalog — works out of the box**

CoreMind ships with 397 channels across 16 airports: KEWR, KJFK, KLGA, KTEB, KIAD, KDCA, KJYO, KAPA, KBOS, KORD, KATL, KLAX, KSFO, KSEA, KMIA, ZNY. No setup needed for these airports.

**Adding more airports:**

For airports with standard mount names, `atc scan` discovers them automatically:
```bash
coremind atc scan KBWI KPHL KPDK
# ✓  kbwi_twr  BWI Tower
# ✓  kbwi_gnd  BWI Ground
```

For airports with obfuscated mount names (KIAD-style), visit the LiveATC search page in a browser. `atc js` walks you through the process:
```bash
coremind atc js      # prints step-by-step instructions + JS snippet
```
1. Open the LiveATC airport search page (`/search/?icao=KXXX`) in your browser
2. Open DevTools → Console, paste the JS snippet shown by `atc js`
3. Save the JSON output to a file, e.g. `~/browser-mounts.json`
4. Scan with the file:
```bash
coremind atc scan KXXX --browser-mounts ~/browser-mounts.json
```

**Manual entry for a single channel:**
```bash
coremind atc add KIAD "Tower" kiad1_twr_1c19c_120250 --freq 120.250
```

**Test a stream without the voice loop:**
```bash
coremind atc test "KIAD tower"
# Checks the PLS endpoint, launches mpv with visible stderr output.
```

The catalog is saved to `~/.coremind/atc-catalog.json`. The Node MCP server picks up changes on the next tool call — no restart needed.
