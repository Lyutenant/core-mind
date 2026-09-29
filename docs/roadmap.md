# CoreMind Roadmap

Planned, in rough priority order:

1. **Porcupine wake word** — custom phrase ("Hey CoreMind") via Picovoice console `.ppn` model; add `PorcupineDetector` implementing `WakeWordDetector`, wire under `wake_word.provider: porcupine`, add `porcupine_model_path` config. Free for personal use; needs an access key.
2. **WebRTC VAD** — optional upgrade from the energy-threshold VAD for noisy rooms.
3. **Long-term memory** — only after the voice loop is fully reliable; deliberately deferred. No vector DB until explicitly planned.
4. **More MCP integrations** — calendars, etc. (home automation ✅ via Home Assistant). All via MCP servers, not bespoke code.
5. **Audio refinements** — Bluetooth speaker support, ReSpeaker XVF3800 features (echo cancellation, DoA) once the base pipeline needs them.
6. **Perception → autonomy ladder** (builds on vision on demand) — the world should be able to *push* to the agent, not just answer on request: sensor/timer-**triggered** capture ("tell me when someone's at the door"), then continuous/contextual perception. A *minimal* reactive/event path gets introduced only when a real closed loop forces it — not as a speculative I/O bus.

Not planned: a native app (the web dashboard is the UI), a database, an agent framework.
