# AI Gateway Release Notes

## v2.3.0 — 2026-08-21

### Added
- **`config.example.json`** — Committed config template with placeholder API keys.
- **Functional endpoint management TUI** — Working `+ Add`, `✎ Edit`, and `✕ Remove` buttons wired to the endpoint edit dialog.
- **`/cors-proxy` endpoint** — llama.cpp/llama-ui compatible CORS proxy for the web UI's external fetches.
- **`/models/load`** — Now validates and sets the requested model as the new default instead of a silent no-op.

### Changed
- Model catalogs are cached for 5 minutes instead of re-downloaded every health-check cycle.
- Smarter routing returns clear 404/503 errors instead of silently falling back to another endpoint.
- Thread-safe `MODELS` list via `get_models()`/`set_models()`.
- Server restart no longer terminates the process on bind failure.
- `build-webui.cmd` guards against a missing `webui-src/` clone.
- README documents `config.example.json` and the web UI build prerequisite.

### Fixed
- Removed silent endpoint fallback that could misroute requests with the wrong API key.
- Eliminated redundant full-catalog downloads every 30 seconds.

**Full Changelog**: https://github.com/Alihkhawaher/ai-gateway/commits/v2.3.0

## v2.2.0 — 2026-08-21

### Added
- **OrcaRouter provider** — New cloud endpoint type for OrcaRouter (`api.orcarouter.ai`), an OpenAI-compatible multi-provider gateway that routes requests across OpenAI, Anthropic, Google Gemini, DeepSeek, xAI Grok, Alibaba Qwen, Moonshot Kimi, MiniMax, and more at provider cost price
- OrcaRouter health checking, model fetching, and path rewriting (same protocol as OpenRouter)
- OrcaRouter option in TUI endpoint type dropdown with auto-filled defaults

**Full Changelog**: https://github.com/Alihkhawaher/ai-gateway/commits/v2.2.0

## v2.1.0 — 2026-08-21

### Added
- **LM Studio-compatible API endpoint** — `/api/v0/models` and `/api/v1/models` for Cline auto-discovery
- **TUI footer key bindings** — Settings, Quit, Save, Restart, Back
- **Server info display** — Main screen shows Listening URL, Web UI URL, and Default model

### Changed
- TUI layout uses consistent ScrollableContainer styling
- Settings screen simplified (read-only endpoint table)
- Top intelligent models persist across health check cycles
- SSL context reuse for better performance

### Fixed
- Top models disappearing after health check cycle
- Unused CSS and variables cleanup

**Full Changelog**: https://github.com/Alihkhawaher/ai-gateway/commits/v2.1.0

## v2.0.0 — 2026-08-06

### What's New

#### Multi-Endpoint Proxy Architecture
AI Gateway v2 transforms from a single-endpoint proxy into a **multi-endpoint AI routing gateway**. Connect to OpenRouter, LM Studio, llama.cpp, **Ollama**, and any custom OpenAI-compatible endpoint simultaneously through a single unified API.

### Highlights
- **Multi-endpoint support** — Connect to multiple AI backends at once
- **Model ID source prefix** — `source/vendor/model` format (e.g., `openrouter/openai/gpt-5.6-luna`, `ollama/llama3`)
- **Ollama support** — Native integration with health checking via `/api/tags`, model discovery, and OpenAI-compatible chat API
- **Health checking** — Background thread checks all endpoints every 30 seconds
- **Model aggregation** — Unified model catalog from all online endpoints
- **Smart routing** — Requests automatically routed based on model ID prefix
- **Auto-discovery** — Automatically detects local llama.cpp (:8080), LM Studio (:1234), and Ollama (:11434)
- **Endpoint management TUI** — Add, edit, remove endpoints via DataTable + modal dialog
- **v1→v2 config migration** — Existing configs automatically migrated
- **Web UI: Show raw model names** — Toggle to display full model identifiers

### Changed
- **Config format** — Now uses `endpoints` array instead of single `host`/`api_key` fields
- **OpenRouter model aggregation** — Only includes curated models, not the full 340+ catalog
- **Path rewriting** — Endpoint-type-aware routing for each backend

### Fixed
- Settings screen back button
- Stale metadata cache from offline endpoints
- DataTable row selection for Edit/Remove
- Thread safety for MODELS list updates

**Full Changelog**: https://github.com/Alihkhawaher/ai-gateway/commits/v2.0.0
