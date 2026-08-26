# Changelog

## [2.3.2] — 2026-08-26

### Fixed
- **Model capabilities not forwarded to web UI / endpoints** — three related bugs:
  - Catalog cache returned shallow copies; `aggregate_models()` mutated the cached dicts, corrupting the cache so models lost their `architecture` metadata (and thus capabilities) on the next 30-second aggregation cycle. Now uses `copy.deepcopy`.
  - `function_calling` was always `False` because tool support was only checked in `input_modalities`, but OpenRouter/OrcaRouter report it in `supported_parameters` (`"tools"`/`"tool_choice"`).
  - `/v1/models` was missing the top-level `modalities` field that the bundled llama.cpp web UI reads to enable image/audio/video uploads (caused "requires a vision-capable model" errors).

### Added
- **llamastash endpoint** — OpenAI-compatible local endpoint at `localhost:11435` (via the `custom` endpoint type).
- New default models: `openrouter/deepseek/deepseek-v4-flash-vision-exp`, `openrouter/z-ai/glm-5.3-flash`.

## [2.3.1] — 2026-08-21

See [changelogs/v2.3.1.md](changelogs/v2.3.1.md) for full details.

### Fixed
- Endpoint `✎ Edit` / `✕ Remove` buttons (wrong `DataTable` cursor attribute)
- Model dropdown refresh (`Select.set_options`)
- Model-index bounds check preventing `IndexError` on Settings open
- Settings screen crash from invalid `get_config` call
- Thread-safe TUI logging via a log queue

### Changed
- Port/address validation with toast feedback and restart prompt
- Endpoint removal now requires confirmation
- Duplicate endpoint names rejected instead of silently overwritten
- Endpoint type auto-fill only fills empty fields
- Top-model fetch now reports results in the TUI log

## [2.3.0] — 2026-08-21

See [changelogs/v2.3.0.md](changelogs/v2.3.0.md) for full details.

### Added
- `config.example.json` — committed config template with placeholder keys
- Functional endpoint management TUI (Add / Edit / Remove)
- `/cors-proxy` endpoint (llama.cpp/llama-ui compatible)
- `/models/load` now sets the model as the new default

### Changed
- Model catalog caching (5-minute TTL) to avoid re-downloading on every health check
- Smarter routing — clear 404/503 instead of silent fallback to another endpoint
- Thread-safe `MODELS` list via `get_models()`/`set_models()`
- Restart path no longer kills the process on bind failure
- `build-webui.cmd` guard for missing `webui-src/`
- README updates (config.example.json + web UI build prerequisite)

### Fixed
- Removed silent endpoint fallback that could misroute requests with the wrong API key
- Eliminated redundant full-catalog downloads every health-check cycle

## [2.2.0] — 2026-08-21

See [changelogs/v2.2.0.md](changelogs/v2.2.0.md) for full details.

### Added
- **OrcaRouter provider** — New cloud endpoint type for OrcaRouter (`api.orcarouter.ai`), an OpenAI-compatible multi-provider gateway supporting OpenAI, Anthropic, Google Gemini, DeepSeek, xAI Grok, Qwen, Kimi, and more
- OrcaRouter health checking, model fetching (`/v1/models`), and path rewriting
- OrcaRouter option in TUI endpoint type dropdown
- New models: DeepSeek V4 Pro, GLM 5.3, Claude Sonnet 5, Qwen 3.8 27B, Kimi K3
- Uncensored model: `obsidian/Qwen3.6-35B-A3B` via OrcaRouter

### Removed
- Outdated models: `qwen/qwen3.7-plus`, `qwen/qwen3.7-flash`, `z-ai/glm-5.2`, `deepseek/deepseek-chat`

## [2.1.0] — 2026-08-21

See [changelogs/v2.1.0.md](changelogs/v2.1.0.md) for full details.

### Added
- LM Studio-compatible API endpoint (`/api/v0/models`, `/api/v1/models`) for Cline auto-discovery
- TUI footer key bindings (Settings, Quit, Save, Restart, Back)
- Server info display on main screen

### Changed
- TUI layout uses consistent ScrollableContainer styling
- Settings screen simplified (read-only endpoint table)
- Top intelligent models persist across health check cycles
- SSL context reuse for better performance

### Fixed
- Top models disappearing after health check cycle
- Unused CSS and variables cleanup

## [2.0.0] — 2026-08-06

### Added
- **Multi-endpoint support** — Connect to OpenRouter, LM Studio, llama.cpp server, and custom OpenAI-compatible endpoints simultaneously
- **Model ID source prefix** — Models use `source/vendor/model` format (e.g., `openrouter/openai/gpt-5.6-luna`, `llama.cpp/qwen3-8b`)
- **Health checking** — Background thread checks all endpoints every 30 seconds with status display in TUI
- **Model aggregation** — Unified model catalog from all online endpoints
- **Smart routing** — Requests automatically routed to the correct endpoint based on model ID prefix
- **Auto-discovery** — Automatically detects local llama.cpp server (:8080) and LM Studio (:1234) on startup
- **Endpoint management TUI** — Add, edit, remove endpoints via DataTable + modal dialog
- **Endpoint status panel** — Main screen shows online/offline status and model counts per endpoint
- **v1→v2 config migration** — Existing v1 configs automatically migrated to v2 format
- **Endpoint type defaults** — Auto-fills name and host when selecting endpoint type
- **Custom endpoint type** — Support for any OpenAI-compatible API endpoint
- **Architecture design document** (`ver2.md`) — Detailed design spec for the multi-endpoint architecture
- **Web UI: Show raw model names setting** — Settings > Display toggle to display full raw model identifiers (e.g. `openrouter/ggml-org/GLM-4.7-Flash-GGUF:Q8_0`) instead of parsed names with badges

### Changed
- **Config format** — Now uses `endpoints` array instead of single `host`/`api_key` top-level fields
- **OpenRouter model aggregation** — Only includes user's curated models, not the full 340+ catalog
- **Path rewriting** — Endpoint-type-aware: OpenRouter gets `/api/v1` prefix, local endpoints use `/v1` directly
- **README.md** — Fully rewritten for v2 architecture with multi-endpoint diagram and documentation
- **run.cmd** — Updated to reflect AI Gateway branding

### Fixed
- Settings screen back button (escape binding now uses `app.pop_screen`)
- Hardcoded "openrouter" prefix in top intelligent models fetch
- Stale metadata cache from offline endpoints (now cleared on each aggregation cycle)
- DataTable row selection for Edit/Remove operations
- Thread safety for MODELS list updates

## [1.0.0] — 2026-07-29

### Added
- Initial release: single-endpoint OpenRouter proxy with SSE streaming
- Terminal TUI with model selection and settings
- Web chat UI (llama.cpp's llama-ui)
- CORS support for cross-origin access
- Automatic model injection
- PWA support with service worker