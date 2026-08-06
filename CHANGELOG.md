# Changelog

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