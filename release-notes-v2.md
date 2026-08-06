# AI Gateway v2.0.0

## What's New

### Multi-Endpoint Proxy Architecture
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