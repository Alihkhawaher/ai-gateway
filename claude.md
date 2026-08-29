# AI Gateway — Project Specification & Workflow

> This file is the authoritative reference for the AI Gateway project. It documents the architecture, key invariants, and the exact workflow for pushing changes to GitHub. Read this before making any changes.

## Overview

AI Gateway is a **multi-endpoint OpenAI-compatible proxy** with a Textual TUI and a bundled web chat UI (llama.cpp's llama-ui). It connects to several AI backends simultaneously, health-checks them, aggregates their models into a unified catalog, and routes requests based on model ID prefixes.

**Single-file architecture:** the entire application lives in `proxy.py` (proxy server + TUI + static file server). There is no framework, no database, and no external runtime dependency beyond Python 3.8+ and the `textual` library (auto-installed on first run).

## Repository Layout

```
.
├── proxy.py              # Main application (proxy + TUI + static server)
├── config.example.json   # Committed template — copy to config.json
├── config.json           # Your settings (endpoints, models, port) — GITIGNORED
├── build-webui.cmd       # One-click web UI build script
├── run.cmd               # Quick-start batch file
├── webui/                # Built web UI static files (llama.cpp llama-ui)
├── webui-src/            # llama.cpp source (gitignored, for rebuilding the UI)
├── changelogs/           # Per-version release notes (vX.Y.Z.md)
├── testing/              # Automated test suite (python testing/test_proxy.py)
├── claude.md             # This file — project spec & workflow
├── .gitignore
└── README.md
```

## Configuration

`config.json` (gitignored) holds the runtime settings. `config.example.json` is the committed template with placeholder API keys.

### Endpoint Types

| Type | Description | Health Check | Model Discovery |
|------|-------------|--------------|-----------------|
| `openrouter` | OpenRouter cloud API | `GET /api/v1/models` (HTTPS) | `/api/v1/models` |
| `orcarouter` | OrcaRouter cloud API | `GET /v1/models` (HTTPS) | `/v1/models` |
| `lmstudio` | LM Studio local server | `GET /v1/models` (HTTP) | `/v1/models` |
| `llama-server` | llama.cpp server | `GET /health` (HTTP) | `/props` |
| `ollama` | Ollama local server | `GET /api/tags` (HTTP) | `/api/tags` |
| `custom` | Any OpenAI-compatible endpoint (e.g. llamastash) | `GET /v1/models` | `/v1/models` |

### Model ID Format

Model IDs use a `source/vendor/model` prefix. The first segment is the endpoint `name`; the rest is the original upstream model ID.

```
openrouter/openai/gpt-5.6-luna   → OpenRouter → openai/gpt-5.6-luna
llamastash/Qwen3.8-27B-DFlash2-Q8_0 → llamastash (custom) → Qwen3.8-27B-DFlash2-Q8_0
```

## Key Invariants — DO NOT REGRESS

These are documented inline in `proxy.py` under the header **"MODEL CAPABILITY FORWARDING — DO NOT REGRESS"**. All three are required and must not be removed or simplified:

1. **`_get_cached_catalog()` must return deep copies (`copy.deepcopy`).**
   `aggregate_models()` mutates the returned dicts (adds `_source`/`_source_host`, overwrites `id`). A shallow copy corrupts the cache, so on the next 30-second aggregation cycle the original model ID can no longer be found and the model loses its `architecture` metadata (capabilities). This was the v2.3.0 regression.

2. **`has_tools` (function_calling) must check `supported_parameters` for `"tools"`/`"tool_choice"`, not only `input_modalities`.**
   OpenRouter/OrcaRouter report tool support in `supported_parameters`, so checking only `input_modalities` makes `function_calling` always `False`.

3. **`/v1/models` must include a top-level `modalities` field (`{vision, audio, video}`) on each model entry.**
   The bundled llama.cpp web UI reads this exact field (`getModelModalities`) to enable image/audio/video uploads. Without it the UI reports "requires a vision-capable model" even for multimodal models.

## Using MCP Servers (tools/resources) in the Web UI

The bundled llama-ui can attach MCP servers to the chat. MCP servers that
speak HTTP/S (Streamable HTTP or SSE) are added directly by URL; the
**"Use llama-server proxy"** toggle routes their traffic through this
gateway's `/cors-proxy` endpoint to bypass browser CORS/mixed-content.

For this to work, the gateway must:

1. Advertise `cors_proxy_enabled: true` in `GET /props` (enables the toggle in
   the web UI — same signal llama-server emits via `--ui-mcp-proxy`).
2. On `/cors-proxy`, forward the MCP session handshake correctly:
   - Forward normal request headers **and** the `x-llama-server-proxy-header-*`
     prefixed headers (with the prefixed ones taking precedence, so the
     correct `Content-Type: application/json` wins over a browser auto-added
     `text/plain`).
   - Forward upstream response headers (notably `Mcp-Session-Id`) back to the
     browser, so stateful MCP bridges can complete their session handshake.

Caveat: the proxy only forwards to HTTP(S) URLs. It does **not** spawn
command/stdio MCP servers (e.g. `@modelcontextprotocol/server-filesystem`).
To use a stdio server, bridge it to HTTP first (e.g. `supergateway`) and add
the resulting localhost URL with the proxy toggle on.

## API Endpoints

| Endpoint | Format | Purpose |
|----------|--------|---------|
| `GET /v1/models` | OpenAI standard | Standard clients, OpenAI Compatible provider |
| `GET /api/v0/models` | LM Studio | Cline LM Studio provider, LM Studio clients |
| `GET /api/v1/models` | LM Studio | Alias for `/api/v0/models` |
| `GET /props` | llama.cpp | Server properties (modalities, context, params) |
| `POST /v1/chat/completions` | OpenAI standard | Chat completions (proxied to correct endpoint) |
| `GET /health` | — | Endpoint statuses |
| `GET/POST /cors-proxy` | llama.cpp | CORS proxy for the web UI's external fetches |

## Testing Convention

- **Every new feature or bug fix must include a test procedure.**
- Automated tests live in `testing/` and use Python's `unittest` (no third-party test runner).
- Run them with:
  ```powershell
  python testing/test_proxy.py
  ```
- Tests must be self-contained: spin up an in-process mock upstream and an
  ephemeral gateway, make real HTTP requests, and assert on behavior. They
  must not require a running gateway, an online model, or any API key.
- Before merging, confirm all tests pass (`Ran N tests ... OK`) and run
  `python -m py_compile proxy.py testing/test_proxy.py`.

## Release Notes Convention

- Release notes live **only** in `changelogs/vX.Y.Z.md` (one file per version).
- There is **no** top-level `CHANGELOG.md`, `release-notes-v2.md`, or `ver2.md` — these were removed. Do not recreate them.
- Each changelog file has `## Fixed`, `## Added`, `## Changed`, and/or `## Removed` sections.

## GitHub Push Workflow

Follow these steps exactly when shipping changes:

1. **Make code/config/doc changes** in the working tree.
2. **Verify** the code compiles:
   ```powershell
   python -m py_compile proxy.py
   ```
3. **Review the diff** to confirm no unintended changes:
   ```powershell
   git diff
   git status
   ```
4. **Stage only the intended files** (never `config.json` — it is gitignored and contains real API keys):
   ```powershell
   git add proxy.py config.example.json README.md changelogs/v2.3.2.md
   ```
5. **Commit** with a descriptive message following the existing convention:
   ```powershell
   git commit -m "vX.Y.Z: <summary>"
   ```
6. **Push** to `origin/master`:
   ```powershell
   git push origin master
   ```
7. **Tag the release** (annotated tag matching the version):
   ```powershell
   git tag -a vX.Y.Z -m "vX.Y.Z: <summary>"
   git push origin vX.Y.Z
   ```
8. **Create the GitHub release** using the `gh` CLI, with the changelog content as the body:
   ```powershell
   gh release create vX.Y.Z --title "vX.Y.Z" --notes-file changelogs/vX.Y.Z.md
   ```

### Versioning

- Follow semantic versioning: `MAJOR.MINOR.PATCH`.
- Bump `PATCH` for bug fixes, `MINOR` for new features, `MAJOR` for breaking changes.
- Always add a corresponding `changelogs/vX.Y.Z.md` file.

## Security Notes

- `config.json` contains real API keys and is **gitignored** — never commit it.
- `config.example.json` uses placeholder keys (`sk-or-v1-YOUR_KEY_HERE`) and is safe to commit.
- The proxy forwards its own API keys upstream; clients do not need to supply keys.