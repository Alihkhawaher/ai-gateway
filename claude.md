# AI Gateway — Project Specification & Workflow

> This file is the authoritative reference for the AI Gateway project. It documents the architecture, key invariants, and the exact workflow for pushing changes to GitHub. Read this before making any changes.

## Overview

AI Gateway is a **multi-endpoint OpenAI-compatible proxy** with a Textual TUI and a bundled web chat UI (llama.cpp's llama-ui). It connects to several AI backends simultaneously, health-checks them, aggregates their models into a unified catalog, and routes requests based on model ID prefixes.

**Single-file architecture:** the entire application lives in `proxy.py` (proxy server + TUI + static file server). There is no framework, no database, and no external runtime dependency beyond Python 3.8+ and the `textual` library (auto-installed on first run). The optional launcher `gateway_tui.py` (split-view TUI: proxy GUI + 1MCP MCP bridge) subclasses `proxy.py`'s TUI classes — it does not modify or duplicate them.

## Repository Layout

```
.
├── proxy.py              # Main application (proxy + TUI + static server)
├── gateway_tui.py        # Split-view TUI launcher (proxy GUI + 1MCP bridge)
├── config.example.json   # Committed template — copy to config.json
├── config.json           # Your settings (endpoints, models, port) — GITIGNORED
├── mcp.json              # MCP servers aggregated by 1MCP — GITIGNORED
├── gateway-stats.json    # All-time stats, counts only — written at run time, GITIGNORED
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
| `strata` | Strata gateway (default `127.0.0.1:47831`) | `GET /v1/models` | `/v1/models` + `/api/native` (per loaded model: vision, context, label) |
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
3. Serve `/cors-proxy` only to the gateway's own web page
   (`_cors_proxy_caller_allowed`): browser requests must be `Sec-Fetch-Site:
   same-origin` (or carry an `Origin` matching `Host`), and `Host` must be an
   IP, `localhost`, a single-label name or a local suffix (`.local`, `.lan`,
   …) so DNS rebinding can't pass as same-origin. Requests with neither header
   (curl, scripts) are allowed. Otherwise any web page could use the gateway
   to reach loopback-only services (1MCP, Strata) and read the replies.

Caveat: the proxy only forwards to HTTP(S) URLs. It does **not** spawn
command/stdio MCP servers itself. To use stdio servers, list them in
`mcp.json` (gitignored) and the split-view TUI (`gateway_tui.py`) launches
[1MCP](https://github.com/1mcp-app/agent), which aggregates **every** listed
server behind ONE streamable-HTTP endpoint on `:8099/mcp` — one port for many
MCPs. Add that single URL in the web UI with the proxy toggle on.

## Endpoint Load Tracking

The TUI shows a per-endpoint **load** column in both endpoint views — the
main screen's endpoint list (`#endpoint-status`) and the Settings screen's
DataTable (`#endpoint-table`). It is driven by a small, thread-safe counter
in `proxy.py`:

- `_endpoint_load` (guarded by `_endpoint_load_lock`) maps endpoint name →
  `{"active": int, "total": int}`.
- `bump_endpoint_load_start(name)` is called in `ProxyHandler._proxy()`
  immediately before the upstream request; `bump_endpoint_load_end(name)` is
  called in the `finally` block, so the in-flight count is always released —
  even on upstream error or timeout.
- `format_endpoint_load(name)` renders `⚡ N active · M total` (busy),
  `idle · M total` (seen, none in flight), or `—` (never used).

`active` = requests in flight to that endpoint right now; `total` = cumulative
proxied requests since start. Only requests routed through `_proxy()` (chat
completions and other proxied paths) are counted — `/cors-proxy` traffic is
not, since it isn't tied to a configured upstream gateway.

## KV Session Swap

A local engine (Strata) holds one conversation in its slot. With `"kv_sessions": {"enabled": true}` on an endpoint, `KVSessionManager` in `proxy.py` keeps the right chat in that slot through the engine's `POST /slots/0?action=save|restore`. `_proxy()` calls `_kv_begin()` just before the upstream request and `_kv_end()` in its `finally`.

Rules that must hold:

1. **`_kv_begin` and `_kv_end` always pair.** `begin` takes the manager's lock and `end` releases it. A missed `end` blocks that endpoint forever, so `_kv_begin` stays inside the `try` whose `finally` calls `_kv_end`.
2. **Swapping never breaks a request.** Any failure is logged and the request is forwarded anyway. A 501 from the engine (slot saving off) or an unreadable `/slots` pauses swapping for `_KV_BACKOFF_S`.
3. **Chat text never leaves memory.** The index (`kv-sessions-<endpoint>.json`, gitignored) holds message hashes and counts only, and session files are never opened. They are deleted only by `max_gb` eviction, and only `gw-*.bin` files listed in the index.
4. **Never save a conversation under the wrong name.** `current` is dropped when the token count from `/slots` no longer matches what was recorded (the engine restarted, or a client used it directly), or when the engine's `host:port` changed.
5. **A save rewrites the whole chat** (about 16.7 KB per token, 8.5 GB at 512k), so saves happen on a switch and after a long idle only, and only when the chat changed since the last save.
6. **A 500 from a save or restore turns swapping off for that engine load** (`_suspend`), until the model is reloaded (Strata gives each load a new key) or `_KV_SUSPEND_S` passes. Strata 0.1.57 can get into a state where every snapshot answers `500 "conversation snapshot: invalid K/V extent"`. After that, `/slots` reports 0 tokens although the engine still holds the chat, so nothing, not even `/slots`, is asked while suspended. The engine's error text (never chat text) is kept in `last_error` and shown in the console.
7. **A gateway restart doesn't forget what the engine holds.** `current` is also kept in the index as `held` (hashes, counts, session id and engine `host:port`, never the key) by `_persist_held()`, which `end()` and the idle loop call before giving the lock back. It is written only when it changed. The first request checks it with rule 4.
8. **Never restore over a long chat that can't be identified.** With no `current` and the engine holding `min_tokens` or more, the restore is skipped: that chat may be this one, newer than its saved copy. If the request continues the skipped saved chat, it takes that chat's id, so the next save replaces that file instead of writing a second copy.

For Strata, the engine's port and per-load API key are read from `~/.strata/data/native-instances/*.json` on each request (`STRATA_INSTANCES_DIR` overrides the folder; tests use it). The key is used in memory only. Strata's host does not forward `/slots`, so these calls go to the engine's own port.

`testing/live_kv_check.py` checks the swap against a real engine with made-up text.

## Live Stats

The console's **Live** panel, its `d` dashboard (`StatsScreen`) and `GET /stats` all draw from `stats_snapshot()`, which reads memory only:

- `GatewayStats` (`_stats`) counts generation requests. `_proxy()` calls `_stats.begin()` before the upstream request, `stat.tap.feed()` on each chunk and `_stats.end()` in its `finally`. All-time totals persist to `gateway-stats.json`, next to `config.json`.
- `_ResponseTap` reads usage and timings out of a response as it passes. It also reads how the reply ended: `finish_reason`, mapped through the `_FINISH_NAMES` allow-list (anything else is `other`), the number of tool calls, and the character counts of the text and the reasoning. `_reply_end()` turns those into the console's **End** column and the log's `ended:` phrase.
- `EngineProbe` (`_engine_probe`) polls an engine's `/metrics` every 2 seconds, and stops after 30 seconds with no reader. For a Strata or `kv_sessions.engine` endpoint, `_stats.end()` also fetches that request's row from the engine, in a background thread.
- The `panel_*` functions and `render_live` are pure: a snapshot goes in and Rich renderables come out. They fetch nothing.

Rules that must hold:

1. **Stats never change or break a request.** The tap only reads, `begin`, `feed` and `end` swallow their own errors, and no request waits on an engine lookup. A byte-for-byte test covers this.
2. **Chat text never leaves the request.** From a request body, the only things kept are the model name, the stream flag, whether tools were offered (yes or no) and the thinking choice as a fixed name (see Thinking Switch). From a reply, it's counts, plus a finish reason taken from the allow-list. Snapshots, `/stats`, the log and `gateway-stats.json` hold counts and names only.
3. **Engine data is an allow-list.** `_engine_view` copies only the fields named in `_ENGINE_*_KEYS`. Add a field there on purpose, and never pass an engine's dict through whole.
4. **Client names are never Rich markup.** App and model names come from clients. A plain `str` cell in a Rich table is parsed as markup, so `[/]` in a name would break the panel: wrap them in `RichText`.
5. **A bad number can't take the console down.** Each panel render has its own `try`, so a failure shows in that panel and nowhere else.
6. **Group tables are capped.** `_agg_group` keeps `_STATS_MAX_KEYS` names apart and counts the rest as `other`, so clients can't grow memory or the totals file without limit.

Tests must not write real files: `testing/test_proxy.py` points `proxy.CONFIG_FILE` and `STRATA_INSTANCES_DIR` at a temp folder on import, because `gateway-stats.json` is written beside the config.

## Tool Call Healing

With `"tool_heal": {"enabled": true}` on an endpoint, `_new_healer()` gives a `/chat/completions` request that offers tools a `ToolHealer`, which reads the reply on its way to the client. `_parse_tool_blocks()` turns calls written as text into real ones, and `guard_tool_call()` / `_guard_command()` check every call that would run. Rule ids and reasons are in `_GUARD_REASONS`. The Settings endpoint dialog (`EndpointEditScreen`) sets `enabled` and `guard`. `_heal_result()` keeps the endpoint's other `tool_heal` keys, and writes nothing for an endpoint that never had the setting and still doesn't. `_new_healer()` reads the config on every request, so a change applies to the next request.

Rules that must hold:

1. **Off is byte for byte, and with nothing to do every character survives.** No healer means the old path. With a healer, a reply with nothing to repair passes raw, except that text which could have started a call is held until it can't and may arrive split differently (`test_replies_with_nothing_to_heal_pass_byte_for_byte`, `test_re_chunked_text_keeps_every_character`).
2. **Never guess.** A tool name is never mapped or re-cased, and parameter bytes are never edited. A call with an unknown name goes through under that name: nothing runs, and the client tells the model what exists. Unknown names must stay safe to send, which is why they aren't renamed to something that would run.
3. **Never heal what isn't whole.** A written call needs its closing tags and `finish_reason` `stop` or `tool_calls`. Anything else stays text. All calls in a reply are healed, or none.
4. **A quotation isn't a call.** Only outside code fences, at the start of a line, in `content`, never from reasoning. One exception, `_calls_after_thinking()`: the reply ended with `stop`, no answer text and no call, and its reasoning holds `</think>` followed by nothing but whole calls in Gemma 4's format (`_parse_gemma_calls()`). The model closed its thinking there itself, so what follows isn't a thought. A Gemma 4 finetune on LM Studio does this (seen with Orion, 2026-11-11): the server doesn't know that tag, keeps everything as reasoning, and the agent stops on an empty reply. Without the tag, with words after the calls, or cut off, the reply passes byte for byte.
5. **Nothing of a call is sent until it's judged.** Structured `tool_calls` deltas are held to the end of the reply. A cut-off stream must send no part of a command.
6. **Broken arguments on a command or file tool never reach the client (A1).** Cline's `experimental_repairToolCall` runs `jsonrepair`, which turns `rm -rf C:\Users\me` into `rm -rf C:Usersme` and runs half a batch. Verified with its real code.
7. **Stopped calls keep the agent loop alive.** `ask_question` if the request offers it, otherwise a tool call to an unknown name that carries the rule. An HTTP error or an SSE error stops the loop in Cline, and plain text ends the turn.
8. **A failure inside the healer hands over what it holds** and passes the rest as it came (`_give_up`). Healing must never break a reply.
9. **No text is kept.** The log and stats get counts and rule ids only. The command appears only in the question sent to the client.
10. **The guard is for accidents, not a security boundary.** False positives break real work, so any new rule needs near misses added to `MUST_PASS` in `CommandGuardTests`.

Test with made-up text only, through `run_healer()`, `sse_text()` and `sse_native()`. Never send test traffic to a real engine: it would displace the chat it holds.

## Context Limit

The gateway never compacts a chat: the client does, and the gateway gives it what it needs to do so.

- `_context_overflow()` recognises an engine's "the prompt doesn't fit" error (status in `_OVERFLOW_STATUSES`, by type, code or the words in `_OVERFLOW_RE`) and `_proxy()` answers `400` with the OpenAI code `context_length_exceeded` (Anthropic's shape on `/messages`), keeping the engine's message and its token counts. Cline 4.1.26 compacts and retries on that code or on "context length / window / limit" and "maximum context"; llama.cpp's "context size" matches none of those, which is why an overflow used to end the run.
- `/v1/models` and `/props` report `loaded_context_length` when the endpoint gives one (LM Studio), with the model's maximum in `max_context_length`.
- A lookup of one model (`/v1/models/<id>`, `/api/v0/models/<id>`, Ollama's `/api/show`) is answered by the gateway with the same window. Until v2.13.1 these fell through to `_proxy()` and came back as a 502, and an app that asks this way then used its own default (128,000 in Cline and Goose). `_note_lookup()` logs which lookups each client makes, once per kind every ten minutes, without the model name.
- `_context_window()` and `window_used` put the share of the window a chat uses into the stats; `_WINDOW_WARN` (90%) is where the console and the log start showing it.

**Context guard** (`"context_guard": {"enabled": true}`, `_ctx_guard()`): a client that compacts only after an overflow loses what it cuts, because the chat no longer fits anything that could read it. Cline 4.1.26's overflow recovery drops messages (`basic` strategy), and its auto-compact didn't fire here even at 140k of the 128k it assumed. At `at` (90%) of the window the gateway answers `context_length_exceeded` before forwarding and keeps the request's messages in `_ctx_pending` for the client's next request. `_ctx_compare()` then tells what that request is: new messages after a cut-down chat is a compact request, and the full chat is put back (`restore`); the same turn with messages cut gets a summary made by the engine from the chat up to the last cut message (`_ctx_summarise()`), appended to the request's last message and sent as the first piece of the reply, so the client stores it. Token counts before forwarding are estimates: request bytes over `_ctx_ratio`, learned in `_record()`.

`gateway.log` (`_write_log_file()`): every `_queue_log_line()` line also goes to a file beside the config, cut back at `_LOG_FILE_MAX`.

Rules that must hold:

1. **Only an error reply is rewritten, and only its body and status.** A reply that isn't an overflow error passes byte for byte, and so does one already in the standard form. A failure while reading it sends the engine's error as it came.
2. **No guessing a limit into existence.** Token counts come from the engine's fields or its message; when it gives none, the message says so without numbers.
3. **Stats keep counts only:** `overflow` (yes or no), `window` and `window_used`.
4. **The guard never locks a client out and never breaks a request.** The request after a "full" is never refused; an unrecognised follow-up, a failed summary or any exception forwards the request as sent.
5. **Held chat text stays in memory:** one follow-up request or `_CTX_PENDING_TTL_S`, never on disk, in the log or in stats. The log gets counts only.
6. **The summary is appended to the last message, never inserted earlier,** so the prompt before it is unchanged, and it reaches the client inside the reply because the gateway keeps no chat state between turns.

## Thinking Switch

The client decides whether a model thinks. `_thinking_asked()` reads that choice from any of the usual fields (`reasoning_effort`, `reasoning`, `chat_template_kwargs`, `thinking`, `think`). For an `lmstudio` endpoint's `/chat/completions`, `_lmstudio_thinking()` writes it as `reasoning_effort`, because that is the only field LM Studio honours there. This was tested live with a Gemma 4 model on 2026-11-11: `"none"` turned thinking off, while `chat_template_kwargs.enable_thinking` and `reasoning.effort` were ignored.

Rules that must hold:

1. **Never a setting the client didn't ask for.** There is no config value for thinking. When the client asks for nothing, or asks for "on" without a level, nothing is added, and LM Studio's load setting decides.
2. **The client's own `reasoning_effort` always wins** and goes through untouched. The client's other fields stay in place.
3. **Only LM Studio's `/chat/completions` is changed.** Every other endpoint type and path gets the body as sent. A failure while translating forwards the request unchanged.
4. **Stats keep only the name of the choice** (`think`: a level from `_THINK_EFFORTS`, `on` or `other`), never a value the client invented.

## API Endpoints

| Endpoint | Format | Purpose |
|----------|--------|---------|
| `GET /v1/models` | OpenAI standard | Standard clients, OpenAI Compatible provider |
| `GET /api/v0/models` | LM Studio | Cline LM Studio provider, LM Studio clients |
| `GET /api/v1/models` | LM Studio | Alias for `/api/v0/models` |
| `GET /v1/models/<id>`, `GET /api/v0/models/<id>` | OpenAI / LM Studio | One model's entry, by the gateway's id or the engine's own (`_send_one_model`). Apps ask this to learn a model's context window |
| `GET /api/tags`, `GET /api/ps`, `POST /api/show` | Ollama | Model list and one model's info (`<arch>.context_length`, `num_ctx`), for apps that ask an Ollama-style server |
| `GET /props` | llama.cpp | Server properties (modalities, context, params) |
| `POST /v1/chat/completions` | OpenAI standard | Chat completions (proxied to correct endpoint) |
| `GET /health` | — | Endpoint statuses |
| `GET /stats` | JSON | Live stats: the gateway's counts, engine numbers and saved chats |
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