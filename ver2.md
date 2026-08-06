# AI Gateway v2 — Multi-Endpoint Aggregation Plan

## Overview

Transform the AI Gateway from a single-endpoint proxy into a **multi-endpoint aggregator** that connects to all configured AI backends simultaneously, checks their availability, and presents a unified model catalog to all clients.

## Current State (v1)

- Single upstream endpoint configured via `host` + `api_key` in `config.json`
- All requests forwarded to that one endpoint
- Model metadata fetched from either OpenRouter or a local backend (never both)
- Model IDs are raw (e.g., `openai/gpt-5.6-luna`, `my-local-model`)

## Target State (v2)

- Multiple endpoints configured and independently enabled/disabled
- Health checking runs continuously — only online endpoints serve models
- Aggregated model list from all online endpoints with source identification
- Model IDs follow `source/vendor/modelname` format
- Smart routing: proxy automatically sends requests to the correct endpoint
- Full backward compatibility with v1 config (auto-migration)

---

## 1. Config Format

### v2 `config.json` Schema

```json
{
  "endpoints": [
    {
      "name": "openrouter",
      "type": "openrouter",
      "host": "openrouter.ai",
      "enabled": true,
      "api_key": "sk-or-v1-..."
    },
    {
      "name": "llama.cpp",
      "type": "llama-server",
      "host": "localhost:8080",
      "enabled": false,
      "api_key": ""
    },
    {
      "name": "lmstudio",
      "type": "lmstudio",
      "host": "localhost:1234",
      "enabled": false,
      "api_key": ""
    }
  ],
  "models": [
    "openrouter/openai/gpt-5.6-luna",
    "openrouter/qwen/qwen3.8-max",
    "llama.cpp/mistral-7b"
  ],
  "model_index": 0,
  "port": 8090,
  "addr": "0.0.0.0",
  "fetch_top_models": false
}
```

### Endpoint Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | **User-configurable short name** — becomes the source prefix in model IDs. Default depends on type: `openrouter`, `lmstudio`, `llama.cpp` |
| `type` | string | Endpoint type: `openrouter`, `lmstudio`, `llama-server`, `custom` |
| `host` | string | Upstream host (with port if non-standard) |
| `enabled` | bool | Whether this endpoint is active |
| `api_key` | string | API key (empty for local endpoints) |

### Endpoint Types

| Type | Default `name` | Default `host` | Health Check | Model Discovery |
|------|----------------|----------------|--------------|-----------------|
| `openrouter` | `openrouter` | `openrouter.ai` | `GET /api/v1/models` (HTTPS) | `/api/v1/models` → full catalog |
| `lmstudio` | `lmstudio` | `localhost:1234` | `GET /v1/models` (HTTP) | `/v1/models` → loaded models |
| `llama-server` | `llama.cpp` | `localhost:8080` | `GET /health` (HTTP) | `/props` → single model + modalities |
| `custom` | `custom` | (user sets) | `GET /v1/models` (HTTP/HTTPS) | `/v1/models` → model list |

### v1 → v2 Migration

On first load, if `config.json` has `host`/`api_key` but no `endpoints`:

```python
def migrate_v1_config(saved: dict) -> dict:
    """Migrate v1 single-host config to v2 endpoints format."""
    host = saved.get("host", "openrouter.ai")
    
    # Detect type from host
    if "openrouter" in host:
        ep_type = "openrouter"
        name = "openrouter"
    elif "1234" in host:
        ep_type = "lmstudio"
        name = "lmstudio"
    else:
        ep_type = "llama-server"
        name = "llama.cpp"
    
    endpoints = [{
        "name": name,
        "type": ep_type,
        "host": host,
        "enabled": True,
        "api_key": saved.get("api_key", ""),
    }]
    
    saved["endpoints"] = endpoints
    # Remove old fields (keep for backward compat reads)
    return saved
```

---

## 2. Model ID Format

### Format: `{source}/{vendor}/{modelname}`

| Endpoint `name` | Upstream model ID | Aggregated model ID |
|---|---|---|
| `openrouter` | `openai/gpt-5.6-luna` | `openrouter/openai/gpt-5.6-luna` |
| `openrouter` | `qwen/qwen3.8-max` | `openrouter/qwen/qwen3.8-max` |
| `llama.cpp` | `mistral-7b` | `llama.cpp/mistral-7b` |
| `llama.cpp` | `llama-3.2-8b-instruct` | `llama.cpp/llama-3.2-8b-instruct` |
| `lmstudio` | `llama-3.2-8b` | `lmstudio/llama-3.2-8b` |

### Parsing Logic

```python
def parse_model_id(model_id: str) -> tuple[str, str]:
    """Parse 'source/vendor/model' into (source_name, original_model_id).
    
    First segment = endpoint name (source)
    Remaining = original model ID as known by the upstream endpoint
    """
    parts = model_id.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    # Fallback: no source prefix, use default endpoint
    return "", model_id

# Examples:
# "openrouter/openai/gpt-5.6-luna" → ("openrouter", "openai/gpt-5.6-luna")
# "llama.cpp/mistral-7b"           → ("llama.cpp", "mistral-7b")
# "mistral-7b"                     → ("", "mistral-7b")  # fallback to default
```

### Collision-Free

Since every model is prefixed with its source `name`, collisions are impossible:
- `openrouter/openai/gpt-4` vs `llama.cpp/gpt-4` — distinct IDs
- Even if two endpoints have the same model name, different source names keep them separate

---

## 3. Health Checking

### Background Thread

A dedicated thread runs every **30 seconds**, checking each enabled endpoint:

```python
def health_check_loop():
    """Continuously check endpoint health."""
    while True:
        for ep in get_config("endpoints"):
            if not ep["enabled"]:
                ep["status"] = "disabled"
                continue
            try:
                status = check_endpoint_health(ep)
                ep["status"] = status  # "online" or "offline"
            except Exception:
                ep["status"] = "offline"
        time.sleep(30)
```

### Per-Type Health Checks

| Type | Check Method | Success Criteria |
|------|-------------|-----------------|
| `openrouter` | `GET https://openrouter.ai/api/v1/models` | Status 200 |
| `lmstudio` | `GET http://{host}/v1/models` | Status 200 |
| `llama-server` | `GET http://{host}/health` or `/props` | Status 200 |
| `custom` | `GET http(s)://{host}/v1/models` | Status 200 |

### Startup Check

On startup, run health checks **immediately** (non-blocking, in background) before the first model list is served. Until checks complete, all enabled endpoints show status `checking`.

### Status Values

| Status | Meaning |
|--------|---------|
| `checking` | Initial state, health check in progress |
| `online` | Endpoint responded successfully |
| `offline` | Endpoint unreachable or returned error |
| `disabled` | Endpoint is disabled in config |

---

## 4. Model Aggregation

### Flow

```
1. Health check identifies online endpoints
2. For each online endpoint, fetch its models:
   - openrouter: GET /api/v1/models → full catalog
   - lmstudio:   GET /v1/models → loaded models
   - llama-server: GET /props → single model info
3. Prefix each model ID with endpoint name
4. Merge into unified MODELS list
5. Build routing table: model_id → (endpoint, original_id)
```

### Data Structures

```python
# Global routing table
# Maps aggregated model ID → (endpoint_name, original_model_id, endpoint_config)
_model_routes: dict[str, tuple[str, str, dict]] = {}

# Aggregated models list (used by TUI and /v1/models)
MODELS: list[str] = []

# Model metadata cache (aggregated from all endpoints)
_model_meta: dict[str, dict] = {}  # aggregated_id → metadata dict
```

### Aggregation Function

```python
def aggregate_models():
    """Fetch models from all online endpoints and build unified list."""
    global MODELS, _model_routes, _model_meta
    
    new_routes = {}
    new_meta = {}
    user_models = list(get_config("models"))
    
    for ep in get_config("endpoints"):
        if ep.get("status") != "online":
            continue
        
        source_name = ep["name"]
        ep_type = ep["type"]
        host = ep["host"]
        
        # Fetch models based on type
        if ep_type == "openrouter":
            models = fetch_openrouter_models(host, ep.get("api_key", ""))
        elif ep_type == "lmstudio":
            models = fetch_lmstudio_models(host)
        elif ep_type == "llama-server":
            models = fetch_llama_server_models(host)
        elif ep_type == "custom":
            models = fetch_custom_models(host, ep.get("api_key", ""))
        else:
            continue
        
        for model in models:
            original_id = model["id"]
            aggregated_id = f"{source_name}/{original_id}"
            
            # Store route
            new_routes[aggregated_id] = (source_name, original_id, ep)
            
            # Store metadata with source info
            model["_source"] = source_name
            model["_source_host"] = host
            model["id"] = aggregated_id
            new_meta[aggregated_id] = model
    
    # Build MODELS list: user models first, then discovered models
    merged = []
    for m in user_models:
        if m in new_routes:
            merged.append(m)
    
    for m in new_routes:
        if m not in merged:
            merged.append(m)
    
    _model_routes = new_routes
    _model_meta = new_meta
    MODELS = merged
```

### Fetch Functions

```python
def fetch_openrouter_models(host: str, api_key: str) -> list[dict]:
    """Fetch all models from OpenRouter."""
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(host, context=ctx, timeout=30)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn.request("GET", "/api/v1/models", headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data.get("data", [])

def fetch_lmstudio_models(host: str) -> list[dict]:
    """Fetch loaded models from LM Studio."""
    conn = http.client.HTTPConnection(host, timeout=10)
    conn.request("GET", "/v1/models")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data.get("data", [])

def fetch_llama_server_models(host: str) -> list[dict]:
    """Fetch model info from llama.cpp server."""
    conn = http.client.HTTPConnection(host, timeout=10)
    conn.request("GET", "/props")
    resp = conn.getresponse()
    props = json.loads(resp.read())
    conn.close()
    
    # llama-server serves one model
    model_path = props.get("model_path", "unknown")
    model_name = os.path.basename(model_path) if "/" in model_path else model_path
    modalities = props.get("modalities", {})
    
    return [{
        "id": model_name,
        "name": model_name,
        "context_length": props.get("default_generation_settings", {}).get("n_ctx", 128000),
        "architecture": {
            "input_modalities": _modalities_to_input(modalities),
        },
    }]

def fetch_custom_models(host: str, api_key: str) -> list[dict]:
    """Fetch models from a custom OpenAI-compatible endpoint."""
    use_ssl = not any(host.startswith(h) for h in ("localhost", "127.0.0.1"))
    if use_ssl:
        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection(host, context=ctx, timeout=30)
    else:
        conn = http.client.HTTPConnection(host, timeout=30)
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    conn.request("GET", "/v1/models", headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data.get("data", [])
```

---

## 5. Smart Request Routing

### Proxy Handler Changes

The core `_proxy()` method needs to:

1. Extract model ID from request body
2. Parse source from model ID (`source/original_id`)
3. Look up endpoint by source name
4. Forward to that endpoint with the original model ID

```python
def _proxy(self):
    # 1. Read body
    body = self._read_body()
    
    # 2. Parse model and determine route
    model_id = None
    original_id = None
    target_endpoint = None
    
    if body:
        try:
            data = json.loads(body)
            model_id = data.get("model", "")
            
            if not model_id:
                # No model specified — use default
                model_id = get_current_model()
                data["model"] = model_id
                body = json.dumps(data).encode("utf-8")
            
            # Parse source prefix
            source, original_id = parse_model_id(model_id)
            
            if source:
                # Find endpoint by name
                target_endpoint = get_endpoint_by_name(source)
            
            if not target_endpoint:
                # Fallback: try to find model in routing table
                route = _model_routes.get(model_id)
                if route:
                    source, original_id, target_endpoint = route
            
            if target_endpoint and original_id != model_id:
                # Strip source prefix for upstream
                data["model"] = original_id
                body = json.dumps(data).encode("utf-8")
                
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    
    # 3. Determine upstream connection
    if target_endpoint:
        host = target_endpoint["host"]
        api_key = target_endpoint.get("api_key", "")
    else:
        # Fallback to first enabled endpoint
        target_endpoint = get_first_enabled_endpoint()
        host = target_endpoint["host"] if target_endpoint else get_config("host")
        api_key = target_endpoint.get("api_key", "") if target_endpoint else ""
    
    # 4. Build headers and forward
    use_ssl = not any(host.startswith(h) for h in ("localhost", "127.0.0.1", "[::1]"))
    # ... rest of proxy logic unchanged ...
```

### Error Handling

| Scenario | Response |
|----------|----------|
| Model not found in any endpoint | `404 {"error": "Model not found: {model_id}"}` |
| Endpoint is offline | `503 {"error": "Endpoint '{name}' is offline"}` |
| All endpoints offline | `503 {"error": "All endpoints are offline"}` |
| No model specified, no default | `400 {"error": "No model specified and no default configured"}` |

---

## 6. `/v1/models` Response Format

### Response Structure

```json
{
  "object": "list",
  "data": [
    {
      "id": "openrouter/openai/gpt-5.6-luna",
      "object": "model",
      "created": 1234567890,
      "owned_by": "openai",
      "source": "openrouter",
      "source_host": "openrouter.ai",
      "name": "GPT-5.6 Luna",
      "context_length": 128000,
      "context_window": 128000,
      "max_model_len": 128000,
      "in_cache": true,
      "path": "openrouter/openai/gpt-5.6-luna",
      "tags": ["vision"],
      "architecture": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"]
      },
      "meta": {
        "capabilities": {
          "vision": true,
          "audio": false,
          "video": false,
          "function_calling": true
        }
      }
    },
    {
      "id": "llama.cpp/mistral-7b",
      "object": "model",
      "owned_by": "llama.cpp",
      "source": "llama.cpp",
      "source_host": "localhost:8080",
      "name": "Mistral 7B",
      "context_length": 8192,
      "tags": [],
      "architecture": {
        "input_modalities": ["text"],
        "output_modalities": ["text"]
      },
      "meta": {
        "capabilities": {
          "vision": false,
          "audio": false,
          "video": false,
          "function_calling": false
        }
      }
    }
  ]
}
```

### Key Fields for Clients

| Field | Purpose | Used by |
|-------|---------|---------|
| `id` | Unique model identifier with source prefix | All clients (for API requests) |
| `name` | Human-readable display name | Web UI, Open WebUI |
| `source` | Which endpoint serves this model | Smart clients |
| `source_host` | Endpoint address | Smart clients |
| `tags` | Capability tags (vision, audio, video) | llama.cpp web UI |
| `architecture.input_modalities` | Input modalities | llama.cpp web UI |
| `meta.capabilities` | Capability flags | Open WebUI |

---

## 7. TUI Changes

### Main Screen

```
┌─ AI Gateway ─────────────────────────────────────────────┐
│ [⚙ Settings]  [✕ Quit]                                   │
│                                                           │
│ Endpoints                                                 │
│   ● openrouter  → openrouter.ai      online   4523 models│
│   ● llama.cpp   → localhost:8080     online   1 model    │
│   ○ lmstudio    → localhost:1234     offline              │
│                                                           │
│ Default: openrouter/openai/gpt-5.6-luna                  │
│                                                           │
│ Proxy Log                                                 │
│   [192.168.1.50] POST /v1/chat/completions → openrouter  │
│   [192.168.1.50] POST /v1/chat/completions → llama.cpp   │
│   ...                                                     │
└───────────────────────────────────────────────────────────┘
```

### Settings Screen — Endpoint Management (Textual TUI)

Uses Textual's `DataTable` for the endpoint list and `ModalScreen` for add/edit dialogs.
Dynamic widget support via `mount()`/`remove()` and `DataTable.add_row()`/`remove_row()`.

```
┌─ Settings ───────────────────────────────────────────────┐
│                                                           │
│ Endpoints                                                 │
│ ┌─────────────────────────────────────────────────────┐  │
│ │ Name       │ Type        │ Host            │ Status  │  │
│ │────────────┼─────────────┼─────────────────┼─────────│  │
│ │ openrouter │ openrouter  │ openrouter.ai   │ ● on    │  │
│ │ llama.cpp  │ llama-server│ localhost:8080  │ ● on    │  │
│ │ lmstudio   │ lmstudio    │ localhost:1234  │ ○ off   │  │
│ └─────────────────────────────────────────────────────┘  │
│ [+ Add]  [✎ Edit]  [✕ Remove]                            │
│                                                           │
│ Default Model                                             │
│ ▼ openrouter/openai/gpt-5.6-luna                          │
│   openrouter/qwen/qwen3.8-max                             │
│   llama.cpp/mistral-7b                                     │
│                                                           │
│ Model Fetching                                            │
│ [✓] Fetch top intelligent models from OpenRouter          │
│                                                           │
│ Server                                                    │
│ Port: [8090]    Bind: [0.0.0.0]                           │
│                                                           │
│ [💾 Save]  [✕ Cancel]  [↻ Restart]                        │
└───────────────────────────────────────────────────────────┘
```

### Add/Edit Endpoint Dialog (ModalScreen)

```
┌─ Add Endpoint ───────────────────────────────────────────┐
│                                                           │
│ Type:     [▼ llama-server          ]                      │
│ Name:     [llama.cpp               ] (short ID prefix)   │
│ Host:     [localhost:8080           ]                     │
│ API Key:  [                        ] (optional)           │
│ Enabled:  [✓]                                             │
│                                                           │
│ [💾 Save]  [✕ Cancel]                                     │
└──────────────────────────────────────────────────────────┘
```

When user selects a type, `name` and `host` auto-fill with defaults:
- `openrouter` → name: `openrouter`, host: `openrouter.ai`
- `lmstudio` → name: `lmstudio`, host: `localhost:1234`
- `llama-server` → name: `llama.cpp`, host: `localhost:8080`
- `custom` → name: `custom`, host: (empty)

**Technical implementation:**
- `DataTable` for endpoint list — dynamic rows via `add_row()`/`remove_row()`
- `ModalScreen` subclass for add/edit form — mounts `Select`, `Input`, `Checkbox`, `Button` widgets
- Type selection auto-fills name/host defaults via `on_select_changed` handler
- Endpoint list is populated from `config.json` endpoints array on screen mount

---

## 8. Implementation Steps

### Step 1: Endpoint Config & Migration
- Add `endpoints` to `DEFAULTS`
- Implement `migrate_v1_config()` for backward compatibility
- Update `load_config()` and `save_config()`
- Add `get_endpoint_by_name()` helper
- Update `config.json` on disk after migration

### Step 2: Health Checking
- Add `health_check_loop()` background thread
- Implement per-type health check functions
- Add `status` field to endpoint state
- Start health check thread on startup
- Update TUI main screen with endpoint status display

### Step 3: Model Aggregation
- Implement per-type model fetch functions
- Implement `aggregate_models()` to build unified list
- Build `_model_routes` routing table
- Update `fetch_model_metadata()` to use aggregation
- Trigger re-aggregation after each health check cycle

### Step 4: Smart Request Routing
- Implement `parse_model_id()` function
- Update `_proxy()` to route based on model source prefix
- Handle default model injection with source-aware routing
- Add proper error responses for offline endpoints
- Strip source prefix before forwarding to upstream

### Step 5: Update `/v1/models` Response
- Update `_handle_models()` to use aggregated model list
- Add `source` and `source_host` fields to response
- Use aggregated metadata for capabilities

### Step 6: Update TUI Settings Screen
- Replace single endpoint dropdown with endpoint manager
- Add endpoint list with enable/disable checkboxes
- Add/Edit/Remove endpoint buttons and dialog
- Update model selector to show source-prefixed models
- Remove old host/api-key fields (now per-endpoint)

### Step 7: Update TUI Main Screen
- Add endpoint status panel showing all endpoints and their status
- Update log to show which endpoint each request routes to
- Show aggregated model count

### Step 8: Update Documentation
- Update `README.md` with v2 features and config format
- Document the `source/vendor/modelname` format
- Document endpoint types and configuration
- Update examples

---

## 9. Backward Compatibility

| v1 Feature | v2 Equivalent | Migration |
|-----------|---------------|-----------|
| `host` field | First endpoint's `host` | Auto-migrated |
| `api_key` field | First endpoint's `api_key` | Auto-migrated |
| `models` list | Same, but IDs now include source prefix | Auto-migrated (prefix added) |
| `model_index` | Same | Unchanged |
| `fetch_top_models` | Same (only for OpenRouter endpoint) | Unchanged |
| CLI args `port` and `addr` | Same | Unchanged |

### Migration on First Load

```python
def load_config():
    # Load saved config
    saved = load_json(CONFIG_FILE)
    
    # Detect v1 format
    if "host" in saved and "endpoints" not in saved:
        saved = migrate_v1_config(saved)
        save_json(CONFIG_FILE, saved)
        print("[proxy] Migrated v1 config to v2 format")
    
    # ... rest of loading
```

---

## 10. File Changes Summary

All changes are in **`proxy.py`** (single-file architecture preserved):

| Section | Change |
|---------|--------|
| Config defaults | Add `endpoints` list, remove `host`/`api_key` from top-level |
| `load_config()` | Add v1→v2 migration |
| `save_config()` | Save `endpoints` format |
| New: Health checking | `health_check_loop()`, `check_endpoint_health()` |
| New: Model aggregation | `aggregate_models()`, `fetch_*_models()` per type |
| New: Routing | `parse_model_id()`, `get_endpoint_by_name()` |
| `_proxy()` | Route requests based on model source prefix |
| `_handle_models()` | Use aggregated list with source fields |
| `_handle_props()` | Use aggregated metadata |
| `SettingsScreen` | Endpoint manager UI |
| `MainScreen` | Endpoint status display |
| `ENDPOINTS` constant | Replaced by dynamic endpoint list from config |

---

## 11. Edge Cases

| Case | Handling |
|------|----------|
| No model in request | Inject default from `model_index`, route to its source endpoint |
| Default model's endpoint offline | Try next available endpoint; if all offline, return 503 |
| Endpoint added but not yet checked | Status = `checking`, models not included until online |
| Model name collision between endpoints | Impossible — source prefix makes IDs unique |
| Endpoint removed from config | Models from that endpoint removed from aggregated list on next cycle |
| OpenRouter returns thousands of models | Only user's `models` list + top models (if enabled) shown; full catalog available for adding |
| Local endpoint serves one model | Shows as single entry with full metadata from `/props` |
| Client sends raw model ID without prefix | Fallback: search routing table for match; if found, route correctly |
| All endpoints offline | `/v1/models` returns empty list; requests return 503 |

---

## 12. Future Considerations (Not in v2.0)

- **Load balancing**: Multiple endpoints of same type (e.g., two llama.cpp servers)
- **Failover**: If primary endpoint fails, try secondary
- **Rate limiting**: Per-endpoint rate tracking
- **Cost tracking**: Track token usage per endpoint
- **Endpoint-specific model lists**: Instead of global `models` list, per-endpoint model preferences