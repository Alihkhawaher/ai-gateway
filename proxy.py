"""
AI Gateway — Multi-endpoint OpenAI-compatible proxy with Textual TUI.

Connects to multiple AI backends (OpenRouter, LM Studio, llama.cpp, etc.)
simultaneously, checks their availability, and presents a unified model
catalog to all clients. Model IDs follow the source:vendor/modelname format.

Includes an integrated web chat UI (llama.cpp's llama-ui) served at /.

Usage:
  python proxy.py              # listens on 0.0.0.0:8090
  python proxy.py 9090         # listens on 0.0.0.0:9090
  python proxy.py 9090 127.0.0.1   # custom bind address
"""

import copy
import json
import http.client
import ipaddress
import mimetypes
import os
import queue
import ssl
import sys
import threading
import subprocess
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote, parse_qs
from concurrent.futures import ThreadPoolExecutor

# ── Config file path ──────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# ── Web UI directory ─────────────────────────────────────────────────────
WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")

# ── Endpoint type defaults ───────────────────────────────────────────────
ENDPOINT_TYPE_DEFAULTS = {
    "openrouter": {"name": "openrouter", "host": "openrouter.ai"},
    "orcarouter": {"name": "orcarouter", "host": "api.orcarouter.ai"},
    "lmstudio":   {"name": "lmstudio",   "host": "localhost:1234"},
    "llama-server": {"name": "llama.cpp", "host": "localhost:8080"},
    "ollama":     {"name": "ollama",      "host": "localhost:11434"},
    "strata":     {"name": "strata",      "host": "127.0.0.1:47831"},
    "custom":     {"name": "custom",      "host": ""},
}

# ── Default values (used if config.json doesn't exist) ────────────────────
DEFAULTS = {
    "endpoints": [
        {
            "name": "openrouter",
            "type": "openrouter",
            "host": "openrouter.ai",
            "enabled": True,
            "api_key": "",
        },
    ],
    "models": [
        "openrouter:openai/gpt-5.6-luna",
        "openrouter:qwen/qwen3.8-max",
        "openrouter:deepseek/deepseek-v4-flash-0731",
        "openrouter:deepseek/deepseek-v4.1-flash",
    ],
    "default_model": "",
    "port": 8090,
    "addr": "0.0.0.0",
    "fetch_top_models": False,
}

# ── Runtime models list (loaded from config) ─────────────────────────────
MODELS = list(DEFAULTS["models"])

# ── Model routing table: aggregated_id → (source_name, original_id, endpoint_dict) ──
_model_routes = {}  # dict[str, tuple[str, str, dict]]

# ── Aggregated model metadata cache ──────────────────────────────────────
_model_meta_lock = threading.Lock()
_model_meta = {}  # aggregated_id → metadata dict

# ── Endpoint runtime status (updated by health checks) ───────────────────
_endpoint_status_lock = threading.Lock()
_endpoint_status = {}  # endpoint_name → "checking" | "online" | "offline" | "disabled"

# ── Endpoint load tracking (in-flight + cumulative proxied requests) ───────
_endpoint_load_lock = threading.Lock()
_endpoint_load = {}  # endpoint_name → {"active": int, "total": int}


def bump_endpoint_load_start(name: str):
    """Mark a request as in-flight for the given endpoint."""
    with _endpoint_load_lock:
        e = _endpoint_load.setdefault(name, {"active": 0, "total": 0})
        e["active"] += 1


def bump_endpoint_load_end(name: str):
    """Mark an in-flight request finished and count it toward the total."""
    with _endpoint_load_lock:
        e = _endpoint_load.setdefault(name, {"active": 0, "total": 0})
        e["active"] = max(0, e["active"] - 1)
        e["total"] += 1


def get_endpoint_load(name: str):
    """Return (active, total) request counts for an endpoint."""
    with _endpoint_load_lock:
        e = _endpoint_load.get(name)
        if not e:
            return 0, 0
        return e["active"], e["total"]


def format_endpoint_load(name: str) -> str:
    """Human-readable load string: active in-flight requests and total served."""
    active, total = get_endpoint_load(name)
    if active > 0:
        return f"⚡ {active} active · {total} total"
    if total > 0:
        return f"idle · {total} total"
    return "—"

# ── Top intelligent models fetched from OpenRouter ────────────────────────
_top_intelligent_models = []
_top_models_lock = threading.Lock()

# ── Thread-safe shared config ──────────────────────────────────────────────
_config_lock = threading.Lock()
_config = {}  # loaded from config.json
_config_load_error = ""  # set when config.json exists but can't be read
_config_save_lock = threading.Lock()

# ── Thread-safe MODELS list ────────────────────────────────────────────────
_models_lock = threading.Lock()

# ── Upstream catalog cache (avoids re-downloading large catalogs) ──────────
_catalog_cache_lock = threading.Lock()
_catalog_cache = {}       # endpoint_name → list of raw model dicts
_catalog_cache_time = {}  # endpoint_name → timestamp
CATALOG_CACHE_TTL = 300   # seconds

# ── TUI log queue (thread-safe bridge from worker threads) ───────────────
_log_queue = queue.Queue()


def _queue_log_line(line: str):
    """Thread-safely enqueue a line for the TUI log widget."""
    try:
        _log_queue.put(line)
    except Exception:
        pass


def load_config():
    """Load config from config.json, merging with defaults. Handles v1→v2 migration."""
    global _config, _config_load_error
    _config_load_error = ""
    with _config_lock:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                # ── v1 → v2 migration ───────────────────────────────────
                if "host" in saved and "endpoints" not in saved:
                    saved = _migrate_v1_config(saved)
                    try:
                        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                            json.dump(saved, f, indent=2)
                        print("[proxy] Migrated v1 config to v2 format")
                    except Exception as e:
                        print(f"[proxy] Could not save migrated config: {e}")
                # ── Merge with defaults ──────────────────────────────────
                for key in DEFAULTS:
                    if key in saved:
                        _config[key] = saved[key]
                # Ensure endpoints list exists
                if "endpoints" not in _config:
                    _config["endpoints"] = list(DEFAULTS["endpoints"])
                print(f"[proxy] Config loaded from {CONFIG_FILE}")
            except Exception as e:
                # Run on defaults in memory, but never save over the file
                # (save_config checks this): one stray comma would otherwise
                # wipe every endpoint and API key on the next save.
                _config_load_error = str(e)
                msg = (f"[proxy] Could not load config: {e} — running on defaults; "
                       f"config.json will not be saved until it is fixed")
                print(msg)
                _queue_log_line(msg)
        else:
            print(f"[proxy] No config file found, using defaults")
        _config = {**DEFAULTS, **_config}
        set_models(list(_config.get("models", DEFAULTS["models"])))


def _migrate_v1_config(saved: dict) -> dict:
    """Migrate v1 single-host config to v2 endpoints format."""
    host = saved.get("host", "openrouter.ai")
    api_key = saved.get("api_key", "")

    # Detect type from host
    if "openrouter" in host:
        ep_type, name = "openrouter", "openrouter"
    elif "1234" in host:
        ep_type, name = "lmstudio", "lmstudio"
    else:
        ep_type, name = "llama-server", "llama.cpp"

    endpoints = [{
        "name": name,
        "type": ep_type,
        "host": host,
        "enabled": True,
        "api_key": api_key,
    }]
    saved["endpoints"] = endpoints

    # Prefix existing model IDs with source name
    old_models = saved.get("models", [])
    saved["models"] = [f"{name}:{m}" for m in old_models]

    # Remove old top-level fields
    saved.pop("host", None)
    saved.pop("api_key", None)
    return saved


def save_config():
    """Save current config to config.json.

    Written to a temp file and swapped in, so a crash or two threads saving
    at once (web UI and TUI) can't leave a half-written file. Skipped while
    config.json is unreadable, so the broken file isn't replaced by defaults.
    """
    if _config_load_error:
        msg = f"[proxy] Not saving config: config.json could not be read ({_config_load_error})"
        print(msg)
        _queue_log_line(msg)
        return
    with _config_lock:
        text = json.dumps(_config, indent=2)
    tmp = CONFIG_FILE + ".tmp"
    try:
        with _config_save_lock:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, CONFIG_FILE)
    except Exception as e:
        print(f"[proxy] Could not save config: {e}")


def get_config(key: str):
    with _config_lock:
        return _config.get(key, DEFAULTS.get(key))


def set_config(key: str, value):
    with _config_lock:
        _config[key] = value


def get_models() -> list:
    """Return a snapshot copy of the aggregated model list."""
    with _models_lock:
        return list(MODELS)


def set_models(models: list):
    """Atomically replace the aggregated model list."""
    global MODELS
    with _models_lock:
        MODELS = list(models)


def get_current_model() -> str:
    """The default model: the one picked by ID, else the first available.

    Kept by ID, not list position: the list is rebuilt every 30 s from the
    endpoints that are online, so a position drifts onto other models.
    """
    models = get_models()
    chosen = get_config("default_model")
    if chosen in models:
        return chosen
    return models[0] if models else ""


# ── Model ID parsing ─────────────────────────────────────────────────────
def parse_model_id(model_id: str) -> tuple:
    """Parse 'source:vendor/model' into (source_name, original_model_id).

    First segment (up to the first ':') = endpoint name (source)
    Remaining = original model ID as known by the upstream endpoint
    (which may itself contain '/', e.g. deepseek/deepseek-v4.1-flash).
    Returns ("", model_id) if no source prefix found.
    """
    parts = model_id.split(":", 1)
    if len(parts) == 2:
        # Check if the first segment matches a configured endpoint name
        with _config_lock:
            ep_names = [ep.get("name", "") for ep in _config.get("endpoints", [])]
        if parts[0] in ep_names:
            return parts[0], parts[1]
    return "", model_id


def get_endpoint_by_name(name: str) -> dict:
    """Find an enabled endpoint config by its name."""
    with _config_lock:
        for ep in _config.get("endpoints", []):
            if ep.get("name") == name and ep.get("enabled", False):
                return ep
    return {}


def find_endpoint_by_name(name: str) -> dict:
    """Find an endpoint config by name regardless of enabled state."""
    with _config_lock:
        for ep in _config.get("endpoints", []):
            if ep.get("name") == name:
                return ep
    return {}


def get_first_enabled_endpoint() -> dict:
    """Get the first enabled endpoint."""
    with _config_lock:
        for ep in _config.get("endpoints", []):
            if ep.get("enabled", False):
                return ep
    return {}


def get_endpoint_status(name: str) -> str:
    """Get runtime status of an endpoint."""
    with _endpoint_status_lock:
        return _endpoint_status.get(name, "disabled")


def set_endpoint_status(name: str, status: str):
    """Set runtime status of an endpoint."""
    with _endpoint_status_lock:
        _endpoint_status[name] = status


# ── Path rewrite: prefix /api/v1 if not already present ───────────────────
def rewrite_path(path: str) -> str:
    if path == "/models/sse":
        return "/api/v1/models"
    if path.startswith("/api/"):
        return path
    if path.startswith("/v1/"):
        return f"/api{path}"
    return f"/api/v1{path}"


# ── CORS headers ─────────────────────────────────────────────────────────
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Max-Age": "86400",
}

# ── MIME types for static serving ─────────────────────────────────────────
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png":  "image/png",
    ".ico":  "image/x-icon",
    ".svg":  "image/svg+xml",
    ".webmanifest": "application/manifest+json",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf":  "font/ttf",
}

# ── Request body size limit ────────────────────────────────────────────────
_MAX_BODY_SIZE = 50 * 1024 * 1024  # 50MB

# ── Shared SSL context for upstream connections ────────────────────────────
_upstream_ssl_ctx = ssl.create_default_context()


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL CAPABILITY FORWARDING — DO NOT REGRESS
# ═══════════════════════════════════════════════════════════════════════════
#
# Model capabilities (vision / audio / video / function_calling) are forwarded
# to clients through THREE mechanisms. All three are REQUIRED and must not be
# removed or simplified — each serves a different client:
#
#   1. `_get_cached_catalog()` MUST return deep copies (copy.deepcopy).
#      `aggregate_models()` mutates the returned dicts (adds _source/_source_host
#      and overwrites `id`). A shallow copy corrupts the cache, so on the next
#      30-second aggregation cycle the original model ID can no longer be found
#      and the model loses its `architecture` metadata (capabilities). This was
#      the v2.3.0 regression.
#
#   2. `has_tools` (function_calling) MUST check `supported_parameters` for
#      "tools"/"tool_choice", NOT only `input_modalities`. OpenRouter/OrcaRouter
#      report tool support in `supported_parameters`, so checking only
#      `input_modalities` makes function_calling always False.
#
#   3. `/v1/models` MUST include a top-level `modalities` field
#      ({vision, audio, video}) on each model entry. The bundled llama.cpp web
#      UI reads this exact field (getModelModalities) to enable image/audio/
#      video uploads. Without it the UI reports "requires a vision-capable
#      model" even for multimodal models.
#
#   4. Reasoning / thinking capability MUST be normalized across gateways by
#      `_extract_reasoning()` and surfaced per model. Each upstream advertises
#      it differently:
#        • OpenRouter : meta["reasoning"] = {mandatory, default_enabled,
#                       supported_efforts, default_effort}  (richest form)
#        • llama.cpp  : default_generation_settings.params.reasoning_format
#                       (and chat_template_caps.supports_reasoning)
#        • Ollama     : model "capabilities" contains "thinking"
#        • LM Studio  : model "capabilities" contains "reasoning"
#        • OrcaRouter : (advertises nothing)
#      The normalized descriptor is exposed as `reasoning` on every /v1/models
#      entry (top-level + meta.capabilities.reasoning), as a "reasoning"
#      capability on /api/v0/models, and drives `reasoning_format` in /props.
#
# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH CHECKING & MODEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def _is_local_host(host: str) -> bool:
    """Check if a host is local."""
    _local = ("localhost", "127.0.0.1", "[::1]")
    return any(host == h or host.startswith(h + ":") for h in _local)


def _is_lan_host_name(host: str) -> bool:
    """True for Host values a DNS-rebinding attacker can't own: IP literals,
    localhost, single-label names (my-pc) and local-only suffixes."""
    host = host.strip().lower()
    if host.startswith("["):
        return True  # IPv6 literal, e.g. [::1]:8090
    name = host.rsplit(":", 1)[0]
    if not name:
        return False
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        pass
    if "." not in name:
        return True
    return name.endswith((".localhost", ".local", ".lan", ".home", ".internal", ".home.arpa"))


def _cors_proxy_caller_allowed(headers) -> bool:
    """Whether a /cors-proxy request comes from the gateway's own web page.

    Browsers label requests with where they come from (Sec-Fetch-Site, and
    Origin on cross-site or non-GET requests), so a page on another site is
    refused: it can't use the gateway to reach services that only listen on
    this PC (1MCP, Strata, router pages) and read their replies. The Host
    check stops DNS rebinding, where an attacker's domain is re-pointed at
    this PC so its page counts as same-origin. Requests with neither header
    are not from a browser page (curl, scripts) and are let through.
    """
    site = (headers.get("Sec-Fetch-Site") or "").lower()
    origin = headers.get("Origin")
    if not site and origin is None:
        return True
    host = (headers.get("Host") or "").strip().lower()
    if not _is_lan_host_name(host):
        return False
    if site and site not in ("same-origin", "none"):
        return False
    if origin is not None and urlparse(origin).netloc.lower() != host:
        return False
    return True


def _dial_host(host: str) -> str:
    """Address to connect to for `host`: localhost → 127.0.0.1.

    On Windows "localhost" resolves to ::1 first, and local AI servers listen
    on IPv4 only, so every connection waited ~2 s for the IPv6 attempt to be
    refused before falling back.
    """
    if host == "localhost" or host.startswith("localhost:"):
        return "127.0.0.1" + host[len("localhost"):]
    return host


def _connect_host(host: str, timeout: int = 10):
    """Create an HTTP or HTTPS connection based on host."""
    if _is_local_host(host):
        return http.client.HTTPConnection(_dial_host(host), timeout=timeout)
    else:
        return http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=timeout)


def check_endpoint_health(ep: dict) -> str:
    """Check if an endpoint is reachable. Returns 'online' or 'offline'."""
    host = ep.get("host", "")
    ep_type = ep.get("type", "")
    api_key = ep.get("api_key", "")

    try:
        if ep_type == "openrouter":
            conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=10)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            conn.request("GET", "/api/v1/models", headers=headers)
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

        elif ep_type == "orcarouter":
            conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=10)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            conn.request("GET", "/v1/models", headers=headers)
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

        elif ep_type == "llama-server":
            conn = http.client.HTTPConnection(_dial_host(host), timeout=10)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

        elif ep_type == "lmstudio":
            conn = http.client.HTTPConnection(_dial_host(host), timeout=10)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

        elif ep_type == "ollama":
            conn = http.client.HTTPConnection(_dial_host(host), timeout=10)
            conn.request("GET", "/api/tags")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

        elif ep_type in ("custom", "strata"):
            conn = _connect_host(host, timeout=10)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            conn.request("GET", "/v1/models", headers=headers)
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

    except Exception:
        pass

    return "offline"


def health_check_loop():
    """Continuously check endpoint health every 30 seconds.

    Endpoints are checked in PARALLEL so one slow or unreachable gateway
    (waiting out its network timeout) doesn't delay all the others. This
    keeps the startup sweep and each refresh cycle fast regardless of how
    many offline endpoints are configured.
    """
    while True:
        # One bad cycle must not end the loop: this is a daemon thread and
        # Textual swallows stderr, so statuses would silently freeze.
        try:
            endpoints = get_config("endpoints")

            # Separate enabled vs disabled, marking disabled immediately.
            enabled_eps = []
            for ep in endpoints:
                if not ep.get("enabled", False):
                    set_endpoint_status(ep.get("name", ""), "disabled")
                else:
                    enabled_eps.append(ep)

            if enabled_eps:
                with ThreadPoolExecutor(max_workers=len(enabled_eps)) as pool:
                    statuses = list(pool.map(check_endpoint_health, enabled_eps))
                for ep, status in zip(enabled_eps, statuses):
                    set_endpoint_status(ep.get("name", ""), status)

            # Re-aggregate models after health check
            aggregate_models()
        except Exception as e:
            _queue_log_line(f"[proxy] Health check cycle failed: {e!r}")
        time.sleep(30)


def _modalities_to_input(modalities: dict) -> list:
    """Convert llama-server modalities dict to input_modalities list."""
    mods = ["text"]
    if modalities.get("vision"):
        mods.append("image")
    if modalities.get("audio"):
        mods.append("audio")
    if modalities.get("video"):
        mods.append("video")
    return mods


# Canonical effort vocabulary, ordered weakest→strongest. Used to normalize
# the differing per-provider effort names into one consistent scale.
_EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _extract_reasoning(meta: dict) -> dict:
    """Normalize thinking/reasoning capability across all gateway types.

    Different upstreams advertise reasoning in different shapes:
      • OpenRouter : meta["reasoning"] = {"mandatory", "default_enabled",
                     "supported_efforts", "default_effort"} — the richest form.
      • llama.cpp  : default_generation_settings.params.reasoning_format
                     (anything other than "none" means reasoning is emitted),
                     and chat_template_caps.supports_reasoning.
      • Ollama     : model "capabilities" contains "thinking".
      • LM Studio  : model "capabilities" contains "reasoning".
      • OrcaRouter : advertises nothing → reasoning unsupported.

    Returns a normalized descriptor ({} when reasoning is unsupported):
        {
          "supported": bool,
          "default_enabled": bool,
          "mandatory": bool,
          "supported_efforts": [str, ...],   # canonical scale
          "default_effort": str | None,
          "source": "<how we learned it>",
        }
    """
    caps = [c.lower() for c in (meta.get("capabilities") or [])]

    # ── 1. Rich per-model descriptor (OpenRouter-style) ────────────────────
    r = meta.get("reasoning")
    if isinstance(r, dict):
        efforts = r.get("supported_efforts") or []
        # Normalize any non-canonical effort names (lowercase, reorder).
        norm_efforts = sorted(
            {str(e).lower() for e in efforts if e},
            key=lambda e: _EFFORT_ORDER.index(e) if e in _EFFORT_ORDER else 99,
        )
        if not norm_efforts:
            # No explicit effort list but the model still reasons.
            norm_efforts = ["low", "medium", "high"]
        return {
            "supported": True,
            "default_enabled": bool(r.get("default_enabled", True)),
            "mandatory": bool(r.get("mandatory", False)),
            "supported_efforts": norm_efforts,
            "default_effort": (r.get("default_effort") or None),
            "source": "model_metadata",
        }

    # ── 2. Capability list (Ollama "thinking" / LM Studio "reasoning") ─────
    if "thinking" in caps or "reasoning" in caps:
        return {
            "supported": True,
            "default_enabled": True,
            "mandatory": False,
            "supported_efforts": [],
            "default_effort": None,
            "source": "capabilities",
        }

    # ── 3. llama.cpp reasoning_format + chat template caps ─────────────────
    gen = meta.get("default_generation_settings") or {}
    params = gen.get("params") or {}
    reasoning_format = params.get("reasoning_format")
    if reasoning_format and str(reasoning_format).lower() != "none":
        return {
            "supported": True,
            "default_enabled": True,
            "mandatory": False,
            "supported_efforts": [],
            "default_effort": None,
            "source": "llama_reasoning_format",
        }

    return {}


def fetch_openrouter_models(host: str, api_key: str) -> list:
    """Fetch all models from OpenRouter."""
    try:
        conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=30)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        conn.request("GET", "/api/v1/models", headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return data.get("data", [])
    except Exception as e:
        print(f"[proxy] Error fetching OpenRouter models: {e}")
        return []


def fetch_lmstudio_models(host: str) -> list:
    """Fetch loaded models from LM Studio via /api/v0/models.

    The OpenAI-style /v1/models endpoint returns only id/object/owned_by,
    so we query /api/v0/models instead, which reports the real context
    sizes (max_context_length, loaded_context_length) plus arch,
    quantization, publisher and capabilities.

    Context fields carried through to the gateway:
      - context_length          → max_context_length (model's max window)
      - loaded_context_length   → the context currently loaded in LM Studio
    """
    try:
        conn = http.client.HTTPConnection(_dial_host(host), timeout=10)
        conn.request("GET", "/api/v0/models")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
    except Exception as e:
        print(f"[proxy] Error fetching LM Studio models: {e}")
        return []

    models = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if not model_id:
            continue
        max_ctx = m.get("max_context_length") or m.get("context_length")
        if max_ctx is None:
            max_ctx = 128000
        loaded_ctx = m.get("loaded_context_length") or max_ctx
        caps = m.get("capabilities", []) or []
        input_mods = ["text"]
        if "tool_use" in caps:
            input_mods.append("tool_use")
        models.append({
            "id": model_id,
            "name": m.get("name", model_id),
            "context_length": max_ctx,
            "loaded_context_length": loaded_ctx,
            "owned_by": m.get("publisher", ""),
            "arch": m.get("arch", ""),
            "quantization": m.get("quantization", ""),
            "capabilities": caps,
            "architecture": {
                "input_modalities": input_mods,
                "output_modalities": ["text"],
            },
        })
    return models


def fetch_llama_server_models(host: str) -> list:
    """Fetch model info from llama.cpp server."""
    try:
        conn = http.client.HTTPConnection(_dial_host(host), timeout=10)
        conn.request("GET", "/props")
        resp = conn.getresponse()
        props = json.loads(resp.read())
        conn.close()

        model_path = props.get("model_path", "unknown")
        # Extract filename from full path (handles both / and \ separators)
        model_name = os.path.basename(model_path)
        # If basename fails (e.g., just a name with no path), use as-is
        if not model_name:
            model_name = model_path
        # Strip .gguf extension for cleaner display
        if model_name.lower().endswith(".gguf"):
            model_name = model_name[:-5]
        modalities = props.get("modalities", {})
        gen_settings = props.get("default_generation_settings", {}) or {}
        chat_caps = props.get("chat_template_caps", {}) or {}

        entry = {
            "id": model_name,
            "name": model_name,
            "context_length": gen_settings.get("n_ctx", 128000),
            "architecture": {
                "input_modalities": _modalities_to_input(modalities),
                "output_modalities": ["text"],
            },
            # Keep the raw generation settings so _extract_reasoning() can read
            # reasoning_format from params for this model.
            "default_generation_settings": gen_settings,
        }
        # If llama.cpp reports reasoning support via the chat template, mark it
        # as a "thinking" capability (same key Ollama uses).
        if chat_caps.get("supports_reasoning"):
            entry["capabilities"] = ["thinking"]
        return [entry]
    except Exception as e:
        print(f"[proxy] Error fetching llama-server models: {e}")
        return []


def fetch_ollama_models(host: str) -> list:
    """Fetch installed models from Ollama via /api/tags."""
    try:
        conn = http.client.HTTPConnection(_dial_host(host), timeout=10)
        conn.request("GET", "/api/tags")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()

        models = []
        for m in data.get("models", []):
            model_name = m.get("name", "")
            if not model_name:
                continue
            # Strip :latest suffix for cleaner display
            if model_name.endswith(":latest"):
                model_name = model_name[:-7]
            # Ollama doesn't provide context_length in /api/tags,
            # use a reasonable default. Newer Ollama versions advertise
            # capabilities (e.g. "thinking", "tools", "vision") per model.
            caps = m.get("capabilities", []) or []
            models.append({
                "id": model_name,
                "name": model_name,
                "context_length": 128000,
                "capabilities": caps,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
            })
        return models
    except Exception as e:
        print(f"[proxy] Error fetching Ollama models: {e}")
        return []


def fetch_orcarouter_models(host: str, api_key: str) -> list:
    """Fetch all models from OrcaRouter."""
    try:
        conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=30)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        conn.request("GET", "/v1/models", headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return data.get("data", [])
    except Exception as e:
        print(f"[proxy] Error fetching OrcaRouter models: {e}")
        return []


def fetch_custom_models(host: str, api_key: str) -> list:
    """Fetch models from a custom OpenAI-compatible endpoint."""
    try:
        conn = _connect_host(host, timeout=15)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        conn.request("GET", "/v1/models", headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        models = data.get("data") if isinstance(data, dict) else None
        return [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []
    except Exception as e:
        print(f"[proxy] Error fetching custom models from {host}: {e}")
        return []


def fetch_strata_models(host: str, api_key: str) -> list:
    """Fetch the models a Strata gateway is serving right now.

    Strata is itself a gateway: which models it serves, and whether each one
    reads images, depends on what is loaded in it, so nothing about them is
    fixed in config. Its /v1/models lists the ready models but only with
    id/label; the app's /api/native reports each loaded model's vision switch
    and context size, matched here by apiModelId or alias. When /api/native
    can't be read (e.g. a Strata on another PC), the models are still listed,
    as text-only.
    """
    models = fetch_custom_models(host, api_key)
    if not models:
        return []

    instances = []
    try:
        conn = _connect_host(host, timeout=10)
        conn.request("GET", "/api/native")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        if resp.status == 200:
            instances = json.loads(body).get("instances", []) or []
    except Exception as e:
        print(f"[proxy] Could not read Strata model details from {host}: {e}")

    by_id = {}
    for inst in instances:
        if inst.get("status") != "ready":
            continue
        for key in [inst.get("apiModelId")] + list(inst.get("aliases") or []):
            if key:
                by_id[key] = inst

    entries = []
    for m in models:
        model_id = m.get("id", "")
        if not model_id:
            continue
        meta = m.get("meta") or {}
        inst = by_id.get(model_id, {})
        input_mods = ["text"]
        if inst.get("vision"):
            input_mods.append("image")
        entries.append({
            "id": model_id,
            "name": inst.get("label") or meta.get("label") or model_id,
            "owned_by": "strata",
            "context_length": inst.get("context") or meta.get("context") or 128000,
            "architecture": {
                "input_modalities": input_mods,
                "output_modalities": ["text"],
            },
        })
    return entries


def _get_cached_catalog(endpoint_name: str, fetch_fn, *args) -> list:
    """Return a catalog for an endpoint, caching large upstream responses.

    OpenRouter/OrcaRouter return hundreds of models; re-downloading that on
    every 30-second health-check cycle is wasteful. This caches the raw
    catalog per endpoint and only refreshes after CATALOG_CACHE_TTL.

    IMPORTANT: We store and return DEEP COPIES so callers can mutate the
    returned dicts (e.g. adding _source / _source_host / overwriting id)
    without corrupting the cache for subsequent aggregations. Before v2.3.0
    this was a shallow copy and the mutate-on-read corrupted the cache,
    causing model capabilities metadata to be lost on the next cycle.
    """
    now = time.time()
    with _catalog_cache_lock:
        cached = _catalog_cache.get(endpoint_name)
        cached_at = _catalog_cache_time.get(endpoint_name, 0)
        if cached is not None and (now - cached_at) < CATALOG_CACHE_TTL:
            return copy.deepcopy(cached)
    models = fetch_fn(*args) or []
    if not models:
        # Fetchers return [] on any error. Caching that would strip every
        # cloud model's capabilities until the TTL ran out, so keep serving
        # the last good catalog and retry on the next cycle.
        return copy.deepcopy(cached) if cached else []
    with _catalog_cache_lock:
        _catalog_cache[endpoint_name] = copy.deepcopy(models)
        _catalog_cache_time[endpoint_name] = time.time()
    return copy.deepcopy(models)


def clear_catalog_cache():
    """Drop cached upstream catalogs (e.g., after endpoint changes)."""
    with _catalog_cache_lock:
        _catalog_cache.clear()
        _catalog_cache_time.clear()


def _collect_endpoint_models(ep: dict, user_models: list, new_routes: dict, new_meta: dict):
    """Add one online endpoint's models to the routes and metadata being built."""
    name = ep.get("name", "")
    ep_type = ep.get("type", "")
    host = ep.get("host", "")
    api_key = ep.get("api_key", "")

    if ep_type in ("openrouter", "orcarouter"):
        # OpenRouter / OrcaRouter: Fetch metadata for user's curated models AND
        # top intelligent models.
        or_original_ids = set()
        for um in user_models:
            if um.startswith(f"{name}:"):
                or_original_ids.add(um[len(name) + 1:])
        # Also include top intelligent models for this endpoint
        with _top_models_lock:
            for tm in _top_intelligent_models:
                if tm.startswith(f"{name}:"):
                    or_original_ids.add(tm[len(name) + 1:])
        if not or_original_ids:
            return
        # Fetch catalog (cached) to get metadata
        if ep_type == "orcarouter":
            all_or = _get_cached_catalog(name, fetch_orcarouter_models, host, api_key)
        else:
            all_or = _get_cached_catalog(name, fetch_openrouter_models, host, api_key)
        or_meta = {m.get("id", ""): m for m in all_or}
        for original_id in or_original_ids:
            meta = or_meta.get(original_id, {"id": original_id, "name": original_id})
            aggregated_id = f"{name}:{original_id}"
            new_routes[aggregated_id] = (name, original_id, ep)
            meta["_source"] = name
            meta["_source_host"] = host
            meta["id"] = aggregated_id
            new_meta[aggregated_id] = meta

    elif ep_type in ("llama-server", "lmstudio", "ollama", "strata", "custom"):
        # Local endpoints: include all discovered models
        if ep_type == "llama-server":
            raw_models = fetch_llama_server_models(host)
        elif ep_type == "lmstudio":
            raw_models = fetch_lmstudio_models(host)
        elif ep_type == "ollama":
            raw_models = fetch_ollama_models(host)
        elif ep_type == "strata":
            raw_models = fetch_strata_models(host, api_key)
        else:
            raw_models = fetch_custom_models(host, api_key)

        for model in raw_models:
            original_id = model.get("id", "")
            if not original_id:
                continue
            aggregated_id = f"{name}:{original_id}"
            new_routes[aggregated_id] = (name, original_id, ep)
            model["_source"] = name
            model["_source_host"] = host
            model["id"] = aggregated_id
            new_meta[aggregated_id] = model


def aggregate_models():
    """Fetch models from all online endpoints and build unified list.

    For cloud endpoints (OpenRouter): Only include models from the user's
    curated models list. Do NOT fetch the full catalog (would be 340+ models).

    For local endpoints (llama-server, LM Studio, custom): Include all
    discovered models since they typically have 1-5 loaded models.
    """
    global _model_routes

    new_routes = {}
    new_meta = {}
    user_models = list(get_config("models"))

    endpoints = get_config("endpoints")
    for ep in endpoints:
        if get_endpoint_status(ep.get("name", "")) != "online":
            continue
        # One endpoint returning something unexpected must not drop the
        # models of all the others.
        try:
            _collect_endpoint_models(ep, user_models, new_routes, new_meta)
        except Exception as e:
            _queue_log_line(f"[proxy] Skipped models from {ep.get('name', '')}: {e!r}")

    # Build MODELS list: user models first, then top intelligent models,
    # then discovered local models
    merged = []
    for m in user_models:
        if m in new_routes and m not in merged:
            merged.append(m)
    # Include top intelligent models (fetched from OpenRouter)
    with _top_models_lock:
        for m in _top_intelligent_models:
            if m in new_routes and m not in merged:
                merged.append(m)
    for m in new_routes:
        if m not in merged:
            merged.append(m)

    _model_routes = new_routes
    with _model_meta_lock:
        _model_meta.clear()
        _model_meta.update(new_meta)

    set_models(merged)
    total = len(merged)
    online_count = sum(1 for ep in endpoints if get_endpoint_status(ep.get("name", "")) == "online")
    print(f"[proxy] Aggregated {total} models from {online_count} online endpoint(s)")


def get_model_meta(model_id: str) -> dict:
    """Get cached metadata for a model, or empty dict if not found."""
    with _model_meta_lock:
        return _model_meta.get(model_id, {})


# ── Auto-discover local endpoints ─────────────────────────────────────────
def auto_discover_local_endpoints():
    """Probe common local AI server ports and add them as endpoints if found."""
    local_probes = [
        {"name": "llama.cpp", "type": "llama-server", "host": "localhost:8080", "probe": "/health"},
        {"name": "lmstudio",  "type": "lmstudio",    "host": "localhost:1234", "probe": "/v1/models"},
        {"name": "ollama",    "type": "ollama",       "host": "localhost:11434", "probe": "/api/tags"},
        {"name": "strata",    "type": "strata",       "host": "127.0.0.1:47831", "probe": "/v1/models"},
    ]
    existing_names = set()
    with _config_lock:
        existing_eps = _config.get("endpoints", [])
        existing_names = {ep.get("name", "") for ep in existing_eps}
        existing_hosts = {ep.get("host", "") for ep in existing_eps}

    for probe in local_probes:
        name = probe["name"]
        host = probe["host"]
        # Skip if already configured (by name or host)
        if name in existing_names or host in existing_hosts:
            continue
        try:
            conn = http.client.HTTPConnection(_dial_host(host), timeout=3)
            conn.request("GET", probe["probe"])
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                new_ep = {
                    "name": name,
                    "type": probe["type"],
                    "host": host,
                    "enabled": True,
                    "api_key": "",
                }
                with _config_lock:
                    _config.setdefault("endpoints", []).append(new_ep)
                print(f"[proxy] Auto-discovered {name} at {host}")
        except Exception:
            pass  # Not running, skip silently


# ── Auto-install textual if missing ────────────────────────────────────────
try:
    import textual  # noqa: F401
except ImportError:
    print("[proxy] textual not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textual"])
    import textual  # noqa: F401

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Header, Footer, Input, Select, Static, Button, Label, Log, Checkbox,
    DataTable,
)


# ═══════════════════════════════════════════════════════════════════════════
#  PROXY HANDLER
# ═══════════════════════════════════════════════════════════════════════════

class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies to upstream AI endpoints."""

    tui_log = None  # Reference to TUI log widget

    # ── CORS preflight ─────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    # ── GET: serve UI or proxy ─────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # Serve the web UI
        if path == "/" or path == "/index.html":
            self._serve_file("index.html")
            return

        # API endpoints handled locally
        if path == "/v1/models" or path == "/models":
            self._handle_models()
            return
        if path == "/api/v0/models" or path == "/api/v1/models":
            self._handle_lmstudio_models()
            return
        if path == "/props":
            query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()} if parsed.query else {}
            self._handle_props(query_params.get("model"))
            return
        if path == "/health" or path == "/v1/health":
            self._handle_health()
            return
        if path == "/slots":
            self._handle_slots()
            return
        if path == "/tools":
            self._handle_tools()
            return
        if path == "/cors-proxy":
            self._handle_cors_proxy()
            return

        # Try to serve as static web UI file
        if self._serve_file(path.lstrip("/")):
            return
        if path.startswith("/_app/"):
            # A web UI file that isn't there (e.g. a page from before an
            # update asking for an old bundle): 404, not an upstream error.
            self.send_error(404, "Not found")
            return

        # Proxy everything else
        self._proxy()

    # ── POST: local API stubs or proxy ─────────────────────────────────────
    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/v1/streams/lookup":
            self._handle_streams_lookup()
            return
        if path in ("/models/load", "/models/unload"):
            self._handle_model_load()
            return
        if path == "/cors-proxy":
            self._handle_cors_proxy()
            return

        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    # ── Static file serving ────────────────────────────────────────────────
    def _serve_file(self, rel_path: str) -> bool:
        root = os.path.normpath(WEBUI_DIR)
        filepath = os.path.normpath(os.path.join(root, rel_path))
        # commonpath, not startswith: the sibling "webui-src" starts with
        # "webui" too, so /..%2fwebui-src/... passed a prefix check.
        try:
            if os.path.commonpath([root, filepath]) != root:
                return False
        except ValueError:  # different drive on Windows
            return False
        if not os.path.isfile(filepath):
            return False

        ext = os.path.splitext(filepath)[1].lower()
        content_type = MIME_TYPES.get(ext)
        if content_type is None:
            content_type, _ = mimetypes.guess_type(filepath)
            if content_type is None:
                content_type = "application/octet-stream"

        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if "/immutable/" in rel_path:
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            elif rel_path in ("sw.js", "manifest.webmanifest", "_app/version.json", "build.json") or ext == ".html":
                # HTML names the hashed bundles of its build, so it must be
                # revalidated: an index.html cached from before a web UI
                # update points at bundles that no longer exist.
                self.send_header("Cache-Control", "no-cache")
            else:
                self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            return False
        return True

    # ── API: /v1/models ────────────────────────────────────────────────────
    def _handle_models(self):
        """Return aggregated model list from all online endpoints."""
        entries = []
        for m in get_models():
            meta = get_model_meta(m)
            context_length = meta.get("context_length", 128000)

            input_mods = meta.get("architecture", {}).get("input_modalities", ["text"])
            output_mods = meta.get("architecture", {}).get("output_modalities", ["text"])
            has_vision = any(mod in input_mods for mod in ("image", "vision"))
            has_audio = "audio" in input_mods
            has_video = "video" in input_mods
            # Tool calling is reported by OpenRouter/OrcaRouter in
            # supported_parameters (e.g. "tools"), not in input_modalities.
            supported_params = meta.get("supported_parameters", [])
            has_tools = (
                any(mod in input_mods for mod in ("tools", "tool_use"))
                or "tools" in supported_params
                or "tool_choice" in supported_params
            )

            tags = []
            if has_vision:
                tags.append("vision")
            if has_audio:
                tags.append("audio")
            if has_video:
                tags.append("video")

            # Normalized reasoning/thinking descriptor ({} when unsupported).
            reasoning = _extract_reasoning(meta)
            if reasoning.get("supported"):
                tags.append("reasoning")

            # Parse source from model ID
            source, original_id = parse_model_id(m)

            entry = {
                "id": m,
                "name": meta.get("name", original_id),
                "object": "model",
                "created": meta.get("created", int(time.time())),
                "owned_by": meta.get("owned_by", original_id.split("/")[0] if "/" in original_id else source),
                "source": source,
                "source_host": meta.get("_source_host", ""),
                "in_cache": True,
                "path": m,
                "context_length": context_length,
                "context_window": context_length,
                "max_model_len": context_length,
                "status": {"value": "loaded"},
                "tags": tags,
                # llama.cpp web UI reads this field directly to detect
                # vision/audio/video support (see getModelModalities in the
                # bundled llama-ui). Without it, the UI reports "requires a
                # vision-capable model" even for multimodal models.
                "modalities": {
                    "vision": has_vision,
                    "audio": has_audio,
                    "video": has_video,
                },
                "architecture": {
                    "input_modalities": input_mods,
                    "output_modalities": output_mods,
                },
                # Normalized reasoning/thinking capability. Exposed top-level
                # (mirrors OpenRouter's own `reasoning` field) so clients can
                # discover supported thinking levels per model.
                "reasoning": reasoning,
                "meta": {
                    "capabilities": {
                        "vision": has_vision,
                        "audio": has_audio,
                        "video": has_video,
                        "function_calling": has_tools,
                        "reasoning": bool(reasoning.get("supported")),
                    },
                },
            }
            entries.append(entry)
        self._send_json({"object": "list", "data": entries})

    # ── API: /api/v0/models (LM Studio compatible) ────────────────────────
    def _handle_lmstudio_models(self):
        """Return aggregated model list in LM Studio /api/v0/models format.

        Cline's LM Studio provider calls GET {baseUrl}/api/v0/models and
        expects each model object to include LM Studio-specific fields
        (type, publisher, arch, quantization, state, max_context_length,
        loaded_context_length).  We map the gateway's existing metadata
        into that shape so Cline's dropdown populates automatically.
        """
        entries = []
        for m in get_models():
            meta = get_model_meta(m)
            context_length = meta.get("context_length", 128000)
            loaded_context = meta.get("loaded_context_length", context_length)

            input_mods = meta.get("architecture", {}).get("input_modalities", ["text"])
            # Tool calling is reported by OpenRouter/OrcaRouter in
            # supported_parameters (e.g. "tools"), not in input_modalities.
            supported_params = meta.get("supported_parameters", [])
            has_tools = (
                any(mod in input_mods for mod in ("tools", "tool_use"))
                or "tools" in supported_params
                or "tool_choice" in supported_params
            )

            source, original_id = parse_model_id(m)

            capabilities = []
            if has_tools:
                capabilities.append("tool_use")
            # LM Studio advertises reasoning support as a "reasoning" capability.
            reasoning = _extract_reasoning(meta)
            if reasoning.get("supported"):
                capabilities.append("reasoning")

            entry = {
                "id": m,
                "object": "model",
                "type": "llm",
                "publisher": meta.get("owned_by", source),
                "arch": meta.get("arch") or meta.get("architecture", {}).get("modality", "unknown"),
                "compatibility_type": "gguf" if meta.get("quantization") else "openai",
                "quantization": meta.get("quantization", ""),
                "state": "loaded",
                "max_context_length": context_length,
                "loaded_context_length": loaded_context,
            }
            if capabilities:
                entry["capabilities"] = capabilities
            entries.append(entry)
        self._send_json({"object": "list", "data": entries})

    # ── API: /props ────────────────────────────────────────────────────────
    def _handle_props(self, model_id=None):
        """Return server properties using aggregated model metadata."""
        model = unquote(model_id) if model_id else get_current_model()
        meta = get_model_meta(model)

        n_ctx = meta.get("context_length", 128000)
        max_completion = meta.get("top_provider", {}).get("max_completion_tokens", -1)

        input_mods = meta.get("architecture", {}).get("input_modalities", ["text"])
        has_vision = any(m in input_mods for m in ("image", "vision"))
        has_audio = "audio" in input_mods
        has_video = "video" in input_mods

        # Drive llama.cpp's reasoning params from the model's real capability
        # instead of the old hardcoded "none" / False.
        reasoning = _extract_reasoning(meta)
        reasoning_format = "auto" if reasoning.get("supported") else "none"

        data = {
            "default_generation_settings": {
                "id": 0, "id_task": -1, "n_ctx": n_ctx,
                "speculative": False, "is_processing": False,
                "params": {
                    "n_predict": max_completion, "seed": 4294967295,
                    "temperature": 0.8, "dynatemp_range": 0.0,
                    "dynatemp_exponent": 1.0, "top_k": 40, "top_p": 0.95,
                    "min_p": 0.05, "top_n_sigma": 0.0,
                    "xtc_probability": 0.0, "xtc_threshold": 0.1,
                    "typ_p": 1.0, "repeat_last_n": 64,
                    "repeat_penalty": 1.0, "presence_penalty": 0.0,
                    "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                    "dry_base": 1.75, "dry_allowed_length": 2,
                    "dry_penalty_last_n": -1,
                    "dry_sequence_breakers": ["\n", ":", "\"", "*"],
                    "mirostat": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
                    "stop": [], "max_tokens": max_completion, "n_keep": 0,
                    "n_discard": 0, "ignore_eos": False, "stream": True,
                    "logit_bias": [], "n_probs": 0, "min_keep": 0,
                    "grammar": "", "grammar_lazy": False,
                    "grammar_triggers": [], "preserved_tokens": [],
                    "chat_format": "chatml", "reasoning_format": reasoning_format,
                    "reasoning_in_content": False, "generation_prompt": "",
                    "samplers": ["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
                    "backend_sampling": False,
                    "speculative.n_max": 0, "speculative.n_min": 0,
                    "speculative.p_min": 0.0, "timings_per_token": False,
                    "post_sampling_probs": False, "lora": [],
                },
                "prompt": "",
                "next_token": {
                    "has_next_token": False, "has_new_line": False,
                    "n_remain": -1, "n_decoded": 0, "stopping_word": "",
                },
            },
            "total_slots": 1,
            "model_path": model,
            "role": "router",
            "modalities": {
                "vision": has_vision, "audio": has_audio, "video": has_video,
            },
            # Normalized reasoning descriptor (llama.cpp UI + other clients).
            "reasoning": reasoning,
            "chat_template": "", "bos_token": "", "eos_token": "",
            "build_info": "ai-gateway v2.0",
            # Advertise the /cors-proxy endpoint so the bundled llama-ui
            # enables the "Use llama-server proxy" option for MCP servers
            # (same mechanism as llama-server's --ui-mcp-proxy flag).
            "cors_proxy_enabled": True,
        }
        self._send_json(data)

    # ── API: /health ───────────────────────────────────────────────────────
    def _handle_health(self):
        # Return endpoint statuses
        endpoints = get_config("endpoints")
        ep_statuses = {}
        for ep in endpoints:
            name = ep.get("name", "")
            ep_statuses[name] = get_endpoint_status(name)
        self._send_json({"status": "ok", "endpoints": ep_statuses})

    def _handle_slots(self):
        self._send_json([])

    def _handle_tools(self):
        self._send_json([])

    def _handle_streams_lookup(self):
        self._send_json({})

    def _handle_model_load(self):
        """Handle /models/load and /models/unload from the web UI.

        /models/load sets the requested model as the new default (default_model)
        so subsequent requests without an explicit model use it. /models/unload
        is a no-op for cloud-aggregated models (they cannot actually be
        unloaded), but we still validate the model is known.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > _MAX_BODY_SIZE:
            self.send_error(413, "Request body too large")
            return
        body = self.rfile.read(content_length) if content_length > 0 else b""

        model = ""
        if body:
            try:
                data = json.loads(body)
                model = data.get("model", "")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        if not model:
            self._send_json({"success": False, "error": "No model specified"}, 400)
            return

        models = get_models()
        if model not in models:
            self._send_json({"success": False, "error": f"Model not found: {model}"}, 404)
            return

        is_unload = self.path == "/models/unload"
        if not is_unload:
            set_config("default_model", model)
            save_config()
            self._log(f"Web UI selected model: {model}")

        self._send_json({"success": True})

    # ── API: /cors-proxy (llama.cpp llama-ui compatible) ──────────────────
    def _handle_cors_proxy(self):
        """Forward cross-origin requests from the bundled web UI.

        The llama.cpp UI avoids browser CORS restrictions by routing external
        fetches through this endpoint: GET /cors-proxy?url=<target>. Custom
        request headers are passed with an `x-llama-server-proxy-header-`
        prefix (verified in webui/sw.js / bundle.js) and are stripped before
        forwarding upstream.
        """
        if not _cors_proxy_caller_allowed(self.headers):
            self._log(f"cors-proxy refused: Origin={self.headers.get('Origin')} "
                      f"Sec-Fetch-Site={self.headers.get('Sec-Fetch-Site')} Host={self.headers.get('Host')}")
            self._send_json({"error": "cors-proxy only serves the gateway's own web page"}, 403)
            return

        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        raw_url = query.get("url", [""])[0]
        if not raw_url:
            self._send_json({"error": "Missing 'url' query parameter"}, 400)
            return

        target = urlparse(raw_url)
        if target.scheme not in ("http", "https"):
            self._send_json({"error": "Only http and https URLs are allowed"}, 400)
            return
        if not target.netloc:
            self._send_json({"error": "Invalid URL"}, 400)
            return

        # Forward request headers upstream. The MCP streamable-HTTP transport
        # carries its session token and Content-Type in prefixed headers
        # (`x-llama-server-proxy-header-*`) when proxying, but the browser may
        # ALSO auto-add plain headers (e.g. `content-type: text/plain` for a
        # JSON string body). We forward normal headers too, but the prefixed
        # (user-intended) headers MUST take precedence so the correct
        # `content-type: application/json` wins over the auto-added text/plain.
        # Hop-by-hop and proxy-internal headers are excluded so we don't
        # corrupt the upstream request or trigger gzip/keep-alive side effects.
        prefix = "x-llama-server-proxy-header-"
        hop_by_hop = {
            "host", "content-length", "connection", "keep-alive",
            "proxy-authenticate", "proxy-authorization", "te",
            "trailer", "transfer-encoding", "upgrade", "accept-encoding",
        }

        upstream_headers = {}
        lower_seen = set()
        # First pass: prefixed (user-supplied) headers. These win.
        for header, value in self.headers.items():
            h = header.lower()
            if h in hop_by_hop or not value:
                continue
            if h.startswith(prefix):
                name = header[len(prefix):]
                if name:
                    upstream_headers[name] = value
                    lower_seen.add(name.lower())
        # Second pass: plain headers (session token, auth, etc.), but never
        # override a header the prefixed headers already provided.
        for header, value in self.headers.items():
            h = header.lower()
            if h in hop_by_hop or not value:
                continue
            if h.startswith(prefix):
                continue
            if h in lower_seen:
                continue
            upstream_headers[header] = value
            lower_seen.add(h)

        # Read body for non-idempotent methods (e.g., MCP tool POSTs).
        request_body = b""
        if self.command not in ("GET", "HEAD"):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > _MAX_BODY_SIZE:
                self.send_error(413, "Request body too large")
                return
            request_body = self.rfile.read(content_length) if content_length > 0 else b""
            if request_body:
                upstream_headers["Content-Length"] = str(len(request_body))

        upstream_path = target.path or "/"
        if target.query:
            upstream_path += "?" + target.query

        use_ssl = target.scheme == "https"
        if use_ssl:
            conn = http.client.HTTPSConnection(target.netloc, context=_upstream_ssl_ctx, timeout=60)
        else:
            conn = http.client.HTTPConnection(_dial_host(target.netloc), timeout=60)

        try:
            conn.request(self.command, upstream_path, body=request_body, headers=upstream_headers)
            response = conn.getresponse()

            self.send_response(response.status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Access-Control-Expose-Headers", "*")
            content_type = response.getheader("Content-Type")
            if content_type:
                self.send_header("Content-Type", content_type)
            # Forward upstream response headers the MCP streamable-HTTP
            # session relies on (e.g. Mcp-Session-Id) back to the browser so
            # it can echo them on subsequent requests. Without this, stateful
            # MCP bridges such as supergateway fail with "No valid session ID
            # provided" on the follow-up POST.
            hop_by_hop = {
                "transfer-encoding", "connection", "keep-alive",
                "proxy-authenticate", "proxy-authorization", "te",
                "trailer", "upgrade",
            }
            for name, value in response.getheaders():
                key = name.lower()
                if key in hop_by_hop or key == "content-type":
                    continue
                self.send_header(name, value)
            data = response.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._log(f"cors-proxy error: {e}")
            try:
                self.send_error(502, f"cors-proxy error: {e}")
            except Exception:
                pass
        finally:
            conn.close()

    # ── JSON response helper ───────────────────────────────────────────────
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── Core proxy logic with smart routing ────────────────────────────────
    def _proxy(self):
        # 1. Read body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > _MAX_BODY_SIZE:
            self.send_error(413, "Request body too large")
            return
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # 2. Parse model and determine route
        target_endpoint = None
        original_id = None
        requested_source = None  # source prefix explicitly present in model ID

        if body:
            try:
                data = json.loads(body)
                model_id = data.get("model", "")

                if not model_id:
                    # No model specified — use default
                    model_id = get_current_model()
                    data["model"] = model_id
                    body = json.dumps(data).encode("utf-8")
                    self._log(f"Injected default model: {model_id}")

                # Parse source prefix
                source, original_id = parse_model_id(model_id)

                if source:
                    requested_source = source
                    target_endpoint = get_endpoint_by_name(source)

                if not target_endpoint:
                    # Fallback: try routing table
                    route = _model_routes.get(model_id)
                    if route:
                        source, original_id, target_endpoint = route

                if target_endpoint and original_id and original_id != model_id:
                    # Strip source prefix for upstream
                    data["model"] = original_id
                    body = json.dumps(data).encode("utf-8")

            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # 3. Determine upstream connection
        if not target_endpoint:
            # If an explicit source was requested but can't be routed, return a
            # clear error instead of silently forwarding to another endpoint.
            if requested_source:
                ep = find_endpoint_by_name(requested_source)
                if ep and not ep.get("enabled", False):
                    self._send_json({"error": {"message": f"Endpoint '{requested_source}' is disabled"}}, 404)
                    return
                status = get_endpoint_status(requested_source)
                self._send_json({"error": {"message": f"Endpoint '{requested_source}' is {status}"}}, 503)
                return
            target_endpoint = get_first_enabled_endpoint()

        if not target_endpoint:
            self._send_json({"error": {"message": "All endpoints are offline or none configured"}}, 503)
            return

        host = target_endpoint.get("host", "")
        api_key = target_endpoint.get("api_key", "")
        ep_name = target_endpoint.get("name", "unknown")
        ep_type = target_endpoint.get("type", "")

        # 4. Rewrite path (OpenRouter needs /api prefix, OrcaRouter and local endpoints use /v1)
        if ep_type == "openrouter":
            path = rewrite_path(self.path)
        else:
            # llama-server, LM Studio, Strata, custom — use path as-is
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if not path.startswith("/v1/") and not path.startswith("/api/"):
                path = f"/v1{path}"

        # 5. Build upstream headers
        fwd_headers = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
        }
        if api_key:
            fwd_headers["Authorization"] = f"Bearer {api_key}"
        else:
            # Still forward any client-provided auth
            auth = self.headers.get("Authorization")
            if auth:
                fwd_headers["Authorization"] = auth

        for hdr in ("HTTP-Referer", "X-Title"):
            val = self.headers.get(hdr)
            if val:
                fwd_headers[hdr] = val
        if body:
            fwd_headers["Content-Length"] = str(len(body))

        # 6. Forward to upstream. The timeout is per socket read; local models
        # can spend minutes on a long prompt before the first token, so they
        # get longer (an endpoint's "timeout" in config overrides both).
        use_ssl = not _is_local_host(host)
        read_timeout = target_endpoint.get("timeout") or (180 if use_ssl else 600)
        if use_ssl:
            conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=read_timeout)
        else:
            conn = http.client.HTTPConnection(_dial_host(host), timeout=read_timeout)

        bump_endpoint_load_start(ep_name)
        headers_sent = False
        streaming = False
        try:
            conn.request(self.command, path, body=body, headers=fwd_headers)
            response = conn.getresponse()
            streaming = "text/event-stream" in (response.getheader("Content-Type") or "")

            # Send status + CORS + upstream response headers
            self.send_response(response.status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            for header, value in response.getheaders():
                if header.lower() in ("transfer-encoding", "connection"):
                    continue
                self.send_header(header, value)
            self.end_headers()
            headers_sent = True

            # Stream response body (SSE-safe): read1 hands over whatever has
            # arrived instead of waiting for a full buffer, so tokens reach
            # the client as the upstream sends them.
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

            self._log(f"→ {ep_name} [{response.status}]")

        except Exception as e:
            self._log(f"Error → {ep_name}: {e}")
            try:
                if not headers_sent:
                    self.send_error(502, f"Bad Gateway ({ep_name}): {e}")
                elif streaming:
                    # The status line is already out: end the stream with an
                    # SSE error event, not a second HTTP response in the body.
                    err = json.dumps({"error": {"message": f"Bad Gateway ({ep_name}): {e}"}})
                    self.wfile.write(f"data: {err}\n\n".encode())
                    self.wfile.flush()
            except Exception:
                pass
            self.close_connection = True
        finally:
            conn.close()
            bump_endpoint_load_end(ep_name)

    def _log(self, msg: str):
        """Queue a request log line for the TUI (thread-safe)."""
        _queue_log_line(f"[{self.address_string()}] {msg}")

    def log_message(self, fmt, *args):
        pass  # Suppress default stderr logging


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — CONFIRM DIALOG (ModalScreen)
# ═══════════════════════════════════════════════════════════════════════════

class ConfirmScreen(ModalScreen):
    """Simple yes/no confirmation dialog."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(self.message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", id="confirm-yes-btn", variant="error", compact=True)
                yield Button("No", id="confirm-no-btn", variant="default", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-yes-btn":
            self.dismiss(True)
        elif event.button.id == "confirm-no-btn":
            self.dismiss(False)

    def action_cancel(self):
        self.dismiss(False)


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — ENDPOINT EDIT DIALOG (ModalScreen)
# ═══════════════════════════════════════════════════════════════════════════

class EndpointEditScreen(ModalScreen):
    """Modal dialog for adding or editing an endpoint."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    ENDPOINT_TYPES = [
        ("OpenRouter", "openrouter"),
        ("OrcaRouter", "orcarouter"),
        ("LM Studio", "lmstudio"),
        ("llama.cpp Server", "llama-server"),
        ("Ollama", "ollama"),
        ("Strata", "strata"),
        ("Custom", "custom"),
    ]

    def __init__(self, endpoint: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint  # None = add new, dict = edit existing
        self.original_name = (endpoint or {}).get("_original_name") or (endpoint or {}).get("name") or None

    def compose(self) -> ComposeResult:
        with Vertical(id="ep-edit-dialog"):
            yield Static("Edit Endpoint" if self.endpoint else "Add Endpoint", classes="screen-title")

            yield Label("Type")
            type_value = self.endpoint.get("type", "openrouter") if self.endpoint else "openrouter"
            yield Select(
                self.ENDPOINT_TYPES,
                value=type_value,
                id="ep-type-select",
            )

            yield Label("Name (short prefix for model IDs)")
            name_value = self.endpoint.get("name", "") if self.endpoint else ""
            yield Input(value=name_value, placeholder="e.g. openrouter, llama.cpp", id="ep-name-input")

            yield Label("Host (address[:port])")
            host_value = self.endpoint.get("host", "") if self.endpoint else ""
            yield Input(value=host_value, placeholder="e.g. openrouter.ai or localhost:8080", id="ep-host-input")

            yield Label("API Key (optional for local endpoints)")
            key_value = self.endpoint.get("api_key", "") if self.endpoint else ""
            yield Input(value=key_value, placeholder="API Key", password=True, id="ep-key-input")

            yield Checkbox(
                "Enabled",
                value=self.endpoint.get("enabled", True) if self.endpoint else True,
                id="ep-enabled-checkbox",
            )

            with Horizontal(id="ep-edit-buttons"):
                yield Button("💾 Save", id="ep-save-btn", variant="success", compact=True)
                yield Button("✕ Cancel", id="ep-cancel-btn", variant="default", compact=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Auto-fill name/host when a type is chosen in add mode, only if empty."""
        if event.select.id == "ep-type-select":
            ep_type = str(event.select.value)
            defaults = ENDPOINT_TYPE_DEFAULTS.get(ep_type, {})
            name_input = self.query_one("#ep-name-input", Input)
            host_input = self.query_one("#ep-host-input", Input)
            if not self.endpoint:
                if not name_input.value.strip():
                    name_input.value = defaults.get("name", "")
                if not host_input.value.strip():
                    host_input.value = defaults.get("host", "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ep-save-btn":
            self._save()
        elif event.button.id == "ep-cancel-btn":
            self.dismiss(None)

    def action_cancel(self):
        self.dismiss(None)

    def _save(self):
        ep_type = str(self.query_one("#ep-type-select", Select).value)
        name = self.query_one("#ep-name-input", Input).value.strip()
        host = self.query_one("#ep-host-input", Input).value.strip()
        api_key = self.query_one("#ep-key-input", Input).value.strip()
        enabled = self.query_one("#ep-enabled-checkbox", Checkbox).value

        if not name:
            self.query_one("#ep-name-input", Input).focus()
            return
        if not host:
            self.query_one("#ep-host-input", Input).focus()
            return

        result = {
            "name": name,
            "type": ep_type,
            "host": host,
            "api_key": api_key,
            "enabled": enabled,
        }
        if self.original_name:
            result["_original_name"] = self.original_name
        self.dismiss(result)


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — SETTINGS SCREEN
# ═══════════════════════════════════════════════════════════════════════════

class SettingsScreen(Screen):
    """Settings screen with endpoint management."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("ctrl+s", "save_settings", "Save"),
        ("ctrl+r", "restart_server", "Restart"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(id="settings-screen"):
            yield Static("Settings", classes="screen-title")

            # ── Endpoint Management ────────────────────────────────────
            yield Static("Endpoints", classes="section-title")
            yield DataTable(id="endpoint-table")
            with Horizontal(id="endpoint-buttons"):
                yield Button("+ Add", id="ep-add-btn", variant="primary", compact=True)
                yield Button("✎ Edit", id="ep-edit-btn", variant="default", compact=True)
                yield Button("✕ Remove", id="ep-remove-btn", variant="error", compact=True)

            # ── Model Settings ─────────────────────────────────────────
            yield Static("Model Settings", classes="section-title")
            yield Label("Default Model")
            current_models = get_models()
            model_options = [(m, m) for m in current_models] if current_models else [("No models available", "")]
            yield Select(
                model_options,
                value=get_current_model() if current_models else "",
                id="model-select",
            )
            yield Checkbox(
                "Fetch top intelligent models from OpenRouter (merges with your models)",
                value=get_config("fetch_top_models"),
                id="fetch-top-models-checkbox",
            )

            # ── Server Settings ────────────────────────────────────────
            yield Static("Server Settings", classes="section-title")
            with Horizontal():
                with Vertical():
                    yield Label("Port")
                    yield Input(
                        value=str(get_config("port")),
                        placeholder="Port (e.g. 8090)",
                        id="port-input",
                    )
                with Vertical():
                    yield Label("Bind Address")
                    yield Input(
                        value=get_config("addr"),
                        placeholder="Bind address (e.g. 0.0.0.0)",
                        id="addr-input",
                    )
        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        self._populate_endpoint_table()
        self._update_model_select()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ep-add-btn":
            self._add_endpoint()
        elif event.button.id == "ep-edit-btn":
            self._edit_endpoint()
        elif event.button.id == "ep-remove-btn":
            self._remove_endpoint()

    def _selected_endpoint(self) -> dict:
        """Return the endpoint dict for the currently selected table row."""
        table = self.query_one("#endpoint-table", DataTable)
        if table.row_count == 0:
            return {}
        try:
            row = table.get_row_at(table.cursor_row)
            row_key = row[0]  # "Name" is the first column
        except Exception:
            return {}
        if row_key is None:
            return {}
        with _config_lock:
            for ep in _config.get("endpoints", []):
                if ep.get("name", "") == row_key:
                    return ep
        return {}

    def _add_endpoint(self):
        self.app.push_screen(EndpointEditScreen(), self._on_endpoint_saved)

    def _edit_endpoint(self):
        ep = self._selected_endpoint()
        if not ep:
            self.notify("No endpoint selected", severity="warning")
            return
        original = dict(ep)
        original["_original_name"] = ep.get("name", "")
        self.app.push_screen(EndpointEditScreen(original), self._on_endpoint_saved)

    def _remove_endpoint(self):
        ep = self._selected_endpoint()
        if not ep:
            self.notify("No endpoint selected", severity="warning")
            return
        name = ep.get("name", "")
        self.app.push_screen(
            ConfirmScreen(f"Remove endpoint '{name}'?"),
            lambda confirmed: self._do_remove_endpoint(name, confirmed),
        )

    def _do_remove_endpoint(self, name: str, confirmed):
        if not confirmed:
            return
        with _config_lock:
            endpoints = list(_config.get("endpoints", []))
            endpoints = [e for e in endpoints if e.get("name", "") != name]
            _config["endpoints"] = endpoints
        clear_catalog_cache()
        save_config()
        set_endpoint_status(name, "disabled")
        self._populate_endpoint_table()
        self._update_model_select()
        _queue_log_line(f"Removed endpoint → {name}")

    def _on_endpoint_saved(self, result) -> None:
        """Callback after the add/edit modal is dismissed with a value."""
        if not result:
            return
        name = result.get("name", "")
        if not name:
            return
        original_name = result.pop("_original_name", None)
        with _config_lock:
            endpoints = list(_config.get("endpoints", []))
            if original_name:
                # Editing: replace the endpoint identified by its original name.
                for i, ep in enumerate(endpoints):
                    if ep.get("name", "") == original_name:
                        endpoints[i] = result
                        break
                else:
                    endpoints.append(result)
            else:
                # Adding: reject duplicate source prefix (name collision).
                if any(ep.get("name", "") == name for ep in endpoints):
                    self.notify(f"Endpoint '{name}' already exists", severity="error")
                    return
                endpoints.append(result)
            _config["endpoints"] = endpoints
        clear_catalog_cache()
        save_config()
        set_endpoint_status(name, "checking")
        self._populate_endpoint_table()
        self._update_model_select()
        _queue_log_line(f"Saved endpoint → {name}")

    def _populate_endpoint_table(self):
        table = self.query_one("#endpoint-table", DataTable)
        table.clear(columns=True)
        table.cursor_type = "row"
        table.add_columns("Name", "Type", "Host", "Status", "Load")
        endpoints = get_config("endpoints")
        for ep in endpoints:
            name = ep.get("name", "")
            status = get_endpoint_status(name)
            status_icon = {"online": "● on", "offline": "○ off", "disabled": "○ off", "checking": "◌ ..."}.get(status, "○ ?")
            table.add_row(
                name,
                ep.get("type", ""),
                ep.get("host", ""),
                status_icon,
                format_endpoint_load(name),
                key=name,
            )

    def action_save_settings(self):
        self._apply_settings()

    def action_restart_server(self):
        self._restart_server()

    def _update_model_select(self):
        """Refresh the model selector dropdown."""
        model_select = self.query_one("#model-select", Select)
        models = get_models()
        if models:
            model_select.set_options([(m, m) for m in models])
            model_select.value = get_current_model()
        else:
            model_select.set_options([("No models available", "")])

    def _apply_settings(self):
        log = ProxyHandler.tui_log
        models = get_models()

        # Model
        model_select = self.query_one("#model-select", Select)
        if model_select.value is not None:
            model_name = str(model_select.value)
            if model_name in models:
                set_config("default_model", model_name)
                _queue_log_line(f"Model set → {model_name}")

        # Fetch top models checkbox
        fetch_checkbox = self.query_one("#fetch-top-models-checkbox", Checkbox)
        fetch_enabled = fetch_checkbox.value
        set_config("fetch_top_models", fetch_enabled)
        if fetch_enabled:
            _queue_log_line("Fetching top intelligent models...")
            threading.Thread(target=fetch_top_intelligent_models, daemon=True).start()

        # Port / addr
        port = self.query_one("#port-input", Input).value.strip()
        addr = self.query_one("#addr-input", Input).value.strip()
        changed = False
        if port.isdigit():
            p = int(port)
            if 1 <= p <= 65535:
                if p != get_config("port"):
                    set_config("port", p)
                    changed = True
            else:
                self.notify(f"Port {p} out of range (1-65535), ignored", severity="error")
        elif port:
            self.notify(f"Invalid port '{port}', ignored — must be a number", severity="error")
        if addr and addr != get_config("addr"):
            set_config("addr", addr)
            changed = True

        save_config()
        if log is not None:
            log.write_line("Config saved to config.json")
        if changed:
            self.notify("Port/address changed — press Ctrl+R (Restart) to apply", severity="warning")

    def _restart_server(self):
        _queue_log_line("Restarting server...")
        global _server
        if _server is not None:
            _server.shutdown()
            _server.server_close()
            _server = None
        try:
            _start_server()
        except OSError as e:
            _queue_log_line(f"Restart failed: {e}")
            return
        _queue_log_line(f"Server restarted on http://{get_config('addr')}:{get_config('port')}")


def fetch_top_intelligent_models(limit: int = 20):
    """Fetch top intelligent models from OpenRouter by intelligence index."""
    global _top_intelligent_models
    endpoints = get_config("endpoints")
    or_ep = next((ep for ep in endpoints if ep.get("type") == "openrouter" and ep.get("enabled")), None)
    if not or_ep:
        msg = "Fetch top models skipped: no enabled OpenRouter endpoint"
        print(f"[proxy] {msg}")
        _queue_log_line(msg)
        return

    ep_name = or_ep.get("name", "openrouter")

    try:
        host = or_ep.get("host", "openrouter.ai")
        api_key = or_ep.get("api_key", "")
        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection(host, context=ctx, timeout=30)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        conn.request("GET", "/api/v1/models?sort=intelligence-high-to-low", headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()

        top_models = [f"{ep_name}:{m['id']}" for m in data.get("data", [])[:limit]]
        with _top_models_lock:
            _top_intelligent_models = top_models

        # Merge with user models: user models first, then top models (no duplicates)
        user_models = list(get_config("models"))
        merged = list(user_models)
        for m in top_models:
            if m not in merged:
                merged.append(m)

        set_models(merged)
        msg = f"Fetched {len(top_models)} top intelligent models (total {len(merged)})"
        print(f"[proxy] {msg}")
        _queue_log_line(msg)
    except Exception as e:
        msg = f"Could not fetch top intelligent models: {e}"
        print(f"[proxy] {msg}")
        _queue_log_line(msg)


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — MAIN SCREEN
# ═══════════════════════════════════════════════════════════════════════════

class MainScreen(Screen):
    """Main screen: toolbar + endpoint status + live log."""

    BINDINGS = [
        ("s", "open_settings", "Settings"),
        ("q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        log = self.query_one("#log", Log)
        ProxyHandler.tui_log = log
        self._update_endpoint_status()
        self._show_server_info()
        # Periodic status refresh
        self.set_interval(5.0, self._update_endpoint_status)
        # Periodic server-info refresh: the aggregated model catalog is merged
        # in asynchronously after mount (OpenRouter top-models fetch), so the
        # "Default" line must be re-rendered to stay in sync with the current
        # model instead of showing the mount-time resolution
        self.set_interval(5.0, self._show_server_info)
        # Drain worker-thread log lines into the Log widget
        self.set_interval(0.25, self._drain_log_queue)

    def _drain_log_queue(self):
        """Move queued log lines (from worker threads) into the Log widget."""
        log = self.query_one("#log", Log)
        try:
            while True:
                line = _log_queue.get_nowait()
                log.write_line(line)
        except queue.Empty:
            pass

    def _show_server_info(self):
        """Display server info after endpoints."""
        info = self.query_one("#server-info", Static)
        addr = get_config("addr")
        port = get_config("port")
        # Use localhost for display when binding to all interfaces
        display_host = "localhost" if addr in ("0.0.0.0", "::") else addr
        url = f"http://{display_host}:{port}"
        model = get_current_model()
        info.update(
            f"Listening : {url}\n"
            f"Web UI    : {url}/\n"
            f"Default   : {model}"
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(id="main-screen"):
            yield Static("Endpoints", classes="section-title")
            yield Static("Loading...", id="endpoint-status")
            yield Static("", id="server-info")
            yield Static("Proxy Log", classes="section-title")
            yield Log(id="log", highlight=True)
        yield Footer()

    def action_open_settings(self):
        self.app.push_screen(SettingsScreen())

    def _update_endpoint_status(self):
        """Refresh the endpoint status display (with per-endpoint load)."""
        status_widget = self.query_one("#endpoint-status", Static)
        endpoints = get_config("endpoints")
        lines = []
        # Column header (aligned to the row template below)
        lines.append(
            f"    {'ENDPOINT':<12} → {'HOST':<24} [{'STATUS':<8}] {'MODELS':>6}   LOAD"
        )
        for ep in endpoints:
            name = ep.get("name", "")
            host = ep.get("host", "")
            status = get_endpoint_status(name)
            if status == "online":
                icon = "●"
            elif status == "checking":
                icon = "◌"
            else:
                icon = "○"
            model_count = sum(1 for m in get_models() if m.startswith(f"{name}:"))
            load_str = format_endpoint_load(name)
            lines.append(
                f"  {icon} {name:<12} → {host:<24} [{status:<8}] {model_count:>6}   {load_str}"
            )
        status_widget.update("\n".join(lines) if lines else "  No endpoints configured")

    def action_quit(self):
        self.app.exit()


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class ProxyTUI(App):
    """Textual app to configure the proxy live while it runs."""

    TITLE = "AI Gateway"
    SUB_TITLE = "Multi-endpoint proxy"

    SCREENS = {
        "main": MainScreen,
        "settings": SettingsScreen,
    }

    CSS = """
    #main-screen {
        padding: 0 2;
        height: 1fr;
    }
    #endpoint-status {
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }
    #server-info {
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }
    #log {
        height: 20;
        width: 100%;
    }
    #settings-screen {
        padding: 0 2;
        height: 1fr;
        overflow-y: auto;
    }
    #settings-screen Label {
        margin-top: 1;
        height: 1;
        color: $text-muted;
    }
    #settings-screen Input {
        height: 1;
        border: none;
        padding: 0 1;
        margin: 0;
    }
    #settings-screen Select {
        height: 1;
        border: none;
        margin: 0;
    }
    #settings-screen Select > SelectCurrent {
        height: 1;
        padding: 0 1;
        border: none;
    }
    #settings-screen Select > SelectOverlay {
        max-height: 10;
    }
    #settings-screen > Horizontal {
        height: auto;
        margin-top: 0;
    }
    #settings-screen > Horizontal > Vertical {
        width: 1fr;
        height: auto;
        margin-right: 1;
    }
    #settings-screen > Horizontal > Vertical:last-child {
        margin-right: 0;
    }
    #settings-screen Button {
        width: auto;
        min-width: 10;
        margin-right: 1;
    }
    #endpoint-table {
        height: auto;
        max-height: 15;
        margin-top: 0;
    }
    #ep-edit-dialog {
        padding: 1 2;
        width: 60;
        height: auto;
        border: solid $primary;
        background: $surface;
    }
    #ep-edit-dialog Label {
        margin-top: 1;
        height: 1;
        color: $text-muted;
    }
    #ep-edit-dialog Input {
        height: 1;
        border: none;
        padding: 0 1;
    }
    #ep-edit-dialog Select {
        height: 1;
        border: none;
    }
    #ep-edit-dialog Select > SelectCurrent {
        height: 1;
        padding: 0 1;
        border: none;
    }
    #ep-edit-buttons {
        height: auto;
        margin-top: 1;
    }
    #ep-edit-buttons Button {
        width: auto;
        min-width: 10;
        margin-right: 1;
    }
    #confirm-dialog {
        padding: 1 2;
        width: 50;
        height: auto;
        border: solid $error;
        background: $surface;
    }
    #confirm-message {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    #confirm-buttons {
        height: auto;
        margin-top: 0;
    }
    #confirm-buttons Button {
        width: auto;
        min-width: 10;
        margin-right: 1;
    }
    .screen-title {
        text-style: bold;
        height: 1;
        margin-bottom: 0;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-top: 1;
    }
    """

    def on_mount(self) -> None:
        self.push_screen("main")


# ═══════════════════════════════════════════════════════════════════════════
#  SERVER LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════

_server = None
_server_thread = None


class _GatewayHTTPServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second process bind the same port and
    # silently share it, so a second instance would split requests with the
    # first instead of failing. Without it, it gets "address in use"; an
    # immediate rebind after closing (TUI restart) still works on Windows.
    allow_reuse_address = os.name != "nt"


def _start_server():
    """Bind and start the HTTP server.

    Raises OSError if the address/port cannot be bound, so callers can decide
    whether to abort (startup) or recover gracefully (TUI restart).
    """
    global _server, _server_thread
    port = get_config("port")
    addr = get_config("addr")
    try:
        _server = _GatewayHTTPServer((addr, port), ProxyHandler)
    except OSError as e:
        print(f"[proxy] ERROR: Cannot bind to {addr}:{port} — {e}")
        print(f"[proxy] Is another instance already running?")
        raise
    _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _server_thread.start()
    print(f"[proxy] Listening on http://{addr}:{port}")
    print(f"[proxy] Web UI: http://{addr}:{port}/")
    print(f"[proxy] Default model: {get_current_model()}")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load saved config (handles v1→v2 migration)
    load_config()

    # CLI overrides
    if len(sys.argv) > 1:
        try:
            set_config("port", int(sys.argv[1]))
        except ValueError:
            print(f"[proxy] ERROR: Invalid port number: {sys.argv[1]}")
            print(f"[proxy] Usage: python proxy.py [port] [bind_address]")
            sys.exit(1)
    if len(sys.argv) > 2:
        set_config("addr", sys.argv[2])

    # Verify webui directory exists
    if not os.path.isdir(WEBUI_DIR):
        print(f"[proxy] WARNING: Web UI directory not found: {WEBUI_DIR}")
        print(f"[proxy] The web UI will not be available. Run the build first.")

    # Auto-discover local endpoints (llama-server, LM Studio)
    auto_discover_local_endpoints()
    save_config()  # Save if new endpoints were discovered

    # Initialize endpoint statuses
    for ep in get_config("endpoints"):
        set_endpoint_status(ep.get("name", ""), "checking")

    # Start health check thread (checks all endpoints + aggregates models)
    print("[proxy] Starting health check thread...")
    threading.Thread(target=health_check_loop, daemon=True).start()

    # Fetch top intelligent models if enabled
    if get_config("fetch_top_models"):
        print("[proxy] Fetching top intelligent models from OpenRouter...")
        threading.Thread(target=fetch_top_intelligent_models, daemon=True).start()

    try:
        _start_server()
    except OSError as e:
        print(f"[proxy] Fatal: could not start server — {e}")
        sys.exit(1)

    try:
        ProxyTUI().run()
    finally:
        if _server:
            _server.shutdown()
            _server.server_close()
