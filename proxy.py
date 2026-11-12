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

import atexit
import collections
import copy
import hashlib
import json
import http.client
import ipaddress
import mimetypes
import os
import queue
import re
import secrets
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
    "force_default_model": False,
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


_LOG_FILE_MAX = 5 * 1024 * 1024     # gateway.log is cut back to its newer half past this
_log_file_lock = threading.Lock()


def _log_file_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(CONFIG_FILE)), "gateway.log")


def _write_log_file(line: str):
    """The console's log lines, kept on disk too: the console forgets them when it closes. They hold
    counts and names only, never chat text. Off with "log_file": false in config.json."""
    try:
        if _config.get("log_file", True) is False:
            return
        path = _log_file_path()
        with _log_file_lock:
            if os.path.exists(path) and os.path.getsize(path) > _LOG_FILE_MAX:
                with open(path, "rb") as f:
                    f.seek(-_LOG_FILE_MAX // 2, os.SEEK_END)
                    f.readline()
                    keep = f.read()
                with open(path, "wb") as f:
                    f.write(keep)
            with open(path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line.replace("\n", " ") + "\n")
    except Exception:  # noqa: BLE001 -- a log must never break what it reports on
        pass


def _queue_log_line(line: str):
    """Thread-safely enqueue a line for the TUI log widget (and the log file)."""
    try:
        _log_queue.put(line)
    except Exception:
        pass
    _write_log_file(str(line))


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


def _forces_default_model(path: str) -> bool:
    """Whether "force default model" applies to this request path.

    Only generation calls (chat/text completions, responses, messages) are
    switched to the default model; embeddings and the like are left alone,
    since a chat model can't serve them.
    """
    if not get_config("force_default_model"):
        return False
    return urlparse(path).path.rstrip("/").endswith(("/completions", "/responses", "/messages"))


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
_REFUSED_BODY_READ = 1024 * 1024   # a refused caller gets this much read, so its reply arrives

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


# ── Gateway access keys ──────────────────────────────────────────────────
# Off unless config.json has "access_keys". Then every API request needs one
# of them, sent the way OpenAI and Anthropic clients send theirs. The web
# page's own files stay open so it can load and ask for the key (llama-ui
# does this on a 401 from /props). A key is never passed to an upstream, and
# never written to the log or the stats: only its name is.

_ACCESS_KEY_MIN = 16
_ACCESS_KEY_HEADERS = ("x-api-key", "api-key")
_access_refusals_logged = {}          # client address → last time a refusal was logged


def _access_keys() -> list:
    """[(name, key)] from the config. Strings and {"name", "key"} both work;
    a key shorter than _ACCESS_KEY_MIN is ignored, so a typo can't open the gateway
    to one-letter guesses."""
    out = []
    raw = get_config("access_keys")
    if not isinstance(raw, list):
        return out
    for i, item in enumerate(raw):
        if isinstance(item, str):
            name, key = f"key {i + 1}", item
        elif isinstance(item, dict):
            name, key = str(item.get("name") or f"key {i + 1}"), item.get("key")
        else:
            continue
        if isinstance(key, str) and len(key.strip()) >= _ACCESS_KEY_MIN:
            out.append((name, key.strip()))
    return out


def _presented_key(headers) -> str:
    """The key a client sent: Authorization: Bearer, x-api-key or api-key."""
    auth = (headers.get("Authorization") or "").strip()
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    for h in _ACCESS_KEY_HEADERS:
        val = (headers.get(h) or "").strip()
        if val:
            return val
    return ""


def access_check(headers):
    """(allowed, key name). Keys off → (True, None). Every configured key is
    compared in constant time, so the answer's timing doesn't leak a prefix."""
    keys = _access_keys()
    if not keys:
        return True, None
    sent = _presented_key(headers).encode()
    found = None
    for name, key in keys:
        if secrets.compare_digest(sent, key.encode()) and found is None:
            found = name
    return found is not None, found


def _access_keys_result(typed: str, existing):
    """The access_keys value to save from the Settings field, and how many typed
    keys were too short. Names already in the config are kept for their keys;
    an empty field leaves a config that never had keys untouched."""
    names = {}
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, dict) and isinstance(item.get("key"), str) and item.get("name"):
                names[item["key"].strip()] = str(item["name"])
    keys, short = [], 0
    for part in (typed or "").replace("\n", ",").split(","):
        key = part.strip()
        if not key:
            continue
        if len(key) < _ACCESS_KEY_MIN:
            short += 1
            continue
        if key in [k if isinstance(k, str) else k["key"] for k in keys]:
            continue
        keys.append({"name": names[key], "key": key} if key in names else key)
    if not keys and existing is None:
        return None, short
    return keys, short


def new_access_key() -> str:
    """A fresh key: 'gw-' and 32 random URL-safe characters."""
    return "gw-" + secrets.token_urlsafe(24)


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
        max_ctx, guessed = _sticky_context(host, model_id, m.get("max_context_length") or m.get("context_length"))
        loaded_ctx = m.get("loaded_context_length") or max_ctx
        caps = m.get("capabilities", []) or []
        input_mods = ["text"]
        if "tool_use" in caps:
            input_mods.append("tool_use")
        models.append({
            "id": model_id,
            "name": m.get("name", model_id),
            "context_length": max_ctx,
            "context_guessed": guessed,
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
            "context_length": gen_settings.get("n_ctx") or _DEFAULT_CONTEXT,
            "context_guessed": not gen_settings.get("n_ctx"),
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
                "context_length": _DEFAULT_CONTEXT,       # /api/tags doesn't say
                "context_guessed": True,
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


_DEFAULT_CONTEXT = 128000      # reported only when an endpoint has never said what a model holds
_known_context = {}             # (host, model id) -> the last context window the endpoint reported
_known_context_lock = threading.Lock()


def _sticky_context(host: str, model_id: str, reported):
    """(context window, guessed). An endpoint that can't be asked right now (busy, restarting) keeps the
    window it last reported: swapping in a default mid-session told clients 128,000 for a 262,144 model
    (2026-11-12). Only a model whose window was never reported gets the default, marked as a guess."""
    key = (host, model_id)
    with _known_context_lock:
        if isinstance(reported, (int, float)) and not isinstance(reported, bool) and reported > 0:
            _known_context[key] = int(reported)
            return int(reported), False
        if key in _known_context:
            return _known_context[key], False
    return _DEFAULT_CONTEXT, True


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
    for inst in sorted(instances, key=lambda i: i.get("status") == "ready"):     # a ready one wins a shared name
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
        context, guessed = _sticky_context(host, model_id, inst.get("context") or meta.get("context"))
        entries.append({
            "id": model_id,
            "name": inst.get("label") or meta.get("label") or model_id,
            "owned_by": "strata",
            "context_length": context,
            "context_guessed": guessed,
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

from rich.console import Group as RichGroup  # rich comes with textual
from rich.table import Table as RichTable
from rich.text import Text as RichText
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Header, Footer, Input, Select, Static, Button, Label, Log, Checkbox,
    DataTable,
)


# ═══════════════════════════════════════════════════════════════════════════
#  KV SESSION SWAP — save and restore a local engine's conversation on disk
# ═══════════════════════════════════════════════════════════════════════════
#
# A local engine (Strata) holds ONE conversation. When several apps each keep
# a long chat there, every switch throws the held conversation away and the
# next request reads the other chat from scratch: 15+ minutes at 500k tokens.
# Strata can save the held conversation to a file and load it back in seconds
# (POST /slots/0?action=save|restore, switched on by "slot_save_path" in its
# run config). This section decides, per request, which conversation the engine
# should hold and swaps it through those two calls.
#
# It never reads a session file: it only passes file names to the engine and
# remembers each conversation as a list of message hashes (the chain).
#
# Opt-in per endpoint in config.json:
#   "kv_sessions": {"enabled": true}                    (other keys: README)

KV_DEFAULTS = {
    "enabled": False,
    "engine": "",           # host:port of the engine's own server; "" = find it (Strata)
    "engine_api_key": "",   # its API key; "" = the key Strata wrote for that instance
    "slot_dir": "",         # where the engine writes sessions (only needed to evict)
    "min_tokens": 30000,    # a conversation shorter than this is cheap to read again: not saved
    "min_gain_chars": 4000, # restore a saved chat only if it matches this many more characters
    "idle_save_s": 600,     # save the held chat after this long without requests (0 = off); a save
                            # rewrites the whole chat (8.5 GB at 512k tokens), so not after every pause
    "max_gb": 0,            # keep at most this much on disk, oldest first (0 = never delete)
}
_KV_KEEP_RATIO = 0.98       # a request that keeps this much of the held chat continues it
_KV_BACKOFF_S = 60          # after the engine can't swap, leave it alone for this long
_KV_SUSPEND_S = 1800        # after a 500 from a save or restore: off until the engine reloads, or this long
_KV_IDLE_POLL_S = 5.0       # how often the idle saver looks
_KV_CALL_TIMEOUT = 900      # a save or restore of a 500k chat takes seconds, not minutes
_kv_managers = {}           # endpoint name → KVSessionManager
_kv_managers_lock = threading.Lock()


def _kv_config(ep: dict):
    """The endpoint's kv_sessions settings with defaults, or None when off."""
    raw = ep.get("kv_sessions")
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return None
    cfg = dict(KV_DEFAULTS)
    cfg.update({k: v for k, v in raw.items() if k in KV_DEFAULTS})
    for key, cast in (("min_tokens", int), ("min_gain_chars", int),
                      ("idle_save_s", float), ("max_gb", float)):
        try:
            cfg[key] = max(0, cast(cfg[key]))
        except (TypeError, ValueError):
            cfg[key] = KV_DEFAULTS[key]
    for key in ("engine", "engine_api_key", "slot_dir"):
        cfg[key] = str(cfg[key] or "").strip()
    return cfg


def _kv_text(content) -> str:
    """A message's content as text for hashing: a string, or the text, tool and
    image parts of a list (an image counts by a short hash of its reference)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif not isinstance(p, dict):
                parts.append(str(p))
            else:
                kind = p.get("type")
                if kind in ("text", "input_text", "output_text") or (kind is None and "text" in p):
                    parts.append(str(p.get("text", "")))
                elif kind in ("image_url", "input_image", "image"):
                    ref = p.get("image_url") or p.get("source") or p.get("url") or ""
                    if isinstance(ref, dict):
                        ref = ref.get("url") or json.dumps(ref, sort_keys=True)
                    parts.append("[image:" + hashlib.sha1(str(ref).encode("utf-8", "replace")).hexdigest()[:12] + "]")
                else:
                    parts.append(json.dumps(p, sort_keys=True, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


class KVChain:
    """A conversation as the hashes of its messages (tools and system prompt
    included, in order) and the running character count at each one."""
    __slots__ = ("hashes", "lens")

    def __init__(self, hashes, lens):
        self.hashes = list(hashes)
        self.lens = list(lens)

    @property
    def total(self) -> int:
        return self.lens[-1] if self.lens else 0


def kv_chain(data, path: str):
    """The KVChain of a chat request (OpenAI chat completions or Anthropic
    messages), or None when the request carries no conversation."""
    if not isinstance(data, dict):
        return None
    msgs = data.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    items = []
    if data.get("tools"):
        items.append(("tools", json.dumps(data["tools"], sort_keys=True, ensure_ascii=False)))
    if path.rstrip("/").endswith("/messages") and data.get("system"):
        items.append(("system", _kv_text(data["system"])))
    for m in msgs:
        if not isinstance(m, dict):
            continue
        text = _kv_text(m.get("content"))
        if m.get("tool_calls"):
            text += "\n" + json.dumps(m["tool_calls"], sort_keys=True, ensure_ascii=False)
        for key in ("name", "tool_call_id"):
            if m.get(key):
                text += f"\n{key}={m[key]}"
        items.append((str(m.get("role", "")), text))
    if not items:
        return None
    hashes, lens, total = [], [], 0
    for label, text in items:
        hashes.append(hashlib.sha1((label + "\x00" + text).encode("utf-8", "replace")).hexdigest()[:20])
        total += len(text)
        lens.append(total)
    return KVChain(hashes, lens)


def _kv_common(hashes, lens, other):
    """(messages, characters) the two chains share from the start."""
    n = 0
    for a, b in zip(hashes, other):
        if a != b:
            break
        n += 1
    return n, (lens[n - 1] if n else 0)


def _kv_instances_dir() -> str:
    return (os.environ.get("STRATA_INSTANCES_DIR")
            or os.path.join(os.path.expanduser("~"), ".strata", "data", "native-instances"))


def _kv_find_strata_instance(model: str):
    """(host:port, api key, slot dir) of the engine server Strata started for
    `model`, read from Strata's own run files (the newest load of that model).
    The key is used in memory only, never copied or logged."""
    folder = _kv_instances_dir()
    try:
        # drafting-*.json are Strata's short-lived probes, not running engines
        names = [n for n in os.listdir(folder) if n.endswith(".json") and not n.startswith("drafting-")]
    except OSError:
        return None
    want = (model or "").strip().lower()
    best = None
    for name in names:
        path = os.path.join(folder, name)
        try:
            mtime = os.path.getmtime(path)
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            port = int(cfg.get("port") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if not isinstance(cfg, dict) or not port:
            continue
        known = {str(cfg.get("model_name", "")).lower()} | {str(a).lower() for a in (cfg.get("aliases") or [])}
        rank = (1 if want and want in known else 0, mtime)
        if best is None or rank > best[0]:
            slot_dir = str(cfg.get("slot_save_path") or "")
            if slot_dir and not os.path.isabs(slot_dir) and cfg.get("cwd"):
                slot_dir = os.path.join(str(cfg["cwd"]), slot_dir)
            best = (rank, (f"{cfg.get('host') or '127.0.0.1'}:{port}", str(cfg.get("api_key") or ""), slot_dir))
    return best[1] if best else None


class _KVError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class _KVTicket:
    """One request's hold on its endpoint's engine (taken by begin, given back by end)."""

    def __init__(self, manager, chain, model, app=""):
        self.manager = manager
        self.chain = chain
        self.model = model
        self.app = app          # which app sent it, for the console's saved-chats table
        self.target = None      # (host, key, slot dir) once the engine was reached
        self.inherit_id = None  # the saved session this request continues, if any
        self.event = None       # what the swap did: continued, restored or new
        self.swap_s = 0.0       # time spent saving and restoring for this request


class KVSessionManager:
    """Keeps the right conversation in one endpoint's engine slot.

    One request at a time per endpoint (the engine has one slot and queues
    requests itself, so nothing waits longer than before). Before a request is
    forwarded: if it continues the conversation the engine already holds, do
    nothing; otherwise save the held one (when long and unsaved), and restore
    the saved conversation that matches the request best. After it: remember
    the engine now holds this request's conversation, unsaved.
    """

    def __init__(self, ep: dict, cfg: dict):
        self.name = ep.get("name", "")
        self.type = ep.get("type", "")
        self.ep_key = ep.get("api_key", "")
        self.cfg = cfg
        self.lock = threading.RLock()
        self.index = {}             # session id → entry (chain, file, tokens, bytes, times)
        self.current = None         # what the engine holds: chain, lens, id, tokens, dirty
        self.unavailable_until = 0.0
        self.last_activity = time.time()
        self.stats = {"saves": 0, "restores": 0, "errors": 0, "continued": 0, "new": 0,
                      "restored_tokens": 0, "save_s": 0.0, "restore_s": 0.0, "written_bytes": 0}
        self.last_event = ""
        self.last_error = None      # the last swap that failed: action, status and the engine's words
        self.suspended = None       # swapping is off for this engine load: reason, since, until
        self._loaded = False
        self._held_sig = None       # what the index file says the engine holds, to write it only on a change
        self._idle_thread = None
        self._last_target = None
        self._slot_dir = ""

    # ── logging and persistence ─────────────────────────────────────────
    def _log(self, msg: str):
        _queue_log_line(f"[kv:{self.name}] {msg}")

    def _index_path(self) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name) or "endpoint"
        return os.path.join(os.path.dirname(os.path.abspath(CONFIG_FILE)), f"kv-sessions-{safe}.json")

    def _load_index(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self._index_path(), "r", encoding="utf-8") as f:
                raw = json.load(f)
            for sid, e in (raw.get("entries") or {}).items():
                if (isinstance(e, dict) and isinstance(e.get("chain"), list) and e["chain"]
                        and isinstance(e.get("lens"), list) and len(e["lens"]) == len(e["chain"])
                        and isinstance(e.get("file"), str)):
                    e["id"] = sid
                    self.index[sid] = e
        except (OSError, ValueError, AttributeError, TypeError):
            return
        if self.current is None:
            self.current = self._held_from(raw.get("held") if isinstance(raw, dict) else None)
        self._held_sig = self._held_signature()

    @staticmethod
    def _held_from(rec):
        """What the engine held when the gateway last ran, as the index file says, or None.
        The first request checks it against the engine's own token count before using it."""
        try:
            if not isinstance(rec, dict):
                return None
            chain, lens, tokens = rec.get("chain"), rec.get("lens"), rec.get("tokens")
            if (not isinstance(chain, list) or not chain or not isinstance(lens, list) or len(lens) != len(chain)
                    or not all(isinstance(h, str) for h in chain) or isinstance(tokens, bool)
                    or not isinstance(tokens, int) or tokens <= 0):
                return None
            sid = rec.get("id")
            return {"chain": list(chain), "lens": [int(n) for n in lens],
                    "id": sid if isinstance(sid, str) and sid else None, "tokens": tokens,
                    "dirty": bool(rec.get("dirty", True)), "app": str(rec.get("app") or ""),
                    "engine": str(rec.get("engine") or "")}
        except (TypeError, ValueError):
            return None

    def _held_record(self):
        """`current` as the index file keeps it: message hashes and counts, never text."""
        cur = self.current
        if not cur or not cur.get("chain"):
            return None
        return {"chain": list(cur["chain"]), "lens": list(cur["lens"]), "id": cur.get("id"),
                "tokens": int(cur.get("tokens") or 0), "dirty": bool(cur.get("dirty")),
                "app": cur.get("app", ""), "engine": cur.get("engine", "")}

    def _held_signature(self):
        rec = self._held_record()
        if rec is None:
            return None
        return (rec["id"], rec["tokens"], rec["dirty"], rec["engine"], len(rec["chain"]), rec["chain"][-1])

    def _save_index(self):
        path = self._index_path()
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "entries": self.index, "held": self._held_record()}, f)
            os.replace(tmp, path)
            self._held_sig = self._held_signature()
        except OSError as e:
            self._log(f"could not write the session index: {e}")

    def _persist_held(self):
        """Write the index when what the engine holds changed, so a gateway restart
        still knows which chat that is (and doesn't restore an older copy over it)."""
        if self._loaded and self._held_signature() != self._held_sig:
            self._save_index()

    # ── talking to the engine ───────────────────────────────────────────
    def _target(self, model: str):
        cfg = self.cfg
        if cfg["engine"]:
            return cfg["engine"], cfg["engine_api_key"] or self.ep_key, cfg["slot_dir"]
        if self.type == "strata":
            found = _kv_find_strata_instance(model)
            if found:
                host, key, slot_dir = found
                return host, cfg["engine_api_key"] or key, cfg["slot_dir"] or slot_dir
        return None

    def _call(self, target, method, path, body=None, timeout=_KV_CALL_TIMEOUT):
        host, key, _slot_dir = target
        conn = http.client.HTTPConnection(_dial_host(host), timeout=timeout)
        try:
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            try:
                parsed = json.loads(raw) if raw else None
            except ValueError:
                parsed = None
            return resp.status, parsed
        finally:
            conn.close()

    def _slot_state(self, target):
        """(tokens the engine holds, busy) from GET /slots, or None."""
        try:
            status, body = self._call(target, "GET", "/slots", timeout=10)
        except Exception:
            return None
        if status != 200 or not isinstance(body, list) or not body or not isinstance(body[0], dict):
            return None
        try:
            held = int(body[0].get("n_prompt_tokens") or 0)
        except (TypeError, ValueError):
            held = 0
        return held, bool(body[0].get("is_processing"))

    def _slot_action(self, target, action: str, filename: str) -> dict:
        status, body = self._call(target, "POST", f"/slots/0?action={action}", {"filename": filename})
        if status == 200 and isinstance(body, dict):
            return body
        msg = ""
        if isinstance(body, dict):
            err = body.get("error")
            msg = err.get("message", "") if isinstance(err, dict) else str(err or "")
        raise _KVError(status, msg or f"HTTP {status}")

    def _mark_unavailable(self, why: str):
        self.unavailable_until = time.time() + _KV_BACKOFF_S
        self.current = None
        self._log(f"session swap paused for {_KV_BACKOFF_S} s: {why}")

    def _note_failure(self, action, error):
        """Keep why a swap failed: the engine's error text, never anything from a chat."""
        message = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", str(error))).strip()[:200]
        self.last_error = {"time": time.time(), "action": action or "swap",
                           "status": getattr(error, "status", None), "message": message}

    def _suspend(self, target, action):
        """A 500 from a save or restore means this engine load can't snapshot at all
        (Strata 0.1.57: "conversation snapshot: invalid K/V extent", from then on). Leave it
        alone until the model is reloaded, which gives it a new key, or for _KV_SUSPEND_S."""
        reason = (self.last_error or {}).get("message", "")
        now = time.time()
        self.suspended = {"load": tuple(target[:2]), "action": action, "reason": reason,
                          "since": now, "until": now + _KV_SUSPEND_S}
        self.current = None
        self._log(f"session swap off until the model is reloaded (or for {_KV_SUSPEND_S // 60} min): "
                  f"the engine could not {action}: {reason}")

    def _suspended_for(self, target) -> bool:
        s = self.suspended
        if not s:
            return False
        if time.time() < s["until"] and s["load"] == tuple(target[:2]):
            return True
        self.suspended = None
        self._log("trying session swap again: " + ("the model was reloaded" if s["load"] != tuple(target[:2]) else "time is up"))
        return False

    # ── the decision (pure: no I/O) ─────────────────────────────────────
    def _plan(self, cur, req: KVChain) -> dict:
        cfg = self.cfg
        cur_chars = 0
        cur_total = 0
        if cur and cur.get("chain"):
            _n, cur_chars = _kv_common(cur["chain"], cur["lens"], req.hashes)
            cur_total = cur["lens"][-1]
        if cur and cur.get("chain") and cur_chars >= _KV_KEEP_RATIO * cur_total:
            return {"keeps": True, "save": False, "restore": None}
        save = bool(cur and cur.get("chain") and cur.get("dirty") and cur["tokens"] >= cfg["min_tokens"])
        best, best_chars = None, 0
        for e in self.index.values():
            _n, chars = _kv_common(e["chain"], e["lens"], req.hashes)
            if chars > best_chars or (best and chars == best_chars and e.get("last_used", 0) > best.get("last_used", 0)):
                best, best_chars = e, chars
        restore = None
        if (best and best_chars >= cur_chars + cfg["min_gain_chars"]
                and int(best.get("tokens", 0)) >= cfg["min_tokens"]):
            restore = best
        return {"keeps": False, "save": save, "restore": restore}

    # ── save / restore ──────────────────────────────────────────────────
    def _save_current(self, target):
        cur = self.current
        sid = cur.get("id") or secrets.token_hex(6)
        fname = f"gw-{sid}.bin"
        t0 = time.time()
        r = self._slot_action(target, "save", fname)
        took = time.time() - t0
        now = time.time()
        before = self.index.get(sid) or {}
        self.index[sid] = {
            "id": sid, "file": fname, "chain": list(cur["chain"]), "lens": list(cur["lens"]),
            "tokens": int(r.get("n_saved") or cur["tokens"]), "bytes": int(r.get("n_written") or 0),
            "saved_at": now, "last_used": now, "app": cur.get("app") or before.get("app", ""),
            "restores": int(before.get("restores", 0)),
        }
        cur["id"] = sid
        cur["dirty"] = False
        self.stats["saves"] += 1
        self.stats["save_s"] += took
        self.stats["written_bytes"] += self.index[sid]["bytes"]
        self._save_index()
        self.last_event = (f"saved {sid}: {self.index[sid]['tokens']:,} tokens, "
                           f"{self.index[sid]['bytes'] / 1e6:,.0f} MB in {took:.1f} s")
        self._log(self.last_event)
        self._evict()
        return took

    def _restore(self, target, entry):
        t0 = time.time()
        r = self._slot_action(target, "restore", entry["file"])
        took = time.time() - t0
        entry["last_used"] = time.time()
        entry["restores"] = int(entry.get("restores", 0)) + 1
        held = self._slot_state(target)
        self.stats["restores"] += 1
        self.stats["restore_s"] += took
        self.stats["restored_tokens"] += int(r.get("n_restored") or 0)
        self.current = {"chain": list(entry["chain"]), "lens": list(entry["lens"]), "id": entry["id"],
                        "tokens": held[0] if held else int(r.get("n_restored") or entry.get("tokens", 0)),
                        "dirty": False, "app": entry.get("app", ""), "engine": target[0]}
        self._save_index()
        self.last_event = f"restored {entry['id']}: {int(r.get('n_restored') or 0):,} tokens in {took:.1f} s"
        self._log(self.last_event)
        return took

    def _evict(self):
        """Delete the oldest session files beyond max_gb. Only files this
        manager made (gw-*.bin in the engine's session folder); never the
        held conversation's; never reads one."""
        limit = int(self.cfg["max_gb"] * (1024 ** 3))
        if limit <= 0:
            return
        total = sum(int(e.get("bytes", 0)) for e in self.index.values())
        if total <= limit:
            return
        if not self._slot_dir:
            self._log("over max_gb but the engine's session folder is unknown (set slot_dir): nothing deleted")
            return
        keep = self.current.get("id") if self.current else None
        for e in sorted(self.index.values(), key=lambda x: x.get("last_used", 0)):
            if total <= limit:
                break
            name = e.get("file", "")
            if e["id"] == keep or os.path.basename(name) != name or not name.startswith("gw-"):
                continue
            try:
                os.remove(os.path.join(self._slot_dir, name))
            except FileNotFoundError:
                pass
            except OSError as ex:
                self._log(f"could not delete {name}: {ex}")
                continue
            total -= int(e.get("bytes", 0))
            del self.index[e["id"]]
            self._log(f"deleted {name} (over {self.cfg['max_gb']:g} GB)")
        self._save_index()

    def _revalidate(self, target):
        """After something went wrong: keep `current` only if the engine still
        holds about as many tokens as we think it does."""
        cur = self.current
        st = self._slot_state(target) if target else None
        if not cur or not st or st[0] == 0 or abs(st[0] - cur["tokens"]) > max(64, 0.02 * cur["tokens"]):
            self.current = None

    # ── a request's life ────────────────────────────────────────────────
    def begin(self, chain: KVChain, model: str = "", app: str = "") -> _KVTicket:
        self.lock.acquire()
        ticket = _KVTicket(self, chain, model, app)
        try:
            self._begin_locked(ticket)
        except Exception as e:  # never let swapping break the request itself
            self.stats["errors"] += 1
            self._note_failure("swap", e)
            self.current = None
            self._log(f"swap skipped: {e}")
        return ticket

    def _begin_locked(self, t: _KVTicket):
        self._load_index()
        if time.time() < self.unavailable_until:
            self.current = None
            return
        target = self._target(t.model)
        if not target:
            self._mark_unavailable("no engine to talk to (set \"engine\" in kv_sessions, or load the model in Strata)")
            return
        if self._suspended_for(target):
            # Not even /slots: after a failed snapshot it reports 0 tokens while it still holds the chat.
            self.current = None
            t.event = None
            return
        st = self._slot_state(target)
        if st is None:
            self._mark_unavailable("the engine's /slots could not be read")
            return
        t.target = self._last_target = target
        self._slot_dir = target[2] or self._slot_dir
        held = st[0]
        cur = self.current
        if cur and ((cur.get("engine") and cur["engine"] != target[0])
                    or held == 0 or abs(held - cur["tokens"]) > max(64, 0.02 * cur["tokens"])):
            self._log("the engine no longer holds the conversation tracked here (restarted, or used directly): not saving it")
            cur = self.current = None
        plan = self._plan(cur, t.chain)
        if plan["restore"] and cur is None and held >= self.cfg["min_tokens"]:
            # The engine holds a long chat this gateway can't name: it may be this very chat, newer
            # than any saved copy. A restore would throw it away unsaved, so let the engine reuse it.
            skipped, plan["restore"] = plan["restore"], None
            _n, chars = _kv_common(skipped["chain"], skipped["lens"], t.chain.hashes)
            if chars >= _KV_KEEP_RATIO * skipped["lens"][-1]:
                t.inherit_id = skipped["id"]  # it carries that saved chat on: its next save replaces that file
            self.last_event = f"left the engine's {held:,}-token chat in place: the gateway can't tell which chat it is"
            self._log(self.last_event + ", so no saved chat was restored over it")
        t.event = "continued" if plan["keeps"] else "new"
        doing = None
        try:
            if plan["save"]:
                doing = "save"
                t.swap_s += self._save_current(target)
            if plan["restore"]:
                doing = "restore"
                t.swap_s += self._restore(target, plan["restore"])
                t.event = "restored"
        except _KVError as e:
            self.stats["errors"] += 1
            self._note_failure(doing, e)
            if e.status == 500:
                self._suspend(target, doing)
                t.target = None
                t.event = None
                return
            if e.status == 501:
                self._mark_unavailable(f"{e} (add \"slot_save_path\" to Strata's run config and load the model again)")
                t.target = None
                t.event = None
                return
            if e.status == 404 and plan["restore"]:
                self.index.pop(plan["restore"]["id"], None)
                self._save_index()
            self._log(f"swap failed: {e}")
            self._revalidate(target)
        # Does this request continue what the engine holds now? Then it inherits its session id.
        cur = self.current
        if cur and cur.get("chain"):
            _n, chars = _kv_common(cur["chain"], cur["lens"], t.chain.hashes)
            if chars >= _KV_KEEP_RATIO * cur["lens"][-1]:
                t.inherit_id = cur.get("id")

    def end(self, t: _KVTicket, ok: bool):
        try:
            if not t.target:
                self.current = None
            else:
                st = self._slot_state(t.target)
                if ok and st:
                    self.current = {"chain": list(t.chain.hashes), "lens": list(t.chain.lens),
                                    "id": t.inherit_id, "tokens": st[0], "dirty": True, "app": t.app,
                                    "engine": t.target[0]}
                    if t.event in ("continued", "new"):
                        self.stats[t.event] += 1
                else:
                    self._revalidate(t.target)
            self.last_activity = time.time()
            self._ensure_idle_thread()
        except Exception as e:
            self._log(f"bookkeeping error: {e}")
            self.current = None
        finally:
            try:
                self._persist_held()
            finally:
                self.lock.release()

    # ── save when idle, so a restart of the engine loses little ─────────
    def _ensure_idle_thread(self):
        if self.cfg["idle_save_s"] <= 0:
            return
        if self._idle_thread and self._idle_thread.is_alive():
            return
        self._idle_thread = threading.Thread(target=self._idle_loop, name=f"kv-idle-{self.name}", daemon=True)
        self._idle_thread.start()

    def _idle_loop(self):
        while True:
            time.sleep(_KV_IDLE_POLL_S)
            wait = self.cfg["idle_save_s"]
            if wait <= 0:
                return
            cur = self.current
            if not cur or not cur.get("dirty") or cur["tokens"] < self.cfg["min_tokens"]:
                continue
            if time.time() - self.last_activity < wait:
                continue
            if not self.lock.acquire(blocking=False):
                continue
            try:
                cur = self.current
                target = self._last_target
                if (cur and target and cur.get("dirty") and cur["tokens"] >= self.cfg["min_tokens"]
                        and time.time() - self.last_activity >= wait):
                    st = self._slot_state(target)
                    if st and not st[1] and abs(st[0] - cur["tokens"]) <= max(64, 0.02 * cur["tokens"]):
                        self._save_current(target)
                    else:
                        self.current = None
            except Exception as e:
                self.stats["errors"] += 1
                self._note_failure("idle save", e)
                self.last_activity = time.time()  # try again after another idle period
                self._log(f"idle save failed: {e}")
                if isinstance(e, _KVError) and e.status == 500 and self._last_target:
                    self._suspend(self._last_target, "save")
            finally:
                try:
                    self._persist_held()
                finally:
                    self.lock.release()

    def _suspension_status(self):
        s = self.suspended
        if not s or time.time() >= s["until"]:
            return None
        return {"action": s["action"], "reason": s["reason"], "since": s["since"], "until": s["until"]}  # never the key

    def status(self) -> dict:
        cur = self.current
        entries = list(self.index.values())
        recent = sorted(entries, key=lambda e: e.get("last_used", 0), reverse=True)[:12]
        return {
            "sessions": len(entries),
            "bytes": sum(int(e.get("bytes", 0)) for e in entries),
            "held": ({"id": cur.get("id"), "tokens": cur["tokens"], "saved": not cur.get("dirty"),
                      "app": cur.get("app", "")} if cur else None),
            "paused": max(0, int(self.unavailable_until - time.time())),
            "suspended": self._suspension_status(),
            "last_error": dict(self.last_error) if self.last_error else None,
            "last_event": self.last_event,
            "chats": [{"id": e["id"], "app": e.get("app", ""), "tokens": int(e.get("tokens", 0)),
                       "bytes": int(e.get("bytes", 0)), "last_used": e.get("last_used", 0),
                       "restores": int(e.get("restores", 0))} for e in recent],
            **self.stats,
        }


def _get_kv_manager(ep: dict, cfg: dict) -> KVSessionManager:
    name = ep.get("name", "")
    with _kv_managers_lock:
        mgr = _kv_managers.get(name)
        if mgr is None:
            mgr = _kv_managers[name] = KVSessionManager(ep, cfg)
        mgr.cfg = cfg
        mgr.type = ep.get("type", "")
        mgr.ep_key = ep.get("api_key", "")
        return mgr


def _kv_begin(ep: dict, command: str, path: str, data, app: str = ""):
    """Take the endpoint's engine for this request (None when swapping is off
    for it). Always pair with _kv_end in a finally."""
    cfg = _kv_config(ep)
    if not cfg or command != "POST" or not isinstance(data, dict):
        return None
    if not path.rstrip("/").endswith(("/chat/completions", "/messages")):
        return None
    try:
        chain = kv_chain(data, path)
    except Exception as e:  # an odd request is forwarded as it is, unmanaged
        _queue_log_line(f"[kv] request not managed: {e}")
        return None
    if chain is None:
        return None
    return _get_kv_manager(ep, cfg).begin(chain, str(data.get("model") or ""), app)


def _kv_end(ticket, ok: bool):
    if ticket is not None:
        ticket.manager.end(ticket, ok)


def kv_status_all() -> dict:
    """Every endpoint with swapping on, including one with no request since a
    restart: its saved chats are still there. Reads the index only, never the engine."""
    for ep in get_config("endpoints") or []:
        cfg = _kv_config(ep)
        if cfg:
            mgr = _get_kv_manager(ep, cfg)
            if not mgr._loaded and mgr.lock.acquire(blocking=False):  # held means a request has loaded it
                try:
                    mgr._load_index()
                finally:
                    mgr.lock.release()
    with _kv_managers_lock:
        managers = dict(_kv_managers)
    return {name: m.status() for name, m in managers.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  LIVE STATS — what passes through the gateway and what the engine reports
# ═══════════════════════════════════════════════════════════════════════════
#
# Counts and names only: tokens, times, model and app names. A response is read
# as it passes for its usage and timing fields, and nothing is ever changed or
# held up. Chat text is never stored, logged or shown.

_STATS_PATHS = ("/chat/completions", "/completions", "/messages", "/responses", "/embeddings")
_STATS_MAX_KEYS = 100       # models or apps kept apart; the rest count as "other"
_STATS_SAVE_EVERY_S = 30    # how often the all-time totals are written
_STATS_DAYS_KEPT = 30
_TAP_MAX_BODY = 8 * 1024 * 1024   # a bigger non-streamed reply is passed on, just not read
_TAP_MAX_LINE = 4 * 1024 * 1024
_READ_RATE_MIN_TOKENS = 1000      # shorter reads are mostly overhead, not a read speed
_ENGINE_POLL_S = 2.0
_ENGINE_IDLE_STOP_S = 30.0        # stop polling when nobody has looked for this long


def _stats_name(value, fallback: str = "unknown", limit: int = 48) -> str:
    """A model or app name as a client sent it, safe to show and store."""
    text = "".join(c for c in str(value or "") if c.isprintable()).strip()
    return text[:limit] or fallback


def _app_name(headers) -> str:
    """Which app sent a request: its X-Title, its site, or its User-Agent."""
    title = headers.get("X-Title")
    if title:
        return _stats_name(title)
    agent = headers.get("User-Agent") or ""
    if agent.startswith("Mozilla"):
        referer = urlparse(headers.get("Referer") or "").netloc
        if referer and referer == (headers.get("Host") or ""):
            return "Web UI"
        return _stats_name(referer, "Browser")
    site = urlparse(headers.get("HTTP-Referer") or "").netloc
    if site:
        return _stats_name(site)
    return _stats_name(agent.split("/")[0].split(" ")[0])


def _percentile(values, pct: float):
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return None
    k = (len(vals) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _number(*values):
    """The first real number among values (a bool is not one)."""
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def _norm_usage(u) -> dict:
    """OpenAI, Anthropic, DeepSeek and Responses usage objects in one shape:
    prompt (cached included), cached, out, cost. Only what the object says."""
    if not isinstance(u, dict):
        return {}
    details = u.get("prompt_tokens_details") or u.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}
    out = {}
    prompt = _number(u.get("prompt_tokens"))
    if prompt is None and _number(u.get("input_tokens")) is not None:
        # Anthropic counts cached and newly cached tokens apart from input_tokens
        prompt = (u["input_tokens"] + (_number(u.get("cache_read_input_tokens")) or 0)
                  + (_number(u.get("cache_creation_input_tokens")) or 0))
    if prompt is not None:
        out["prompt"] = prompt
    cached = _number(details.get("cached_tokens"), u.get("cache_read_input_tokens"), u.get("prompt_cache_hit_tokens"))
    if cached is not None:
        out["cached"] = cached
    generated = _number(u.get("completion_tokens"), u.get("output_tokens"))
    if generated is not None:
        out["out"] = generated
    cost = _number(u.get("cost"))
    if cost is not None:
        out["cost"] = cost
    return out


class _ResponseTap:
    """Reads usage and speed out of a response as it passes. It never changes
    the response; anything it can't read is skipped."""

    def __init__(self):
        self.first_byte = None
        self.first_token = None
        self.last_token = None
        self.deltas = 0         # streamed pieces: roughly tokens, when no usage comes
        self.usage = {}
        self.timings = {}
        self.bytes = 0
        self.finish_reason = None   # how the reply ended, as one of the names in _FINISH_NAMES
        self.text_chars = 0         # lengths only: the text itself is never kept
        self.reasoning_chars = 0
        self._call_keys = set()     # one key per tool call the model made (deltas of one call share it)
        self._line = bytearray()
        self._body = bytearray()
        self._too_big = False

    @property
    def calls(self) -> int:
        return len(self._call_keys)

    def feed(self, chunk: bytes, streaming: bool):
        try:
            now = time.time()
            if self.first_byte is None:
                self.first_byte = now
            self.bytes += len(chunk)
            if streaming:
                self._feed_sse(chunk, now)
            elif not self._too_big:
                if len(self._body) + len(chunk) > _TAP_MAX_BODY:
                    self._too_big = True
                    self._body = bytearray()
                else:
                    self._body.extend(chunk)
        except Exception:
            pass

    def _feed_sse(self, chunk: bytes, now: float):
        self._line.extend(chunk)
        if b"\n" not in chunk:
            if len(self._line) > _TAP_MAX_LINE:
                self._line = bytearray()
            return
        lines = self._line.split(b"\n")
        self._line = lines.pop()
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                self._event(json.loads(payload), now)
            except ValueError:
                continue

    def _event(self, obj, now: float):
        if not isinstance(obj, dict):
            return
        for holder in (obj, obj.get("message"), obj.get("response")):
            if isinstance(holder, dict) and isinstance(holder.get("usage"), dict):
                self.usage.update(_norm_usage(holder["usage"]))
        if isinstance(obj.get("timings"), dict):
            self.timings = obj["timings"]
        if self._is_token(obj):
            if self.first_token is None:
                self.first_token = now
            self.last_token = now
            self.deltas += 1
        try:
            self._read_end(obj)
        except Exception:  # an odd shape costs only these counts
            pass

    # ── how the reply ended: names and counts, never the text ───────────
    def _read_end(self, obj: dict):
        end = _finish_name(obj.get("done_reason"))  # Ollama's own API
        choices = obj.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            end = _finish_name(choice.get("finish_reason")) or end
            part = choice.get("delta") if isinstance(choice.get("delta"), dict) else choice.get("message")
            if isinstance(part, dict):
                self._count_part(part)
            elif isinstance(choice.get("text"), str):
                self.text_chars += len(choice["text"])
        elif "type" not in obj and isinstance(obj.get("message"), dict):
            self._count_part(obj["message"])  # Ollama: {"message": {...}, "done_reason": ...}
        kind = obj.get("type")
        if kind == "message" or (kind == "message_start" and isinstance(obj.get("message"), dict)):
            msg = obj if kind == "message" else obj["message"]  # Anthropic: a whole reply, or its start
            end = _finish_name(msg.get("stop_reason")) or end
            for i, block in enumerate(msg.get("content") or []):
                self._count_block(block, ("block", i))
        elif kind == "message_delta" and isinstance(obj.get("delta"), dict):
            end = _finish_name(obj["delta"].get("stop_reason")) or end
        elif kind == "content_block_start":
            self._count_block(obj.get("content_block"), ("block", obj.get("index")))
        elif kind == "content_block_delta" and isinstance(obj.get("delta"), dict):
            delta = obj["delta"]
            if isinstance(delta.get("text"), str):
                self.text_chars += len(delta["text"])
            if isinstance(delta.get("thinking"), str):
                self.reasoning_chars += len(delta["thinking"])
        elif kind == "response.output_text.delta" and isinstance(obj.get("delta"), str):
            self.text_chars += len(obj["delta"])
        elif kind == "response.reasoning_text.delta" and isinstance(obj.get("delta"), str):
            self.reasoning_chars += len(obj["delta"])
        elif kind == "response.output_item.added" and isinstance(obj.get("item"), dict):
            if obj["item"].get("type") == "function_call":
                self._call_keys.add(("item", obj["item"].get("id") or obj.get("output_index")))
        elif kind in ("response.completed", "response.incomplete") or obj.get("object") == "response":
            resp = obj.get("response") if isinstance(obj.get("response"), dict) else obj
            if obj.get("object") == "response":  # a whole reply: count what it holds
                for item in resp.get("output") or []:
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        self._call_keys.add(("item", item.get("id")))
            status = resp.get("status")
            details = resp.get("incomplete_details") if isinstance(resp.get("incomplete_details"), dict) else {}
            if status == "incomplete":
                end = _finish_name(details.get("reason")) or "other"
            elif status == "completed":
                end = "tool_calls" if self._call_keys else "stop"
        if end:
            self.finish_reason = end

    def _count_part(self, part: dict):
        content = part.get("content")
        if isinstance(content, str):
            self.text_chars += len(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    self.text_chars += len(p["text"])
        for key in ("reasoning_content", "reasoning"):
            if isinstance(part.get(key), str):
                self.reasoning_chars += len(part[key])
        calls = part.get("tool_calls")
        if isinstance(calls, list):
            for i, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                index = call.get("index")
                if isinstance(index, int) and not isinstance(index, bool):
                    self._call_keys.add(("index", index))  # every delta of one streamed call
                elif call.get("id"):
                    self._call_keys.add(("id", str(call["id"])))
                else:
                    self._call_keys.add(("whole", len(self._call_keys), i))  # a whole call, no id (Ollama)
        if isinstance(part.get("function_call"), dict) and part["function_call"].get("name"):
            self._call_keys.add(("function_call", 0))

    def _count_block(self, block, key):
        if not isinstance(block, dict):
            return
        if block.get("type") in ("tool_use", "server_tool_use"):
            self._call_keys.add(key)
        elif block.get("type") == "text" and isinstance(block.get("text"), str):
            self.text_chars += len(block["text"])
        elif block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
            self.reasoning_chars += len(block["thinking"])

    @staticmethod
    def _is_token(obj: dict) -> bool:
        choices = obj.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            if isinstance(delta, dict):
                return any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls", "function_call"))
            return bool(choices[0].get("text"))
        kind = obj.get("type")
        return isinstance(kind, str) and (kind == "content_block_delta" or kind.endswith(".delta"))

    def finish(self):
        """Read a non-streamed reply's body, once it has all passed."""
        if self._body:
            try:
                self._event(json.loads(bytes(self._body)), 0.0)
            except ValueError:
                pass
            self.first_token = self.last_token = None  # one body: no token timing in it
            self.deltas = 0
            self._body = bytearray()


# How a reply ended, in the words of each API, as the few names the console shows.
# An allow-list: anything else is "other", so a server's own words are never shown.
_FINISH_NAMES = {
    "stop": "stop", "end_turn": "stop", "stop_sequence": "stop", "eos": "stop", "eos_token": "stop",
    "length": "length", "max_tokens": "length", "max_output_tokens": "length",
    "model_context_window_exceeded": "context", "context_length_exceeded": "context",
    "tool_calls": "tool_calls", "function_call": "tool_calls", "tool_use": "tool_calls",
    "content_filter": "filtered", "refusal": "filtered",
}


def _finish_name(value):
    if not isinstance(value, str) or not value.strip():
        return None
    return _FINISH_NAMES.get(value.strip().lower(), "other")


def _reply_end(rec: dict):
    """(short label, long phrase, warn) for how a generation ended, or None when
    there was no reply to judge (an error, or a request that isn't a generation)."""
    if rec.get("cancelled") or rec.get("error"):
        return None
    finish, calls = rec.get("finish"), rec.get("calls") or 0
    said = (rec.get("text_chars") or 0) + (rec.get("reasoning_chars") or 0)
    if finish is None:
        if not (calls or said or rec.get("out")):
            return None
        return "no end", "the reply ended without saying why (cut off?)", True
    if finish == "length":
        return "cut off", "cut off at the token limit", True
    if finish == "context":
        return "ctx full", "the context is full", True
    if finish == "filtered":
        return "filtered", "stopped by a content filter", True
    if calls:
        return f"{calls} call{'s' if calls > 1 else ''}", f"{calls} tool call{'s' if calls > 1 else ''}", False
    if finish == "tool_calls":
        return "0 calls", "said tool_calls but sent none", True
    if finish == "other":
        return "other", "ended for another reason", True
    if not rec.get("text_chars"):
        return "empty", "no answer text" + (", only reasoning" if rec.get("reasoning_chars") else ""), True
    if rec.get("tools"):
        return "no call", "an answer with no tool call, which ends an agent's turn", False
    return "answer", "an answer", False


def _new_agg() -> dict:
    return {"requests": 0, "errors": 0, "prompt": 0, "cached": 0, "out": 0, "cost": 0.0, "saved_s": 0.0, "last": 0.0,
            "healed": 0, "stopped": 0}


def _agg_add(agg: dict, rec: dict):
    agg["requests"] += 1
    agg["errors"] += 1 if rec.get("error") else 0
    for key in ("prompt", "cached", "out", "cost", "saved_s", "healed", "stopped"):
        agg[key] += rec.get(key) or 0
    agg["last"] = rec["time"]


def _agg_group(groups: dict, key: str) -> dict:
    """The group's counters; past _STATS_MAX_KEYS, new names share one "other"."""
    if key not in groups and len(groups) >= _STATS_MAX_KEYS:
        key = "other"
    return groups.setdefault(key, _new_agg())


class _RequestStats:
    """One request on its way through the gateway."""

    def __init__(self, ep: dict, path: str, model: str, client: str, app: str, stream: bool):
        self.ep = ep
        self.ep_name = ep.get("name", "unknown")
        self.path = path
        self.model = _stats_name(model)
        self.client = client
        self.app = app
        self.stream = stream
        self.tools = False          # did the request offer tools (yes or no)
        self.think = None           # what the client asked of thinking (see _thinking_asked)
        self.overflow = None        # the engine said the prompt doesn't fit: {"prompt", "limit"} token counts
        self.body_size = 0          # the request's size in bytes, for the context guard's bytes-per-token
        self.started = time.time()
        self.tap = _ResponseTap()
        self.status = None
        self.failed = False
        self.cancelled = False      # the client left halfway (pressed Stop)
        self.swap_s = None
        self.kv_event = None
        self.heal = None            # what the tool call healer did: counts and a rule id


class GatewayStats:
    """Everything the console shows about traffic. Thread-safe, in memory, with
    the all-time totals in gateway-stats.json beside config.json (gitignored)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.started = time.time()
        self.active = {}
        self.recent = collections.deque(maxlen=200)
        self.session = _new_agg()
        self.by_endpoint = {}
        self.by_model = {}
        self.by_app = {}
        self.samples = {}           # endpoint or "*" → {"ttft", "tok_s", "read"} deques
        self.minutes = collections.deque(maxlen=120)  # [minute, requests, in, out]
        self.life = None            # the all-time totals, loaded on first use
        self.read_rate = {}         # endpoint → recent read speed, tokens/s
        self._dirty = False
        self._last_save = 0.0
        self._next_id = 0
        self._pending = set()       # threads still asking the engine about a request

    # ── all-time totals on disk ─────────────────────────────────────────
    @staticmethod
    def _path() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(CONFIG_FILE)), "gateway-stats.json")

    def _ensure_loaded(self):
        if self.life is not None:
            return
        life = {"since": time.time(), **_new_agg(), "by_endpoint": {}, "by_model": {}, "by_app": {}, "days": {}}
        try:
            with open(self._path(), "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                for key in list(_new_agg()) + ["since"]:
                    if isinstance(saved.get(key), (int, float)):
                        life[key] = saved[key]
                for key in ("by_endpoint", "by_model", "by_app", "days"):
                    if isinstance(saved.get(key), dict):
                        life[key] = {k: {**_new_agg(), **{f: n for f, n in v.items() if f in _new_agg() and isinstance(n, (int, float))}}
                                     for k, v in saved[key].items() if isinstance(v, dict)}
                if isinstance(saved.get("read_rate"), dict):
                    self.read_rate = {k: v for k, v in saved["read_rate"].items() if isinstance(v, (int, float))}
        except (OSError, ValueError, TypeError):
            pass
        self.life = life

    def save(self, force: bool = False):
        with self.lock:
            if self.life is None or not self._dirty:
                return
            if not force and time.time() - self._last_save < _STATS_SAVE_EVERY_S:
                return
            self._last_save = time.time()
            self._dirty = False
            text = json.dumps({**self.life, "read_rate": self.read_rate})
        path = self._path()
        tmp = f"{path}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    # ── a request's life ────────────────────────────────────────────────
    def begin(self, ep: dict, command: str, path: str, data, client: str, headers):
        """Start counting a generation request; None for anything else."""
        try:
            if command != "POST" or not path.rstrip("/").endswith(_STATS_PATHS):
                return None
            data = data if isinstance(data, dict) else {}
            req = _RequestStats(ep, path, data.get("model"), client, _app_name(headers), bool(data.get("stream")))
            req.tools = bool(data.get("tools") or data.get("functions"))  # offered at all: yes or no, nothing else
            req.think = _thinking_asked(data)  # the client's thinking choice: a name from a fixed list
            with self.lock:
                self._next_id += 1
                req.id = self._next_id
                self.active[req.id] = req
            return req
        except Exception:
            return None

    def end(self, req):
        """The request is over. An engine with its own numbers is asked about
        it in the background, so nothing waits for that."""
        if req is None:
            return
        try:
            req.tap.finish()
            ep = req.ep
            kv = ep.get("kv_sessions") if isinstance(ep.get("kv_sessions"), dict) else {}
            if (ep.get("type") == "strata" or kv.get("engine")) and req.status and req.status < 400 and not req.failed:
                worker = threading.Thread(target=self._finish_with_engine, args=(req,), daemon=True)
                with self.lock:
                    self._pending.add(worker)
                worker.start()
            else:
                self._record(req, None)
        except Exception:
            with self.lock:
                self.active.pop(getattr(req, "id", None), None)

    def _finish_with_engine(self, req):
        row = None
        try:
            for attempt in range(2):
                row = _engine_probe.request_row(req.ep, req.model, req.started)
                if row or attempt:
                    break
                time.sleep(0.4)
        except Exception:
            pass
        try:
            self._record(req, row)
        except Exception:
            pass
        finally:
            with self.lock:
                self.active.pop(req.id, None)
                self._pending.discard(threading.current_thread())

    def wait_idle(self, timeout: float = 5.0):
        """For tests: wait for the background engine lookups to finish."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if not self._pending:
                    return True
            time.sleep(0.01)
        return False

    def _record(self, req, row):
        tap = req.tap
        usage, timings = tap.usage, tap.timings
        row = row or {}
        prompt = _number(row.get("prompt_tokens"), usage.get("prompt"))
        if prompt is None and _number(timings.get("prompt_n")) is not None:
            prompt = timings["prompt_n"] + (_number(timings.get("cache_n")) or 0)
        cached = _number(row.get("reused"), usage.get("cached"), timings.get("cache_n"))
        out = _number(row.get("output_tokens"), usage.get("out"), timings.get("predicted_n"))
        estimated = out is None and tap.deltas > 0
        if estimated:
            out = tap.deltas
        ttft = tap.first_token - req.started if tap.first_token else None
        if ttft is None and req.stream and tap.first_byte:
            ttft = tap.first_byte - req.started
        tok_s = _number(row.get("decode_tok_s"), timings.get("predicted_per_second"))
        if tok_s is None and tap.first_token and tap.last_token and tap.last_token - tap.first_token > 0.2 and (out or 0) > 1:
            tok_s = (out - 1) / (tap.last_token - tap.first_token)
        read_n = None
        if row.get("prompt_ms") and _number(row.get("prompt_read")) is not None:
            read_n = row["prompt_read"]
            read_s = row["prompt_ms"] / 1000
        elif timings.get("prompt_ms") and _number(timings.get("prompt_n")) is not None:
            read_n = timings["prompt_n"]
            read_s = timings["prompt_ms"] / 1000
        read_tok_s = read_n / read_s if read_n and read_s > 0 else None
        now = time.time()
        heal = req.heal or {}
        rec = {
            "healed": heal.get("healed", 0), "heal_unknown": heal.get("unknown", 0), "stopped": heal.get("stopped", 0),
            "heal_left": heal.get("left", 0), "heal_rule": heal.get("rule"),
            "time": req.started, "endpoint": req.ep_name, "model": req.model, "app": req.app, "client": req.client,
            "status": req.status, "error": req.failed or not req.status or req.status >= 400,
            "cancelled": req.cancelled, "stream": req.stream, "duration_s": round(now - req.started, 2),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "prompt": prompt, "cached": cached, "out": out, "out_estimated": estimated,
            "tok_s": round(tok_s, 1) if tok_s else None, "read_tok_s": round(read_tok_s, 1) if read_tok_s else None,
            "read": read_n, "cost": usage.get("cost"), "bytes": tap.bytes,
            "swap_s": round(req.swap_s, 2) if req.swap_s else None, "swap": req.kv_event,
            "expert_hit": row.get("hit_rate"), "drafts": row.get("drafts_offered"),
            "drafts_accepted": row.get("drafts_accepted"),
            "finish": tap.finish_reason, "calls": tap.calls, "tools": req.tools, "think": req.think,
            "overflow": bool(req.overflow), "overflow_early": bool((req.overflow or {}).get("early")),
            "window": None, "window_used": None,
            "text_chars": tap.text_chars, "reasoning_chars": tap.reasoning_chars,
        }
        if req.body_size and req.status == 200:
            _ctx_learn(req.ep_name, req.body_size, prompt)
        over = req.overflow or {}
        window = over.get("limit") or _context_window(req.ep_name, req.model)
        held = over.get("prompt") or ((prompt or 0) + (out or 0))
        if window and held:
            rec["window"], rec["window_used"] = window, round(held / window, 3)
        with self.lock:
            self._ensure_loaded()
            self.active.pop(req.id, None)
            if read_tok_s and read_n >= _READ_RATE_MIN_TOKENS:
                old = self.read_rate.get(req.ep_name)
                self.read_rate[req.ep_name] = read_tok_s if old is None else old * 0.7 + read_tok_s * 0.3
            rate = self.read_rate.get(req.ep_name) or _engine_probe.read_speed(req.ep_name)
            rec["saved_s"] = round(cached / rate, 1) if cached and rate else 0.0
            self.recent.appendleft(rec)
            day = time.strftime("%Y-%m-%d", time.localtime(req.started))
            for agg in (self.session, self.life, _agg_group(self.by_endpoint, req.ep_name),
                        _agg_group(self.by_model, req.model), _agg_group(self.by_app, req.app),
                        _agg_group(self.life["by_endpoint"], req.ep_name), _agg_group(self.life["by_model"], req.model),
                        _agg_group(self.life["by_app"], req.app), self.life["days"].setdefault(day, _new_agg())):
                _agg_add(agg, rec)
            for old_day in sorted(self.life["days"])[:-_STATS_DAYS_KEPT]:
                del self.life["days"][old_day]
            for key in ("*", req.ep_name):
                bucket = self.samples.setdefault(key, {"ttft": collections.deque(maxlen=300), "tok_s": collections.deque(maxlen=300),
                                                       "read": collections.deque(maxlen=300)})
                if rec["ttft_s"] is not None:
                    bucket["ttft"].append(rec["ttft_s"])
                if rec["tok_s"]:
                    bucket["tok_s"].append(rec["tok_s"])
                if read_tok_s and read_n >= _READ_RATE_MIN_TOKENS:
                    bucket["read"].append(read_tok_s)
            minute = int(req.started // 60)
            if not self.minutes or self.minutes[-1][0] != minute:
                self.minutes.append([minute, 0, 0, 0])
            self.minutes[-1][1] += 1
            self.minutes[-1][2] += rec["prompt"] or 0
            self.minutes[-1][3] += rec["out"] or 0
            self._dirty = True
        if rec["status"]:  # a request that never got a reply has its own error line already
            _queue_log_line(f"[{req.client}] {_log_line(rec)}")
        self.save()

    # ── what the console reads ──────────────────────────────────────────
    def snapshot(self) -> dict:
        with self.lock:
            self._ensure_loaded()
            now = time.time()
            today = self.life["days"].get(time.strftime("%Y-%m-%d", time.localtime(now)), _new_agg())
            days = sorted(self.life["days"].items())[-14:]
            everything = self.samples.get("*", {})
            top = lambda groups, n=10: dict(sorted(groups.items(), key=lambda kv: -kv[1]["requests"])[:n])
            last_minute = int(now // 60)
            per_minute = {m[0]: m for m in self.minutes}
            return {
                "uptime_s": round(now - self.started),
                "active": [{"endpoint": r.ep_name, "model": r.model, "app": r.app, "elapsed_s": round(now - r.started, 1)}
                           for r in self.active.values()],
                "session": dict(self.session),
                "today": dict(today),
                "all_time": {k: self.life[k] for k in list(_new_agg()) + ["since"]},
                "ttft_p50": _percentile(everything.get("ttft", ()), 50),
                "ttft_p95": _percentile(everything.get("ttft", ()), 95),
                "tok_s_p50": _percentile(everything.get("tok_s", ()), 50),
                "tok_s_best": max(everything.get("tok_s", ()), default=None),
                "read_tok_s_p50": _percentile(everything.get("read", ()), 50),
                "by_endpoint": {name: {**agg,
                                       "ttft_p50": _percentile(self.samples.get(name, {}).get("ttft", ()), 50),
                                       "tok_s_p50": _percentile(self.samples.get(name, {}).get("tok_s", ()), 50)}
                                for name, agg in self.by_endpoint.items()},
                "by_model": top(self.by_model),
                "by_app": top(self.by_app),
                "all_time_by_app": top(self.life["by_app"]),
                # Engine lookups finish in their own threads, so requests can land out of order
                "recent": sorted(self.recent, key=lambda r: r["time"], reverse=True)[:30],
                "per_minute": [{"requests": per_minute.get(m, [0, 0, 0, 0])[1], "in": per_minute.get(m, [0, 0, 0, 0])[2],
                                "out": per_minute.get(m, [0, 0, 0, 0])[3]} for m in range(last_minute - 59, last_minute + 1)],
                "per_day": [{"day": d, "requests": a["requests"], "in": a["prompt"], "out": a["out"]} for d, a in days],
            }


_WINDOW_WARN = 0.9          # from here a chat is close to the model's context limit


def _log_line(rec: dict) -> str:
    """One finished request for the proxy log."""
    head = f"→ {rec['endpoint']} [{rec['status']}]"
    bits = ["stopped by the client"] if rec.get("cancelled") else []
    if rec.get("prompt") is not None:
        cached = rec.get("cached")
        share = f" ({cached * 100 / rec['prompt']:.0f}% cached)" if cached is not None and rec["prompt"] else ""
        bits.append(f"{fmt_count(rec['prompt'])} in{share}")
    if rec.get("out") is not None:
        bits.append(f"{'~' if rec.get('out_estimated') else ''}{fmt_count(rec['out'])} out")
    if rec.get("ttft_s") is not None:
        bits.append(f"first token {fmt_secs(rec['ttft_s'])}")
    if rec.get("tok_s"):
        bits.append(f"{rec['tok_s']:.1f} t/s")
    if rec.get("overflow_early"):
        bits.append(f"context {(rec.get('window_used') or 0) * 100:.0f}% full: the client was told to compact early")
    elif rec.get("overflow"):
        bits.append("context full: the client was told to compact")
    elif (rec.get("window_used") or 0) >= _WINDOW_WARN:
        bits.append(f"context {rec['window_used'] * 100:.0f}% full of {fmt_count(rec.get('window'))}")
    if rec.get("think"):
        bits.append("thinking " + {"none": "off (asked)", "on": "on (asked)"}.get(rec["think"], f"{rec['think']} (asked)"))
    end = _reply_end(rec)
    if end:
        bits.append(f"ended: {end[1]}")
    if rec.get("expert_hit") is not None:
        bits.append(f"experts {rec['expert_hit'] * 100:.0f}% hit")
    if rec.get("swap") == "restored":
        bits.append(f"chat restored, swap {fmt_secs(rec.get('swap_s'))}")
    elif rec.get("swap_s"):
        bits.append(f"saved held chat, {fmt_secs(rec['swap_s'])}")
    if rec.get("healed"):
        unknown = f", {rec['heal_unknown']} with a name the app doesn't have" if rec.get("heal_unknown") else ""
        bits.append(f"healed {rec['healed']} tool call{'' if rec['healed'] == 1 else 's'}{unknown}")
    if rec.get("stopped"):
        bits.append(f"stopped a tool call: rule {rec.get('heal_rule')}")
    if rec.get("heal_left"):
        bits.append(f"tool call left as text{': rule ' + rec['heal_rule'] if rec.get('heal_rule') else ''}")
    return head + ("  " + " · ".join(bits) if bits else "")


def fmt_count(n) -> str:
    if n is None:
        return "—"
    n = float(n)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= limit:
            return f"{n / limit:.1f}{suffix}"
    return f"{n:.0f}"


def fmt_secs(s) -> str:
    if s is None:
        return "—"
    s = float(s)
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60)}s"
    if s < 86400:
        return f"{int(s // 3600)}h {int(s % 3600 // 60)}m"
    return f"{int(s // 86400)}d {int(s % 86400 // 3600)}h"


def fmt_bytes(n) -> str:
    if n is None:
        return "—"
    n = float(n)
    for limit, suffix in ((1024 ** 4, "TB"), (1024 ** 3, "GB"), (1024 ** 2, "MB"), (1024, "KB")):
        if abs(n) >= limit:
            return f"{n / limit:.1f} {suffix}"
    return f"{n:.0f} B"


_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_BAR_PARTS = " ▏▎▍▌▋▊▉█"


def spark(values, width: int = 0, top=None) -> str:
    """A one-line chart. A gap in the data (None) shows as a space."""
    vals = list(values or [])
    if width:
        vals = vals[-width:]
    real = [v for v in vals if isinstance(v, (int, float))]
    if not real:
        return ""
    peak = top if top else max(real)
    if peak <= 0:
        return "".join(" " if not isinstance(v, (int, float)) else _SPARK_CHARS[0] for v in vals)
    out = []
    for v in vals:
        if not isinstance(v, (int, float)):
            out.append(" ")
        else:
            out.append(_SPARK_CHARS[max(0, min(len(_SPARK_CHARS) - 1, int(v / peak * (len(_SPARK_CHARS) - 1) + 0.5)))])
    return "".join(out)


def meter(fraction, width: int = 12) -> str:
    """A bar made of eighths of a cell, padded with a light shade."""
    fraction = max(0.0, min(1.0, float(fraction or 0)))
    eighths = int(fraction * width * 8 + 0.5)
    full, part = eighths // 8, eighths % 8
    text = "█" * full + (_BAR_PARTS[part] if part else "")
    return text + "░" * (width - len(text))


_ENGINE_LIVE_KEYS = ("state", "queued", "phase", "prompt_tokens", "prompt_read", "prompt_total", "generated", "max_tokens",
                     "elapsed_s", "tok_s", "tok_s_mean", "prefill_tok_s_mean")
_ENGINE_ROW_KEYS = ("time", "duration_s", "finish", "prompt_tokens", "reused", "output_tokens", "first_engine_token_s",
                    "prompt_read", "prompt_ms", "decode_ms", "decode_tok_s", "hit_rate", "pcie_share", "drafts_offered", "drafts_accepted")
_ENGINE_TOTAL_KEYS = ("since", "requests", "prompt_tokens", "reused", "output_tokens", "prompt_ms", "decode_ms",
                      "drafts_offered", "drafts_accepted")
_ENGINE_PARK_KEYS = ("enabled", "budget_mib", "slots", "parked", "bytes", "evictions", "parks", "restores")
_ENGINE_INFO_KEYS = ("version", "max_context", "kv", "kv_resident", "expert_slots", "expert_cache_mib", "arena_mib",
                     "vram_free_mib", "pool_workers")
_ENGINE_HISTORY_KEYS = ("tok_s", "gpu_util", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used", "disk_read_mb", "prefill_tok_s_mean")


def _pick(source, keys) -> dict:
    source = source if isinstance(source, dict) else {}
    return {k: source.get(k) for k in keys}


def _engine_view(m: dict) -> dict:
    """The parts of an engine's /metrics the console uses, by allow-list: numbers
    and status words, nothing a client wrote."""
    static = m.get("hardware_static") if isinstance(m.get("hardware_static"), dict) else {}
    engine = m.get("engine") if isinstance(m.get("engine"), dict) else {}
    return {
        "model": _stats_name(engine.get("model"), ""),
        "info": _pick(engine, _ENGINE_INFO_KEYS),
        "live": _pick(m.get("live"), _ENGINE_LIVE_KEYS),
        "totals": _pick(m.get("totals"), _ENGINE_TOTAL_KEYS),
        "parking": _pick(m.get("conversation_cache"), _ENGINE_PARK_KEYS),
        "hardware": dict(m["hardware"]) if isinstance(m.get("hardware"), dict) else {},
        "gpu_name": _stats_name(static.get("gpu_name"), ""),
        "cpu_name": _stats_name(static.get("cpu_name"), ""),
        "history": _pick(m.get("history"), _ENGINE_HISTORY_KEYS),
        "requests": [_pick(r, _ENGINE_ROW_KEYS) for r in (m.get("requests") or []) if isinstance(r, dict)],
    }


class EngineProbe:
    """Polls the /metrics of engines that have one (Strata), only while someone
    is looking, and answers from memory so the console never waits on it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}          # endpoint name → {"ok", "at", ...view}
        self.last_read = 0.0
        self.thread = None

    @staticmethod
    def _target(ep: dict, model: str = ""):
        """(host, key) of the engine's own server for this endpoint, or None."""
        kv = ep.get("kv_sessions") if isinstance(ep.get("kv_sessions"), dict) else {}
        engine = str(kv.get("engine") or "").strip()
        if engine:
            return engine, kv.get("engine_api_key") or ep.get("api_key", "")
        if ep.get("type") == "strata":
            found = _kv_find_strata_instance(model)
            if found:
                return found[0], kv.get("engine_api_key") or found[1]
        return None

    @staticmethod
    def _get_metrics(target) -> dict:
        host, key = target
        conn = http.client.HTTPConnection(_dial_host(host), timeout=3)
        try:
            conn.request("GET", "/metrics", headers={"Authorization": f"Bearer {key}"} if key else {})
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status}")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or "engine" not in parsed:
                raise ValueError("no engine metrics here")
            return parsed
        finally:
            conn.close()

    def _poll_one(self, ep: dict, model: str = ""):
        name = ep.get("name", "")
        target = self._target(ep, model)
        if not target:
            return None
        try:
            view = {"ok": True, "at": time.time(), **_engine_view(self._get_metrics(target))}
        except Exception as e:
            view = {"ok": False, "at": time.time(), "error": str(e)[:120]}
        with self.lock:
            self.data[name] = view
        return view

    def _loop(self):
        while time.time() - self.last_read < _ENGINE_IDLE_STOP_S:
            for ep in list(get_config("endpoints") or []):
                if ep.get("enabled") and get_endpoint_status(ep.get("name", "")) == "online":
                    self._poll_one(ep)
            time.sleep(_ENGINE_POLL_S)
        with self.lock:
            self.thread = None

    def read(self) -> dict:
        """The latest numbers per endpoint; starts polling if it had stopped."""
        self.last_read = time.time()
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, name="engine-probe", daemon=True)
                self.thread.start()
            return dict(self.data)

    def read_speed(self, name: str):
        """The engine's overall prompt reading speed since it loaded, tokens/s."""
        totals = (self.data.get(name) or {}).get("totals") or {}
        read = (totals.get("prompt_tokens") or 0) - (totals.get("reused") or 0)
        if read >= _READ_RATE_MIN_TOKENS and totals.get("prompt_ms"):
            return read / (totals["prompt_ms"] / 1000)
        return None

    def request_row(self, ep: dict, model: str, started: float):
        """The engine's own numbers for the request that started at `started`."""
        view = self._poll_one(ep, model)
        if not view or not view.get("ok"):
            return None
        for row in view.get("requests", []):  # newest first
            if isinstance(row.get("time"), (int, float)) and row["time"] >= started - 1:
                return row
        return None


_stats = GatewayStats()
_engine_probe = EngineProbe()
atexit.register(lambda: _stats.save(force=True))


def stats_snapshot() -> dict:
    """Everything the console and GET /stats show. Fast: reads memory only."""
    return {"gateway": _stats.snapshot(), "engines": _engine_probe.read(), "kv_sessions": kv_status_all()}


# ═══════════════════════════════════════════════════════════════════════════
#  TOOL CALL HEALING — gives damaged tool calls their shape back, and stops
#  the ones that could do damage because of an error
# ═══════════════════════════════════════════════════════════════════════════
#
# Off unless an endpoint has "tool_heal": {"enabled": true}, and then only for
# /chat/completions requests that offer tools. Replies are read in memory and
# nothing about them is kept: the log and stats get counts and rule ids only.
#
# The guard is for accidents (bad escaping, cut-off replies, a lost path), not a
# security boundary: a command can always hide what it does. Client approval
# settings stay the real protection.

_HEAL_MAX_HOLD = 8 * 1024 * 1024    # past this, a held reply is passed on as it came
_HEAL_KEEPALIVE_S = 10.0            # an idle client may give up while a long call is held
_TOOL_TRIGGERS = ("<tool_call>", "<function=")
_FENCES = ("```", "~~~")
_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_PARAM_CLOSE_RE = re.compile(r"</parameter>\s*(?=<parameter=|</function>)")
_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?\Z")

# Tools that run or write something: their calls are always checked.
_GUARDED_TOOLS = {"run_commands", "run_command", "execute_command", "bash", "shell", "powershell",
                  "terminal", "editor", "edit", "write_to_file", "replace_in_file", "apply_patch"}
_COMMAND_KEYS = ("commands", "command", "cmd", "script")

_GUARD_REASONS = {
    "A1": "its arguments are broken, and repairing them could change what it does",
    "T1": "the reply was cut off, so the call may be incomplete",
    "S01": "it deletes a path that is only a variable, which may be empty",
    "S02": "it deletes a root, home or whole working folder",
    "S03": "its path has lost its backslashes",
    "S04": "it uses a PowerShell variable that was never set, so the path starts at the drive root",
    "S05": "it deletes a drive, system or top-level folder",
    "S06": "it deletes everything a recursive listing finds, with no filter",
    "S09": "its redirect would empty a file",
    "S10": "its path contains a control character, usually a damaged \\n or \\t",
    "S11": "it throws away uncommitted work",
    "S12": "it rewrites or deletes git history that may not be recoverable",
    "S13": "it deletes an unfiltered list of files",
    "S14": "it writes directly to a disk",
    "S15": "it changes permissions on a root folder",
    "S16": "it shuts the machine down or deletes registry keys",
    "S17": "the real command is hidden, so it can't be checked",
    "S18": "its path climbs out of the working folder",
    "S19": "an unquoted space seems to have split its path",
    "S20": "it deletes an empty path",
    "R1": "the same call was already made several times in a row",
    "R2": "the reply made the same call many times over",
}
_STOP_NAMES = {"A1": "broken_arguments_send_valid_json", "T1": "reply_was_cut_off_send_a_shorter_call",
               "R1": "same_call_repeated_in_a_row_use_its_result_or_change_the_call",
               "R2": "same_call_many_times_in_one_reply_send_it_once"}
_REPEAT_LIMIT = 3           # identical calls in a row that may run; Cline 4.1.26 ends the task at 5


def _call_key(name, arguments) -> str:
    """A call as a hash, for comparing only: the name and its arguments, with JSON spacing and key
    order left out. Nothing of the call is kept."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = arguments.strip()
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(f"{name}\0{arguments}".encode("utf-8", "replace")).hexdigest()


def _repeat_run(messages):
    """(the calls of the chat's last assistant turn as hashes, how many turns in a row made exactly
    those calls). Tool results are stepped over; anything else between two turns (the user spoke, the
    model answered in words) ends the run."""
    run, count = None, 0
    for msg in reversed(messages if isinstance(messages, list) else []):
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "tool":
            continue
        calls = msg.get("tool_calls") if role == "assistant" else None
        if not isinstance(calls, list) or not calls:
            break
        keys = tuple(_call_key(str((c.get("function") or {}).get("name") or ""), (c.get("function") or {}).get("arguments", ""))
                     for c in calls if isinstance(c, dict))
        if run is None:
            run = keys
        elif keys != run:
            break
        count += 1
    return run, count

_SHELL_WRAPPERS = {"sudo", "command", "builtin", "exec", "time", "nohup", "nice", "env", "&", "call", "start", "timeout", "npx", "bunx",
                   "do", "then", "else", "elif", "if", "while", "until", "!"}
_WRAPPER_VALUE_FLAGS = {"nice": {"-n"}, "sudo": {"-u", "-g", "-p"}, "timeout": {"-k", "-s"}, "env": {"-u"}}
_DELETE_VERBS = {"rm", "rmdir", "rd", "del", "erase", "ri", "remove-item", "unlink", "shred", "rimraf"}
_CMD_DELETE_VERBS = {"rd", "rmdir", "del", "erase"}
_DISK_VERBS = {"fdisk", "parted", "wipefs", "sgdisk", "diskpart", "clear-disk", "format-volume", "initialize-disk"}
_SYSTEM_VERBS = {"shutdown", "reboot", "halt", "poweroff", "stop-computer", "restart-computer"}
_LISTING_VERBS = {"get-childitem", "gci", "dir", "ls"}
_POSIX_SYSTEM_DIRS = ("etc", "bin", "sbin", "lib", "lib64", "boot", "sys", "proc", "dev")     # at any depth
_POSIX_SHALLOW_DIRS = ("usr", "var", "opt", "root", "home", "users")                         # the top two levels
_HOME_CACHES = (".npm", ".cache", ".pnpm-store", ".yarn", ".nuget", ".gradle")
_TEMP_PATH_RE = re.compile(r"(/tmp/|/var/tmp/|\$\{?tmpdir|\$env:te?mp|%te?mp%|/dev/null$|nul$)", re.I)
_HEREDOC_RE = re.compile(r"<<-?[ \t]*(['\"]?)([\w.-]+)\1")
_HERE_STRING_RE = re.compile(r"@'[ \t]*\r?\n.*?\n'@|@\"[ \t]*\r?\n.*?\n\"@", re.S)
_WINDOWS_SYSTEM_DIRS = ("windows", "program files", "program files (x86)", "programdata")
_PS_SAFE_VARS = {"_", "psitem", "home", "pwd", "pshome", "true", "false", "null"}
_WIDE_TARGETS = {".", "./", "..", "../", "*", ".*", "*.*", "**", "./*", "./.*", "../*", "~", "~/", "~/*", "/", "/*", "/.*",
                 "$home", "${home}", "$home/", "$home/*", "$pwd", "${pwd}", "$(pwd)", "$env:userprofile", "$env:home",
                 "%userprofile%", "%homepath%", "%cd%", "%systemroot%", "%windir%", "%programfiles%", "$env:systemroot",
                 "$env:windir", "$env:programfiles"}


def _split_shell(cmd: str, shell: str):
    """A command line as the shell would read it: a list of simple commands, each
    {"words": [(text, quoted, lost_backslash)], "redirects": [...], "piped": whether
    the next command reads this one's output, "nested": commands inside $(...)}."""
    esc = "`" if shell == "powershell" else "^" if shell == "cmd" else "\\"
    segments, nested = [], []
    state = {"words": [], "redirects": [], "cur": None, "quoted": False, "lost": False, "redirect": False, "glued": None}

    def end_word():
        if state["cur"] is None:
            return
        text = "".join(state["cur"])
        if state["redirect"]:
            state["redirects"].append(text)
            state["redirect"] = False
        else:
            state["words"].append((text, state["quoted"], state["lost"]))
        state.update(cur=None, quoted=False, lost=False)

    def end_segment(piped=False):
        end_word()
        if state["words"] or state["redirects"]:
            segments.append({"words": state["words"], "redirects": state["redirects"], "piped": piped, "glued": state["glued"]})
        state.update(words=[], redirects=[], redirect=False, glued=None)

    def add(text):
        if state["cur"] is None:
            state["cur"] = []
        state["cur"].append(text)

    i, n, quote = 0, len(cmd), None
    while i < n:
        c = cmd[i]
        if quote == "'":
            if c == "'":
                if shell == "powershell" and cmd[i + 1:i + 2] == "'":
                    add("'")
                    i += 2
                    continue
                quote = None
            else:
                add(c)
            i += 1
            continue
        if quote == '"':
            if c == '"':
                quote = None
            elif c == esc and shell != "cmd" and i + 1 < n and (shell == "powershell" or cmd[i + 1] in '"$`\\\n'):
                add(cmd[i + 1])
                i += 1
            elif c == "$" and cmd[i + 1:i + 2] == "(":
                end = _matching_paren(cmd, i + 1)
                nested.append(cmd[i + 2:end])
                add(cmd[i:end + 1])
                i = end
            else:
                add(c)
            i += 1
            continue
        if c == '"' or (c == "'" and shell != "cmd"):
            quote = c
            state["quoted"] = True
            add("")
            i += 1
            continue
        if c == esc and i + 1 < n:
            if cmd[i + 1] in "\r\n":    # a line continuation
                i += 2
                continue
            if shell == "posix" and cmd[i + 1].isalnum():
                state["lost"] = True    # bash drops this backslash: C:\Users becomes C:Users
            add(cmd[i + 1])
            i += 2
            continue
        if c == "#" and shell != "cmd" and state["cur"] is None:
            end = cmd.find("\n", i)
            i = n if end < 0 else end
            continue
        if c in " \t":
            # A tab or line break right against a word may be a \t or \n that JSON decoded out of a path
            if c == "\t" and state["cur"] is not None and not cmd[i + 1:i + 2].isspace() and i + 1 < n:
                state["glued"] = "tab"
            end_word()
        elif c in "\r\n":
            if state["cur"] is not None and not state["quoted"] and cmd[i:i + 2] != "\r\n" and cmd[i + 1:].strip():
                state["glued"] = "line"
            end_segment()
        elif c == ";" and shell != "cmd":
            end_segment()
        elif c == "{" and cmd[i + 1:i + 2] == "}":
            add("{}")
            i += 1
        elif c == "&":
            if shell == "posix" and cmd[i + 1:i + 2] == ">":    # &> file: both streams into a file
                end_word()
                i += 2 if cmd[i + 2:i + 3] == ">" else 1
                state["redirect"] = True
            elif cmd[i + 1:i + 2] == "&":
                end_segment()
                i += 1
            elif shell == "powershell" and state["cur"] is None and not state["words"]:
                state["words"].append(("&", False, False))
            else:
                end_segment()
        elif c == "|":
            if cmd[i + 1:i + 2] == "|":
                end_segment()
                i += 1
            else:
                end_segment(piped=True)
        elif c in "(){}" and shell != "cmd":
            end_segment()
        elif c == "$" and shell != "cmd" and cmd[i + 1:i + 2] == "(":
            end = _matching_paren(cmd, i + 1)
            nested.append(cmd[i + 2:end])
            add(cmd[i:end + 1])
            i = end
        elif c == "$" and shell != "cmd" and cmd[i + 1:i + 2] == "{":
            end = cmd.find("}", i)
            end = n - 1 if end < 0 else end
            add(cmd[i:end + 1])
            i = end
        elif c == ">":
            if state["cur"] is not None and "".join(state["cur"]) in ("1", "2", "*"):
                state["cur"] = None
            end_word()
            if cmd[i + 1:i + 2] == ">":
                i += 1
            if cmd[i + 1:i + 2] == "&":     # 2>&1
                i += 2
                while i + 1 < n and cmd[i + 1].isdigit():
                    i += 1
            else:
                state["redirect"] = True
        else:
            add(c)
        i += 1
    end_segment()
    return segments, nested


def _matching_paren(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _verb_of(words: list):
    """(the program's name in lower case, its arguments), past env assignments and wrappers."""
    i = 0
    while i < len(words):
        text, quoted, _lost = words[i]
        if not quoted and re.match(r"[A-Za-z_]\w*=", text):
            i += 1
            continue
        name = re.split(r"[\\/]", text)[-1].lower()
        name = re.sub(r"\.(exe|cmd|bat|com)$", "", name)
        if name in _SHELL_WRAPPERS and not quoted and i + 1 < len(words):
            i += 1
            while i < len(words) and name != "&" and not words[i][1]:
                word = words[i][0]
                if word.startswith("-"):
                    i += 2 if word in _WRAPPER_VALUE_FLAGS.get(name, ()) else 1
                elif name in ("nice", "timeout") and re.fullmatch(r"\d+(\.\d+)?[smhd]?", word):
                    i += 1
                else:
                    break
            continue
        return name, words[i + 1:]
    return "", []


def _assigned_vars(cmd: str, shell: str) -> set:
    """Variables this command sets itself. Each command runs in its own shell, so these
    are the only ordinary ones that can hold a value."""
    if shell == "powershell":
        found = re.findall(r"\$(\w+)\s*=[^=]", cmd) + re.findall(r"foreach\s*\(\s*\$(\w+)", cmd, re.I)
    elif shell == "cmd":
        found = re.findall(r"\bset\s+\"?(\w+)=", cmd, re.I) + re.findall(r"\bfor\s+%%?(\w)\b", cmd, re.I)
    else:
        found = re.findall(r"(?:^|[;&|\s(])(\w+)=", cmd) + re.findall(r"\bfor\s+(\w+)\s+in\b", cmd) + re.findall(r"\bread\s+(?:-\w+\s+)*(\w+)", cmd)
    return {name.lower() for name in found}


def _bad_target(target, recursive: bool, shell: str, assigned: set):
    """Why deleting this path is an accident waiting to happen, as a rule id, or None."""
    text, _quoted, lost = target
    if any(ord(ch) < 32 for ch in text):
        return "S10"
    s = text.strip()
    if not s:
        return "S20"
    if lost:
        return "S03"
    if shell == "powershell":
        s = re.sub(r"^\$\(\$(\w+)\)", r"$\1", s)        # "$($root)\build" is $root\build
    low = s.lower().replace("\\", "/")
    low = re.sub(r"^/mnt/([a-z])(/|$)", r"\1:/", low)   # WSL's view of a Windows drive
    if low in _WIDE_TARGETS or re.fullmatch(r"/+\*?", low):
        return "S02"
    if re.match(r"hk(lm|cu|cr|u|cc):", low):
        return "S16"
    if re.match(r"[a-z]:/*(\*|\*\.\*|\.\*)?\Z", low):
        return "S05"
    if re.match(r"[a-z]:[^/]", low):
        return "S03"
    pattern = {"powershell": r"\$\{?(env:)?([A-Za-z_]\w*)\}?(.*)\Z",
               "cmd": r"%()([A-Za-z_]\w*)%(.*)\Z"}.get(shell, r"\$\{?()([A-Za-z_]\w*|[0-9@*])(?::?[-=+?][^}]*)?\}?(.*)\Z")
    var = re.match(pattern, s, re.S)
    if var:
        name, rest = var.group(2).lower(), var.group(3)
        if name in assigned or (shell == "powershell" and name in _PS_SAFE_VARS):
            return None
        if rest in ("", "/", "\\", "/*", "\\*", "/.", "\\.", "/.*", "\\.*"):
            return "S01"
        if shell == "powershell" and not var.group(1):
            return "S04"
    if not recursive:
        return None
    parts = [p for p in low.split("/") if p]
    if parts and parts[-1] == ".." or parts.count("..") >= 2:
        return "S18"
    meaningful = [p for p in parts if p.strip("*.") and p != "~"]
    if low.startswith("~/") and len(meaningful) <= 1 and parts[-1] not in _HOME_CACHES:
        return "S02"
    if low.startswith("/") and not low.startswith("//"):
        if len(meaningful) <= 1 or meaningful[0] in _POSIX_SYSTEM_DIRS or (meaningful[0] in _POSIX_SHALLOW_DIRS and len(meaningful) <= 2):
            return "S02"
    drive = re.match(r"[a-z]:/(.*)\Z", low)
    if drive:
        inner = [p for p in drive.group(1).split("/") if p.strip("*.")]
        if len(inner) <= 1 or inner[0] in _WINDOWS_SYSTEM_DIRS or (inner[0] == "users" and len(inner) <= 2):
            return "S05"
    return None


def _check_delete(verb: str, args: list, shell: str, assigned: set):
    """The rule a delete command breaks, or None."""
    targets, recursive, filtered, skip = [], verb == "rimraf", False, False
    for word in args:
        if skip:
            skip = False
            continue
        text, quoted, _lost = word
        low = text.lower()
        if not quoted and low.startswith("-") and len(low) > 1:
            if low == "-whatif":
                return None
            if low in ("-path", "-literalpath", "-lp"):
                continue
            if low in ("-filter", "-include", "-exclude"):
                filtered, skip = True, True
                continue
            if low == "--recursive" or low.startswith("-rec") or (re.fullmatch(r"-[a-z]{1,4}", low) and "r" in low and not low.startswith("-fo")):
                recursive = True
            continue
        if not quoted and shell != "posix" and verb in _CMD_DELETE_VERBS and re.fullmatch(r"/[a-z](:\S*)?", low):
            recursive = recursive or low.startswith("/s")
            continue
        parts = [p for p in text.split(",") if p] if shell == "powershell" and not quoted else []
        targets.extend((part, quoted, word[2]) for part in parts or [text])
    for target in targets:
        rule = _bad_target(target, recursive, shell, assigned)
        if rule and not (filtered and rule in ("S02", "S01")):
            return rule
    if recursive and len(targets) >= 2 and not targets[0][1]:
        # C:\work\My Project read as two paths. On Linux, /tmp/a dist is a normal list.
        first = targets[0][0]
        windows = bool(re.match(r"[A-Za-z]:[\\/]|\\", first))
        if windows or first.startswith("/"):
            for text, quoted, _lost in targets[1:]:
                relative = not quoted and not re.match(r"([A-Za-z]:)?[\\/]|\.", text) and "*" not in text
                if relative and (windows or re.search(r"[\\/]", text)):
                    return "S19"
    return None


def _check_git(args: list):
    words = [w[0] for w in args]
    i = 0
    while i < len(words) and words[i].startswith("-"):
        i += 2 if words[i] in ("-c", "-C") else 1
    if i >= len(words):
        return None
    sub, rest = words[i].lower(), words[i + 1:]
    low = [w.lower() for w in rest]
    if sub == "reset" and "--hard" in low:
        return "S11"
    if sub == "checkout" and ("." in rest or "*" in rest) and (low[:1] == ["."] or "--" in low):
        return "S11"
    if sub == "restore" and ("." in rest or "*" in rest) and "--staged" not in low and "-s" not in low:
        return "S11"
    if sub == "clean":
        flags = [w for w in rest if w.startswith("-")]
        dry = any(w == "--dry-run" or re.fullmatch(r"-[a-zA-Z]*n[a-zA-Z]*", w) for w in flags)
        force = any(w == "--force" or re.fullmatch(r"-[a-zA-Z]*f[a-zA-Z]*", w) for w in flags)
        ignored_only = any(re.fullmatch(r"-[a-zA-Z]*X[a-zA-Z]*", w) for w in flags)     # build output, easily made again
        named = any(not w.startswith("-") and w not in (".", "*") for w in rest)
        if force and not dry and not ignored_only and not named:
            return "S11"
    if sub == "rm" and any(w in (".", "*") for w in rest) and "--cached" not in low:
        return "S11"
    if sub == "push":
        forced = any(w == "--force" or w == "-f" or re.fullmatch(r"-[a-z]*f[a-z]*", w) for w in low)
        names = [w for w in low if not w.startswith("-")]
        if forced and (len(names) < 2 or any(re.search(r"(^|[:+])(main|master)$", w) for w in names[1:])):
            return "S12"
        if any(w.startswith("+") and re.search(r"(main|master)$", w) for w in names[1:]):
            return "S12"
    if sub == "branch" and "-D" in rest:
        return "S12"
    if sub == "stash" and low[:1] in (["drop"], ["clear"]):
        return "S12"
    return None


def _check_find(args: list, assigned: set):
    words = [w[0] for w in args]
    low = [w.lower() for w in words]
    deletes = "-delete" in low
    for flag in ("-exec", "-execdir", "-ok"):
        if flag in low:
            after = low[low.index(flag) + 1:low.index(flag) + 2]
            deletes = deletes or (after and re.split(r"[\\/]", after[0])[-1] in _DELETE_VERBS)
    if not deletes:
        return None
    start = next((w for w in args if not w[0].startswith("-")), None)
    if start and start[0] not in (".", "./", ".\\"):     # a named folder is as scoped as rm -rf build
        return "S13" if _bad_target(start, True, "posix", assigned) else None
    filters = ("-name", "-iname", "-path", "-ipath", "-wholename", "-regex", "-iregex", "-newer", "-mtime", "-mmin", "-size", "-user")
    return None if any(f in low for f in filters) else "S13"


def _check_simple(segment: dict, shell: str, assigned: set, depth: int):
    """One simple command against the rules, or None."""
    words, redirects = segment["words"], segment["redirects"]
    for target in redirects:
        if any(ord(ch) < 32 for ch in target):
            return "S10"
        if re.match(r"/dev/(sd|hd|nvme|vd|xvd|disk|mmcblk)", target.lower()):
            return "S14"
    lasting = [target for target in redirects if not _TEMP_PATH_RE.search(target)]
    if not words:
        return "S09" if lasting else None       # "> file" alone empties it
    verb, args = _verb_of(words)
    low = [a[0].lower() for a in args]
    if redirects and verb and verb not in ("echo", "printf") and any(a[0] == target for a in args for target in redirects):
        return "S09"                            # sort a > a: the shell empties a first
    if lasting and (verb in (":", "true") or (verb == "type" and low[:1] == ["nul"]) or (verb == "cat" and low[:1] == ["/dev/null"])):
        return "S09"
    if verb in _DELETE_VERBS:
        rule = _check_delete(verb, args, shell, assigned)
        if rule:
            return rule
        glued = segment.get("glued")
        ends_on_path = args and (args[-1][1] or not args[-1][0].startswith(("-", "/" if shell != "posix" else "-")))
        # In bash a line ending in a path is a normal script, so only a tab says damage there.
        if ends_on_path and (glued == "tab" or (glued == "line" and shell != "posix")):
            return "S10"
        return None
    if verb == "git":
        return _check_git(args)
    if verb == "find":
        return _check_find(args, assigned)
    helping = bool({"--help", "/?"} & set(low))     # not -h: that means halt to shutdown
    if (verb.startswith("mkfs") or verb in _DISK_VERBS) and not helping:
        return "S14"
    if verb == "format" and any(re.fullmatch(r"[a-z]:", w) for w in low):
        return "S14"
    if verb == "dd" and any(re.match(r"of=/dev/(?!null\b|zero\b|stdout\b|stderr\b)", w) for w in low):
        return "S14"
    if verb in ("chmod", "chown", "icacls", "takeown") and any(w in ("-r", "--recursive", "/t", "/r") or re.fullmatch(r"-[a-z]*r[a-z]*", w) for w in low):
        for arg in args:
            if not arg[0].startswith("-") and _bad_target(arg, True, shell, assigned) in ("S02", "S05", "S18"):
                return "S15"
    if verb == "shutdown" and (helping or {"/a", "-a", "-c"} & set(low)):      # cancels one
        return None
    if verb in _SYSTEM_VERBS or (verb == "init" and low[:1] in (["0"], ["6"])):
        return "S16"
    if verb == "reg" and low[:1] == ["delete"]:
        return "S16"
    if verb == "remove-itemproperty" and any(w.startswith(("hklm:", "hkcu:")) for w in low):
        return "S16"
    if verb in ("iex", "invoke-expression"):
        return "S17"
    if words[0][0] == "&" and len(words) > 1 and words[1][0].startswith("$") and words[1][0][1:].lower() not in assigned:
        return "S17"
    if verb == "eval" and len(args) == 1 and re.fullmatch(r"\$\{?\w+\}?", args[0][0]):
        return "S17"
    if verb == "wsl":
        i = 0
        while i < len(args) and args[i][0].startswith("-"):
            flag = args[i][0].lower()
            if flag in ("--", "-e", "--exec"):
                i += 1
                break
            i += 2 if flag in ("-d", "--distribution", "-u", "--user", "--cd") else 1
        return _guard_command(" ".join(a[0] for a in args[i:]), "posix", depth + 1)
    if verb in ("bash", "sh", "zsh", "dash", "ksh", "powershell", "pwsh", "cmd"):
        inner_shell = "powershell" if verb in ("powershell", "pwsh") else "cmd" if verb == "cmd" else "posix"
        if inner_shell == "powershell" and any(w == "-ec" or (w.startswith(("-e", "/e")) and "encodedcommand".startswith(w[1:])) for w in low):
            return "S17"
        flag = next((i for i, w in enumerate(low) if w in ("-c", "-command", "/c", "/k") or (inner_shell == "posix" and re.fullmatch(r"-[a-z]*c", w))), None)
        if flag is not None and flag + 1 < len(args):
            inner = " ".join(a[0] for a in args[flag + 1:])
            if inner.startswith("$(") and inner_shell == "posix":
                return "S17"
            return _guard_command(inner, inner_shell, depth + 1)
    return None


def _guard_command(cmd: str, shell: str, depth: int = 0):
    """The rule this command line breaks, or None."""
    if depth > 3 or not isinstance(cmd, str) or not cmd.strip():
        return None
    fed_to_shells = []
    if shell == "posix":
        cmd, fed_to_shells = _strip_heredocs(cmd)
    elif shell == "powershell":
        cmd = _HERE_STRING_RE.sub("''", cmd)        # here-strings are text
    # A JSON \t or \r inside a Windows path decodes to a control character.
    if re.search(r"[A-Za-z]:[\t\r\f\b\v]|[A-Za-z]:[\\/][^\s\"']*[\t\r\f\b\v]\S", cmd):
        return "S10"
    if re.search(r"(?:^|[\s;&|])(?:bash|sh|zsh|dash|source|\.)\s+<\(\s*(?:curl|wget)\b", cmd):
        return "S17"
    segments, nested = _split_shell(cmd, shell)
    assigned = _assigned_vars(cmd, shell)
    for inner in nested + fed_to_shells:
        rule = _guard_command(inner, shell, depth + 1)
        if rule:
            return rule
    pipeline = []
    for segment in segments:
        rule = _check_simple(segment, shell, assigned, depth)
        if rule:
            return rule
        verb, args = _verb_of(segment["words"])
        if pipeline and verb in ("sh", "bash", "zsh", "dash"):
            if any(v in ("curl", "wget", "iwr", "irm", "invoke-webrequest", "invoke-restmethod") for v, _a in pipeline):
                return "S17"
        if pipeline and verb == "xargs":
            target = next((a[0].lower() for a in args if not a[0].startswith("-")), "")
            filtered = any(v in ("grep", "rg", "findstr") or (v == "find" and "-name" in [w[0].lower() for w in a]) for v, a in pipeline)
            if re.split(r"[\\/]", target)[-1] in _DELETE_VERBS and not filtered:
                return "S13"
        if pipeline and verb in ("set-content", "sc", "out-file", "tee", "tee-object"):     # Get-Content a | Set-Content a
            read = {a[0].lower() for v, stage in pipeline if v in ("get-content", "gc", "cat", "type") for a in stage}
            if any(a[0].lower() in read for a in args if not a[0].startswith("-")):
                return "S09"
        pipeline = pipeline + [(verb, args)] if segment["piped"] else []
    return _check_listing_delete(segments)


def _strip_heredocs(cmd: str):
    """(the command without its heredoc bodies, the bodies a shell will run). A body is
    text unless it feeds a shell: cat <<EOF prints it, bash <<EOF runs it."""
    if "<<" not in cmd:
        return cmd, []
    lines, out, run, i = cmd.split("\n"), [], [], 0
    while i < len(lines):
        line = lines[i]
        i += 1
        match = _HEREDOC_RE.search(line)
        if not match or line[:match.start()].count('"') % 2 or line[:match.start()].count("'") % 2:
            out.append(line)
            continue
        body = []
        while i < len(lines) and lines[i].strip() != match.group(2):
            body.append(lines[i])
            i += 1
        i += 1
        out.append(line[:match.start()] + line[match.end():])
        if re.search(r"(?:^|[|;&\s])(?:bash|sh|zsh|dash)\b", line[:match.start()]):
            run.append("\n".join(body))
    return "\n".join(out), run


def _check_listing_delete(segments: list):
    """Get-ChildItem -Recurse | Remove-Item, with nothing that narrows the list."""
    listed = False
    for segment in segments:
        verb, args = _verb_of(segment["words"])
        low = [a[0].lower() for a in args]
        if verb in ("where-object", "where", "?"):
            return None
        if verb in _LISTING_VERBS and any(w.startswith("-rec") for w in low):
            filtered = any(w.startswith(("-filt", "-inc", "-ex")) for w in low)
            named = any(not w.startswith("-") and w.replace("\\", "/") not in (".", "./", "*", "./*") for w in low)
            listed = listed or not (filtered or named)
        elif listed and verb in _DELETE_VERBS:
            targets = [w for w in low if not w.startswith("-")]
            if not targets or targets[0] in ("$_", "$psitem", "$_.fullname", "$psitem.fullname"):
                return "S06"
    return None


def _commands_in(args) -> list:
    """Every command in a tool's arguments, in the shapes clients accept."""
    found = []
    if isinstance(args, str):
        return [args]
    if isinstance(args, list):
        for item in args:
            found.extend(_commands_in(item))
        return found
    if isinstance(args, dict):
        if isinstance(args.get("command"), str) and isinstance(args.get("args"), list):
            quoted = [f'"{a}"' if isinstance(a, str) and (" " in a or not a) else str(a) for a in args["args"]]
            return [" ".join([args["command"]] + quoted)]
        for key in _COMMAND_KEYS:
            if key in args:
                found.extend(_commands_in(args[key]))
    return found


def _paths_in(args):
    if not isinstance(args, dict):
        return
    for key in ("path", "file_path", "target_file"):
        if isinstance(args.get(key), str):
            yield args[key]
    for key in ("files", "edits", "paths", "file_paths"):
        items = args.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    yield item
                else:
                    yield from _paths_in(item)


def guard_tool_call(name: str, arguments: str, shell: str, allow=()):
    """(rule id, the command it is about), or (None, None) when the call may go ahead."""
    guarded = (name or "").lower() in _GUARDED_TOOLS
    if isinstance(arguments, str) and not arguments.strip():
        return None, None       # nothing to run, and nothing a repair could invent
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        # A client's lenient repair can turn C:\Users into C:Users, or run half a batch.
        hits = guarded or any(f'"{key}"' in (arguments or "") for key in _COMMAND_KEYS)
        return ("A1", None) if hits and "A1" not in allow else (None, None)
    tool_shell = {"bash": "posix", "powershell": "powershell"}.get((name or "").lower(), shell)
    commands = [args] if isinstance(args, str) and guarded else _commands_in(args) if isinstance(args, dict) else _commands_in(args) if guarded else []
    for command in commands:
        rule = _guard_command(command, tool_shell)
        if rule and rule not in allow:
            return rule, command
    if "S10" not in allow:
        for path in _paths_in(args):
            if any(ord(ch) < 32 for ch in path):
                return "S10", path
    return None, None


def _parse_tool_blocks(text: str, tools=None):
    """Tool calls written out as text: ([{name, arguments}], any prose after them), or
    None unless every call is complete. Parameter bytes are never edited."""
    calls, pos, n = [], 0, len(text)
    tools = tools or {}

    def skip(at):
        while at < n and text[at].isspace():
            at += 1
        return at

    while True:
        pos = skip(pos)
        wrapped = text.startswith("<tool_call>", pos)
        if wrapped:
            pos = skip(pos + len("<tool_call>"))
        if text.startswith("<function=", pos):
            close = text.find(">", pos)
            if close < 0:
                return None
            name = text[pos + len("<function="):close].strip()
            if not _TOOL_NAME_RE.match(name):
                return None
            pos, params = close + 1, {}
            schema = (tools.get(name) or {}).get("parameters")
            known = schema.get("properties") if isinstance(schema, dict) else None
            while True:
                pos = skip(pos)
                if text.startswith("</function>", pos):
                    pos += len("</function>")
                    break
                if not text.startswith("<parameter=", pos):
                    return None
                close = text.find(">", pos)
                if close < 0:
                    return None
                key = text[pos + len("<parameter="):close].strip()
                if not _TOOL_NAME_RE.match(key) or key in params:
                    return None
                start = close + 1
                start += 2 if text.startswith("\r\n", start) else 1 if text.startswith("\n", start) else 0
                end = _PARAM_CLOSE_RE.search(text, start)
                # The tag itself inside a value: a parameter the tool doesn't have isn't a real next one.
                while end and isinstance(known, dict):
                    following = re.compile(r"<parameter=([^>]*)>").match(text, end.end())
                    if not following or following.group(1).strip() in known:
                        break
                    end = _PARAM_CLOSE_RE.search(text, end.end())
                if not end:
                    return None
                value = text[start:end.start()]
                params[key] = value[:-2] if value.endswith("\r\n") else value[:-1] if value.endswith("\n") else value
                pos = end.start() + len("</parameter>")
            calls.append({"name": name, "params": params})
        elif wrapped and text.startswith("{", pos):     # {"name": ..., "arguments": {...}}
            close = text.find("</tool_call>", pos)
            if close < 0:
                return None
            try:
                obj = json.loads(text[pos:close])
            except ValueError:
                return None
            name = obj.get("name") if isinstance(obj, dict) else None
            args = obj.get("arguments", obj.get("parameters", {})) if isinstance(obj, dict) else None
            if not isinstance(name, str) or not _TOOL_NAME_RE.match(name) or not isinstance(args, (dict, str)):
                return None
            calls.append({"name": name, "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)})
            pos = close
        else:
            return None
        pos = skip(pos)
        if text.startswith("</tool_call>", pos):
            pos = skip(pos + len("</tool_call>"))
        elif wrapped and pos < n and not text.startswith(_TOOL_TRIGGERS, pos):
            return None
        if not text.startswith(_TOOL_TRIGGERS, pos):
            break
    trailing = text[pos:]
    return None if any(t in trailing for t in _TOOL_TRIGGERS) else (calls, trailing)


_GEMMA_OPEN, _GEMMA_CLOSE, _GEMMA_QUOTE = "<|tool_call>", "<tool_call|>", '<|"|>'
_THINK_END = "</think>"
_GEMMA_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_GEMMA_SCALAR_RE = re.compile(r"true|false|null|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_gemma_calls(text: str):
    """Gemma 4's own call format, <|tool_call>call:name{key:value,...}<tool_call|>, one or more
    and nothing else: [{name, arguments}] or None. Strings sit between <|"|> marks and are
    taken as written; anything this can't read whole means None."""
    n = len(text)

    def skip(at):
        while at < n and text[at].isspace():
            at += 1
        return at

    def value(at):
        at = skip(at)
        if text.startswith(_GEMMA_QUOTE, at):
            end = text.find(_GEMMA_QUOTE, at + len(_GEMMA_QUOTE))
            if end < 0:
                raise ValueError("open string")
            return text[at + len(_GEMMA_QUOTE):end], end + len(_GEMMA_QUOTE)
        if text.startswith("{", at):
            out, at = {}, skip(at + 1)
            if text.startswith("}", at):
                return out, at + 1
            while True:
                at = skip(at)
                if text.startswith(_GEMMA_QUOTE, at):
                    key, at = value(at)
                else:
                    m = _GEMMA_KEY_RE.match(text, at)
                    if not m:
                        raise ValueError("no key")
                    key, at = m.group(0), m.end()
                at = skip(at)
                if not text.startswith(":", at) or key in out:
                    raise ValueError("no colon")
                out[key], at = value(at + 1)
                at = skip(at)
                if text.startswith("}", at):
                    return out, at + 1
                if not text.startswith(",", at):
                    raise ValueError("no comma")
                at += 1
        if text.startswith("[", at):
            out, at = [], skip(at + 1)
            if text.startswith("]", at):
                return out, at + 1
            while True:
                item, at = value(at)
                out.append(item)
                at = skip(at)
                if text.startswith("]", at):
                    return out, at + 1
                if not text.startswith(",", at):
                    raise ValueError("no comma")
                at += 1
        m = _GEMMA_SCALAR_RE.match(text, at)
        if not m:
            raise ValueError("no value")
        return json.loads(m.group(0)), m.end()

    calls, pos = [], skip(0)
    try:
        while pos < n:
            if not text.startswith(_GEMMA_OPEN, pos):
                return None
            pos = skip(pos + len(_GEMMA_OPEN))
            if not text.startswith("call:", pos):
                return None
            brace = text.find("{", pos)
            name = text[pos + len("call:"):brace].strip() if brace > 0 else ""
            if not _TOOL_NAME_RE.match(name):
                return None
            args, pos = value(brace)
            pos = skip(pos)
            if not isinstance(args, dict) or not text.startswith(_GEMMA_CLOSE, pos):
                return None
            calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False)})
            pos = skip(pos + len(_GEMMA_CLOSE))
    except (ValueError, RecursionError):
        return None
    return calls or None


def _calls_after_thinking(reasoning: str):
    """The calls a model wrote straight after closing its thinking with </think>, where the
    server (LM Studio with a Gemma 4 finetune) didn't see thinking end and so left them in the
    reasoning. Only when that closing tag is there and everything after it is whole calls:
    a call the model merely thought about is never one."""
    at = reasoning.rfind(_THINK_END)
    if at < 0:
        return None
    return _parse_gemma_calls(reasoning[at + len(_THINK_END):])


def _decode_params(call: dict, tools: dict) -> str:
    """The call's arguments as a JSON string, typed by the tool's schema where known. A
    value that doesn't fit stays as written, and the client tells the model what's wrong."""
    if "arguments" in call:
        return call["arguments"]
    function = tools.get(call["name"]) or {}
    parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
    schema = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
    out = {}
    for key, value in call["params"].items():
        spec = schema.get(key) if isinstance(schema.get(key), dict) else {}
        kinds = spec.get("type")
        kinds = {kinds} if isinstance(kinds, str) else set(kinds) if isinstance(kinds, list) else set()
        if "string" in kinds and not kinds & {"array", "object", "number", "integer", "boolean"}:
            out[key] = value
            continue
        stripped = value.strip()
        if stripped[:1] in ("[", "{", '"') or stripped in ("true", "false", "null") or _NUMBER_RE.match(stripped):
            try:
                out[key] = json.loads(stripped)
                continue
            except ValueError:
                pass
        out[key] = value
    return json.dumps(out, ensure_ascii=False)


def _tools_shell(tools: dict) -> str:
    """Which shell the client runs commands in, from how its tool describes itself."""
    for function in tools.values():
        description = function.get("description") if isinstance(function.get("description"), str) else ""
        said = re.search(r"run through ([^;]*)", description, re.I)
        text = (said.group(1) if said else "").lower()
        if "powershell" in text or "pwsh" in text:
            return "powershell"
        if "cmd.exe" in text:
            return "cmd"
    return "posix"


class ToolHealer:
    """Reads one /chat/completions reply on its way to the client, in memory.

    A reply with nothing to repair passes unchanged; only text that could have started
    a call may arrive split differently. From the moment a reply looks like a tool call
    it is held, because a client must only ever receive a call that is whole and safe to
    run. A failure in here hands over what is held, as it came."""

    def __init__(self, cfg: dict, data: dict):
        self.allow = set(cfg.get("allow_rules") or []) if cfg.get("guard", True) else None
        self.tools = {}
        for tool in data.get("tools") or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                self.tools[function["name"]] = function
        self.shell = _tools_shell(self.tools)
        self.repeat_limit, self._run, self._run_count = 0, None, 0
        try:
            if cfg.get("stop_repeats"):
                limit = cfg.get("repeat_limit")
                self.repeat_limit = limit if isinstance(limit, int) and not isinstance(limit, bool) and 2 <= limit <= 20 else _REPEAT_LIMIT
                self._run, self._run_count = _repeat_run(data.get("messages"))
        except Exception:  # noqa: BLE001 - a chat this can't read is a chat it leaves alone
            self.repeat_limit = 0
        self.result = {}            # counts and a rule id, for stats and the log
        self._line = bytearray()
        self._event_lines = []
        self._tail = ""             # sent content stops here: this much could still start a call
        self._mid_line = False
        self._fence = False
        self._text = None           # held content, from where a call seems to start
        self._text_size = 0
        self._native = {}           # index -> id, name, arguments of structured calls
        self._held = []             # raw events that wait for the verdict
        self._after = []            # raw events after the finish reason
        self._finish = None
        self._finish_obj = {}
        self._template = {}
        self._rewrite_native = False
        self._resolved = False
        self._done = False
        self._broken = False
        self._last_write = time.time()
        self._reasoning = []        # the reasoning, in memory for this reply only: a call may hide after </think>
        self._reasoning_size = 0
        self._said = False          # any answer text at all

    # ── building replies ────────────────────────────────────────────────
    def _chunk(self, delta: dict, finish=None, extra=None) -> bytes:
        obj = {**self._template, "object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **(extra or {})}
        return b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"

    def _tool_chunks(self, calls: list) -> bytes:
        out = [self._chunk({"tool_calls": [{"index": i, "id": f"call_gw_{i + 1}", "type": "function",
                                            "function": {"name": call["name"], "arguments": call["arguments"]}}]})
               for i, call in enumerate(calls)]
        usage = {k: self._finish_obj[k] for k in ("usage", "timings") if k in self._finish_obj}
        out.append(self._chunk({}, "tool_calls", usage))
        return b"".join(out) + b"".join(self._after)

    def _stopped_call(self, rule: str, command) -> dict:
        reason = _GUARD_REASONS.get(rule, "it looked unsafe")
        function = self.tools.get("ask_question")
        properties = ((function or {}).get("parameters") or {}).get("properties") or {}
        if rule not in _STOP_NAMES and "question" in properties and "options" in properties:
            shown = f"\n\n{command[:300]}" if command else ""
            return {"name": "ask_question", "arguments": json.dumps({
                "question": f"The gateway stopped a command because {reason} (rule {rule}).{shown}\n\nWhat should happen next?",
                "options": ["Skip it and try another way", "I'll run it myself"]}, ensure_ascii=False)}
        slug = _STOP_NAMES.get(rule, "destructive_command_was_not_run")
        return {"name": f"gateway_stopped_{rule}_{slug}", "arguments": "{}"}

    # ── the verdict ─────────────────────────────────────────────────────
    def _judge(self, calls: list):
        if self.allow is not None:
            for call in calls:
                rule, command = guard_tool_call(call["name"], call["arguments"], self.shell, self.allow)
                if rule:
                    return rule, command
        return self._repeated(calls), None

    def _repeated(self, calls: list):
        """R1: these exact calls were the chat's last `repeat_limit` turns already. R2: one reply makes
        the same call more often than that. The model is told instead, before the client's own loop
        detection ends the task. Compared by hash; a different argument is a different call."""
        if not self.repeat_limit or not calls:
            return None
        try:
            keys = tuple(_call_key(call["name"], call["arguments"]) for call in calls)
            if keys == self._run and self._run_count >= self.repeat_limit:
                return "R1"
            if max(keys.count(key) for key in set(keys)) > self.repeat_limit:
                return "R2"
        except Exception:  # noqa: BLE001
            pass
        return None

    def _stop(self, rule: str, command=None) -> bytes:
        self.result = {"stopped": 1, "rule": rule}
        return self._tool_chunks([self._stopped_call(rule, command)])

    def _was_cut(self) -> bool:
        """An engine may leave the finish reason out: [DONE] then says the reply ended well."""
        return self._finish not in ("stop", "tool_calls", "function_call") if self._finish else not self._done

    def _resolve_native(self) -> bytes:
        calls = [self._native[i] for i in sorted(self._native)]
        if self.allow is not None and self._was_cut() and "T1" not in self.allow:
            for call in calls:
                if call["name"].lower() in _GUARDED_TOOLS or any(f'"{k}"' in call["arguments"] for k in _COMMAND_KEYS):
                    return self._stop("T1")
        rule, command = self._judge(calls)
        if rule:
            return self._stop(rule, command)
        if self._rewrite_native:
            return self._tool_chunks(calls)
        return b"".join(self._held) + b"".join(self._after)

    def _resolve_text(self) -> bytes:
        text = "".join(self._text)
        parsed = _parse_tool_blocks(text, self.tools)
        if parsed is None or self._was_cut():
            self.result = {"left": 1, "rule": "T1"} if self._was_cut() else {"left": 1}
            return self._flush_text(text)
        blocks, trailing = parsed
        calls = [{"name": b["name"], "arguments": _decode_params(b, self.tools)} for b in blocks]
        rule, command = self._judge(calls)
        if rule:
            return self._stop(rule, command)
        self.result = {"healed": len(calls), "unknown": sum(1 for c in calls if c["name"] not in self.tools)}
        prose = self._chunk({"content": trailing}) if trailing.strip() else b""
        return prose + self._tool_chunks(calls)

    def _flush_text(self, text: str) -> bytes:
        finish = b"".join(self._held)
        return (self._chunk({"content": text}) if text else b"") + finish + b"".join(self._after)

    def _resolve(self) -> bytes:
        if self._resolved:
            return b""
        self._resolved = True
        if self._native:
            return (self._chunk({"content": self._tail + "".join(self._text or [])}) if self._tail or self._text else b"") + self._resolve_native()
        if self._text is not None:
            return self._resolve_text()
        return self._flush_text(self._tail)

    # ── content as it arrives ───────────────────────────────────────────
    def _scan(self, content: str) -> str:
        """What can be sent now. From a call's opening on, content is held."""
        buf, pos, sent = self._tail + content, 0, []
        self._tail = ""
        while pos < len(buf):
            end = buf.find("\n", pos)
            line = buf[pos:] if end < 0 else buf[pos:end + 1]
            if self._mid_line:
                sent.append(line)
                self._mid_line = end < 0
                pos += len(line)
                continue
            stripped = line.lstrip(" \t")
            indent = len(line) - len(stripped)
            if not self._fence and indent <= 3 and stripped.startswith(_TOOL_TRIGGERS):
                self._text = [buf[pos:]]
                self._text_size = len(buf) - pos
                return "".join(sent)
            if stripped.startswith(_FENCES):
                self._fence = not self._fence
                self._mid_line = end < 0
            elif end < 0:
                undecided = not stripped or any(t.startswith(stripped) for t in _TOOL_TRIGGERS + _FENCES)
                if undecided and indent <= 3:
                    self._tail = buf[pos:]
                    return "".join(sent)
                self._mid_line = True
            sent.append(line)
            pos += len(line)
        return "".join(sent)

    def _hold_text(self, content: str):
        self._text.append(content)
        self._text_size += len(content)
        if self._text_size > _HEAL_MAX_HOLD:
            raise OverflowError("held reply too long")

    # ── one SSE event ───────────────────────────────────────────────────
    def _holding(self) -> bool:
        return self._text is not None or bool(self._native)

    def _event(self, raw: bytes) -> bytes:
        data = None
        for line in raw.split(b"\n"):
            if line.startswith(b"data:"):
                data = line[5:].strip()
                break
        if data is None:
            return raw
        if data == b"[DONE]":
            self._done = True
            return self._resolve() + raw
        try:
            obj = json.loads(data)
        except ValueError:
            return raw
        if not isinstance(obj, dict) or not isinstance(obj.get("choices"), list):
            return raw
        for key in ("id", "created", "model"):
            if key in obj:
                self._template[key] = obj[key]
        choices = obj["choices"]
        if not choices:                                 # usage arrives after the finish reason
            if self._holding() or self._tail:
                self._after.append(raw)
                return b""
            return raw
        if len(choices) != 1:
            raise ValueError("more than one choice")
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        out = []
        content = delta.get("content")
        pass_raw = True
        for key in ("reasoning_content", "reasoning"):
            thought = delta.get(key)
            if isinstance(thought, str) and thought and self._reasoning_size <= _HEAL_MAX_HOLD:
                self._reasoning.append(thought)
                self._reasoning_size += len(thought)
                break
        if isinstance(content, str) and content:
            self._said = True
            if self._text is not None:
                self._hold_text(content)
                sent = ""
            else:
                sent = self._scan(content)
            if sent != content:
                pass_raw = False
                rest = {k: v for k, v in delta.items() if k != "content"}
                if sent:
                    rest["content"] = sent
                if any(v for k, v in rest.items()):
                    out.append(b"data: " + json.dumps({**obj, "choices": [{**choice, "delta": rest, "finish_reason": None}]},
                                                      ensure_ascii=False).encode("utf-8") + b"\n\n")
        calls = delta.get("tool_calls")
        finish_in_held = False
        if isinstance(calls, list) and calls:
            self._note_native(calls)
            if pass_raw:                # the raw event carries any finish reason with it
                self._held.append(raw)
                finish_in_held = True
            else:
                self._held.append(b"data: " + json.dumps({**obj, "choices": [{**choice, "delta": {"tool_calls": calls}, "finish_reason": None}]},
                                                         ensure_ascii=False).encode("utf-8") + b"\n\n")
            pass_raw = False
        finish = choice.get("finish_reason")
        if finish:
            self._finish = finish
            self._finish_obj = obj
            if self._holding() or self._tail:
                if not finish_in_held:
                    self._held.append(self._finish_event(obj, choice))
                return b"".join(out)
            if finish == "stop" and not self._said and not self._resolved:
                rescued = self._rescue_from_reasoning()
                if rescued is not None:
                    self._resolved = True
                    return b"".join(out) + rescued
            if not pass_raw:
                out.append(self._finish_event(obj, choice))
        return raw if pass_raw else b"".join(out)

    def _rescue_from_reasoning(self):
        """A reply that ended with no answer and no call, but whose reasoning closes with
        </think> and whole calls: send those calls. None when there is nothing to rescue."""
        if self._reasoning_size > _HEAL_MAX_HOLD:
            return None
        calls = _calls_after_thinking("".join(self._reasoning))
        if not calls:
            return None
        rule, command = self._judge(calls)
        if rule:
            return self._stop(rule, command)
        self.result = {"healed": len(calls), "unknown": sum(1 for c in calls if c["name"] not in self.tools)}
        return self._tool_chunks(calls)

    @staticmethod
    def _finish_event(obj: dict, choice: dict) -> bytes:
        return b"data: " + json.dumps({**obj, "choices": [{**choice, "delta": {}}]}, ensure_ascii=False).encode("utf-8") + b"\n\n"

    def _note_native(self, calls: list):
        for item in calls:
            if not isinstance(item, dict):
                continue
            index = item.get("index", 0)
            call = self._native.setdefault(index, {"name": "", "arguments": ""})
            function = item.get("function") if isinstance(item.get("function"), dict) else {}
            if isinstance(function.get("name"), str):
                call["name"] += function["name"]
            args = function.get("arguments")
            if isinstance(args, str):
                call["arguments"] += args
            elif args is not None:      # an object where a string belongs: send it once, as a string
                call["arguments"] = json.dumps(args, ensure_ascii=False)
                self._rewrite_native = True

    # ── the stream ──────────────────────────────────────────────────────
    def feed(self, chunk: bytes) -> bytes:
        if self._broken:
            return chunk
        self._line.extend(chunk)
        out = []
        try:
            while True:
                at = self._line.find(b"\n")
                if at < 0:
                    break
                line = bytes(self._line[:at + 1])
                del self._line[:at + 1]
                self._event_lines.append(line)
                if not line.strip():
                    raw = b"".join(self._event_lines)
                    self._event_lines = []
                    try:
                        out.append(self._event(raw))
                    except OverflowError:       # this event's text is already among what is held
                        return b"".join(out) + self._give_up()
                    except Exception:
                        return b"".join(out) + self._give_up(raw)
            if len(self._line) > _HEAL_MAX_HOLD:
                raise OverflowError("line too long")
            sent = b"".join(out)
            now = time.time()
            if sent:
                self._last_write = now
            elif self._holding() and now - self._last_write >= _HEAL_KEEPALIVE_S:
                self._last_write = now
                sent = self._chunk({})
            return sent
        except Exception:
            return b"".join(out) + self._give_up()

    def _give_up(self, current: bytes = b"") -> bytes:
        """Something unexpected: from here on the reply passes as it came."""
        self._broken = True
        self._resolved = True
        self.result = {}
        if self._text is not None or self._tail:
            head = self._chunk({"content": self._tail + "".join(self._text or [])})
        else:
            head = b""
        out = head + b"".join(self._held) + b"".join(self._after) + current + b"".join(self._event_lines) + bytes(self._line)
        self._held, self._after, self._event_lines, self._line = [], [], [], bytearray()
        self._text, self._tail = None, ""
        return out

    def finish(self) -> bytes:
        """The upstream has ended. A reply that never finished is never healed."""
        if self._broken:
            return b""
        out = b""
        leftover = b"".join(self._event_lines) + bytes(self._line)
        self._event_lines, self._line = [], bytearray()
        try:
            if leftover.strip():
                out = self._event(leftover)
                leftover = b""
            return out + self._resolve()
        except OverflowError:
            return out + self._give_up()
        except Exception:
            return out + self._give_up(leftover)

    # ── a whole reply ───────────────────────────────────────────────────
    def rewrite_body(self, raw: bytes) -> bytes:
        """The same for a reply that isn't streamed; unchanged replies stay byte for byte."""
        try:
            obj = json.loads(raw)
            choices = obj.get("choices") if isinstance(obj, dict) else None
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
                return raw
            choice, message = choices[0], choices[0]["message"]
            finish = choice.get("finish_reason")
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                found = []
                for item in calls:
                    function = item.get("function") or {}
                    args = function.get("arguments", "")
                    found.append({"name": str(function.get("name") or ""), "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)})
                rule, command = self._judge(found)
                if not rule and self.allow is not None and finish == "length":
                    rule = "T1" if any(c["name"].lower() in _GUARDED_TOOLS for c in found) and "T1" not in self.allow else None
                if rule:
                    return self._stopped_body(obj, choice, message, rule, command, None)
                if any(not isinstance((c.get("function") or {}).get("arguments", ""), str) for c in calls):
                    for item, call in zip(calls, found):
                        item["function"]["arguments"] = call["arguments"]
                    return json.dumps(obj, ensure_ascii=False).encode("utf-8")
                return raw
            content = message.get("content")
            if not isinstance(content, str) or not content:
                thought = message.get("reasoning_content") or message.get("reasoning")
                found = _calls_after_thinking(thought) if finish == "stop" and isinstance(thought, str) else None
                if not found:
                    return raw
                rule, command = self._judge(found)
                if rule:
                    return self._stopped_body(obj, choice, message, rule, command, None)
                self.result = {"healed": len(found), "unknown": sum(1 for c in found if c["name"] not in self.tools)}
                message["tool_calls"] = [{"id": f"call_gw_{i + 1}", "type": "function",
                                          "function": {"name": c["name"], "arguments": c["arguments"]}} for i, c in enumerate(found)]
                choice["finish_reason"] = "tool_calls"
                return json.dumps(obj, ensure_ascii=False).encode("utf-8")
            prose = self._scan(content)
            if self._text is None:
                return raw
            text = "".join(self._text)
            parsed = _parse_tool_blocks(text, self.tools)
            cut = finish not in ("stop", "tool_calls", "function_call", None)
            if parsed is None or cut:
                self.result = {"left": 1, "rule": "T1"} if cut else {"left": 1}
                return raw
            blocks, trailing = parsed
            found = [{"name": b["name"], "arguments": _decode_params(b, self.tools)} for b in blocks]
            rule, command = self._judge(found)
            if rule:
                return self._stopped_body(obj, choice, message, rule, command, prose)
            self.result = {"healed": len(found), "unknown": sum(1 for c in found if c["name"] not in self.tools)}
            message["content"] = (prose + trailing) if (prose + trailing).strip() else None
            message["tool_calls"] = [{"id": f"call_gw_{i + 1}", "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}} for i, c in enumerate(found)]
            choice["finish_reason"] = "tool_calls"
            return json.dumps(obj, ensure_ascii=False).encode("utf-8")
        except Exception:
            self.result = {}
            return raw

    def _stopped_body(self, obj, choice, message, rule, command, prose) -> bytes:
        self.result = {"stopped": 1, "rule": rule}
        call = self._stopped_call(rule, command)
        message["content"] = prose if prose and prose.strip() else None
        message["tool_calls"] = [{"id": "call_gw_1", "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]
        choice["finish_reason"] = "tool_calls"
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _new_healer(ep: dict, command: str, path: str, data):
    """A healer for this request, or None: most requests pass untouched."""
    try:
        cfg = ep.get("tool_heal")
        if not isinstance(cfg, dict) or not cfg.get("enabled") or command != "POST" or not isinstance(data, dict):
            return None
        if not path.rstrip("/").endswith("/chat/completions") or data.get("tool_choice") == "none" or data.get("n") not in (None, 1):
            return None
        tools = data.get("tools")
        return ToolHealer(cfg, data) if isinstance(tools, list) and tools else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  CONTEXT LIMIT — a full context reaches the client in the words it knows
# ═══════════════════════════════════════════════════════════════════════════
#
#  When a prompt no longer fits, OpenAI answers 400 with the code
#  "context_length_exceeded", and clients are written against that: Cline
#  4.1.26 compacts and retries on that code, or on words like "context
#  length / window / limit" or "maximum context". Local engines say it their
#  own way (llama.cpp: "exceeds the available context size", type
#  exceed_context_size_error), which such a client reads as a plain failure
#  and stops. The gateway rewrites only the error body of a reply that is
#  already an error; nothing else about the request or the reply changes.

_OVERFLOW_STATUSES = (400, 413, 422, 500)
_OVERFLOW_MAX_BODY = 256 * 1024
_OVERFLOW_TYPES = ("exceed_context_size_error", "context_length_exceeded", "context_window_exceeded")
_OVERFLOW_RE = re.compile(
    r"context[\s_]*(?:size|length|window|limit)|maximum\s+context|context\s+overflow|"
    r"too\s+many\s+tokens|prompt\s+is\s+too\s+long|input\s+is\s+too\s+long|"
    r"tokens?\s+exceeds?|exceeds?\s+the\s+(?:available\s+|maximum\s+|model'?s?\s+)?context|\bn_ctx\b", re.I)
_OVERFLOW_LIMIT_RE = re.compile(r"(?:context(?:\s+(?:size|length|window|limit))?|n_ctx|maximum)\D{0,40}?(\d[\d,]{2,})", re.I)
_OVERFLOW_PROMPT_RE = re.compile(r"(?:prompt|request|messages?|input|resulted\s+in|n_prompt_tokens|keep)\D{0,60}?(\d[\d,]{2,})", re.I)


def _whole_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\d[\d,]*", value.strip()):
        return int(value.replace(",", ""))
    return None


def _context_overflow(raw: bytes, status: int, path: str = ""):
    """Is this error reply an engine saying the prompt doesn't fit? Then (body, info): the body to
    send (None when the reply is already in the standard form and goes as it came) and
    {"prompt", "limit"} token counts where the engine gave them. Otherwise None."""
    if status not in _OVERFLOW_STATUSES or not raw or len(raw) >= _OVERFLOW_MAX_BODY:
        return None
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    err = obj.get("error")
    detail = err if isinstance(err, dict) else {}
    message = next((m for m in (detail.get("message"), err, obj.get("message"), obj.get("detail")) if isinstance(m, str) and m), "")
    kind, code = str(detail.get("type") or obj.get("type") or ""), str(detail.get("code") or "")
    if not (code == "context_length_exceeded" or kind in _OVERFLOW_TYPES or _OVERFLOW_RE.search(message)):
        return None
    limit = _whole_number(detail.get("n_ctx"))
    prompt = _whole_number(detail.get("n_prompt_tokens"))
    if limit is None:
        found = _OVERFLOW_LIMIT_RE.search(message)
        limit = _whole_number(found.group(1)) if found else None
    if prompt is None:
        found = _OVERFLOW_PROMPT_RE.search(message)
        prompt = _whole_number(found.group(1)) if found else None
    if prompt is not None and prompt == limit:
        prompt = None                       # one number read twice says nothing about the prompt
    info = {"prompt": prompt, "limit": limit}
    anthropic = path.rstrip("/").endswith("/messages")
    if code == "context_length_exceeded" and not anthropic:
        return None, info                   # already what clients expect
    said = re.sub(r"\s+", " ", message).strip()[:300]
    if anthropic:
        size = f"{prompt} tokens > {limit} maximum" if prompt and limit else "it does not fit the model's context window"
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": f"prompt is too long: {size} (engine: {said})"}}
    else:
        if prompt and limit:
            text = (f"This model's maximum context length is {limit} tokens. However, your messages resulted in "
                    f"{prompt} tokens. Please reduce the length of the messages.")
        elif limit:
            text = f"This model's maximum context length is {limit} tokens, and your messages do not fit. Please reduce the length of the messages."
        else:
            text = "This model's maximum context length was exceeded. Please reduce the length of the messages."
        body = {"error": {"message": f"{text} (engine: {said})", "type": "invalid_request_error", "param": "messages",
                          "code": "context_length_exceeded",
                          **({"n_prompt_tokens": prompt} if prompt is not None else {}), **({"n_ctx": limit} if limit is not None else {})}}
    return json.dumps(body, ensure_ascii=False).encode("utf-8"), info


def _find_model_entry(entries: list, wanted: str):
    """The entry a client means by a model name: the gateway's id (endpoint:model), or the engine's own id
    when only one endpoint offers it. None when nothing matches, or the bare name is offered by several."""
    wanted = (wanted or "").strip()
    for entry in entries:
        if entry.get("id") == wanted:
            return entry
    bare = [e for e in entries if parse_model_id(str(e.get("id") or ""))[1] == wanted]
    return bare[0] if len(bare) == 1 else None


_LOOKUP_PATHS = ("/v1/models", "/models", "/api/v0/models", "/api/v1/models", "/props", "/api/tags", "/api/ps", "/api/show")
_LOOKUP_LOG_EVERY_S = 600
_lookup_seen = {}
_lookup_lock = threading.Lock()


def _note_lookup(client: str, method: str, path: str):
    """Log which model lookups an app makes: that is how to tell where it gets a model's context window
    from (or that it never asks). One line per client and kind of lookup every ten minutes; the model
    name in a per-model lookup is left out of the key so a poll can't flood the log."""
    try:
        base = next((p for p in sorted(_LOOKUP_PATHS, key=len, reverse=True) if path == p or path.startswith(p + "/")), None)
        if base is None:
            return
        kind = f"{method} {base}" + ("/<model>" if path != base else "")
        now = time.time()
        with _lookup_lock:
            if now - _lookup_seen.get((client, kind), 0) < _LOOKUP_LOG_EVERY_S:
                return
            _lookup_seen[(client, kind)] = now
            for key in [k for k, t in _lookup_seen.items() if now - t > 6 * _LOOKUP_LOG_EVERY_S]:
                del _lookup_seen[key]
        _queue_log_line(f"[{client}] model lookup: {kind}")
    except Exception:  # noqa: BLE001
        pass


def _context_window(ep_name: str, model: str):
    """How many tokens the model at this endpoint can hold as loaded now, or None when unknown."""
    try:
        for gateway_id in get_models():
            source, original = parse_model_id(gateway_id)
            if source == ep_name and original == model:
                meta = get_model_meta(gateway_id)
                if meta.get("context_guessed"):
                    return None             # a default, not something the endpoint said
                return _whole_number(meta.get("loaded_context_length")) or _whole_number(meta.get("context_length"))
    except Exception:  # noqa: BLE001 -- a number for the console, never worth a failed request
        pass
    return None


# ── Context guard: say "full" while the whole chat still fits ────────────────
#
#  A client that only compacts after the engine refuses the prompt compacts too
#  late: by then the chat no longer fits, so nothing can read all of it, and what
#  the client cuts is lost (Cline 4.1.26, seen 2026-11-12 at 260k of 262k).
#  With "context_guard": {"enabled": true} on an endpoint the gateway answers
#  context_length_exceeded at `at` (90%) of the window, before forwarding, and
#  keeps that request's messages in memory for the client's next request:
#    - a compact request (new messages after a cut-down history): the full
#      history is put back, so the client's compaction reads all of it;
#    - the same turn again with messages cut out: the engine summarises what was
#      cut (it still fits, and mostly from its cache), and the summary goes into
#      that turn and into the reply, where the client keeps it in its own history.
#  Anything unclear passes as the client sent it. Chat text is held in memory
#  only, for one follow-up request or _CTX_PENDING_TTL_S, never written or logged.

CTX_GUARD_DEFAULTS = {"enabled": False, "at": 0.9, "summary_tokens": 2000}
_CTX_PENDING_TTL_S = 900
_CTX_MIN_DROPPED_CHARS = 2000       # less than this was cut: not worth a summary
_CTX_MIN_LEARN_TOKENS = 500
_CTX_SUMMARY_MAX_CHARS = 24000
_CTX_SUMMARY_PROMPT = (
    "Stop here and do not continue the task. Part of the conversation above is about to be removed to free up "
    "context. Write a summary that lets the work continue without it: the user's goal and constraints, decisions "
    "made and why, what has been done (files, commands, results), exact names, paths and values that will be needed "
    "again, what failed, and what remains to be done. Be specific and concise. Reply with the summary only.")
# Not copied from the request answered "full" into the summary request; everything else is, so the engine
# renders the same prompt start and reads it from its cache instead of from scratch.
_CTX_NOT_CARRIED = ("messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "n", "tool_choice",
                    "temperature", "response_format")
_ctx_lock = threading.Lock()
_ctx_ratio = {}          # endpoint name -> request bytes per prompt token, learned from finished requests
_ctx_pending = {}        # (client address, endpoint name) -> the request that was answered "full"


def _ctx_guard_config(ep: dict):
    raw = ep.get("context_guard")
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return None
    cfg = dict(CTX_GUARD_DEFAULTS, enabled=True)
    at = raw.get("at")
    if isinstance(at, (int, float)) and not isinstance(at, bool) and 0.5 <= at <= 0.99:
        cfg["at"] = float(at)
    size = raw.get("summary_tokens")
    if isinstance(size, int) and not isinstance(size, bool) and 200 <= size <= 8000:
        cfg["summary_tokens"] = size
    return cfg


def _ctx_size(body: bytes, data) -> int:
    """The request's size in bytes without inline images, which are large but cost few tokens."""
    size = len(body)
    try:
        for message in data.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            for part in content if isinstance(content, list) else []:
                url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) and isinstance(part.get("image_url"), dict) else None
                if isinstance(url, str) and url.startswith("data:"):
                    size -= len(url)
    except Exception:  # noqa: BLE001
        pass
    return max(size, 0)


def _ctx_learn(ep_name: str, size: int, prompt_tokens):
    if not size or not isinstance(prompt_tokens, (int, float)) or prompt_tokens < _CTX_MIN_LEARN_TOKENS:
        return
    ratio = size / prompt_tokens
    if not 1.0 <= ratio <= 40.0:
        return
    with _ctx_lock:
        old = _ctx_ratio.get(ep_name)
        _ctx_ratio[ep_name] = ratio if old is None else old * 0.7 + ratio * 0.3


def _ctx_estimate(ep_name: str, size: int):
    """About how many tokens a request of this size is, or None before one request has finished."""
    with _ctx_lock:
        ratio = _ctx_ratio.get(ep_name)
    return int(size / ratio) if ratio else None


def _msg_key(message) -> str:
    return hashlib.sha1(json.dumps(message, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _ctx_compare(full: list, new: list):
    """How the client's next request relates to the one answered "full": which of the full chat's
    messages it kept, which it cut, and what it added after them. None when it isn't that chat."""
    full_keys = [_msg_key(m) for m in full]
    known = set(full_keys)
    matched = [_msg_key(m) in known for m in new]
    if not any(matched):
        return None
    last = max(i for i, hit in enumerate(matched) if hit)
    kept = {_msg_key(m) for m, hit in zip(new, matched) if hit}
    dropped = [i for i, key in enumerate(full_keys) if key not in kept]
    first = new[0] if new and isinstance(new[0], dict) else {}
    return {"kept": len(kept), "trailing": new[last + 1:], "dropped": dropped,
            "dropped_chars": sum(len(json.dumps(full[i], ensure_ascii=False)) for i in dropped),
            "new_system": first if first.get("role") == "system" and not matched[0] else None}


def _ctx_guard(cfg: dict, ep_name: str, client: str, data: dict, body: bytes):
    """What to do with this request: None (forward it), or {"action": "full" | "restore" | "summarise", ...}."""
    key, now, messages = (client, ep_name), time.time(), data["messages"]
    with _ctx_lock:
        pending = _ctx_pending.pop(key, None)
        for stale in [k for k, v in _ctx_pending.items() if now - v["time"] > _CTX_PENDING_TTL_S]:
            del _ctx_pending[stale]
    if pending and now - pending["time"] <= _CTX_PENDING_TTL_S:
        # The request right after a "full" is never answered "full" again, whatever it is.
        cmp = _ctx_compare(pending["messages"], messages)
        shape = (f"{len(messages)} messages, {cmp['kept']} from the full chat of {len(pending['messages'])}, "
                 f"{len(cmp['dropped'])} cut, {len(cmp['trailing'])} new at the end, tools {'offered' if data.get('tools') else 'not offered'}"
                 if cmp else f"{len(messages)} messages, none from the full chat")
        if not cmp or cmp["dropped_chars"] < _CTX_MIN_DROPPED_CHARS:
            return {"action": "pass", "shape": shape}
        if cmp["trailing"]:
            full = pending["messages"]
            if cmp["new_system"]:
                full = [cmp["new_system"]] + (full[1:] if full and isinstance(full[0], dict) and full[0].get("role") == "system" else full)
            return {"action": "restore", "shape": shape, "messages": list(full) + list(cmp["trailing"]), "cut": len(cmp["dropped"])}
        return {"action": "summarise", "shape": shape, "prefix": pending["messages"][:cmp["dropped"][-1] + 1],
                "tools": pending.get("tools"), "params": pending.get("params") or {}, "cut": len(cmp["dropped"])}
    window = _context_window(ep_name, str(data.get("model") or ""))
    tokens = _ctx_estimate(ep_name, _ctx_size(body, data))
    if window and tokens and tokens >= cfg["at"] * window:
        with _ctx_lock:
            _ctx_pending[key] = {"messages": messages, "tools": data.get("tools"), "time": now,
                                 "params": {k: v for k, v in data.items() if k not in _CTX_NOT_CARRIED}}
        return {"action": "full", "tokens": tokens, "window": window}
    return None


def _ctx_summary_messages(prefix: list) -> list:
    messages = list(prefix)
    last = messages[-1] if messages else None
    if isinstance(last, dict) and last.get("role") == "user" and isinstance(last.get("content"), str):
        messages[-1] = {**last, "content": last["content"] + "\n\n" + _CTX_SUMMARY_PROMPT}   # roles keep alternating
    else:
        messages.append({"role": "user", "content": _CTX_SUMMARY_PROMPT})
    return messages


def _ctx_note(summary: str) -> str:
    return ("[Context note from the gateway: earlier messages of this conversation were removed to make room. "
            "Summary of what was removed:]\n" + summary.strip() + "\n[End of context note]")


def _ctx_append(data: dict, note: str):
    """Put the note at the end of the request's last message, so everything before it is unchanged."""
    last = data["messages"][-1]
    content = last.get("content")
    if isinstance(content, list):
        last["content"] = content + [{"type": "text", "text": note}]
    else:
        last["content"] = (content + "\n\n" if isinstance(content, str) and content else "") + note


# ═══════════════════════════════════════════════════════════════════════════
#  THINKING SWITCH — the client decides, the gateway carries it
# ═══════════════════════════════════════════════════════════════════════════
#
#  Clients turn a model's thinking on or off in different ways. LM Studio's
#  /v1/chat/completions honours only `reasoning_effort`: tested 2026-11-11
#  with a Gemma 4 model, "none" turned thinking off, while
#  chat_template_kwargs.enable_thinking and reasoning.effort were ignored.
#  So for LM Studio the gateway puts the client's own choice into
#  reasoning_effort. It never sets one the client didn't ask for, never
#  overrides a reasoning_effort the client sent, and leaves the client's
#  other fields in place.

_THINK_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _think_level(value):
    if isinstance(value, str) and value.strip():
        value = value.strip().lower()
        return value if value in _THINK_EFFORTS else "other"
    return None


def _thinking_asked(data):
    """What the client asked for: a level from _THINK_EFFORTS ("none" is off), "on"
    (thinking wanted, no level), "other" (a value the gateway doesn't know), or None."""
    if not isinstance(data, dict):
        return None
    if data.get("reasoning_effort") is not None:                     # OpenAI
        return _think_level(data["reasoning_effort"]) or "other"
    reasoning = data.get("reasoning")                                # OpenRouter, Responses API
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            return "none"
        level = _think_level(reasoning.get("effort"))
        if level:
            return level
        if reasoning.get("enabled") is True or reasoning.get("max_tokens"):
            return "on"
    kwargs = data.get("chat_template_kwargs")                        # llama.cpp, vLLM
    if isinstance(kwargs, dict):
        for key in ("enable_thinking", "thinking"):
            if kwargs.get(key) is False:
                return "none"
            if kwargs.get(key) is True:
                return "on"
    thinking = data.get("thinking")                                  # Anthropic
    if isinstance(thinking, dict):
        kind = str(thinking.get("type") or "").lower()
        if kind == "disabled":
            return "none"
        if kind in ("enabled", "adaptive"):
            return "on"
    think = data.get("think")                                        # Ollama
    if think is False:
        return "none"
    if think is True:
        return "on"
    return _think_level(think)


def _lmstudio_thinking(data) -> bool:
    """Carry the client's thinking choice to LM Studio in the one field it honours.
    True when the request was changed."""
    if not isinstance(data, dict) or data.get("reasoning_effort") is not None:
        return False                 # the client's own reasoning_effort goes through as sent
    asked = _thinking_asked(data)
    if asked in _THINK_EFFORTS:
        data["reasoning_effort"] = asked
        return True
    return False                     # "on", "other" or nothing: LM Studio's load setting decides


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
        if self._access_denied(path):
            return

        # API endpoints handled locally
        _note_lookup(self.address_string(), "GET", path)
        if path == "/v1/models" or path == "/models":
            self._handle_models()
            return
        if path == "/api/v0/models" or path == "/api/v1/models":
            self._handle_lmstudio_models()
            return
        for prefix, handler in (("/v1/models/", self._handle_models), ("/models/", self._handle_models),
                                ("/api/v0/models/", self._handle_lmstudio_models),
                                ("/api/v1/models/", self._handle_lmstudio_models)):
            if path.startswith(prefix) and len(path) > len(prefix):
                handler(only=path[len(prefix):])
                return
        if path == "/api/tags":
            self._handle_ollama_tags()
            return
        if path == "/api/ps":
            self._handle_ollama_tags(loaded_only=True)
            return
        if path == "/props":
            query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()} if parsed.query else {}
            self._handle_props(query_params.get("model"))
            return
        if path == "/health" or path == "/v1/health":
            self._handle_health()
            return
        if path == "/stats":
            self._send_json(stats_snapshot())
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
        if self._access_denied():
            return

        if path == "/v1/streams/lookup":
            self._handle_streams_lookup()
            return
        if path == "/api/show":
            _note_lookup(self.address_string(), "POST", path)
            self._handle_ollama_show()
            return
        if path in ("/models/load", "/models/unload"):
            self._handle_model_load()
            return
        if path == "/cors-proxy":
            self._handle_cors_proxy()
            return

        self._proxy()

    def do_PUT(self):
        if not self._access_denied():
            self._proxy()

    def do_DELETE(self):
        if not self._access_denied():
            self._proxy()

    def _access_denied(self, path=None) -> bool:
        """Answer 401 when access keys are on and the request has none of them.
        True when the request was answered here. With `path`, a file of the web
        page is still served, so the page can load and ask for the key."""
        try:
            ok, _name = access_check(self.headers)
        except Exception:
            ok = False                                   # a broken check never opens the gateway
        if ok:
            return False
        if path is not None and self._serve_file(path.lstrip("/")):
            return True
        client = self.address_string()
        now = time.time()
        if now - _access_refusals_logged.get(client, 0) >= 60:
            if len(_access_refusals_logged) > 1000:
                _access_refusals_logged.clear()
            _access_refusals_logged[client] = now
            why = "a wrong access key" if _presented_key(self.headers) else "no access key"
            _queue_log_line(f"[{client}] refused {self.command} {urlparse(self.path).path}: {why} "
                            "(further refusals from this address are not logged for a minute)")
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if 0 < length <= _REFUSED_BODY_READ:
                self.rfile.read(length)
        except (ValueError, OSError):
            pass
        body = json.dumps({"error": {"message": "This gateway needs an API key. Send it as "
                                                "'Authorization: Bearer <key>' or 'x-api-key'.",
                                     "type": "invalid_request_error", "code": "invalid_api_key"}}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("WWW-Authenticate", 'Bearer realm="AI Gateway"')
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass
        return True

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
    def _handle_models(self, only=None):
        """Return aggregated model list from all online endpoints (or, with `only`, that one model)."""
        entries = []
        for m in get_models():
            meta = get_model_meta(m)
            # What fits now: a model loaded with less than its maximum must say so, or clients compact too late.
            max_context = meta.get("context_length", 128000)
            context_length = meta.get("loaded_context_length") or max_context

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
                "max_context_length": max_context,
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
        if only is not None:
            return self._send_one_model(entries, only)
        self._send_json({"object": "list", "data": entries})

    def _send_one_model(self, entries, wanted):
        """GET <list>/<model id>: that model's entry, as apps ask for it before a chat to learn its context
        window. Without this the lookup was forwarded to an engine and came back as an error, and the app
        fell back to its own default (128,000 in Cline and Goose)."""
        match = _find_model_entry(entries, wanted)
        if match is None:
            self._send_json({"error": {"message": f"The model '{wanted}' does not exist", "type": "invalid_request_error",
                                       "param": "model", "code": "model_not_found"}}, 404)
        else:
            self._send_json(match)

    # ── API: /api/v0/models (LM Studio compatible) ────────────────────────
    def _handle_lmstudio_models(self, only=None):
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
        if only is not None:
            return self._send_one_model(entries, only)
        self._send_json({"object": "list", "data": entries})

    # ── Ollama-style model info, for apps that ask an Ollama server what a model can hold ──
    def _ollama_models(self):
        out = []
        for m in get_models():
            meta = get_model_meta(m)
            window = meta.get("loaded_context_length") or meta.get("context_length") or 128000
            out.append({"name": m, "model": m, "window": int(window), "arch": str(meta.get("arch") or "llm"),
                        "quantization": str(meta.get("quantization") or ""),
                        "tools": "tool_use" in (meta.get("capabilities") or [])
                                 or "tools" in ((meta.get("supported_parameters") or []))})
        return out

    def _handle_ollama_tags(self, loaded_only=False):
        models = [{"name": m["name"], "model": m["model"], "size": 0, "digest": "",
                   "details": {"format": "gguf", "family": m["arch"], "quantization_level": m["quantization"]},
                   **({"context_length": m["window"]} if loaded_only else {})} for m in self._ollama_models()]
        self._send_json({"models": models})

    def _handle_ollama_show(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            asked = json.loads(self.rfile.read(length) or b"{}") if length <= _MAX_BODY_SIZE else {}
        except ValueError:
            asked = {}
        wanted = str((asked.get("model") or asked.get("name") or "") if isinstance(asked, dict) else "")
        match = _find_model_entry([{"id": m["name"], **m} for m in self._ollama_models()], wanted)
        if match is None:
            self._send_json({"error": f"model '{wanted}' not found"}, 404)
            return
        arch = match["arch"]
        self._send_json({
            "details": {"format": "gguf", "family": arch, "quantization_level": match["quantization"]},
            "model_info": {"general.architecture": arch, f"{arch}.context_length": match["window"]},
            "parameters": f"num_ctx {match['window']}",
            "capabilities": ["completion"] + (["tools"] if match["tools"] else []),
        })

    # ── API: /props ────────────────────────────────────────────────────────
    def _handle_props(self, model_id=None):
        """Return server properties using aggregated model metadata."""
        model = unquote(model_id) if model_id else get_current_model()
        meta = get_model_meta(model)

        n_ctx = meta.get("loaded_context_length") or meta.get("context_length", 128000)   # as loaded now
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
        health = {"status": "ok", "endpoints": ep_statuses}
        kv = kv_status_all()
        if kv:
            health["kv_sessions"] = kv
        self._send_json(health)

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
            # Closing with the body unread resets the connection on Windows, and the client can lose this reply
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                if 0 < length <= _REFUSED_BODY_READ:
                    self.rfile.read(length)
            except (ValueError, OSError):
                pass
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
        # override a header the prefixed headers already provided. With access
        # keys on, a plain key header is the gateway's own (llama-ui adds it),
        # so it stays here; the MCP server's auth comes prefixed.
        keys_on = bool(_access_keys())
        for header, value in self.headers.items():
            h = header.lower()
            if h in hop_by_hop or not value:
                continue
            if keys_on and h in ("authorization",) + _ACCESS_KEY_HEADERS:
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

    def _ctx_summarise(self, host, use_ssl, path, headers, model, verdict, cfg, timeout):
        """Ask the engine for a summary of the chat up to the last message the client cut. The prompt is the
        start of the chat the engine just held, so it mostly comes from its cache. None when it can't be made."""
        base = {**(verdict.get("params") or {}), "model": model, "messages": _ctx_summary_messages(verdict["prefix"]),
                "max_tokens": cfg["summary_tokens"], "temperature": 0.2, "stream": False}
        tools = verdict.get("tools")
        plain = {k: v for k, v in base.items() if k != "tools"}
        # First exactly as the client asked (same tools, no tool_choice): the same prompt start, so it comes
        # from the engine's cache. Only if that gives no text: forbid tools, then drop them.
        attempts = [base, {**base, "tool_choice": "none"}, plain] if tools else [base]
        for body in attempts:
            raw = json.dumps(body).encode("utf-8")
            if use_ssl:
                conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=timeout)
            else:
                conn = http.client.HTTPConnection(_dial_host(host), timeout=timeout)
            try:
                conn.request("POST", path, body=raw, headers={**headers, "Content-Length": str(len(raw))})
                response = conn.getresponse()
                data = response.read(_TAP_MAX_BODY)
                if response.status != 200:
                    continue
                message = json.loads(data)["choices"][0]["message"]
                text = message.get("content") if isinstance(message.get("content"), str) else ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                if text:
                    return text[:_CTX_SUMMARY_MAX_CHARS]
            except Exception:  # noqa: BLE001 -- try the plainer request, then give up
                continue
            finally:
                conn.close()
        return None

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
        req_json = None

        if body:
            try:
                data = json.loads(body)
                req_json = data
                model_id = data.get("model", "")

                if not model_id:
                    # No model specified — use default
                    model_id = get_current_model()
                    data["model"] = model_id
                    body = json.dumps(data).encode("utf-8")
                    self._log(f"Injected default model: {model_id}")
                elif _forces_default_model(self.path):
                    # "Force" is ticked next to the default model: every
                    # client gets it, whatever model it asked for.
                    forced = get_current_model()
                    if forced and forced != model_id:
                        self._log(f"Forced default model: {forced} (client asked for {model_id})")
                        model_id = forced
                        data["model"] = model_id
                        body = json.dumps(data).encode("utf-8")

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

        # The client's thinking choice, in the form LM Studio honours (see THINKING SWITCH).
        if (ep_type == "lmstudio" and isinstance(req_json, dict)
                and urlparse(self.path).path.rstrip("/").endswith("/chat/completions")):
            try:
                if _lmstudio_thinking(req_json):
                    body = json.dumps(req_json).encode("utf-8")
            except Exception as e:  # the request goes as the client sent it
                self._log(f"thinking switch skipped: {e}")

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
        elif not _access_keys():
            # Still forward any client-provided auth. With access keys on,
            # the client's key is the gateway's, and never goes upstream.
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

        # Context guard (see CONTEXT LIMIT): say "full" while the whole chat still fits, and give
        # the client's next request what it cut. Any doubt or failure forwards the request as sent.
        ctx_note = None
        guard = None
        if (self.command == "POST" and isinstance(req_json, dict) and isinstance(req_json.get("messages"), list)
                and req_json["messages"] and path.rstrip("/").endswith("/chat/completions")):
            guard = _ctx_guard_config(target_endpoint)
        if guard:
            try:
                verdict = _ctx_guard(guard, ep_name, self.address_string(), req_json, body)
            except Exception as e:  # noqa: BLE001
                verdict = None
                self._log(f"context guard skipped: {type(e).__name__}")
            action = (verdict or {}).get("action")
            if action == "full":
                tokens, window = verdict["tokens"], verdict["window"]
                stat = _stats.begin(target_endpoint, self.command, path, req_json, self.address_string(), self.headers)
                if stat:
                    stat.status = 400
                    stat.overflow = {"prompt": tokens, "limit": window, "early": True}
                self._log(f"{ep_name}: context guard: about {tokens:,} of {window:,} tokens ({tokens * 100 // window}%), "
                          f"answered context_length_exceeded so the client compacts while the chat still fits")
                self._send_json({"error": {
                    "message": (f"This model's maximum context length is {window} tokens. Your messages resulted in about "
                                f"{tokens} tokens, past the {guard['at'] * 100:.0f}% mark where the gateway asks for compaction "
                                f"while the whole conversation still fits. Please reduce the length of the messages."),
                    "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded",
                    "n_prompt_tokens": tokens, "n_ctx": window}}, 400)
                _stats.end(stat)
                return
            if action:
                self._log(f"{ep_name}: context guard: the request after 'full' has {verdict['shape']}")
            try:
                if action == "restore":
                    req_json["messages"] = verdict["messages"]
                    body = json.dumps(req_json).encode("utf-8")
                    self._log(f"{ep_name}: context guard: put the {verdict['cut']} cut messages back, so the client's "
                              f"compaction reads the whole chat")
                elif action == "summarise":
                    summary = self._ctx_summarise(host, use_ssl, path, fwd_headers, req_json.get("model"), verdict, guard, read_timeout)
                    if summary:
                        ctx_note = _ctx_note(summary)
                        _ctx_append(req_json, ctx_note)
                        body = json.dumps(req_json).encode("utf-8")
                        self._log(f"{ep_name}: context guard: summarised {verdict['cut']} cut messages into "
                                  f"{len(summary):,} characters and added them to this turn")
                    else:
                        self._log(f"{ep_name}: context guard: no summary could be made; the request goes as the client cut it")
            except Exception as e:  # noqa: BLE001
                ctx_note = None
                self._log(f"context guard failed, request forwarded as sent: {type(e).__name__}")
            if body:
                fwd_headers["Content-Length"] = str(len(body))
        if use_ssl:
            conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=read_timeout)
        else:
            conn = http.client.HTTPConnection(_dial_host(host), timeout=read_timeout)

        bump_endpoint_load_start(ep_name)
        headers_sent = False
        streaming = False
        kv_ticket = None
        kv_ok = False
        client_gone = False
        stat = _stats.begin(target_endpoint, self.command, path, req_json, self.address_string(), self.headers)
        if stat and guard:
            stat.body_size = _ctx_size(body, req_json)      # with the reply's token count: bytes per token
        healer = _new_healer(target_endpoint, self.command, path, req_json)
        try:
            # Before forwarding: put the right conversation in the engine's slot
            # (kv_sessions endpoints only; holds the engine until _kv_end).
            kv_ticket = _kv_begin(target_endpoint, self.command, path, req_json, stat.app if stat else "")
            if stat and kv_ticket:
                stat.kv_event, stat.swap_s = kv_ticket.event, kv_ticket.swap_s
            conn.request(self.command, path, body=body, headers=fwd_headers)
            response = conn.getresponse()
            streaming = "text/event-stream" in (response.getheader("Content-Type") or "")
            # The engine took the request: it holds this conversation now, even
            # if the client presses Stop halfway through the stream.
            kv_ok = response.status < 400
            if stat:
                stat.status = response.status
            if response.status >= 400:
                healer = None

            # A reply that isn't streamed has to be whole before it can be judged.
            whole_body = None
            if response.status in _OVERFLOW_STATUSES and not streaming:
                whole_body = bytearray(response.read(_OVERFLOW_MAX_BODY))    # an error body: small, read to see why
            elif (healer or ctx_note) and not streaming:
                whole_body = bytearray()
                while len(whole_body) <= _TAP_MAX_BODY:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    whole_body.extend(chunk)
                else:
                    healer = None       # too big to judge: the rest follows below as it came
            first_bytes = bytes(whole_body) if whole_body is not None else b""
            if first_bytes and stat:
                stat.tap.feed(first_bytes, False)
            replaced = healer.rewrite_body(first_bytes) if healer and not streaming else None
            send_status = response.status
            if ctx_note and not streaming and response.status == 200 and first_bytes:
                try:                                         # the summary in front of the answer, where the client keeps it
                    whole = json.loads(replaced if replaced is not None else first_bytes)
                    message = whole["choices"][0]["message"]
                    message["content"] = ctx_note + "\n\n" + (message.get("content") or "")
                    replaced = json.dumps(whole, ensure_ascii=False).encode("utf-8")
                except Exception:  # noqa: BLE001
                    pass
            if response.status in _OVERFLOW_STATUSES and first_bytes:
                try:
                    overflow = _context_overflow(first_bytes, response.status, path)
                except Exception:  # noqa: BLE001 -- the engine's error then goes as it came
                    overflow = None
                if overflow:
                    if overflow[0] is not None:
                        replaced, send_status = overflow[0], 400
                    if stat:
                        stat.overflow = overflow[1]
                        stat.status = send_status
                    self._log(f"{ep_name}: the prompt doesn't fit the model's context"
                              + (f" ({overflow[1]['prompt']:,} of {overflow[1]['limit']:,} tokens)" if overflow[1]["prompt"] and overflow[1]["limit"] else "")
                              + ": answered with context_length_exceeded so the client can compact")

            # Send status + CORS + upstream response headers
            self.send_response(send_status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            for header, value in response.getheaders():
                if header.lower() in ("transfer-encoding", "connection"):
                    continue
                if header.lower() == "content-length" and (healer or replaced is not None or ctx_note):
                    continue            # the healer may change the length
                self.send_header(header, value)
            if replaced is not None:
                self.send_header("Content-Length", str(len(replaced)))
            self.end_headers()
            headers_sent = True
            if ctx_note and streaming and response.status == 200:
                first = {"id": "chatcmpl-gateway-context", "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": str(req_json.get("model") or ""),
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": ctx_note + "\n\n"}, "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(first, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.wfile.flush()

            # Stream response body (SSE-safe): read1 hands over whatever has
            # arrived instead of waiting for a full buffer, so tokens reach
            # the client as the upstream sends them.
            pending = replaced if replaced is not None else first_bytes
            while True:
                if pending:
                    chunk, pending = pending, b""
                    outgoing = chunk
                elif replaced is not None:
                    break
                else:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    if stat:
                        stat.tap.feed(chunk, streaming)
                    outgoing = healer.feed(chunk) if healer else chunk
                try:
                    if outgoing:
                        self.wfile.write(outgoing)
                        self.wfile.flush()
                except OSError:
                    client_gone = True
                    raise
            if healer and streaming:
                outgoing = healer.finish()
                try:
                    if outgoing:
                        self.wfile.write(outgoing)
                        self.wfile.flush()
                except OSError:
                    client_gone = True
                    raise
            if healer and stat:
                stat.heal = healer.result

            if not stat:  # a counted request gets a fuller line when its numbers are in
                self._log(f"→ {ep_name} [{response.status}]")

        except Exception as e:
            if stat:
                stat.failed = not client_gone
                stat.cancelled = client_gone
            self._log(f"Error → {ep_name}: {e}")
            try:
                if not headers_sent:
                    # OpenAI-style JSON, not an HTML error page: API clients
                    # (Cline, the OpenAI SDK) show error.message to the user.
                    if isinstance(e, ConnectionRefusedError):
                        msg = (f"Endpoint '{ep_name}' refused the connection at {host}: "
                               f"it isn't running or isn't listening there")
                    else:
                        msg = f"Bad Gateway ({ep_name}): {e}"
                    self._send_json({"error": {"message": msg, "type": "upstream_error",
                                               "code": 502}}, 502)
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
            _kv_end(kv_ticket, kv_ok)
            _stats.end(stat)

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

def _heal_label(ep: dict) -> str:
    """The Settings table's Heal cell: whether tool call healing (and its guard) is on."""
    cfg = ep.get("tool_heal")
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return "—"
    label = "✓ no guard" if cfg.get("guard", True) is False else "✓"
    return label + " ↻" if cfg.get("stop_repeats") else label


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

            heal = self._heal_cfg()
            heal_box = Checkbox("Heal tool calls", value=bool(heal.get("enabled")), id="ep-heal-checkbox")
            heal_box.tooltip = ("Turn tool calls a model writes out as text into real calls, so the app runs them. "
                                "For local models (tool_heal in config.json).")
            yield heal_box
            guard_box = Checkbox("Check commands before they run (guard)", value=heal.get("guard", True) is not False,
                                 id="ep-guard-checkbox", disabled=not heal.get("enabled"))
            guard_box.tooltip = ("Stop tool calls that would delete a root or home folder, run a damaged path, "
                                 "or a cut-off command. Works only with healing on.")
            yield guard_box
            repeat_box = Checkbox("Stop a call the model keeps repeating", value=bool(heal.get("stop_repeats")),
                                  id="ep-repeat-checkbox", disabled=not heal.get("enabled"))
            repeat_box.tooltip = (f"After the same call {_REPEAT_LIMIT} times in a row, the next one isn't run: the model is told "
                                  "to use the result it has or change the call. Works only with healing on "
                                  "(stop_repeats and repeat_limit in config.json).")
            yield repeat_box
            ctx = (self.endpoint or {}).get("context_guard")
            ctx_box = Checkbox("Compact early (say 'full' at 90%)",
                               value=bool(isinstance(ctx, dict) and ctx.get("enabled")), id="ep-ctx-checkbox")
            ctx_box.tooltip = ("For apps that only compact after the context overflows. The gateway answers 'context full' "
                               "while the whole chat still fits, then gives the app's next request what it cut "
                               "(context_guard in config.json).")
            yield ctx_box

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

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "ep-heal-checkbox":
            self.query_one("#ep-guard-checkbox", Checkbox).disabled = not event.value  # the guard runs inside the healer
            self.query_one("#ep-repeat-checkbox", Checkbox).disabled = not event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ep-save-btn":
            self._save()
        elif event.button.id == "ep-cancel-btn":
            self.dismiss(None)

    def action_cancel(self):
        self.dismiss(None)

    def _heal_cfg(self) -> dict:
        cfg = (self.endpoint or {}).get("tool_heal")
        return cfg if isinstance(cfg, dict) else {}

    def _heal_result(self, heal_on: bool, guard_on: bool, repeats_on=None):
        """The endpoint's tool_heal after the dialog: its other keys (allow_rules, repeat_limit) kept,
        and nothing written for an endpoint that never had it and still doesn't."""
        heal = dict(self._heal_cfg())
        if not heal_on and not heal:
            return None
        heal["enabled"] = heal_on
        if repeats_on:
            heal["stop_repeats"] = True
        elif repeats_on is not None:
            heal.pop("stop_repeats", None)  # off is the default
        if guard_on:
            heal.pop("guard", None)  # on is the default
        else:
            heal["guard"] = False
        return heal

    def _save(self):
        ep_type = str(self.query_one("#ep-type-select", Select).value)
        name = self.query_one("#ep-name-input", Input).value.strip()
        host = self.query_one("#ep-host-input", Input).value.strip()
        api_key = self.query_one("#ep-key-input", Input).value.strip()
        enabled = self.query_one("#ep-enabled-checkbox", Checkbox).value
        heal = self._heal_result(self.query_one("#ep-heal-checkbox", Checkbox).value,
                                 self.query_one("#ep-guard-checkbox", Checkbox).value,
                                 self.query_one("#ep-repeat-checkbox", Checkbox).value)

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
        if heal is not None:
            result["tool_heal"] = heal
        ctx_on = self.query_one("#ep-ctx-checkbox", Checkbox).value
        ctx = (self.endpoint or {}).get("context_guard")
        if ctx_on or isinstance(ctx, dict):         # its other keys (at, summary_tokens) are kept
            result["context_guard"] = {**(ctx if isinstance(ctx, dict) else {}), "enabled": ctx_on}
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
            with Horizontal(id="default-model-row"):
                yield Select(
                    model_options,
                    value=get_current_model() if current_models else "",
                    id="model-select",
                )
                force_box = Checkbox(
                    "Force",
                    value=bool(get_config("force_default_model")),
                    id="force-model-checkbox",
                )
                force_box.tooltip = "Use the default model for every chat request, whatever model the client asks for"
                yield force_box
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

            # ── Access keys ────────────────────────────────────────────
            yield Static("Access Keys", classes="section-title")
            yield Label("Keys apps must send to use the gateway, separated by commas. Empty = open to everyone on the network.")
            with Horizontal(id="access-keys-row"):
                yield Input(
                    value=", ".join(k for _, k in _access_keys()),
                    placeholder="Empty: no key needed",
                    id="access-keys-input",
                )
                yield Button("+ New key", id="access-key-new-btn", variant="primary", compact=True)
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
        elif event.button.id == "access-key-new-btn":
            box = self.query_one("#access-keys-input", Input)
            current = box.value.strip().rstrip(",").strip()
            box.value = f"{current}, {new_access_key()}" if current else new_access_key()
            self.notify("New key added. Press Ctrl+S to save it, then give it to your apps.")

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
                # Settings the dialog doesn't show (timeout, kv_sessions) stay.
                for i, ep in enumerate(endpoints):
                    if ep.get("name", "") == original_name:
                        endpoints[i] = {**ep, **result}
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
        table.add_columns("Name", "Type", "Host", "Status", "Heal", "Load")
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
                _heal_label(ep),
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

        # Force default model checkbox
        force = self.query_one("#force-model-checkbox", Checkbox).value
        if force != bool(get_config("force_default_model")):
            set_config("force_default_model", force)
            _queue_log_line("Default model is now forced for all chat requests" if force
                            else "Default model no longer forced")

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

        # Access keys
        typed = self.query_one("#access-keys-input", Input).value
        result, short = _access_keys_result(typed, get_config("access_keys"))
        if short:
            self.notify(f"{short} key(s) shorter than {_ACCESS_KEY_MIN} characters ignored. "
                        "Press '+ New key' for a strong one.", severity="error")
        if result != get_config("access_keys"):
            set_config("access_keys", result)
            n = len(result or [])
            _queue_log_line(f"Access keys: {n} key(s), apps must send one" if n
                            else "Access keys off: the gateway is open to everyone on the network")

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

# ── Console rendering: pure functions from a stats snapshot to Rich text ──
# (Rich Text, never markup: model and app names come from clients.)

_GIB = 1024 ** 3
_LABEL_WIDTH = 10
# Plain ANSI red and friends are too dark to read on the console's background
_RED, _RED_BOLD = "#ff6b6b", "bold #ff6b6b"
_YELLOW, _YELLOW_BOLD = "#ffd166", "bold #ffd166"
_GREEN, _GREEN_BOLD = "#5fd787", "bold #5fd787"
_CYAN = "#5fd7ff"
_MAGENTA = "#d7a0ff"


def _line(label: str, *parts) -> RichText:
    """A labelled line. A part is plain text, or (text, style)."""
    line = RichText()
    line.append(f"{label:<{_LABEL_WIDTH}}", style="dim")
    for part in parts:
        if isinstance(part, tuple):
            line.append(str(part[0]), style=part[1] or "")
        else:
            line.append(str(part))
    return line


def _lines(rows) -> RichText:
    return RichText("\n", no_wrap=True, overflow="ellipsis").join(rows)


def _pct(part, whole):
    return part * 100 / whole if part is not None and whole else None


def _pct_style(value) -> str:
    if value is None:
        return "dim"
    return _GREEN if value >= 80 else _YELLOW if value >= 40 else "dim"  # a first read at 0% is normal


def _temp_style(c) -> str:
    return "dim" if c is None else _RED_BOLD if c >= 90 else _YELLOW if c >= 85 else _GREEN


def _load_style(fraction) -> str:
    return "dim" if fraction is None else _RED_BOLD if fraction >= 0.97 else _YELLOW if fraction >= 0.9 else _CYAN


def _fmt_pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}%"


def _ago(ts, now=None) -> str:
    if not ts:
        return "—"
    return fmt_secs(max(0, (now or time.time()) - ts)) + " ago"


def _main_engine(snap: dict):
    """(endpoint name, engine view) of the engine to feature: the first that
    answers, or (name, None) when one is configured but silent."""
    engines = snap.get("engines") or {}
    for name, view in engines.items():
        if view.get("ok"):
            return name, view
    return (next(iter(engines)), None) if engines else (None, None)


def _main_kv(snap: dict, name) -> dict:
    kv = snap.get("kv_sessions") or {}
    if name in kv:
        return kv[name]
    return next(iter(kv.values()), {}) if kv else {}


def _expert_hits(engine: dict):
    """(last request's expert cache hit %, average of the requests the engine lists)."""
    rates = [r["hit_rate"] * 100 for r in engine.get("requests", []) if isinstance(r.get("hit_rate"), (int, float))]
    return (rates[0], sum(rates) / len(rates)) if rates else (None, None)


def _state_parts(engine: dict, spark_width: int = 14) -> list:
    """The engine's state as line parts: reading, generating, idle or not loaded."""
    live = engine.get("live") or {}
    state = live.get("state")
    if state == "reading":
        read, total, rate = live.get("prompt_read"), live.get("prompt_total"), live.get("prefill_tok_s_mean")
        parts = [("◐ reading ", _YELLOW_BOLD)]
        if read is not None and total:
            parts += [(meter(read / total, 10), _YELLOW), f" {fmt_count(read)}/{fmt_count(total)}"]
            if rate:
                parts.append(f" · {rate:.0f} t/s · {fmt_secs((total - read) / rate)} left")
        return parts
    if state == "generating":
        tok_s = live.get("tok_s")
        parts = [("● generating ", _GREEN_BOLD)]
        if tok_s:
            parts.append((f"{tok_s:.1f} t/s ", "bold"))
        if live.get("generated"):
            parts.append(f"· {fmt_count(live['generated'])} tokens ")
        return parts + [(spark((engine.get("history") or {}).get("tok_s"), spark_width), _GREEN)]
    if state == "unloaded":
        return [("○ not loaded", "dim")]
    return [("○ idle", "dim")]


def _plural(count, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _chat_parts(kv: dict, engine, show_saved: bool = True) -> list:
    held = kv.get("held")
    if not held:
        return [("none tracked yet", "dim")]  # after a restart the engine may still hold one
    context = ((engine or {}).get("info") or {}).get("max_context")
    parts = [(str(held.get("id") or "new")[:6], "bold")]
    if held.get("app"):
        parts.append(f" {held['app'][:14]}")
    parts.append(f" {fmt_count(held.get('tokens'))}")
    if context:
        fraction = (held.get("tokens") or 0) / context
        parts += [f"/{fmt_count(context)} ", (meter(fraction, 8), _load_style(fraction)), f" {fraction * 100:.0f}%"]
    if show_saved and not held.get("saved"):
        parts.append((" unsaved", "dim"))
    return parts


def _gpu_parts(engine: dict) -> list:
    hw = engine.get("hardware") or {}
    if hw.get("gpu_util") is None and hw.get("gpu_temp") is None:
        return [("no GPU readings", "dim")]
    used, total = hw.get("gpu_mem_used"), hw.get("gpu_mem_total")
    fraction = used / total if used is not None and total else None
    parts = [f"{hw.get('gpu_util', 0):.0f}% ", (f"{hw.get('gpu_temp', 0):.0f}°C", _temp_style(hw.get("gpu_temp")))]
    if hw.get("gpu_power") is not None:
        parts.append(f" {hw['gpu_power']:.0f}/{hw.get('gpu_power_limit') or 0:.0f} W")
    if fraction is not None:
        parts += [" · VRAM ", (f"{used / _GIB:.1f}/{total / _GIB:.1f} GB", _load_style(fraction))]
    return parts


def render_live(snap: dict) -> RichText:
    """The live panel beside the endpoints on the main screen."""
    gw = snap.get("gateway") or {}
    name, engine = _main_engine(snap)
    kv = _main_kv(snap, name)
    rows = []
    if engine:
        rows.append(_line("Engine", *_state_parts(engine, spark_width=6)))
        rows.append(_line("Chat", *_chat_parts(kv, engine, show_saved=False)))
        totals = engine.get("totals") or {}
        prompt = _pct(totals.get("reused"), totals.get("prompt_tokens"))
        hit = _expert_hits(engine)[0]
        drafts = _pct(totals.get("drafts_accepted"), totals.get("drafts_offered"))
        rows.append(_line("Cache", "prompt ", (_fmt_pct(prompt), _pct_style(prompt)),
                          " · experts ", (_fmt_pct(hit, 0), _pct_style(hit)),
                          " · drafts ", (_fmt_pct(drafts, 0), _CYAN)))
    else:
        rows.append(_line("Engine", ("no engine metrics" if name else "no engine with metrics", "dim")))
        session = gw.get("session") or {}
        prompt = _pct(session.get("cached"), session.get("prompt"))
        rows.append(_line("Cache", "prompt ", (_fmt_pct(prompt), _pct_style(prompt))))
    if kv:
        rows.append(_line("Swaps", f"{kv.get('continued', 0)} kept · {kv.get('restores', 0)} restored · "
                                   f"{_plural(kv.get('sessions', 0), 'chat')}, {fmt_bytes(kv.get('bytes', 0))}"))
        if kv.get("suspended"):
            rows.append(_line("Swap", ("off until the model is reloaded: ", _YELLOW_BOLD), (kv["suspended"]["reason"], _YELLOW)))
    life = gw.get("all_time") or {}
    if life.get("saved_s"):
        rows.append(_line("Saved", (f"≈ {fmt_secs(life['saved_s'])}", _GREEN_BOLD), " of prompt reading"))
    if engine:
        rows.append(_line("GPU", *_gpu_parts(engine)))
    today = gw.get("today") or {}
    errors = today.get("errors", 0)
    rows.append(_line("Today", f"{today.get('requests', 0)} req · {fmt_count(today.get('prompt', 0))} in · "
                               f"{fmt_count(today.get('out', 0))} out · ", (_plural(errors, "error"), _RED_BOLD if errors else "dim")))
    if today.get("healed") or today.get("stopped"):
        rows.append(_line("Tools", f"{_plural(today.get('healed', 0), 'call')} healed · ",
                          (f"{today.get('stopped', 0)} stopped", _YELLOW_BOLD if today.get("stopped") else "dim")))
    recent = gw.get("recent") or []
    if recent:
        last = recent[0]
        route = f"{last['app'][:14]} → {last['endpoint'][:12]} "
        if last.get("error"):
            rows.append(_line("Last", route, (f"failed {last.get('status') or ''}".strip(), _RED_BOLD)))
        else:
            speed = f"{last['tok_s']:.1f} t/s" if last.get("tok_s") else fmt_secs(last.get("duration_s"))
            rows.append(_line("Last", route + f"{fmt_count(last.get('prompt'))} in {fmt_count(last.get('out'))} out {speed}"))
    return _lines(rows)


def panel_engine(snap: dict):
    gw = snap.get("gateway") or {}
    name, engine = _main_engine(snap)
    kv = _main_kv(snap, name)
    rows = []
    if engine:
        info = engine.get("info") or {}
        live = engine.get("live") or {}
        rows.append(_line("Model", (engine.get("model") or "—", "bold"),
                          (f"  engine {info['version']}" if info.get("version") else "", "dim")))
        rows.append(_line("State", *_state_parts(engine)))
        if live.get("elapsed_s") is not None:
            rows.append(_line("This turn", f"{fmt_secs(live['elapsed_s'])} so far"))
        rows.append(_line("Chat", *_chat_parts(kv, engine)))
    else:
        message = "not reachable" if name else "none configured with metrics"
        rows.append(_line("State", (f"engine {message}", "dim")))
    active = gw.get("active") or []
    queued = (engine.get("live", {}).get("queued") or 0) if engine else 0
    if active:
        rows.append(_line("In flight", f"{len(active)} · " + ", ".join(sorted({a['app'][:14] for a in active}))[:60]))
    else:
        rows.append(_line("In flight", ("none", "dim")))
    if queued:
        rows.append(_line("Queued", (str(queued), _YELLOW)))
    rows.append(_line("Gateway", f"up {fmt_secs(gw.get('uptime_s'))}"))
    if kv.get("last_event"):
        rows.append(_line("Last swap", (kv["last_event"], "dim")))
    if kv.get("paused"):
        rows.append(_line("Swap", (f"paused for {kv['paused']} s (see the log)", _YELLOW)))
    if kv.get("suspended"):
        s = kv["suspended"]
        left = fmt_secs(max(0, s["until"] - time.time()))
        rows.append(_line("Swap", (f"off until the model is reloaded, or for {left}", _YELLOW_BOLD)))
    error = kv.get("last_error")
    if error:
        rows.append(_line("Swap error", (f"{error['action']} failed {_ago(error['time'], time.time())}: "
                                         f"{error['status'] or ''} {error['message']}".replace("  ", " "), _RED if kv.get("suspended") else "dim")))
    return _lines(rows)


def panel_cache(snap: dict):
    gw = snap.get("gateway") or {}
    name, engine = _main_engine(snap)
    kv = _main_kv(snap, name)
    rows = []
    if engine:
        totals = engine.get("totals") or {}
        prompt = _pct(totals.get("reused"), totals.get("prompt_tokens"))
        rows.append(_line("Prompt", (_fmt_pct(prompt), _pct_style(prompt)), " ",
                          (meter((prompt or 0) / 100, 12), _pct_style(prompt)),
                          f" {fmt_count(totals.get('reused'))} of {fmt_count(totals.get('prompt_tokens'))} reused"))
        last_hit, avg_hit = _expert_hits(engine)
        info = engine.get("info") or {}
        expert_tail = (f" · {fmt_count(info.get('expert_slots'))} in VRAM ({fmt_bytes((info.get('expert_cache_mib') or 0) * 1024 ** 2)})"
                       if info.get("expert_slots") else "")
        rows.append(_line("Experts", (_fmt_pct(last_hit), _pct_style(last_hit)), " last",
                          f" · {_fmt_pct(avg_hit)} avg{expert_tail}"))
        drafts = _pct(totals.get("drafts_accepted"), totals.get("drafts_offered"))
        rows.append(_line("Drafts", (_fmt_pct(drafts), _CYAN), " ", (meter((drafts or 0) / 100, 12), _CYAN),
                          f" {fmt_count(totals.get('drafts_accepted'))} of {fmt_count(totals.get('drafts_offered'))} kept"))
        parking = engine.get("parking") or {}
        if parking.get("enabled"):
            rows.append(_line("In RAM", f"{parking.get('parked', 0)} of {parking.get('slots', 0)} chats parked · "
                                        f"{fmt_bytes(parking.get('bytes'))} of {fmt_bytes((parking.get('budget_mib') or 0) * 1024 ** 2)}"
                                        f" · {parking.get('evictions', 0)} evicted"))
    session = gw.get("session") or {}
    if session.get("prompt"):
        share = _pct(session.get("cached"), session.get("prompt"))
        rows.append(_line("Gateway", (_fmt_pct(share), _pct_style(share)),
                          f" of {fmt_count(session['prompt'])} tokens cached, all endpoints"))
    if kv:
        seen = kv.get("continued", 0) + kv.get("restores", 0) + kv.get("new", 0)
        ready = _pct(kv.get("continued", 0) + kv.get("restores", 0), seen)
        rows.append(_line("Chats", (_fmt_pct(ready), _pct_style(ready)),
                          f" ready: {kv.get('continued', 0)} kept, {kv.get('restores', 0)} restored, {kv.get('new', 0)} new"))
        avg_save = kv.get("save_s", 0) / kv["saves"] if kv.get("saves") else None
        avg_restore = kv.get("restore_s", 0) / kv["restores"] if kv.get("restores") else None
        rows.append(_line("On disk", f"{_plural(kv.get('sessions', 0), 'chat')}, {fmt_bytes(kv.get('bytes', 0))} · "
                                     f"{fmt_bytes(kv.get('written_bytes', 0))} written"))
        rows.append(_line("Swap time", f"save {fmt_secs(avg_save)} · restore {fmt_secs(avg_restore)} (average)"))
    life = gw.get("all_time") or {}
    if life.get("saved_s"):
        rows.append(_line("Saved", (f"≈ {fmt_secs(life['saved_s'])}", _GREEN_BOLD),
                          f" of reading, {fmt_count(life.get('cached'))} tokens not read again"))
    if not rows:
        rows.append(_line("Prompt", ("no requests yet", "dim")))
    return _lines(rows)


def panel_speed(snap: dict):
    gw = snap.get("gateway") or {}
    _name, engine = _main_engine(snap)
    rows = []
    median, best = gw.get("tok_s_p50"), gw.get("tok_s_best")
    rows.append(_line("Output", (f"{median:.1f} t/s" if median else "—", "bold"), " median",
                      f" · {best:.1f} best" if best else ""))
    history = (engine or {}).get("history") or {}
    if history.get("tok_s"):
        rows.append(_line("  60 s", (spark(history["tok_s"], 44) or "", _GREEN)))
    totals = (engine or {}).get("totals") or {}
    read = (totals.get("prompt_tokens") or 0) - (totals.get("reused") or 0)
    read_rate = gw.get("read_tok_s_p50") or (read / (totals["prompt_ms"] / 1000) if totals.get("prompt_ms") else None)
    rows.append(_line("Reading", f"{read_rate:.0f} t/s" if read_rate else "—", " prompt speed"))
    rows.append(_line("First tok", (fmt_secs(gw.get("ttft_p50")), "bold"), " median",
                      f" · {fmt_secs(gw.get('ttft_p95'))} p95" if gw.get("ttft_p95") is not None else ""))
    per_minute = [m["requests"] for m in gw.get("per_minute") or []]
    rows.append(_line("Req/min", (spark(per_minute, 44) if any(per_minute) else "none in the last hour", _CYAN if any(per_minute) else "dim")))
    per_day = gw.get("per_day") or []
    if per_day:
        rows.append(_line("Tok/day", (spark([d["in"] + d["out"] for d in per_day], 14), _MAGENTA),
                          f"  {fmt_count(per_day[-1]['in'] + per_day[-1]['out'])} today"))
    return _lines(rows)


def panel_hardware(snap: dict):
    _name, engine = _main_engine(snap)
    if not engine:
        return _lines([_line("Hardware", ("no engine metrics", "dim"))])
    hw = engine.get("hardware") or {}
    history = engine.get("history") or {}
    rows = []
    gpu = (engine.get("gpu_name") or "GPU").replace("NVIDIA ", "")
    rows.append(_line("GPU", f"{hw.get('gpu_util', 0):.0f}% ", (spark(history.get("gpu_util"), 26, top=100), _GREEN), f" {gpu}"))
    rows.append(_line("Temp", (f"{hw.get('gpu_temp', 0):.0f}°C ", _temp_style(hw.get("gpu_temp"))),
                      (spark(history.get("gpu_temp"), 26, top=100), _temp_style(hw.get("gpu_temp")))))
    if hw.get("gpu_power") is not None and hw.get("gpu_power_limit"):
        rows.append(_line("Power", (meter(hw["gpu_power"] / hw["gpu_power_limit"], 12), _CYAN),
                          f" {hw['gpu_power']:.0f} / {hw['gpu_power_limit']:.0f} W"))
    if hw.get("gpu_mem_used") is not None and hw.get("gpu_mem_total"):
        fraction = hw["gpu_mem_used"] / hw["gpu_mem_total"]
        rows.append(_line("VRAM", (meter(fraction, 12), _load_style(fraction)),
                          f" {hw['gpu_mem_used'] / _GIB:.1f} / {hw['gpu_mem_total'] / _GIB:.1f} GB"))
    if hw.get("gpu_pcie_gen"):
        rows.append(_line("PCIe", f"Gen{hw['gpu_pcie_gen']} x{hw.get('gpu_pcie_width')} · "
                                  f"rx {hw.get('gpu_pcie_rx_mb') or 0:.0f} · tx {hw.get('gpu_pcie_tx_mb') or 0:.0f} MB/s"))
    if hw.get("cpu") is not None:
        rows.append(_line("CPU", f"{hw['cpu']:.0f}% ", (spark(history.get("cpu"), 26, top=100), _CYAN)))
    if hw.get("ram_used") is not None and hw.get("ram_total"):
        fraction = hw["ram_used"] / hw["ram_total"]
        rows.append(_line("RAM", (meter(fraction, 12), _load_style(fraction)),
                          f" {hw['ram_used'] / _GIB:.1f} / {hw['ram_total'] / _GIB:.1f} GB"))
    if hw.get("disk_read_mb") is not None:
        rows.append(_line("Disk", f"read {hw['disk_read_mb']:.0f} · write {hw.get('disk_write_mb') or 0:.0f} MB/s"))
    return _lines(rows)


def _table(*columns) -> RichTable:
    """A borderless table. A column is a name: ending in > it is right-aligned;
    ending in * it takes the spare width, and gives way first when space is short."""
    table = RichTable(box=None, expand=True, pad_edge=False, show_edge=False, header_style="bold dim", padding=(0, 1, 0, 0))
    for col in columns:
        name = col.rstrip(">*")
        table.add_column(name, justify="right" if col.endswith(">") else "left", no_wrap=True, overflow="ellipsis",
                         min_width=len(name) or None, ratio=1 if col.endswith("*") else None)
    return table


def _cached_cell(agg: dict):
    share = _pct(agg.get("cached"), agg.get("prompt"))
    return RichText(_fmt_pct(share, 0), style=_pct_style(share))


def panel_totals(snap: dict):
    gw = snap.get("gateway") or {}
    table = _table("*", "This run>", "Today>", "All time>")
    groups = [gw.get("session") or {}, gw.get("today") or {}, gw.get("all_time") or {}]
    table.add_row("Requests", *[f"{g.get('requests', 0):,}" for g in groups])
    table.add_row("Errors", *[RichText(f"{g.get('errors', 0):,}", style=_RED_BOLD if g.get("errors") else "dim") for g in groups])
    table.add_row("Tokens in", *[fmt_count(g.get("prompt", 0)) for g in groups])
    table.add_row("  cached", *[_cached_cell(g) for g in groups])
    table.add_row("Tokens out", *[fmt_count(g.get("out", 0)) for g in groups])
    table.add_row("Time saved", *[RichText("≈ " + fmt_secs(g.get("saved_s", 0)) if g.get("saved_s") else "—", style=_GREEN) for g in groups])
    if any(g.get("cost") for g in groups):
        table.add_row("Cloud cost", *[f"${g.get('cost', 0):.2f}" for g in groups])
    if any(g.get("healed") or g.get("stopped") for g in groups):
        table.add_row("Calls healed", *[f"{g.get('healed', 0):,}" for g in groups])
        table.add_row("Calls stopped", *[RichText(f"{g.get('stopped', 0):,}", style=_YELLOW_BOLD if g.get("stopped") else "dim") for g in groups])
    since = (gw.get("all_time") or {}).get("since")
    if since:
        return RichGroup(table, RichText(f"All time since {time.strftime('%d %b %Y', time.localtime(since))}", style="dim"))
    return table


def panel_chats(snap: dict):
    name, _engine = _main_engine(snap)
    kv = _main_kv(snap, name)
    chats = kv.get("chats") or []
    if not kv:
        return _lines([_line("Off", ("turn on kv_sessions on an endpoint to swap chats to disk", "dim"))])
    if not chats:
        return _lines([_line("None yet", ("a long chat is saved the first time you switch away from it", "dim"))])
    held = (kv.get("held") or {}).get("id")
    now = time.time()
    table = _table("", "Chat", "App*", "Tokens>", "Size>", "Restored>", "Used>")
    for chat in chats[:8]:
        table.add_row(RichText("●", style=_GREEN) if chat["id"] == held else "", chat["id"][:8], RichText(chat.get("app", "")[:16] or "—"),
                      fmt_count(chat["tokens"]), fmt_bytes(chat["bytes"]), f"{chat['restores']}×", _ago(chat["last_used"], now))
    return table


def panel_endpoints(snap: dict):
    gw = snap.get("gateway") or {}
    table = _table("Endpoint*", "Status", "Now>", "Requests>", "Errors>", "In>", "Cached>", "Out>", "First tok>", "Speed>", "Cost>")
    stats = gw.get("by_endpoint") or {}
    for ep in get_config("endpoints") or []:
        name = ep.get("name", "")
        status = get_endpoint_status(name)
        style = _GREEN if status == "online" else _YELLOW if status == "checking" else "dim"
        agg = stats.get(name) or {}
        active, _total = get_endpoint_load(name)
        table.add_row(RichText(name), RichText(status, style=style),
                      RichText(str(active), style=_YELLOW_BOLD) if active else "—",
                      f"{agg.get('requests', 0):,}" if agg else "—",
                      RichText(str(agg.get("errors", 0)), style=_RED_BOLD if agg.get("errors") else "dim") if agg else "—",
                      fmt_count(agg.get("prompt")), _cached_cell(agg), fmt_count(agg.get("out")),
                      fmt_secs(agg.get("ttft_p50")), f"{agg['tok_s_p50']:.1f} t/s" if agg.get("tok_s_p50") else "—",
                      f"${agg['cost']:.2f}" if agg.get("cost") else "—")
    return table


def _model_short(model: str) -> str:
    """The model's name without its vendor prefix."""
    return model.rstrip("/").rsplit("/", 1)[-1] or model


def _group_table(title: str, groups: dict, short=str):
    table = _table(title + "*", "Requests>", "In>", "Cached>", "Out>", "Used>")
    now = time.time()
    for name, agg in list(groups.items())[:6]:
        # Names come from clients: a plain str cell would be read as Rich markup
        table.add_row(RichText(short(name)), f"{agg['requests']:,}", fmt_count(agg["prompt"]), _cached_cell(agg),
                      fmt_count(agg["out"]), _ago(agg["last"], now))
    if not groups:
        table.add_row(RichText("none yet", style="dim"), "", "", "", "", "")
    return table


def panel_models(snap: dict):
    return _group_table("Model", (snap.get("gateway") or {}).get("by_model") or {}, _model_short)


def panel_apps(snap: dict):
    gw = snap.get("gateway") or {}
    return _group_table("App", gw.get("by_app") or gw.get("all_time_by_app") or {})


def panel_recent(snap: dict):
    table = _table("Time", "App", "Model*", "In>", "Cached>", "Out>", "First tok>", "Speed>", "Experts>", "Swap", "End",
                   "Status>")
    recent = (snap.get("gateway") or {}).get("recent") or []
    for r in recent[:12]:
        end = _reply_end(r)
        cached = _pct(r.get("cached"), r.get("prompt"))
        status = "stopped" if r.get("cancelled") else str(r.get("status") or "failed")
        if r.get("stopped"):
            status = f"guard {r.get('heal_rule')}"
        elif r.get("healed"):
            status += " healed"
        if r.get("overflow"):
            status, end = ("ctx early" if r.get("overflow_early") else "ctx full"), ("compact", "", True)
        used = r.get("window_used") or 0
        prompt_cell = fmt_count(r.get("prompt")) + (f" {used * 100:.0f}%" if used >= _WINDOW_WARN else "")
        swap = r.get("swap") or ""
        swap_text = {"restored": f"restored {fmt_secs(r.get('swap_s'))}", "continued": "kept",
                     "new": f"new, saved {fmt_secs(r.get('swap_s'))}" if r.get("swap_s") else "new"}.get(swap, "—")
        hit = r["expert_hit"] * 100 if isinstance(r.get("expert_hit"), (int, float)) else None
        table.add_row(time.strftime("%H:%M:%S", time.localtime(r["time"])), RichText(r["app"][:14]), RichText(_model_short(r["model"])),
                      RichText(prompt_cell, style=_YELLOW_BOLD if used >= _WINDOW_WARN else ""),
                      RichText(_fmt_pct(cached, 0), style=_pct_style(cached)),
                      ("~" if r.get("out_estimated") else "") + fmt_count(r.get("out")), fmt_secs(r.get("ttft_s")),
                      f"{r['tok_s']:.1f} t/s" if r.get("tok_s") else "—", RichText(_fmt_pct(hit, 0), style=_pct_style(hit)),
                      RichText(swap_text, style=_GREEN if swap == "restored" else "dim"),
                      RichText(end[0], style=_YELLOW if end[2] else "dim") if end else RichText("—", style="dim"),
                      RichText(status, style=_RED_BOLD if r.get("error") else "dim" if r.get("cancelled") else _GREEN))
    if not recent:
        table.add_row(RichText("no requests yet", style="dim"), *[""] * 11)
    return table


_DASHBOARD_PANELS = (
    ("panel-engine", "Engine", panel_engine, False),
    ("panel-cache", "Cache", panel_cache, False),
    ("panel-speed", "Speed", panel_speed, False),
    ("panel-hardware", "Hardware", panel_hardware, False),
    ("panel-totals", "Totals", panel_totals, False),
    ("panel-chats", "Saved chats", panel_chats, False),
    ("panel-endpoints", "Endpoints", panel_endpoints, True),
    ("panel-models", "Models (this run)", panel_models, False),
    ("panel-apps", "Apps (this run)", panel_apps, False),
    ("panel-recent", "Recent requests", panel_recent, True),
)


class StatsScreen(Screen):
    """The full dashboard: what the engine is doing, how the caches are paying
    off, the hardware, and where the traffic comes from."""

    BINDINGS = [
        ("d", "app.pop_screen", "Back"),
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(id="stats-scroll"):
            with Grid(id="stats-grid"):
                for widget_id, title, _render, wide in _DASHBOARD_PANELS:
                    panel = Static("", id=widget_id, classes="panel panel-wide" if wide else "panel")
                    panel.border_title = title
                    yield panel
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_panels()
        self.set_interval(1.0, self._refresh_panels)
        self.query_one("#stats-scroll").focus()

    def on_resize(self, event) -> None:
        self.query_one("#stats-grid").set_class(event.size.width < 110, "narrow")

    def _refresh_panels(self):
        snap = stats_snapshot()
        for widget_id, title, render, _wide in _DASHBOARD_PANELS:
            try:
                self.query_one(f"#{widget_id}", Static).update(render(snap))
            except Exception as e:  # a bad number must never take the console down
                self.query_one(f"#{widget_id}", Static).update(RichText(f"{title}: {e}", style="dim"))


class MainScreen(Screen):
    """Main screen: endpoints, live stats and the proxy log."""

    BINDINGS = [
        ("d", "open_dashboard", "Dashboard"),
        ("s", "open_settings", "Settings"),
        ("q", "quit", "Quit"),
    ]

    def _compose_overview(self) -> ComposeResult:
        """The top of the main screen: endpoints beside live stats. A subclass
        with its own layout yields this too, so the widget IDs stay the same."""
        with Horizontal(id="overview"):
            with Vertical(id="overview-left"):
                yield Static("Endpoints", classes="section-title")
                yield Static("Loading...", id="endpoint-status")
                yield Static("", id="server-info")
            with Vertical(id="overview-right"):
                yield Static("Live", classes="section-title")
                yield Static("", id="live-stats")

    def on_mount(self) -> None:
        log = self.query_one("#log", Log)
        ProxyHandler.tui_log = log
        self._update_endpoint_status()
        self._show_server_info()
        self._update_live_stats()
        self.set_interval(1.0, self._update_live_stats)
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
        forced = "  (forced for all chat requests)" if get_config("force_default_model") else ""
        n_keys = len(_access_keys())
        access = f"API key needed ({n_keys} key{'s' if n_keys != 1 else ''})" if n_keys else "open, no key needed"
        info.update(RichText(
            f"Listening : {url}\n"
            f"Web UI    : {url}/\n"
            f"Default   : {model}{forced}\n"
            f"Access    : {access}"
        ))

    def _update_live_stats(self):
        try:
            self.query_one("#live-stats", Static).update(render_live(stats_snapshot()))
        except Exception as e:  # a bad number must never take the console down
            self.query_one("#live-stats", Static).update(RichText(f"Stats unavailable: {e}", style="dim"))

    def on_resize(self, event) -> None:
        self.query_one("#overview").set_class(event.size.width < 120, "narrow")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(id="main-screen"):
            yield from self._compose_overview()
            yield Static("Proxy Log", classes="section-title")
            yield Log(id="log", highlight=True)
        yield Footer()

    def action_open_dashboard(self):
        self.app.push_screen(StatsScreen())

    def action_open_settings(self):
        self.app.push_screen(SettingsScreen())

    def _update_endpoint_status(self):
        """Refresh the endpoint status display (with per-endpoint load)."""
        status_widget = self.query_one("#endpoint-status", Static)
        endpoints = get_config("endpoints")
        lines = []
        # Column header (aligned to the row template below)
        lines.append(
            f"    {'ENDPOINT':<12} → {'HOST':<18} [{'STATUS':<8}] {'MODELS':>6}   LOAD"
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
                f"  {icon} {name:<12} → {host:<18} [{status:<8}] {model_count:>6}   {load_str}"
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
    #overview {
        height: auto;
        width: 100%;
    }
    #overview-left {
        width: 76;
        height: auto;
        margin-right: 2;
    }
    #overview-right {
        width: 1fr;
        height: auto;
    }
    #overview.narrow {
        layout: vertical;
    }
    #overview.narrow #overview-left, #overview.narrow #overview-right {
        width: 100%;
        margin-right: 0;
    }
    #live-stats {
        height: auto;
        width: 100%;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    #stats-scroll {
        height: 1fr;
        padding: 0 1;
    }
    #stats-grid {
        grid-size: 2;
        grid-rows: auto;
        grid-gutter: 0 1;
        height: auto;
    }
    #stats-grid.narrow {
        grid-size: 1;
    }
    .panel {
        height: auto;
        min-height: 3;
        border: round $primary;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 0 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    #stats-grid .panel {
        height: 100%;
    }
    .panel-wide {
        column-span: 2;
    }
    #stats-grid.narrow .panel-wide {
        column-span: 1;
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
    #settings-screen #default-model-row Select {
        width: 1fr;
    }
    #settings-screen #access-keys-input {
        width: 1fr;
        margin-right: 1;
    }
    #settings-screen #force-model-checkbox {
        width: auto;
        height: 1;
        border: none;
        padding: 0 1;
        margin: 0 0 0 1;
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
