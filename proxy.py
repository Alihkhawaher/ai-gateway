"""
AI Gateway — Multi-endpoint OpenAI-compatible proxy with Textual TUI.

Connects to multiple AI backends (OpenRouter, LM Studio, llama.cpp, etc.)
simultaneously, checks their availability, and presents a unified model
catalog to all clients. Model IDs follow the source/vendor/modelname format.

Includes an integrated web chat UI (llama.cpp's llama-ui) served at /.

Usage:
  python proxy.py              # listens on 0.0.0.0:8090
  python proxy.py 9090         # listens on 0.0.0.0:9090
  python proxy.py 9090 127.0.0.1   # custom bind address
"""

import json
import http.client
import mimetypes
import os
import ssl
import sys
import threading
import subprocess
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote, parse_qs

# ── Config file path ──────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# ── Web UI directory ─────────────────────────────────────────────────────
WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")

# ── Endpoint type defaults ───────────────────────────────────────────────
ENDPOINT_TYPE_DEFAULTS = {
    "openrouter": {"name": "openrouter", "host": "openrouter.ai"},
    "lmstudio":   {"name": "lmstudio",   "host": "localhost:1234"},
    "llama-server": {"name": "llama.cpp", "host": "localhost:8080"},
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
        "openrouter/openai/gpt-5.6-luna",
        "openrouter/qwen/qwen3.8-max",
        "openrouter/deepseek/deepseek-v4-flash-0731",
    ],
    "model_index": 0,
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

# ── Top intelligent models fetched from OpenRouter ────────────────────────
_top_intelligent_models = []
_top_models_lock = threading.Lock()

# ── Thread-safe shared config ──────────────────────────────────────────────
_config_lock = threading.Lock()
_config = {}  # loaded from config.json


def load_config():
    """Load config from config.json, merging with defaults. Handles v1→v2 migration."""
    global _config, MODELS
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
                print(f"[proxy] Could not load config: {e}")
        else:
            print(f"[proxy] No config file found, using defaults")
        _config = {**DEFAULTS, **_config}
        MODELS = list(_config.get("models", DEFAULTS["models"]))


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
    saved["models"] = [f"{name}/{m}" for m in old_models]

    # Remove old top-level fields
    saved.pop("host", None)
    saved.pop("api_key", None)
    return saved


def save_config():
    """Save current config to config.json."""
    with _config_lock:
        data = dict(_config)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[proxy] Could not save config: {e}")


def get_config(key: str):
    with _config_lock:
        return _config.get(key, DEFAULTS.get(key))


def set_config(key: str, value):
    with _config_lock:
        _config[key] = value


def get_current_model() -> str:
    with _config_lock:
        idx = max(0, min(_config.get("model_index", 0), len(MODELS) - 1))
        return MODELS[idx] if MODELS else ""


# ── Model ID parsing ─────────────────────────────────────────────────────
def parse_model_id(model_id: str) -> tuple:
    """Parse 'source/vendor/model' into (source_name, original_model_id).

    First segment = endpoint name (source)
    Remaining = original model ID as known by the upstream endpoint.
    Returns ("", model_id) if no source prefix found.
    """
    parts = model_id.split("/", 1)
    if len(parts) == 2:
        # Check if the first segment matches a configured endpoint name
        with _config_lock:
            ep_names = [ep.get("name", "") for ep in _config.get("endpoints", [])]
        if parts[0] in ep_names:
            return parts[0], parts[1]
    return "", model_id


def get_endpoint_by_name(name: str) -> dict:
    """Find an endpoint config by its name."""
    with _config_lock:
        for ep in _config.get("endpoints", []):
            if ep.get("name") == name and ep.get("enabled", False):
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
    "Access-Control-Allow-Headers": "Authorization, Content-Type, HTTP-Referer, X-Title",
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
#  HEALTH CHECKING & MODEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def _is_local_host(host: str) -> bool:
    """Check if a host is local."""
    _local = ("localhost", "127.0.0.1", "[::1]")
    return any(host == h or host.startswith(h + ":") for h in _local)


def _connect_host(host: str, timeout: int = 10):
    """Create an HTTP or HTTPS connection based on host."""
    if _is_local_host(host):
        return http.client.HTTPConnection(host, timeout=timeout)
    else:
        return http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=timeout)


def check_endpoint_health(ep: dict) -> str:
    """Check if an endpoint is reachable. Returns 'online' or 'offline'."""
    host = ep.get("host", "")
    ep_type = ep.get("type", "")
    api_key = ep.get("api_key", "")

    try:
        if ep_type == "openrouter":
            ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(host, context=ctx, timeout=10)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            conn.request("GET", "/api/v1/models", headers=headers)
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

        elif ep_type == "llama-server":
            conn = http.client.HTTPConnection(host, timeout=10)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

        elif ep_type == "lmstudio":
            conn = http.client.HTTPConnection(host, timeout=10)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return "online" if resp.status == 200 else "offline"

        elif ep_type == "custom":
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
    """Continuously check endpoint health every 30 seconds."""
    while True:
        endpoints = get_config("endpoints")
        for ep in endpoints:
            name = ep.get("name", "")
            if not ep.get("enabled", False):
                set_endpoint_status(name, "disabled")
                continue
            status = check_endpoint_health(ep)
            set_endpoint_status(name, status)
        # Re-aggregate models after health check
        aggregate_models()
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


def fetch_openrouter_models(host: str, api_key: str) -> list:
    """Fetch all models from OpenRouter."""
    try:
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
    except Exception as e:
        print(f"[proxy] Error fetching OpenRouter models: {e}")
        return []


def fetch_lmstudio_models(host: str) -> list:
    """Fetch loaded models from LM Studio."""
    try:
        conn = http.client.HTTPConnection(host, timeout=10)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return data.get("data", [])
    except Exception as e:
        print(f"[proxy] Error fetching LM Studio models: {e}")
        return []


def fetch_llama_server_models(host: str) -> list:
    """Fetch model info from llama.cpp server."""
    try:
        conn = http.client.HTTPConnection(host, timeout=10)
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

        return [{
            "id": model_name,
            "name": model_name,
            "context_length": props.get("default_generation_settings", {}).get("n_ctx", 128000),
            "architecture": {
                "input_modalities": _modalities_to_input(modalities),
                "output_modalities": ["text"],
            },
        }]
    except Exception as e:
        print(f"[proxy] Error fetching llama-server models: {e}")
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
        return data.get("data", [])
    except Exception as e:
        print(f"[proxy] Error fetching custom models from {host}: {e}")
        return []


def aggregate_models():
    """Fetch models from all online endpoints and build unified list.

    For cloud endpoints (OpenRouter): Only include models from the user's
    curated models list. Do NOT fetch the full catalog (would be 340+ models).

    For local endpoints (llama-server, LM Studio, custom): Include all
    discovered models since they typically have 1-5 loaded models.
    """
    global MODELS, _model_routes

    new_routes = {}
    new_meta = {}
    user_models = list(get_config("models"))
    user_model_set = set(user_models)

    endpoints = get_config("endpoints")
    for ep in endpoints:
        name = ep.get("name", "")
        if get_endpoint_status(name) != "online":
            continue

        ep_type = ep.get("type", "")
        host = ep.get("host", "")
        api_key = ep.get("api_key", "")

        if ep_type == "openrouter":
            # OpenRouter: Only fetch metadata for user's curated models.
            # Build list of original IDs that belong to this endpoint.
            or_original_ids = []
            for um in user_models:
                if um.startswith(f"{name}/"):
                    or_original_ids.append(um[len(name) + 1:])
            if not or_original_ids:
                continue
            # Fetch full catalog to get metadata, but only keep user's models
            all_or = fetch_openrouter_models(host, api_key)
            or_meta = {m.get("id", ""): m for m in all_or}
            for original_id in or_original_ids:
                meta = or_meta.get(original_id, {"id": original_id, "name": original_id})
                aggregated_id = f"{name}/{original_id}"
                new_routes[aggregated_id] = (name, original_id, ep)
                meta["_source"] = name
                meta["_source_host"] = host
                meta["id"] = aggregated_id
                new_meta[aggregated_id] = meta

        elif ep_type in ("llama-server", "lmstudio", "custom"):
            # Local endpoints: include all discovered models
            if ep_type == "llama-server":
                raw_models = fetch_llama_server_models(host)
            elif ep_type == "lmstudio":
                raw_models = fetch_lmstudio_models(host)
            else:
                raw_models = fetch_custom_models(host, api_key)

            for model in raw_models:
                original_id = model.get("id", "")
                if not original_id:
                    continue
                aggregated_id = f"{name}/{original_id}"
                new_routes[aggregated_id] = (name, original_id, ep)
                model["_source"] = name
                model["_source_host"] = host
                model["id"] = aggregated_id
                new_meta[aggregated_id] = model

    # Build MODELS list: user models first, then discovered local models
    merged = []
    for m in user_models:
        if m in new_routes and m not in merged:
            merged.append(m)
    for m in new_routes:
        if m not in merged:
            merged.append(m)

    _model_routes = new_routes
    with _model_meta_lock:
        _model_meta.clear()
        _model_meta.update(new_meta)

    MODELS = merged
    total = len(MODELS)
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
            conn = http.client.HTTPConnection(host, timeout=3)
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

        # Try to serve as static web UI file
        if self._serve_file(path.lstrip("/")):
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

        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    # ── Static file serving ────────────────────────────────────────────────
    def _serve_file(self, rel_path: str) -> bool:
        filepath = os.path.normpath(os.path.join(WEBUI_DIR, rel_path))
        if not filepath.startswith(os.path.normpath(WEBUI_DIR)):
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
            elif rel_path in ("sw.js", "manifest.webmanifest", "_app/version.json", "build.json"):
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
        for m in MODELS:
            meta = get_model_meta(m)
            context_length = meta.get("context_length", 128000)

            input_mods = meta.get("architecture", {}).get("input_modalities", ["text"])
            output_mods = meta.get("architecture", {}).get("output_modalities", ["text"])
            has_vision = any(mod in input_mods for mod in ("image", "vision"))
            has_audio = "audio" in input_mods
            has_video = "video" in input_mods
            has_tools = any(mod in input_mods for mod in ("tools", "tool_use"))

            tags = []
            if has_vision:
                tags.append("vision")
            if has_audio:
                tags.append("audio")
            if has_video:
                tags.append("video")

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
                "architecture": {
                    "input_modalities": input_mods,
                    "output_modalities": output_mods,
                },
                "meta": {
                    "capabilities": {
                        "vision": has_vision,
                        "audio": has_audio,
                        "video": has_video,
                        "function_calling": has_tools,
                    },
                },
            }
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
                    "chat_format": "chatml", "reasoning_format": "none",
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
            "chat_template": "", "bos_token": "", "eos_token": "",
            "build_info": "ai-gateway v2.0",
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
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > _MAX_BODY_SIZE:
            self.send_error(413, "Request body too large")
            return
        body = self.rfile.read(content_length) if content_length > 0 else b""
        if body:
            try:
                data = json.loads(body)
                model = data.get("model", "")
                if model:
                    self._log(f"Web UI selected model: {model}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        self._send_json({"success": True})

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
            target_endpoint = get_first_enabled_endpoint()

        if not target_endpoint:
            self._send_json({"error": {"message": "All endpoints are offline or none configured"}}, 503)
            return

        host = target_endpoint.get("host", "")
        api_key = target_endpoint.get("api_key", "")
        ep_name = target_endpoint.get("name", "unknown")
        ep_type = target_endpoint.get("type", "")

        # 4. Rewrite path (OpenRouter needs /api prefix, local endpoints don't)
        if ep_type == "openrouter":
            path = rewrite_path(self.path)
        else:
            # llama-server, LM Studio, custom — use path as-is
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

        # 6. Forward to upstream
        use_ssl = not _is_local_host(host)
        if use_ssl:
            conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=180)
        else:
            conn = http.client.HTTPConnection(host, timeout=180)

        try:
            conn.request(self.command, path, body=body, headers=fwd_headers)
            response = conn.getresponse()

            # Send status + CORS + upstream response headers
            self.send_response(response.status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            for header, value in response.getheaders():
                if header.lower() in ("transfer-encoding", "connection"):
                    continue
                self.send_header(header, value)
            self.end_headers()

            # Stream response body (SSE-safe)
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

            self._log(f"→ {ep_name} [{response.status}]")

        except Exception as e:
            self._log(f"Error → {ep_name}: {e}")
            try:
                self.send_error(502, f"Bad Gateway ({ep_name}): {e}")
            except Exception:
                pass
        finally:
            conn.close()

    def _log(self, msg: str):
        if ProxyHandler.tui_log is not None:
            try:
                ProxyHandler.tui_log.write_line(f"[{self.address_string()}] {msg}")
            except Exception:
                pass
        else:
            print(f"[proxy] {msg}")

    def log_message(self, fmt, *args):
        pass  # Suppress default stderr logging


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — ENDPOINT EDIT DIALOG (ModalScreen)
# ═══════════════════════════════════════════════════════════════════════════

class EndpointEditScreen(ModalScreen):
    """Modal dialog for adding or editing an endpoint."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    ENDPOINT_TYPES = [
        ("OpenRouter", "openrouter"),
        ("LM Studio", "lmstudio"),
        ("llama.cpp Server", "llama-server"),
        ("Custom", "custom"),
    ]

    def __init__(self, endpoint: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint  # None = add new, dict = edit existing

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
                Button("💾 Save", id="ep-save-btn", variant="success", compact=True)
                Button("✕ Cancel", id="ep-cancel-btn", variant="default", compact=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Auto-fill name and host when type is selected."""
        if event.select.id == "ep-type-select":
            ep_type = str(event.select.value)
            defaults = ENDPOINT_TYPE_DEFAULTS.get(ep_type, {})
            name_input = self.query_one("#ep-name-input", Input)
            host_input = self.query_one("#ep-host-input", Input)
            if not self.endpoint:
                name_input.value = defaults.get("name", "")
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
        self.dismiss(result)


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — SETTINGS SCREEN
# ═══════════════════════════════════════════════════════════════════════════

class SettingsScreen(Screen):
    """Settings screen with endpoint management."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(id="settings-screen"):
            yield Static("Settings", classes="screen-title")

            # ── Endpoint Management ────────────────────────────────────
            yield Static("Endpoints", classes="section-title")
            yield DataTable(id="endpoint-table")
            with Horizontal(id="ep-buttons"):
                Button("+ Add", id="ep-add-btn", variant="success", compact=True)
                Button("✎ Edit", id="ep-edit-btn", variant="primary", compact=True)
                Button("✕ Remove", id="ep-remove-btn", variant="error", compact=True)

            # ── Model Settings ─────────────────────────────────────────
            yield Static("Model Settings", classes="section-title")
            yield Label("Default Model")
            yield Select(
                [(m, m) for m in MODELS],
                value=MODELS[get_config("model_index")] if MODELS else "",
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

            with Horizontal(id="settings-buttons"):
                Button("💾 Save", id="apply-btn", variant="success", compact=True)
                Button("✕ Cancel", id="back-btn", variant="default", compact=True)
                Button("↻ Restart Server", id="restart-btn", variant="warning", compact=True)
        yield Footer()

    def on_mount(self) -> None:
        self._populate_endpoint_table()

    def _populate_endpoint_table(self):
        table = self.query_one("#endpoint-table", DataTable)
        table.clear(columns=True)
        table.cursor_type = "row"
        table.add_columns("Name", "Type", "Host", "Status")
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
                key=name,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "ep-add-btn":
            self._add_endpoint()
        elif event.button.id == "ep-edit-btn":
            self._edit_endpoint()
        elif event.button.id == "ep-remove-btn":
            self._remove_endpoint()
        elif event.button.id == "apply-btn":
            self._apply_settings()
        elif event.button.id == "restart-btn":
            self._restart_server()

    def _add_endpoint(self):
        def on_result(result):
            if result:
                endpoints = list(get_config("endpoints"))
                endpoints.append(result)
                set_config("endpoints", endpoints)
                self._populate_endpoint_table()
                self._update_model_select()
        self.app.push_screen(EndpointEditScreen(), on_result)

    def _edit_endpoint(self):
        table = self.query_one("#endpoint-table", DataTable)
        if table.cursor_row is None:
            return
        row_key = table.get_row_at(table.cursor_row)
        ep_name = str(row_key[0]) if row_key else ""
        endpoints = list(get_config("endpoints"))
        ep = next((e for e in endpoints if e.get("name") == ep_name), None)
        if not ep:
            return

        def on_result(result):
            if result:
                for i, e in enumerate(endpoints):
                    if e.get("name") == ep_name:
                        endpoints[i] = result
                        break
                set_config("endpoints", endpoints)
                self._populate_endpoint_table()
                self._update_model_select()
        self.app.push_screen(EndpointEditScreen(endpoint=ep), on_result)

    def _remove_endpoint(self):
        table = self.query_one("#endpoint-table", DataTable)
        if table.cursor_row is None:
            return
        row_key = table.get_row_at(table.cursor_row)
        ep_name = str(row_key[0]) if row_key else ""
        endpoints = list(get_config("endpoints"))
        endpoints = [e for e in endpoints if e.get("name") != ep_name]
        set_config("endpoints", endpoints)
        self._populate_endpoint_table()
        self._update_model_select()

    def _update_model_select(self):
        """Refresh the model selector dropdown."""
        model_select = self.query_one("#model-select", Select)
        new_options = [(m, m) for m in MODELS]
        model_select.clear()
        for label, value in new_options:
            model_select.add_option((label, value))
        if MODELS:
            idx = min(get_config("model_index"), len(MODELS) - 1)
            model_select.value = MODELS[idx]

    def _apply_settings(self):
        log = ProxyHandler.tui_log

        # Model
        model_select = self.query_one("#model-select", Select)
        if model_select.value is not None:
            model_name = str(model_select.value)
            if model_name in MODELS:
                set_config("model_index", MODELS.index(model_name))
                log.write_line(f"Model set → {model_name}")

        # Fetch top models checkbox
        fetch_checkbox = self.query_one("#fetch-top-models-checkbox", Checkbox)
        fetch_enabled = fetch_checkbox.value
        set_config("fetch_top_models", fetch_enabled)
        if fetch_enabled:
            log.write_line("Fetching top intelligent models...")
            threading.Thread(target=fetch_top_intelligent_models, daemon=True).start()

        # Port / addr
        port = self.query_one("#port-input", Input).value.strip()
        addr = self.query_one("#addr-input", Input).value.strip()
        if port.isdigit():
            set_config("port", int(port))
        if addr:
            set_config("addr", addr)

        save_config()
        log.write_line("Config saved to config.json")

    def _restart_server(self):
        log = ProxyHandler.tui_log
        log.write_line("Restarting server...")
        global _server
        if _server is not None:
            _server.shutdown()
            _server.server_close()
            _server = None
        _start_server()
        log.write_line(f"Server restarted on http://{get_config('addr')}:{get_config('port')}")


def fetch_top_intelligent_models(limit: int = 20):
    """Fetch top intelligent models from OpenRouter by intelligence index."""
    global _top_intelligent_models, MODELS
    endpoints = get_config("endpoints")
    or_ep = next((ep for ep in endpoints if ep.get("type") == "openrouter" and ep.get("enabled")), None)
    if not or_ep:
        print("[proxy] No OpenRouter endpoint enabled, skipping top models fetch")
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

        top_models = [f"{ep_name}/{m['id']}" for m in data.get("data", [])[:limit]]
        with _top_models_lock:
            _top_intelligent_models = top_models

        # Merge with user models: user models first, then top models (no duplicates)
        user_models = list(get_config("models"))
        merged = list(user_models)
        for m in top_models:
            if m not in merged:
                merged.append(m)

        with _config_lock:
            MODELS = merged
        print(f"[proxy] Fetched {len(top_models)} top intelligent models, merged total: {len(MODELS)}")
    except Exception as e:
        print(f"[proxy] Could not fetch top intelligent models: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — MAIN SCREEN
# ═══════════════════════════════════════════════════════════════════════════

class MainScreen(Screen):
    """Main screen: toolbar + endpoint status + live log."""

    BINDINGS = [("q", "quit", "Quit")]

    def on_mount(self) -> None:
        log = self.query_one("#log", Log)
        ProxyHandler.tui_log = log
        log.write_line("TUI ready — proxy is running.")
        log.write_line(f"Listening : http://{get_config('addr')}:{get_config('port')}")
        log.write_line(f"Web UI    : http://{get_config('addr')}:{get_config('port')}/")
        log.write_line(f"Default   : {get_current_model()}")
        self._update_endpoint_status()
        # Periodic status refresh
        self.set_interval(5.0, self._update_endpoint_status)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Button("⚙ Settings", id="menu-settings", variant="primary", compact=True),
            Button("✕ Quit", id="menu-quit", variant="error", compact=True),
            id="main-toolbar",
        )
        yield Vertical(
            Static("Endpoints", classes="screen-title"),
            Static("Loading...", id="endpoint-status"),
            id="endpoint-panel",
        )
        yield Vertical(
            Static("Proxy Log", classes="screen-title"),
            Log(id="log", highlight=True),
            id="log-panel",
        )
        yield Footer()

    def _update_endpoint_status(self):
        """Refresh the endpoint status display."""
        status_widget = self.query_one("#endpoint-status", Static)
        endpoints = get_config("endpoints")
        lines = []
        for ep in endpoints:
            name = ep.get("name", "")
            host = ep.get("host", "")
            status = get_endpoint_status(name)
            if status == "online":
                icon = "●"
                style = "green"
            elif status == "checking":
                icon = "◌"
                style = "yellow"
            else:
                icon = "○"
                style = "red"
            model_count = sum(1 for m in MODELS if m.startswith(f"{name}/"))
            lines.append(f"  {icon} {name:<12} → {host:<24} [{status}]  {model_count} models")
        status_widget.update("\n".join(lines) if lines else "  No endpoints configured")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "menu-settings":
            self.app.push_screen(SettingsScreen())
        elif event.button.id == "menu-quit":
            self.app.exit()

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
    #main-toolbar {
        dock: top;
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    #main-toolbar Button {
        min-width: 10;
        margin-right: 1;
    }
    #endpoint-panel {
        height: auto;
        width: 100%;
        padding: 0 1;
    }
    #endpoint-status {
        height: auto;
        width: 100%;
    }
    #log-panel {
        height: 1fr;
        width: 100%;
    }
    #log {
        height: 1fr;
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
    #settings-buttons {
        height: auto;
        margin-top: 1;
    }
    #ep-buttons {
        height: auto;
        margin-top: 0;
    }
    #ep-buttons Button {
        width: auto;
        min-width: 8;
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


def _start_server():
    global _server, _server_thread
    port = get_config("port")
    addr = get_config("addr")
    try:
        _server = ThreadingHTTPServer((addr, port), ProxyHandler)
    except OSError as e:
        print(f"[proxy] ERROR: Cannot bind to {addr}:{port} — {e}")
        print(f"[proxy] Is another instance already running?")
        sys.exit(1)
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

    _start_server()
    try:
        ProxyTUI().run()
    finally:
        if _server:
            _server.shutdown()
            _server.server_close()