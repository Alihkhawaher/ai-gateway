"""
OpenRouter reverse proxy with a Textual TUI for live configuration.

The proxy server runs in a background thread; the Textual TUI runs in the
main thread. A toolbar lets you open a Settings screen to change the model,
API key, host, port, and bind address at runtime without restarting.

Includes an integrated web chat UI (llama.cpp's llama-ui) served at /.

If `textual` is not installed, it is installed automatically via pip.

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

# ── OpenRouter model metadata cache ──────────────────────────────────────
_model_meta_lock = threading.Lock()
_model_meta = {}  # model_id -> dict from OpenRouter /api/v1/models


def fetch_model_metadata():
    """Fetch model metadata from OpenRouter and cache it."""
    global _model_meta
    try:
        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection("openrouter.ai", context=ctx, timeout=30)
        conn.request("GET", "/api/v1/models")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        with _model_meta_lock:
            _model_meta = {m["id"]: m for m in data.get("data", [])}
        print(f"[proxy] Fetched metadata for {len(_model_meta)} models from OpenRouter")
    except Exception as e:
        print(f"[proxy] Could not fetch model metadata: {e}")


def get_model_meta(model_id: str) -> dict:
    """Get cached metadata for a model, or empty dict if not found."""
    with _model_meta_lock:
        return _model_meta.get(model_id, {})

# ── Auto-install textual if missing ────────────────────────────────────────
try:
    import textual  # noqa: F401
except ImportError:
    print("[proxy] textual not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textual"])
    import textual  # noqa: F401

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Input, Select, Static, Button, Label, Log,
)

# ── Available upstream endpoints ──────────────────────────────────────────
ENDPOINTS = [
    ("OpenRouter", "openrouter.ai"),
    ("LM Studio (localhost:1234)", "localhost:1234"),
    ("llama-server (localhost:8080)", "localhost:8080"),
]

# ── Default values (used if config.json doesn't exist) ────────────────────
DEFAULTS = {
    "model_index": 0,
    "models": [
        "xiaomi/mimo-v2.5-pro",
        "deepseek/deepseek-v4-flash-0731",
    ],
    "api_key": "",
    "port": 8090,
    "addr": "0.0.0.0",
    "host": "openrouter.ai",
}

# ── Runtime models list (loaded from config) ─────────────────────────────
MODELS = list(DEFAULTS["models"])

# ── Thread-safe shared config ──────────────────────────────────────────────
_config_lock = threading.Lock()
_config = dict(DEFAULTS)


def load_config():
    """Load config from config.json, merging with defaults."""
    global _config, MODELS
    with _config_lock:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for key in DEFAULTS:
                    if key in saved:
                        _config[key] = saved[key]
                print(f"[proxy] Config loaded from {CONFIG_FILE}")
            except Exception as e:
                print(f"[proxy] Could not load config: {e}")
        else:
            print(f"[proxy] No config file found, using defaults")
        # Sync MODELS list from config
        MODELS = list(_config.get("models", DEFAULTS["models"]))


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
        idx = max(0, min(_config["model_index"], len(MODELS) - 1))
        return MODELS[idx]


# ── Path rewrite: prefix /api/v1 if not already present ───────────────────
def rewrite_path(path: str) -> str:
    # Compatibility shim: some clients (Open WebUI, LibreChat, etc.) request
    # /models/sse expecting an SSE stream of models. OpenRouter serves the
    # model list at /api/v1/models — map it there.
    if path == "/models/sse":
        return "/api/v1/models"
    if path.startswith("/api/"):
        return path
    if path.startswith("/v1/"):
        return f"/api{path}"  # /v1/chat/completions -> /api/v1/chat/completions
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


class ProxyHandler(BaseHTTPRequestHandler):
    # Reference to the TUI log (set at runtime) so proxy activity shows in the UI.
    tui_log = None

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
            # Parse query params for model-specific props
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

        # Proxy everything else to OpenRouter
        self._proxy()

    # ── POST: local API stubs or proxy ─────────────────────────────────────
    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # Local stubs for POST endpoints
        if path == "/v1/streams/lookup":
            self._handle_streams_lookup()
            return
        if path in ("/models/load", "/models/unload"):
            self._handle_model_load()
            return

        # Proxy everything else to OpenRouter
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    # ── Static file serving ────────────────────────────────────────────────
    def _serve_file(self, rel_path: str) -> bool:
        """Serve a file from the webui directory. Returns True if served."""
        filepath = os.path.normpath(os.path.join(WEBUI_DIR, rel_path))
        # Security: ensure the resolved path is within WEBUI_DIR
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
            # Immutable cache for hashed assets, no-cache for sw.js/manifest
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
        """Return model list in OpenAI-compatible format with llama.cpp extensions."""
        entries = []
        for m in MODELS:
            meta = get_model_meta(m)
            entries.append({
                "id": m,
                "name": meta.get("name", m),
                "object": "model",
                "created": meta.get("created", int(time.time())),
                "owned_by": m.split("/")[0] if "/" in m else "openrouter",
                "in_cache": True,
                "path": m,
                "status": {
                    "value": "loaded",
                },
            })
        self._send_json({"object": "list", "data": entries})

    # ── API: /props ────────────────────────────────────────────────────────
    def _handle_props(self, model_id=None):
        """Return server properties using real OpenRouter model metadata."""
        model = unquote(model_id) if model_id else get_current_model()
        meta = get_model_meta(model)

        # Extract context length from OpenRouter metadata
        n_ctx = meta.get("context_length", 128000)
        max_completion = meta.get("top_provider", {}).get("max_completion_tokens", -1)

        # Map OpenRouter input modalities to llama.cpp modalities
        input_mods = meta.get("architecture", {}).get("input_modalities", ["text"])
        has_vision = any(m in input_mods for m in ("image", "vision"))
        has_audio = "audio" in input_mods
        has_video = "video" in input_mods

        data = {
            "default_generation_settings": {
                "id": 0,
                "id_task": -1,
                "n_ctx": n_ctx,
                "speculative": False,
                "is_processing": False,
                "params": {
                    "n_predict": max_completion,
                    "seed": 4294967295,
                    "temperature": 0.8,
                    "dynatemp_range": 0.0,
                    "dynatemp_exponent": 1.0,
                    "top_k": 40,
                    "top_p": 0.95,
                    "min_p": 0.05,
                    "top_n_sigma": 0.0,
                    "xtc_probability": 0.0,
                    "xtc_threshold": 0.1,
                    "typ_p": 1.0,
                    "repeat_last_n": 64,
                    "repeat_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "dry_multiplier": 0.0,
                    "dry_base": 1.75,
                    "dry_allowed_length": 2,
                    "dry_penalty_last_n": -1,
                    "dry_sequence_breakers": ["\n", ":", "\"", "*"],
                    "mirostat": 0,
                    "mirostat_tau": 5.0,
                    "mirostat_eta": 0.1,
                    "stop": [],
                    "max_tokens": max_completion,
                    "n_keep": 0,
                    "n_discard": 0,
                    "ignore_eos": False,
                    "stream": True,
                    "logit_bias": [],
                    "n_probs": 0,
                    "min_keep": 0,
                    "grammar": "",
                    "grammar_lazy": False,
                    "grammar_triggers": [],
                    "preserved_tokens": [],
                    "chat_format": "chatml",
                    "reasoning_format": "none",
                    "reasoning_in_content": False,
                    "generation_prompt": "",
                    "samplers": ["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
                    "backend_sampling": False,
                    "speculative.n_max": 0,
                    "speculative.n_min": 0,
                    "speculative.p_min": 0.0,
                    "timings_per_token": False,
                    "post_sampling_probs": False,
                    "lora": [],
                },
                "prompt": "",
                "next_token": {
                    "has_next_token": False,
                    "has_new_line": False,
                    "n_remain": -1,
                    "n_decoded": 0,
                    "stopping_word": "",
                },
            },
            "total_slots": 1,
            "model_path": model,
            "role": "router",
            "modalities": {
                "vision": has_vision,
                "audio": has_audio,
                "video": has_video,
            },
            "chat_template": "",
            "bos_token": "",
            "eos_token": "",
            "build_info": "openrouter-proxy v1.0",
        }
        self._send_json(data)

    # ── API: /health ───────────────────────────────────────────────────────
    def _handle_health(self):
        self._send_json({"status": "ok"})

    # ── API: /slots ────────────────────────────────────────────────────────
    def _handle_slots(self):
        self._send_json([])

    # ── API: /tools ────────────────────────────────────────────────────────
    def _handle_tools(self):
        self._send_json([])

    # ── API: /v1/streams/lookup ────────────────────────────────────────────
    def _handle_streams_lookup(self):
        """Stub for background streaming session lookup (not supported)."""
        self._send_json({})

    # ── API: /models/load, /models/unload ──────────────────────────────────
    def _handle_model_load(self):
        """Stub for model load/unload (all models always available via OpenRouter)."""
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

    # ── Core proxy logic ───────────────────────────────────────────────────
    def _proxy(self):
        # 1. Read body (with size limit)
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > _MAX_BODY_SIZE:
            self.send_error(413, "Request body too large")
            return
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # 2. Inject current model if missing / empty
        if body:
            try:
                data = json.loads(body)
                if isinstance(data, dict) and ("model" not in data or not data["model"]):
                    current = get_current_model()
                    data["model"] = current
                    body = json.dumps(data).encode("utf-8")
                    self._log(f"Injected model: {current}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # 3. Rewrite path
        path = rewrite_path(self.path)

        # 4. Build upstream headers
        fwd_headers = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
            "Authorization": f"Bearer {get_config('api_key')}",
        }
        for hdr in ("HTTP-Referer", "X-Title"):
            val = self.headers.get(hdr)
            if val:
                fwd_headers[hdr] = val
        if body:
            fwd_headers["Content-Length"] = str(len(body))

        # 5. Forward to upstream
        host = get_config("host")
        _local_hosts = ("localhost", "127.0.0.1", "[::1]")
        use_ssl = not any(host == h or host.startswith(h + ":") for h in _local_hosts)
        if use_ssl:
            conn = http.client.HTTPSConnection(host, context=_upstream_ssl_ctx, timeout=180)
        else:
            conn = http.client.HTTPConnection(host, timeout=180)

        try:
            conn.request(self.command, path, body=body, headers=fwd_headers)
            response = conn.getresponse()

            # 6. Send status + CORS + upstream response headers
            self.send_response(response.status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            for header, value in response.getheaders():
                if header.lower() in ("transfer-encoding", "connection"):
                    continue
                self.send_header(header, value)
            self.end_headers()

            # 7. Stream response body (SSE-safe)
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

        except Exception as e:
            self._log(f"Error: {e}")
            try:
                self.send_error(502, f"Bad Gateway: {e}")
            except Exception:
                pass  # Headers already sent or connection broken
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

    # ── Suppress default stderr logging; use stdout instead ────────────────
    def log_message(self, fmt, *args):
        self._log(f"{self.address_string()} - {fmt % args}")


# ── Settings screen ────────────────────────────────────────────────────────
class SettingsScreen(Screen):
    """A separate screen for editing proxy settings."""

    BINDINGS = [("escape", "pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(id="settings-screen"):
            yield Static("Settings", classes="screen-title")
            yield Static("Endpoint Settings", classes="section-title")
            yield Label("Endpoint")
            current_host = get_config("host")
            preset_values = [v for _, v in ENDPOINTS]
            endpoint_options = [(label, value) for label, value in ENDPOINTS]
            endpoint_options.append(("Custom", "__custom__"))
            select_value = current_host if current_host in preset_values else "__custom__"
            yield Select(
                endpoint_options,
                value=select_value,
                id="endpoint-select",
            )
            yield Label("Endpoint Address (host[:port])")
            yield Input(
                value=current_host,
                placeholder="e.g. openrouter.ai or 192.168.1.10:1234",
                id="host-input",
            )
            yield Label("Model")
            yield Select(
                [(m, m) for m in MODELS],
                value=MODELS[get_config("model_index")],
                id="model-select",
            )
            yield Label("API Key")
            yield Input(
                value=get_config("api_key"),
                placeholder="API Key (not needed for local endpoints)",
                password=True,
                id="api-key-input",
            )
            yield Static("Server Settings", classes="section-title")
            yield Horizontal(
                Vertical(
                    Label("Port"),
                    Input(
                        value=str(get_config("port")),
                        placeholder="Port (e.g. 8090)",
                        id="port-input",
                    ),
                ),
                Vertical(
                    Label("Bind Address"),
                    Input(
                        value=get_config("addr"),
                        placeholder="Bind address (e.g. 0.0.0.0)",
                        id="addr-input",
                    ),
                ),
            )
            yield Horizontal(
                Button("💾 Save", id="apply-btn", variant="success", compact=True),
                Button("✕ Cancel", id="back-btn", variant="default", compact=True),
                Button("↻ Restart Server", id="restart-btn", variant="warning", compact=True),
                id="settings-buttons",
            )
        yield Footer()

    def on_select_changed(self, event: Select.Changed) -> None:
        # Picking a preset endpoint fills the editable host field.
        if event.select.id == "endpoint-select":
            value = event.select.value
            if value and value != "__custom__":
                self.query_one("#host-input", Input).value = str(value)

    def on_input_changed(self, event: Input.Changed) -> None:
        # Typing a custom address flips the dropdown to "Custom".
        if event.input.id == "host-input":
            preset_values = [v for _, v in ENDPOINTS]
            endpoint_select = self.query_one("#endpoint-select", Select)
            text = event.input.value.strip()
            if text in preset_values:
                if endpoint_select.value != text:
                    endpoint_select.value = text
            elif endpoint_select.value != "__custom__":
                endpoint_select.value = "__custom__"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-btn":
            self._apply_settings()
        elif event.button.id == "restart-btn":
            self._restart_server()
        elif event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "quit-btn":
            self.app.exit()

    def _apply_settings(self):
        log = ProxyHandler.tui_log
        # Model
        model_select = self.query_one("#model-select", Select)
        if model_select.value is not None:
            model_name = str(model_select.value)
            if model_name in MODELS:
                set_config("model_index", MODELS.index(model_name))
                log.write_line(f"Model set → {model_name}")
        # API key
        api_key = self.query_one("#api-key-input", Input).value.strip()
        if api_key:
            set_config("api_key", api_key)
            log.write_line("API key updated.")
        # Host — use the editable input (presets just fill it)
        host = self.query_one("#host-input", Input).value.strip()
        if host:
            set_config("host", host)
            log.write_line(f"Target host → {host}")
        # Port / addr are applied on restart
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


# ── Main screen ────────────────────────────────────────────────────────────
class MainScreen(Screen):
    """Main screen: toolbar + live log."""

    BINDINGS = [("q", "quit", "Quit")]

    def on_mount(self) -> None:
        log = self.query_one("#log", Log)
        ProxyHandler.tui_log = log
        log.write_line("TUI ready — proxy is running.")
        log.write_line(f"Listening : http://{get_config('addr')}:{get_config('port')}")
        log.write_line(f"Web UI    : http://{get_config('addr')}:{get_config('port')}/")
        log.write_line(f"Target    : {get_config('host')}")
        log.write_line(f"Model     : {get_current_model()}")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Button("⚙ Settings", id="menu-settings", variant="primary", compact=True),
            Button("✕ Quit", id="menu-quit", variant="error", compact=True),
            id="main-toolbar",
        )
        yield Vertical(
            Static("Proxy Log", classes="screen-title"),
            Log(id="log", highlight=True),
            id="log-panel",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "menu-settings":
            self.app.push_screen(SettingsScreen())
        elif event.button.id == "menu-quit":
            self.app.exit()

    def action_quit(self):
        self.app.exit()


# ── Textual TUI app ────────────────────────────────────────────────────────
class ProxyTUI(App):
    """Textual app to configure the proxy live while it runs."""

    TITLE = "OpenRouter Proxy"
    SUB_TITLE = "Live configuration"

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
    /* compact=True on the buttons keeps them 1 line tall (like the settings
       screen buttons) instead of Textual's default 3-line bordered buttons. */
    #main-toolbar Button {
        min-width: 10;
        margin-right: 1;
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
    /* IMPORTANT: inner Verticals MUST stay height: auto.
       Textual's Vertical/Horizontal containers default to height: 1fr, and an
       auto-height parent with fr-height children expands to fill ALL available
       space (Textual issue #3063). Without `height: auto` here, the Port/Bind
       row stretches to fill the whole screen, pushing the buttons to the
       bottom and leaving a huge gap. Do not remove this rule. */
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


# ── Server lifecycle (background thread) ───────────────────────────────────
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
    print(f"[proxy] Target: {get_config('host')}")
    sys.stdout.flush()


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load saved config (if any), then allow CLI overrides.
    load_config()
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

    # Fetch model metadata from OpenRouter in background (non-blocking)
    threading.Thread(target=fetch_model_metadata, daemon=True).start()

    _start_server()
    try:
        ProxyTUI().run()
    finally:
        if _server:
            _server.shutdown()
            _server.server_close()
