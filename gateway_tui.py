"""
Gateway Split-View TUI.

Runs the AI Gateway proxy (from proxy.py, started headlessly) with its FULL
existing GUI (endpoint status, server info, settings) in the top pane, and
Supergateway (streamable HTTP MCP server) in the bottom pane.

This is achieved by subclassing proxy.py's own TUI classes — proxy.py itself
is never modified:
  - SplitMainScreen(proxy.MainScreen): overrides only compose() to wrap the
    existing widgets (same IDs) in a top pane and add a supergateway pane.
    All inherited logic (status refresh, log draining, settings) keeps working.
  - GatewayApp(proxy.ProxyTUI): inherits all CSS/screens, swaps in the split
    main screen.
"""

import os
import queue
import subprocess
import sys
import threading

# ── Resolve paths relative to THIS script's directory ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import the proxy as a module (its __main__ guard is bypassed).
import proxy  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Supergateway subprocess (relative filesystem root)
# ═══════════════════════════════════════════════════════════════════════════

def build_supergateway_cmd() -> list:
    """Build the supergateway command list.

    The filesystem server root is 'supergateway' — relative to the project
    directory (SCRIPT_DIR), NOT an absolute hardcoded path. The inner npx
    command resolves it against the subprocess cwd (set to SCRIPT_DIR).
    """
    stdio_arg = "npx -y @modelcontextprotocol/server-filesystem supergateway"
    return [
        "npx", "-y", "supergateway",
        "--stdio", stdio_arg,
        "--outputTransport", "streamableHttp",
        "--port", "8099",
    ]


def start_supergateway(q: "queue.Queue") -> subprocess.Popen:
    """Spawn supergateway and stream its stdout/stderr into the queue."""
    cmd_list = build_supergateway_cmd()

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        # On Windows, npx is a .cmd shim that CreateProcess cannot execute
        # directly, so run it through the shell. list2cmdline quotes the
        # --stdio value correctly.
        cmd: list = subprocess.list2cmdline(cmd_list)
        shell = True
    else:
        cmd = cmd_list
        shell = False

    proc = subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
        shell=shell,
    )

    def _reader():
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if line.strip():
                q.put(line)
        q.put("<< supergateway process exited >>")

    threading.Thread(target=_reader, daemon=True).start()
    return proc


def stop_supergateway(proc: subprocess.Popen):
    """Terminate supergateway and its process tree."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        except Exception:
            proc.terminate()
    else:
        proc.terminate()


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — split main screen (subclasses proxy.py's MainScreen)
# ═══════════════════════════════════════════════════════════════════════════

from textual.app import ComposeResult  # noqa: E402
from textual.containers import Vertical  # noqa: E402
from textual.widgets import Footer, Header, Log, Static  # noqa: E402


class SplitMainScreen(proxy.MainScreen):
    """Main screen with the full proxy GUI on top and supergateway below.

    Only compose() is overridden — the inherited on_mount(),
    _update_endpoint_status(), _show_server_info(), _drain_log_queue() and
    the 's' (settings) binding all keep working because the original widget
    IDs (#endpoint-status, #server-info, #log) are preserved.
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="split-root"):
            # ── Top pane: the original proxy GUI (same widget IDs) ─────────
            with Vertical(id="proxy-pane"):
                yield Static("Endpoints", classes="section-title")
                yield Static("Loading...", id="endpoint-status")
                yield Static("", id="server-info")
                yield Static("Proxy Log", classes="section-title")
                yield Log(id="log", highlight=True)
            # ── Bottom pane: supergateway ──────────────────────────────────
            with Vertical(id="sg-pane"):
                yield Static("Supergateway  (streamableHttp :8099, root: ./supergateway)",
                             classes="section-title")
                yield Log(id="supergateway-log", highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        # Run the original MainScreen mount logic (tui_log hookup,
        # status refresh interval, proxy log queue draining).
        super().on_mount()
        # Drain supergateway output into its log pane.
        self.set_interval(0.1, self._drain_sg_queue)

    def _drain_sg_queue(self):
        sg_log = self.query_one("#supergateway-log", Log)
        try:
            while True:
                line = self.app.sg_lines.get_nowait()
                sg_log.write_line(line)
        except queue.Empty:
            pass

    def action_quit(self):
        stop_supergateway(self.app.sg_proc)
        self.app.exit()


# ═══════════════════════════════════════════════════════════════════════════
#  TUI — application (subclasses proxy.py's ProxyTUI)
# ═══════════════════════════════════════════════════════════════════════════

class GatewayApp(proxy.ProxyTUI):
    """The proxy's own app, with the split main screen swapped in."""

    TITLE = "AI Gateway"
    SUB_TITLE = "Proxy + Supergateway"

    SCREENS = {
        "main": SplitMainScreen,
        "settings": proxy.SettingsScreen,
    }

    CSS = proxy.ProxyTUI.CSS + """
    #split-root {
        height: 1fr;
        padding: 0 2;
    }
    #proxy-pane {
        height: 2fr;
        border: round $primary;
        padding: 0 1;
    }
    #sg-pane {
        height: 1fr;
        border: round $secondary;
        padding: 0 1;
    }
    #log {
        height: 1fr;
    }
    #supergateway-log {
        height: 1fr;
    }
    """

    def __init__(self, sg_lines: "queue.Queue", sg_proc: subprocess.Popen):
        super().__init__()
        self.sg_lines = sg_lines
        self.sg_proc = sg_proc


# ═══════════════════════════════════════════════════════════════════════════
#  startup
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Start the proxy headlessly (same as proxy.py's __main__, no TUI) ────
    proxy.load_config()

    if not os.path.isdir(proxy.WEBUI_DIR):
        print(f"[proxy] WARNING: Web UI directory not found: {proxy.WEBUI_DIR}")

    proxy.auto_discover_local_endpoints()
    proxy.save_config()

    for ep in proxy.get_config("endpoints"):
        proxy.set_endpoint_status(ep.get("name", ""), "checking")

    print("[proxy] Starting health check thread...")
    threading.Thread(target=proxy.health_check_loop, daemon=True).start()

    if proxy.get_config("fetch_top_models"):
        print("[proxy] Fetching top intelligent models from OpenRouter...")
        threading.Thread(target=proxy.fetch_top_intelligent_models, daemon=True).start()

    try:
        proxy._start_server()
    except OSError as e:
        print(f"[proxy] Fatal: could not start server — {e}")
        sys.exit(1)

    # ── Start supergateway subprocess (relative root: ./supergateway) ──────
    print("[gateway-tui] Launching supergateway (relative root: ./supergateway)")
    sg_lines = queue.Queue()
    sg_proc = start_supergateway(sg_lines)

    # ── Run the split-view TUI (full proxy GUI + supergateway pane) ────────
    try:
        GatewayApp(sg_lines, sg_proc).run()
    finally:
        stop_supergateway(sg_proc)
        try:
            if proxy._server is not None:
                proxy._server.shutdown()
                proxy._server.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    main()