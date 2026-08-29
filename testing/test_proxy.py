"""
Comprehensive tests for the AI Gateway's MCP support:

  * /props advertises cors_proxy_enabled (enables "Use llama-server proxy").
  * /cors-proxy forwards GET and POST requests to an upstream MCP server.
  * Prefixed headers (x-llama-server-proxy-header-*) are stripped and sent as
    normal headers.
  * Plain request headers are forwarded (so MCP's Mcp-Session-Id reaches the
    server).
  * Prefixed (user-intended) headers win over browser auto-added plain headers
    (e.g. Content-Type: application/json beats auto text/plain).
  * Upstream response headers (Mcp-Session-Id) are forwarded back to the client.
  * Hop-by-hop headers are excluded in both directions.
  * Error handling: missing url, invalid scheme, oversized body.

Run with:
  python testing/test_proxy.py
"""

import json
import os
import sys
import threading
import unittest
import urllib.parse
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Allow running from any directory: put the repo root (parent of testing/)
# on sys.path so `import proxy` resolves regardless of the current working
# directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proxy


# ── Mock upstream MCP server ──────────────────────────────────────────────
class MockUpstream(BaseHTTPRequestHandler):
    """A controllable upstream that records requests and simulates a stateful
    streamable-HTTP MCP server (issues a session id on initialize)."""

    requests = []           # list of dicts: method, path, headers, body
    session_id = "session-abc-123"
    logs = []

    def _record(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        MockUpstream.requests.append({
            "method": self.command,
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": body.decode("utf-8", "replace"),
        })

    def do_GET(self):
        self._record()
        self._json(200, {"ok": True, "method": "GET"})

    def do_POST(self):
        self._record()
        # Simulate MCP: initialize (no session header) issues a session id.
        if not self.headers.get("Mcp-Session-Id"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Mcp-Session-Id", MockUpstream.session_id)
            resp = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2025-03-26"},
            }).encode()
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self._json(200, {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"echoSession": self.headers.get("Mcp-Session-Id")},
            })

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_server(handler, port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t, srv.server_address[1]


def request(port, method, path, body=None, headers=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        data = r.read()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, data
    finally:
        conn.close()


class GatewayMCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up_srv, cls.up_t, cls.up_port = start_server(MockUpstream)
        cls.gw_srv, cls.gw_t, cls.gw_port = start_server(proxy.ProxyHandler)
        cls.up_url = f"http://127.0.0.1:{cls.up_port}/mcp"
        cls.proxy_path = "/cors-proxy?url=" + urllib.parse.quote(cls.up_url, safe="")

    @classmethod
    def tearDownClass(cls):
        cls.gw_srv.shutdown(); cls.gw_srv.server_close()
        cls.up_srv.shutdown(); cls.up_srv.server_close()

    def setUp(self):
        MockUpstream.requests.clear()

    # ── /props ────────────────────────────────────────────────────────────
    def test_props_advertises_cors_proxy(self):
        status, _, body = request(self.gw_port, "GET", "/props")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIs(data.get("cors_proxy_enabled"), True)

    # ── /cors-proxy basic forwarding ──────────────────────────────────────
    def test_proxy_forwards_get(self):
        status, _, _ = request(self.gw_port, "GET", self.proxy_path)
        self.assertEqual(status, 200)
        self.assertEqual(MockUpstream.requests[0]["method"], "GET")
        self.assertEqual(MockUpstream.requests[0]["path"], "/mcp")

    def test_proxy_forwards_post_body(self):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        status, _, _ = request(
            self.gw_port, "POST", self.proxy_path, body=body,
            headers={"X-Llama-Server-Proxy-Header-Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(MockUpstream.requests[0]["method"], "POST")
        self.assertEqual(MockUpstream.requests[0]["body"], body)

    # ── Header handling ───────────────────────────────────────────────────
    def test_prefixed_headers_are_stripped(self):
        status, _, _ = request(
            self.gw_port, "POST", self.proxy_path,
            body="{}",
            headers={
                "X-Llama-Server-Proxy-Header-Content-Type": "application/json",
                "X-Llama-Server-Proxy-Header-X-Custom": "custom-value",
            },
        )
        self.assertEqual(status, 200)
        up_headers = MockUpstream.requests[0]["headers"]
        self.assertEqual(up_headers.get("Content-Type"), "application/json")
        self.assertEqual(up_headers.get("X-Custom"), "custom-value")
        # No prefixed headers leak upstream.
        for k in up_headers:
            self.assertNotIn("x-llama-server-proxy-header-", k.lower())

    def test_plain_headers_are_forwarded(self):
        status, _, _ = request(
            self.gw_port, "POST", self.proxy_path,
            body="{}",
            headers={
                "Content-Type": "application/json",
                "Mcp-Session-Id": "session-xyz",
            },
        )
        self.assertEqual(status, 200)
        up_headers = MockUpstream.requests[0]["headers"]
        self.assertEqual(up_headers.get("Mcp-Session-Id"), "session-xyz")

    def test_prefixed_content_type_beats_auto_text_plain(self):
        """A browser may auto-add content-type: text/plain; the prefixed
        JSON content-type must win."""
        status, _, _ = request(
            self.gw_port, "POST", self.proxy_path,
            body=json.dumps({"a": 1}),
            headers={
                "Content-Type": "text/plain;charset=UTF-8",
                "X-Llama-Server-Proxy-Header-Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 200)
        up_headers = MockUpstream.requests[0]["headers"]
        self.assertEqual(up_headers.get("Content-Type"), "application/json")

    def test_hop_by_hop_request_headers_excluded(self):
        status, _, _ = request(
            self.gw_port, "POST", self.proxy_path,
            body="{}",
            headers={
                "Content-Type": "application/json",
                "Connection": "close",
                "Accept-Encoding": "gzip",
            },
        )
        self.assertEqual(status, 200)
        up_headers = MockUpstream.requests[0]["headers"]
        self.assertNotIn("Connection", {k.lower(): k for k in up_headers})
        self.assertNotIn("Accept-Encoding", {k.lower(): k for k in up_headers})

    # ── Response header forwarding (session handshake) ────────────────────
    def test_response_session_header_is_forwarded(self):
        """The Mcp-Session-Id issued by the upstream must reach the browser."""
        status, headers, _ = request(
            self.gw_port, "POST", self.proxy_path,
            body=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            headers={"X-Llama-Server-Proxy-Header-Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("mcp-session-id"), MockUpstream.session_id)

    def test_full_session_round_trip(self):
        """initialize returns a session id; the client echoes it and the
        upstream receives it."""
        # 1. initialize (no session)
        status, headers, _ = request(
            self.gw_port, "POST", self.proxy_path,
            body=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            headers={"X-Llama-Server-Proxy-Header-Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        sid = headers.get("mcp-session-id")
        self.assertIsNotNone(sid)

        # 2. echo the session id back like the MCP SDK does
        status, _, body = request(
            self.gw_port, "POST", self.proxy_path,
            body=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            headers={
                "Content-Type": "application/json",
                "Mcp-Session-Id": sid,
            },
        )
        self.assertEqual(status, 200)
        # The upstream saw the echoed session id.
        self.assertEqual(MockUpstream.requests[-1]["headers"].get("Mcp-Session-Id"), sid)
        data = json.loads(body)
        self.assertEqual(data["result"]["echoSession"], sid)

    # ── Error handling ────────────────────────────────────────────────────
    def test_missing_url_returns_400(self):
        status, _, body = request(self.gw_port, "GET", "/cors-proxy")
        self.assertEqual(status, 400)
        self.assertIn("url", json.loads(body)["error"].lower())

    def test_invalid_scheme_returns_400(self):
        status, _, body = request(
            self.gw_port, "GET",
            "/cors-proxy?url=" + urllib.parse.quote("file:///etc/passwd", safe=""),
        )
        self.assertEqual(status, 400)
        self.assertIn("http", json.loads(body)["error"].lower())

    def test_oversized_body_returns_413(self):
        original = proxy._MAX_BODY_SIZE
        proxy._MAX_BODY_SIZE = 16
        try:
            status, _, _ = request(
                self.gw_port, "POST", self.proxy_path,
                body="x" * 64,
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(status, 413)
        finally:
            proxy._MAX_BODY_SIZE = original


if __name__ == "__main__":
    unittest.main(verbosity=2)