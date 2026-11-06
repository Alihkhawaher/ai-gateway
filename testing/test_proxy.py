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
import tempfile
import threading
import time
import unittest
import urllib.parse
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Allow running from any directory: put the repo root (parent of testing/)
# on sys.path so `import proxy` resolves regardless of the current working
# directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proxy

# Tests never touch the real config, stats or session files, or a real engine.
_TEST_HOME = tempfile.TemporaryDirectory()
proxy.CONFIG_FILE = os.path.join(_TEST_HOME.name, "config.json")
_NO_INSTANCES = os.path.join(_TEST_HOME.name, "no-strata-instances")
os.environ["STRATA_INSTANCES_DIR"] = _NO_INSTANCES


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


# ── Mock Strata gateway ───────────────────────────────────────────────────
class MockStrata(BaseHTTPRequestHandler):
    """A Strata host: /v1/models lists the ready models (id + label only);
    /api/native reports each loaded model's vision switch and context."""

    instances = []
    native_status = 200

    def do_GET(self):
        if self.path == "/v1/models":
            ready = [i for i in MockStrata.instances if i["status"] == "ready"]
            self._json(200, {"object": "list", "data": [
                {"id": i["apiModelId"], "object": "model", "owned_by": "strata",
                 "meta": {"label": i["label"], "aliases": i["aliases"]}}
                for i in ready
            ]})
        elif self.path == "/api/native":
            if MockStrata.native_status != 200:
                self._json(MockStrata.native_status, {"error": "Open the local Strata app to use host controls."})
            else:
                self._json(200, {"instances": MockStrata.instances})
        else:
            self._json(404, {"error": "not found"})

    _json = MockUpstream._json

    def log_message(self, *args):
        pass


# ── Mock streaming chat upstream ──────────────────────────────────────────
class MockStream(BaseHTTPRequestHandler):
    """An OpenAI-style upstream: /v1/models lists model "m"; chat streams
    three SSE events 0.5 s apart (chunked), or drops the connection after the
    first one when fail_after_first is set. /api/v0/models answers like a
    broken LM Studio ({"data": null})."""

    protocol_version = "HTTP/1.1"
    fail_after_first = False
    last_body = None

    def do_GET(self):
        if self.path == "/api/v0/models":
            self._json(200, {"data": None})
        else:
            self._json(200, {"object": "list", "data": [{"id": "m", "object": "model"}]})

    def do_POST(self):
        MockStream.last_body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i in range(3):
            data = f'data: {{"n": {i}}}\n\n'.encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
            if MockStream.fail_after_first:
                self.close_connection = True  # drop mid-stream, no final chunk
                return
            time.sleep(0.5)
        self.wfile.write(b"0\r\n\r\n")

    _json = MockUpstream._json

    def log_message(self, *args):
        pass


def strata_instance(model_id, vision, context, status="ready"):
    return {"apiModelId": model_id, "aliases": [model_id], "label": f"Label {model_id}",
            "vision": vision, "context": context, "status": status}


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

    # ── Caller check: only the gateway's own web page ─────────────────────
    def _post_as(self, headers):
        return request(self.gw_port, "POST", self.proxy_path, body="{}", headers=headers)

    def test_own_page_is_allowed(self):
        status, _, _ = self._post_as({"Sec-Fetch-Site": "same-origin",
                                      "Origin": f"http://127.0.0.1:{self.gw_port}"})
        self.assertEqual(status, 200)
        self.assertEqual(len(MockUpstream.requests), 1)

    def test_own_page_by_lan_name_is_allowed(self):
        status, _, _ = self._post_as({"Host": "my-pc:8090", "Sec-Fetch-Site": "same-origin",
                                      "Origin": "http://my-pc:8090"})
        self.assertEqual(status, 200)

    def test_other_site_is_refused(self):
        status, _, _ = self._post_as({"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(MockUpstream.requests, [])

    def test_other_site_get_is_refused(self):
        status, _, _ = request(self.gw_port, "GET", self.proxy_path,
                               headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)
        self.assertEqual(MockUpstream.requests, [])

    def test_other_site_without_fetch_metadata_is_refused(self):
        """Older browsers send Origin but not Sec-Fetch-Site."""
        status, _, _ = self._post_as({"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(MockUpstream.requests, [])

    def test_dns_rebinding_is_refused(self):
        """An attacker's domain re-pointed at this PC looks same-origin to
        the browser; its public Host name gives it away."""
        rebound = f"attacker.example.com:{self.gw_port}"
        status, _, _ = self._post_as({"Host": rebound, "Sec-Fetch-Site": "same-origin",
                                      "Origin": f"http://{rebound}"})
        self.assertEqual(status, 403)
        self.assertEqual(MockUpstream.requests, [])

    def _refused_post(self, s, body: bytes, late: bool):
        head = (f"POST /cors-proxy?url=http://example.invalid/ HTTP/1.1\r\nHost: attacker.example.com:{self.gw_port}\r\n"
                f"Sec-Fetch-Site: cross-site\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode()
        s.sendall(head)
        if late:
            time.sleep(0.2)  # long enough for a refusal that ignores the body to have closed
        s.sendall(body)

    def test_a_refused_post_always_gets_its_answer(self):
        """Closing with a body unread resets the connection on Windows, and the
        client can lose the reply. The refusal must reach it."""
        import socket
        with socket.create_connection(("127.0.0.1", self.gw_port), timeout=5) as s:
            self._refused_post(s, b'{"x": 1}', late=True)
            reply = s.recv(4096)
        self.assertIn(b" 403 ", reply.split(b"\r\n")[0])

    def test_a_refused_body_is_not_read_as_a_new_request(self):
        import socket
        with socket.create_connection(("127.0.0.1", self.gw_port), timeout=5) as s:
            self._refused_post(s, b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", late=False)
            time.sleep(0.3)
            reply = s.recv(65536)
        self.assertIn(b" 403 ", reply.split(b"\r\n")[0])
        self.assertEqual(reply.count(b"HTTP/1."), 1)

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


class StrataEndpointTests(unittest.TestCase):
    """A Strata endpoint's capabilities follow the model loaded in Strata."""

    @classmethod
    def setUpClass(cls):
        cls.st_srv, cls.st_t, cls.st_port = start_server(MockStrata)
        cls.gw_srv, cls.gw_t, cls.gw_port = start_server(proxy.ProxyHandler)
        cls.host = f"127.0.0.1:{cls.st_port}"

    @classmethod
    def tearDownClass(cls):
        cls.gw_srv.shutdown(); cls.gw_srv.server_close()
        cls.st_srv.shutdown(); cls.st_srv.server_close()

    def setUp(self):
        MockStrata.native_status = 200
        MockStrata.instances = [
            strata_instance("iq3_s", vision=True, context=131072),
            strata_instance("q8_0", vision=False, context=32768),
            strata_instance("loading", vision=True, context=8192, status="loading"),
        ]

    def _by_id(self):
        return {m["id"]: m for m in proxy.fetch_strata_models(self.host, "")}

    def test_vision_and_context_come_from_loaded_model(self):
        models = self._by_id()
        self.assertEqual(set(models), {"iq3_s", "q8_0"})  # not-ready models are left out
        self.assertIn("image", models["iq3_s"]["architecture"]["input_modalities"])
        self.assertEqual(models["iq3_s"]["context_length"], 131072)
        self.assertEqual(models["iq3_s"]["name"], "Label iq3_s")
        self.assertEqual(models["q8_0"]["architecture"]["input_modalities"], ["text"])
        self.assertEqual(models["q8_0"]["context_length"], 32768)

    def test_switching_model_in_strata_is_picked_up(self):
        self.assertIn("image", self._by_id()["iq3_s"]["architecture"]["input_modalities"])
        MockStrata.instances = [strata_instance("iq3_s", vision=False, context=65536)]
        models = self._by_id()
        self.assertEqual(models["iq3_s"]["architecture"]["input_modalities"], ["text"])
        self.assertEqual(models["iq3_s"]["context_length"], 65536)

    def test_without_native_details_models_are_text_only(self):
        MockStrata.native_status = 403
        models = self._by_id()
        self.assertEqual(set(models), {"iq3_s", "q8_0"})
        self.assertEqual(models["iq3_s"]["architecture"]["input_modalities"], ["text"])

    def test_gateway_models_report_vision_for_strata(self):
        """End to end: the gateway's /v1/models sets modalities.vision, the
        field the bundled web UI reads to allow image uploads."""
        saved = proxy.get_config("endpoints")
        proxy.set_config("endpoints", [{"name": "strata", "type": "strata", "host": self.host,
                                        "enabled": True, "api_key": ""}])
        try:
            proxy.set_endpoint_status("strata", proxy.check_endpoint_health(
                proxy.get_config("endpoints")[0]))
            proxy.aggregate_models()
            status, _, body = request(self.gw_port, "GET", "/v1/models")
            self.assertEqual(status, 200)
            models = {m["id"]: m for m in json.loads(body)["data"]}
            self.assertTrue(models["strata:iq3_s"]["modalities"]["vision"])
            self.assertEqual(models["strata:iq3_s"]["context_length"], 131072)
            self.assertFalse(models["strata:q8_0"]["modalities"]["vision"])
        finally:
            proxy.set_config("endpoints", saved)
            proxy.set_endpoint_status("strata", "disabled")
            proxy.aggregate_models()


class GatewayReliabilityTests(unittest.TestCase):
    """Config safety, localhost dialing, streaming, aggregation resilience,
    catalog caching, default model, static file containment, port binding."""

    @classmethod
    def setUpClass(cls):
        cls.up_srv, cls.up_t, cls.up_port = start_server(MockStream)
        cls.gw_srv, cls.gw_t, cls.gw_port = start_server(proxy.ProxyHandler)

    @classmethod
    def tearDownClass(cls):
        cls.gw_srv.shutdown(); cls.gw_srv.server_close()
        cls.up_srv.shutdown(); cls.up_srv.server_close()

    def setUp(self):
        MockStream.fail_after_first = False
        self._saved = (proxy.CONFIG_FILE, proxy._config, proxy.get_models())
        proxy._config = dict(proxy._config)
        self._tmp = tempfile.TemporaryDirectory()
        proxy.CONFIG_FILE = os.path.join(self._tmp.name, "config.json")

    def tearDown(self):
        for name in ("mock", "broken"):
            proxy.set_endpoint_status(name, "disabled")
        proxy.CONFIG_FILE, proxy._config, models = self._saved
        proxy._config_load_error = ""
        proxy.aggregate_models()
        proxy.set_models(models)
        self._tmp.cleanup()

    def _use_mock_endpoint(self, extra=()):
        eps = [{"name": "mock", "type": "custom", "host": f"localhost:{self.up_port}",
                "enabled": True, "api_key": ""}] + list(extra)
        proxy.set_config("endpoints", eps)
        for ep in eps:
            proxy.set_endpoint_status(ep["name"], "online")
        proxy.aggregate_models()

    def _stream_chat(self):
        conn = HTTPConnection("127.0.0.1", self.gw_port, timeout=10)
        try:
            t0 = time.time()
            conn.request("POST", "/v1/chat/completions",
                         body=json.dumps({"model": "mock:m", "stream": True, "messages": []}),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            first = resp.fp.readline()
            first_at = time.time() - t0
            rest = resp.read()
            return resp.status, first, first_at, rest
        finally:
            conn.close()

    # ── config.json ───────────────────────────────────────────────────────
    def test_broken_config_is_not_overwritten(self):
        broken = '{"endpoints": [{"name": "mine", "api_key": "k"},], }'
        with open(proxy.CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(broken)
        proxy.load_config()
        self.assertTrue(proxy._config_load_error)
        proxy.save_config()
        with open(proxy.CONFIG_FILE, encoding="utf-8") as f:
            self.assertEqual(f.read(), broken)

    def test_config_save_round_trips_without_temp_file(self):
        with open(proxy.CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"endpoints": [], "models": [], "default_model": ""}, f)
        proxy.load_config()
        proxy.set_config("default_model", "mock:m")
        proxy.save_config()
        with open(proxy.CONFIG_FILE, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["default_model"], "mock:m")
        self.assertEqual(os.listdir(self._tmp.name), ["config.json"])

    # ── localhost dialing ─────────────────────────────────────────────────
    def test_localhost_is_dialed_as_ipv4(self):
        self.assertEqual(proxy._dial_host("localhost:1234"), "127.0.0.1:1234")
        self.assertEqual(proxy._dial_host("localhost"), "127.0.0.1")
        self.assertEqual(proxy._dial_host("localhostx:1"), "localhostx:1")
        self.assertEqual(proxy._dial_host("api.example.com"), "api.example.com")

    # ── streaming ─────────────────────────────────────────────────────────
    def test_stream_events_arrive_as_sent(self):
        self._use_mock_endpoint()
        status, first, first_at, rest = self._stream_chat()
        self.assertEqual(status, 200)
        self.assertEqual(first, b'data: {"n": 0}\n')
        # The upstream sends event 0 at once and the last one ~1 s later; a
        # buffered relay would hold event 0 until the end.
        self.assertLess(first_at, 0.45)
        self.assertIn(b'"n": 2', rest)

    def test_stream_failure_ends_with_sse_error_not_second_response(self):
        MockStream.fail_after_first = True
        self._use_mock_endpoint()
        status, first, _, rest = self._stream_chat()
        self.assertEqual(status, 200)
        self.assertEqual(first, b'data: {"n": 0}\n')
        self.assertIn(b'"error"', rest)
        self.assertNotIn(b"HTTP/1.", rest)
        self.assertNotIn(b"<html", rest.lower())

    def test_unreachable_endpoint_gives_json_error(self):
        """A stopped upstream (e.g. LM Studio closed while a client still
        asks for its model) gets an OpenAI-style JSON error naming it."""
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); closed_port = s.getsockname()[1]; s.close()
        proxy.set_config("endpoints", [{"name": "gone", "type": "lmstudio", "host": f"localhost:{closed_port}",
                                        "enabled": True, "api_key": ""}])
        status, headers, body = request(self.gw_port, "POST", "/v1/chat/completions",
                                        body=json.dumps({"model": "gone:some-model", "messages": []}),
                                        headers={"Content-Type": "application/json"})
        self.assertEqual(status, 502)
        self.assertIn("application/json", headers.get("content-type", ""))
        message = json.loads(body)["error"]["message"]
        self.assertIn("'gone'", message)
        self.assertIn("isn't running", message)

    # ── force default model ───────────────────────────────────────────────
    def _closed_port(self):
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        return port

    def _post(self, path, model):
        return request(self.gw_port, "POST", path, body=json.dumps({"model": model, "messages": []}),
                       headers={"Content-Type": "application/json"})

    def test_forced_default_model_overrides_client_model(self):
        """A client stuck on a model of a stopped endpoint (Cline asking for
        lmstudio:... while LM Studio is closed) gets the default instead."""
        gone = {"name": "gone", "type": "lmstudio", "host": f"127.0.0.1:{self._closed_port()}",
                "enabled": True, "api_key": ""}
        self._use_mock_endpoint(extra=[gone])
        proxy.set_config("default_model", "mock:m")
        proxy.set_config("force_default_model", True)
        status, _, _ = self._post("/v1/chat/completions", "gone:qwen")
        self.assertEqual(status, 200)
        self.assertEqual(MockStream.last_body["model"], "m")

    def test_without_force_client_model_is_kept(self):
        gone = {"name": "gone", "type": "lmstudio", "host": f"127.0.0.1:{self._closed_port()}",
                "enabled": True, "api_key": ""}
        self._use_mock_endpoint(extra=[gone])
        proxy.set_config("default_model", "mock:m")
        proxy.set_config("force_default_model", False)
        status, _, _ = self._post("/v1/chat/completions", "gone:qwen")
        self.assertEqual(status, 502)

    def test_force_leaves_non_chat_requests_alone(self):
        self._use_mock_endpoint()
        proxy.set_config("default_model", "mock:m")
        proxy.set_config("force_default_model", True)
        status, _, _ = self._post("/v1/embeddings", "mock:embed-model")
        self.assertEqual(status, 200)
        self.assertEqual(MockStream.last_body["model"], "embed-model")

    # ── aggregation / catalog ─────────────────────────────────────────────
    def test_one_bad_endpoint_does_not_drop_the_others(self):
        broken = {"name": "broken", "type": "lmstudio", "host": f"127.0.0.1:{self.up_port}",
                  "enabled": True, "api_key": ""}
        self._use_mock_endpoint(extra=[broken])  # raises inside the LM Studio parser
        self.assertIn("mock:m", proxy.get_models())

    def test_failed_catalog_fetch_keeps_last_good(self):
        good = [{"id": "a", "architecture": {"input_modalities": ["text", "image"]}}]
        results = [good, []]
        ttl = proxy.CATALOG_CACHE_TTL
        proxy.CATALOG_CACHE_TTL = 0  # force a refetch on every call
        try:
            self.assertEqual(proxy._get_cached_catalog("t-keep", lambda: results.pop(0)), good)
            self.assertEqual(proxy._get_cached_catalog("t-keep", lambda: results.pop(0)), good)
        finally:
            proxy.CATALOG_CACHE_TTL = ttl
            proxy.clear_catalog_cache()

    def test_failed_first_catalog_fetch_is_not_cached(self):
        good = [{"id": "a"}]
        results = [[], good]
        try:
            self.assertEqual(proxy._get_cached_catalog("t-retry", lambda: results.pop(0)), [])
            self.assertEqual(proxy._get_cached_catalog("t-retry", lambda: results.pop(0)), good)
        finally:
            proxy.clear_catalog_cache()

    # ── default model ─────────────────────────────────────────────────────
    def test_default_model_is_kept_by_id(self):
        proxy.set_config("default_model", "b:2")
        proxy.set_models(["a:1", "b:2", "c:3"])
        self.assertEqual(proxy.get_current_model(), "b:2")
        proxy.set_models(["x:0", "a:1", "b:2"])  # list reshuffled
        self.assertEqual(proxy.get_current_model(), "b:2")
        proxy.set_models(["a:1", "c:3"])  # its endpoint went offline
        self.assertEqual(proxy.get_current_model(), "a:1")
        proxy.set_models(["a:1", "b:2"])  # and came back
        self.assertEqual(proxy.get_current_model(), "b:2")

    def test_web_ui_model_pick_sets_default_by_id(self):
        self._use_mock_endpoint()
        status, _, _ = request(self.gw_port, "POST", "/models/load",
                               body=json.dumps({"model": "mock:m"}),
                               headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(proxy.get_config("default_model"), "mock:m")
        with open(proxy.CONFIG_FILE, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["default_model"], "mock:m")

    # ── static files ──────────────────────────────────────────────────────
    def test_static_files_stay_inside_webui(self):
        webui = os.path.join(self._tmp.name, "webui")
        src = os.path.join(self._tmp.name, "webui-src")
        os.makedirs(webui); os.makedirs(src)
        with open(os.path.join(webui, "index.html"), "w") as f:
            f.write("<html>ui</html>")
        with open(os.path.join(src, "secret.txt"), "w") as f:
            f.write("SECRET")
        saved = proxy.WEBUI_DIR
        proxy.WEBUI_DIR = webui
        proxy.set_config("endpoints", [])  # unmatched paths fall through to the proxy
        try:
            _, _, body = request(self.gw_port, "GET", "/..%2fwebui-src/secret.txt")
            self.assertNotIn(b"SECRET", body)
            status, _, body = request(self.gw_port, "GET", "/index.html")
            self.assertEqual(status, 200)
            self.assertIn(b"ui", body)
        finally:
            proxy.WEBUI_DIR = saved

    def test_web_ui_update_is_picked_up_at_once(self):
        """index.html must be revalidated (it names the build's hashed
        bundles); a missing UI asset is a 404, not a proxied upstream error."""
        webui = os.path.join(self._tmp.name, "webui")
        os.makedirs(os.path.join(webui, "_app", "immutable"))
        with open(os.path.join(webui, "index.html"), "w") as f:
            f.write('<script src="/_app/immutable/bundle.new.js"></script>')
        with open(os.path.join(webui, "_app", "immutable", "bundle.new.js"), "w") as f:
            f.write("//")
        saved = proxy.WEBUI_DIR
        proxy.WEBUI_DIR = webui
        proxy.set_config("endpoints", [])
        try:
            for path in ("/", "/index.html"):
                _, headers, _ = request(self.gw_port, "GET", path)
                self.assertEqual(headers.get("cache-control"), "no-cache")
            _, headers, _ = request(self.gw_port, "GET", "/_app/immutable/bundle.new.js")
            self.assertIn("immutable", headers.get("cache-control", ""))
            status, _, _ = request(self.gw_port, "GET", "/_app/immutable/bundle.old.js")
            self.assertEqual(status, 404)
        finally:
            proxy.WEBUI_DIR = saved

    # ── port binding ──────────────────────────────────────────────────────
    @unittest.skipUnless(os.name == "nt", "SO_REUSEADDR port sharing is Windows behaviour")
    def test_second_instance_cannot_share_the_port(self):
        first = proxy._GatewayHTTPServer(("127.0.0.1", 0), proxy.ProxyHandler)
        try:
            with self.assertRaises(OSError):
                proxy._GatewayHTTPServer(("127.0.0.1", first.server_address[1]), proxy.ProxyHandler)
        finally:
            first.server_close()


class SettingsScreenTests(unittest.TestCase):
    """The TUI's Settings screen, run headless."""

    def setUp(self):
        self._saved = (proxy.CONFIG_FILE, proxy._config, proxy.get_models())
        self._tmp = tempfile.TemporaryDirectory()
        proxy.CONFIG_FILE = os.path.join(self._tmp.name, "config.json")
        proxy._config = dict(proxy._config)
        proxy.set_config("endpoints", [])

    def tearDown(self):
        proxy.CONFIG_FILE, proxy._config, models = self._saved
        proxy.set_models(models)
        self._tmp.cleanup()

    def test_force_checkbox_sits_next_to_default_model_and_saves(self):
        import asyncio
        from textual.widgets import Checkbox, Select

        proxy.set_models(["a:1", "b:2"])
        proxy.set_config("default_model", "b:2")
        proxy.set_config("force_default_model", False)
        layout = {}

        async def run():
            app = proxy.ProxyTUI()
            async with app.run_test(size=(120, 40)) as pilot:
                screen = proxy.SettingsScreen()
                await app.push_screen(screen)
                await pilot.pause()
                box = screen.query_one("#force-model-checkbox", Checkbox)
                select = screen.query_one("#model-select", Select)
                layout.update(box=box.region, select=select.region)
                box.value = True
                screen.action_save_settings()
                await pilot.pause()

        asyncio.run(run())
        self.assertEqual(layout["box"].y, layout["select"].y)  # same row
        self.assertGreater(layout["box"].x, layout["select"].x)  # to its right
        self.assertGreater(layout["box"].width, 0)
        self.assertTrue(proxy.get_config("force_default_model"))
        with open(proxy.CONFIG_FILE, encoding="utf-8") as f:
            self.assertIs(json.load(f)["force_default_model"], True)

    def test_editing_an_endpoint_keeps_settings_the_dialog_does_not_show(self):
        import types

        kv = {"enabled": True, "min_tokens": 1}
        proxy.set_config("endpoints", [{"name": "st", "type": "strata", "host": "h:1", "api_key": "",
                                        "enabled": True, "timeout": 900, "kv_sessions": kv}])
        screen = types.SimpleNamespace(notify=lambda *a, **k: None, _populate_endpoint_table=lambda: None,
                                       _update_model_select=lambda: None)
        proxy.SettingsScreen._on_endpoint_saved(screen, {"name": "st", "type": "strata", "host": "h:2", "api_key": "",
                                                         "enabled": False, "_original_name": "st"})
        ep = proxy.get_config("endpoints")[0]
        self.assertEqual((ep["host"], ep["enabled"]), ("h:2", False))
        self.assertEqual((ep["timeout"], ep["kv_sessions"]), (900, kv))
        proxy.set_endpoint_status("st", "disabled")


# ── Mock one-slot engine (Strata's server) ────────────────────────────────
class MockEngine(BaseHTTPRequestHandler):
    """A one-slot engine: chat replaces what the slot holds (reusing the common
    start), /slots reports it, save and restore move it to and from files.
    Serves the model API and the slot API on one port."""

    protocol_version = "HTTP/1.1"
    held = ""
    saved = {}          # file name → text
    calls = []          # what happened, in order
    slot_dir = ""
    api_key = ""
    fail = {}           # "save" / "restore" → (status, message)
    chat_status = 200
    chat_delay = 0.0
    file_bytes = 1024
    active = 0
    max_active = 0
    metrics = None      # what /metrics answers, when set
    lock = threading.Lock()

    @classmethod
    def reset(cls, slot_dir):
        cls.held, cls.saved, cls.calls = "", {}, []
        cls.slot_dir, cls.api_key, cls.fail = slot_dir, "", {}
        cls.chat_status, cls.chat_delay, cls.file_bytes = 200, 0.0, 1024
        cls.active = cls.max_active = 0
        cls.metrics = None

    def _allowed(self):
        if MockEngine.api_key and self.headers.get("Authorization") != f"Bearer {MockEngine.api_key}":
            self._json(401, {"error": {"message": "Invalid API key"}})
            return False
        return True

    def do_GET(self):
        if self.path == "/slots":
            if self._allowed():
                MockEngine.calls.append(("slots",))
                self._json(200, [{"id": 0, "n_ctx": 524288, "is_processing": False,
                                  "n_prompt_tokens": len(MockEngine.held) // 4}])
        elif self.path == "/metrics" and MockEngine.metrics is not None:
            if self._allowed():
                self._json(200, MockEngine.metrics)
        else:
            self._json(200, {"object": "list", "data": [{"id": "m", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/slots/0":
            if self._allowed():
                self._slot(urllib.parse.parse_qs(parsed.query)["action"][0], body["filename"])
        else:
            self._chat(body)

    def _slot(self, action, name):
        MockEngine.calls.append((action, name))
        if action in MockEngine.fail:
            status, message = MockEngine.fail[action]
            self._json(status, {"error": {"code": status, "message": message}})
        elif action == "save":
            MockEngine.saved[name] = MockEngine.held
            with open(os.path.join(MockEngine.slot_dir, name), "wb") as f:
                f.write(b"\0" * MockEngine.file_bytes)
            self._json(200, {"id_slot": 0, "filename": name, "n_saved": len(MockEngine.held) // 4,
                             "n_written": MockEngine.file_bytes, "timings": {"save_ms": 1}})
        elif name not in MockEngine.saved:
            self._json(404, {"error": {"code": 404, "message": f"no saved session named {name}"}})
        else:
            MockEngine.held = MockEngine.saved[name]
            self._json(200, {"id_slot": 0, "filename": name, "n_restored": len(MockEngine.held) // 4,
                             "n_read": MockEngine.file_bytes, "timings": {"restore_ms": 1}})

    def _chat(self, body):
        with MockEngine.lock:
            MockEngine.active += 1
            MockEngine.max_active = max(MockEngine.max_active, MockEngine.active)
        try:
            if MockEngine.chat_delay:
                time.sleep(MockEngine.chat_delay)
            if MockEngine.chat_status != 200:
                MockEngine.calls.append(("refused",))
                self._json(MockEngine.chat_status, {"error": {"message": "refused"}})
                return
            text = "\n".join(str(m.get("content")) for m in body.get("messages", []))
            reused = len(os.path.commonprefix([MockEngine.held, text]))
            MockEngine.calls.append(("chat", reused, len(text)))
            MockEngine.held = text
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                data = b'data: {"n": 1}\n\n'
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
            else:
                self._json(200, {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        finally:
            with MockEngine.lock:
                MockEngine.active -= 1

    _json = MockUpstream._json

    def log_message(self, *args):
        pass


class KVChainTests(unittest.TestCase):
    """How a request becomes a chain of message hashes."""

    PATH = "/v1/chat/completions"

    def test_same_messages_give_the_same_chain(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}]
        a = proxy.kv_chain({"messages": msgs}, self.PATH)
        b = proxy.kv_chain({"messages": [dict(m) for m in msgs], "temperature": 0.2}, self.PATH)
        self.assertEqual(a.hashes, b.hashes)
        self.assertEqual(a.lens, [1, 6])
        self.assertEqual(a.total, 6)

    def test_a_longer_chat_extends_the_chain(self):
        short = proxy.kv_chain({"messages": [{"role": "user", "content": "hi"}]}, self.PATH)
        long = proxy.kv_chain({"messages": [{"role": "user", "content": "hi"},
                                            {"role": "assistant", "content": "yes"},
                                            {"role": "user", "content": "more"}]}, self.PATH)
        self.assertEqual(long.hashes[:1], short.hashes)
        self.assertEqual(proxy._kv_common(short.hashes, short.lens, long.hashes), (1, 2))

    def test_tools_are_part_of_the_chain(self):
        msgs = [{"role": "user", "content": "hi"}]
        plain = proxy.kv_chain({"messages": msgs}, self.PATH)
        tooled = proxy.kv_chain({"messages": msgs, "tools": [{"type": "function"}]}, self.PATH)
        self.assertNotEqual(plain.hashes[0], tooled.hashes[0])

    def test_text_parts_read_like_plain_text_and_images_count_by_hash(self):
        plain = proxy.kv_chain({"messages": [{"role": "user", "content": "hi"}]}, self.PATH)
        parts = proxy.kv_chain({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}, self.PATH)
        self.assertEqual(plain.hashes, parts.hashes)
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}}
        other = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "B" * 5000}}
        one = proxy.kv_chain({"messages": [{"role": "user", "content": [img]}]}, self.PATH)
        two = proxy.kv_chain({"messages": [{"role": "user", "content": [other]}]}, self.PATH)
        self.assertNotEqual(one.hashes, two.hashes)
        self.assertLess(one.total, 100)

    def test_anthropic_system_prompt_counts(self):
        msgs = [{"role": "user", "content": "hi"}]
        a = proxy.kv_chain({"system": "one", "messages": msgs}, "/v1/messages")
        b = proxy.kv_chain({"system": "two", "messages": msgs}, "/v1/messages")
        self.assertNotEqual(a.hashes[0], b.hashes[0])
        self.assertEqual(len(a.hashes), 2)

    def test_no_conversation_gives_no_chain(self):
        self.assertIsNone(proxy.kv_chain({"messages": []}, self.PATH))
        self.assertIsNone(proxy.kv_chain({"prompt": "x"}, self.PATH))
        self.assertIsNone(proxy.kv_chain(["not", "a", "dict"], self.PATH))

    def test_settings_get_defaults_and_bad_numbers_are_ignored(self):
        self.assertIsNone(proxy._kv_config({"name": "x"}))
        self.assertIsNone(proxy._kv_config({"kv_sessions": {"enabled": False}}))
        cfg = proxy._kv_config({"kv_sessions": {"enabled": True, "min_tokens": "oops", "max_gb": -5, "x": 1}})
        self.assertEqual(cfg["min_tokens"], proxy.KV_DEFAULTS["min_tokens"])
        self.assertEqual(cfg["max_gb"], 0)
        self.assertNotIn("x", cfg)


class KVSessionSwapTests(unittest.TestCase):
    """End to end through the gateway against a mock one-slot engine."""

    @classmethod
    def setUpClass(cls):
        cls.en_srv, cls.en_t, cls.en_port = start_server(MockEngine)
        cls.gw_srv, cls.gw_t, cls.gw_port = start_server(proxy.ProxyHandler)

    @classmethod
    def tearDownClass(cls):
        cls.gw_srv.shutdown(); cls.gw_srv.server_close()
        cls.en_srv.shutdown(); cls.en_srv.server_close()

    def setUp(self):
        self._saved = (proxy.CONFIG_FILE, proxy._config, proxy._KV_IDLE_POLL_S)
        proxy._config = dict(proxy._config)
        self._tmp = tempfile.TemporaryDirectory()
        proxy.CONFIG_FILE = os.path.join(self._tmp.name, "config.json")
        self.slot_dir = os.path.join(self._tmp.name, "slots")
        os.mkdir(self.slot_dir)
        MockEngine.reset(self.slot_dir)
        proxy._kv_managers.clear()
        proxy.set_config("force_default_model", False)
        self._use({})

    def tearDown(self):
        for m in proxy._kv_managers.values():
            m.cfg = {**m.cfg, "idle_save_s": 0}  # lets an idle-save thread end
        proxy._kv_managers.clear()
        proxy.set_endpoint_status("eng", "disabled")
        proxy._stats.wait_idle()
        proxy.CONFIG_FILE, proxy._config, proxy._KV_IDLE_POLL_S = self._saved
        os.environ["STRATA_INSTANCES_DIR"] = _NO_INSTANCES
        self._tmp.cleanup()

    def _use(self, kv, ep_type="custom", enabled=True):
        settings = {"enabled": enabled, "engine": f"127.0.0.1:{self.en_port}", "min_tokens": 100,
                    "min_gain_chars": 50, "idle_save_s": 0, "slot_dir": self.slot_dir}
        settings.update(kv)
        proxy.set_config("endpoints", [{"name": "eng", "type": ep_type, "host": f"localhost:{self.en_port}",
                                        "enabled": True, "api_key": "", "kv_sessions": settings}])
        proxy.set_endpoint_status("eng", "online")

    @staticmethod
    def _conv(tag, *more):
        msgs = [{"role": "system", "content": "You help."}, {"role": "user", "content": tag * 2000}]
        for i, text in enumerate(more):
            msgs.append({"role": "assistant" if i % 2 == 0 else "user", "content": text})
        return msgs

    def _chat(self, msgs, stream=False, model="eng:m", headers=None):
        status, _, body = request(self.gw_port, "POST", "/v1/chat/completions",
                                  body=json.dumps({"model": model, "messages": msgs, "stream": stream}),
                                  headers={"Content-Type": "application/json", **(headers or {})})
        # The gateway finishes its bookkeeping just after the client has the
        # whole reply, still holding the lock it took before forwarding.
        manager = proxy._kv_managers.get("eng")
        if manager:
            with manager.lock:
                pass
        return status, body

    @staticmethod
    def _steps():
        return [c for c in MockEngine.calls if c[0] != "slots"]

    def _names(self, action):
        return [c[1] for c in MockEngine.calls if c[0] == action]

    def test_switching_saves_the_held_chat_and_restores_the_other(self):
        a_len = len("You help.\n" + "a" * 2000)
        self.assertEqual(self._chat(self._conv("a"))[0], 200)
        self.assertEqual(self._names("save"), [])  # first chat: nothing to save yet
        self.assertEqual(self._chat(self._conv("b"))[0], 200)
        self.assertEqual(len(self._names("save")), 1)
        a_file = self._names("save")[0]
        self.assertTrue(a_file.startswith("gw-") and a_file.endswith(".bin"))
        MockEngine.calls.clear()
        self.assertEqual(self._chat(self._conv("a", "ok", "more"))[0], 200)
        steps = self._steps()
        self.assertEqual([s[0] for s in steps], ["save", "restore", "chat"])  # in this order
        self.assertEqual(steps[1][1], a_file)
        self.assertNotEqual(steps[0][1], a_file)
        self.assertEqual(steps[2][1], a_len)  # the whole first chat was reused

    def test_continuing_the_held_chat_touches_nothing(self):
        self._chat(self._conv("a"))
        MockEngine.calls.clear()
        self._chat(self._conv("a", "ok", "more"))
        self._chat(self._conv("a", "ok", "more", "fine", "again"))
        self.assertEqual([s[0] for s in self._steps()], ["chat", "chat"])

    def test_regenerating_or_editing_the_last_message_is_not_a_switch(self):
        self._chat(self._conv("a", "ok", "more"))
        MockEngine.calls.clear()
        self._chat(self._conv("a", "ok", "more"))           # regenerate
        self._chat(self._conv("a", "ok", "changed"))        # edit: 98% of the chat stays
        self.assertEqual([s[0] for s in self._steps()], ["chat", "chat"])

    def test_a_short_side_request_is_never_saved_and_the_long_chat_comes_back(self):
        self._chat(self._conv("a"))
        self._chat([{"role": "user", "content": "Title this chat."}])
        self.assertEqual(len(self._names("save")), 1)  # the long chat
        MockEngine.calls.clear()
        self._chat(self._conv("a", "ok", "more"))
        self.assertEqual([s[0] for s in self._steps()], ["restore", "chat"])  # side request not saved

    def test_a_saved_chat_keeps_its_file(self):
        self._chat(self._conv("a"))
        self._chat(self._conv("b"))
        a_file = self._names("save")[0]
        self._chat(self._conv("a", "ok", "more"))
        MockEngine.calls.clear()
        self._chat(self._conv("c"))
        self.assertEqual(self._names("save"), [a_file])  # same file, newer state
        self.assertIn("ok\nmore", MockEngine.saved[a_file])
        self.assertEqual(proxy.kv_status_all()["eng"]["sessions"], 2)

    def test_an_already_saved_chat_is_not_saved_again(self):
        self._chat(self._conv("a"))
        self._chat(self._conv("b"))
        self._chat(self._conv("a"))  # restored, nothing new since
        MockEngine.calls.clear()
        self._chat(self._conv("b"))
        steps = [s[0] for s in self._steps()]
        self.assertEqual(steps.count("save"), 1)  # only the chat that changed
        self.assertEqual(steps[-1], "chat")

    def test_engine_restart_forgets_the_held_chat(self):
        self._chat(self._conv("a"))
        MockEngine.held = ""  # the model was unloaded and loaded again
        MockEngine.calls.clear()
        self.assertEqual(self._chat(self._conv("b"))[0], 200)
        self.assertEqual(self._names("save"), [])

    def test_someone_using_the_engine_directly_is_not_saved_under_this_chats_name(self):
        self._chat(self._conv("a"))
        MockEngine.held = "x" * 100000
        MockEngine.calls.clear()
        self._chat(self._conv("b"))
        self.assertEqual(self._names("save"), [])

    def test_engine_without_slot_saving_still_answers_and_is_left_alone(self):
        MockEngine.fail = {"save": (501, "slot save/restore is disabled (start the server with --slot-save-path DIR)")}
        self._chat(self._conv("a"))
        self.assertEqual(self._chat(self._conv("b"))[0], 200)
        self.assertEqual(len(self._names("save")), 1)
        self.assertGreater(proxy.kv_status_all()["eng"]["paused"], 0)
        MockEngine.calls.clear()
        self.assertEqual(self._chat(self._conv("c"))[0], 200)
        self.assertEqual([c[0] for c in MockEngine.calls], ["chat"])  # not even /slots

    def test_a_failed_save_does_not_stop_the_request(self):
        MockEngine.fail = {"save": (507, "not enough disk space")}
        self._chat(self._conv("a"))
        self.assertEqual(self._chat(self._conv("b"))[0], 200)
        self.assertEqual(proxy.kv_status_all()["eng"]["errors"], 1)
        self.assertEqual(proxy.kv_status_all()["eng"]["sessions"], 0)

    def test_a_missing_file_is_forgotten(self):
        self._chat(self._conv("a"))
        self._chat(self._conv("b"))
        self.assertEqual(proxy.kv_status_all()["eng"]["sessions"], 1)
        MockEngine.saved.clear()
        MockEngine.calls.clear()
        self.assertEqual(self._chat(self._conv("a", "ok", "more"))[0], 200)
        self.assertEqual(len(self._names("restore")), 1)
        names = {e["file"] for e in proxy._kv_managers["eng"].index.values()}
        self.assertNotIn(self._names("restore")[0], names)

    def test_requests_are_served_one_at_a_time(self):
        MockEngine.chat_delay = 0.2
        results = []
        threads = [threading.Thread(target=lambda t=t: results.append(self._chat(self._conv(t))[0]))
                   for t in "abc"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, [200, 200, 200])
        self.assertEqual(MockEngine.max_active, 1)

    def test_endpoint_without_the_setting_is_untouched(self):
        self._use({}, enabled=False)
        self._chat(self._conv("a"))
        self._chat(self._conv("b"))
        self.assertEqual([c[0] for c in MockEngine.calls], ["chat", "chat"])  # no /slots at all
        self.assertEqual(proxy.kv_status_all(), {})

    def test_other_requests_are_not_managed(self):
        status, _, _ = request(self.gw_port, "POST", "/v1/embeddings",
                               body=json.dumps({"model": "eng:m", "input": "x"}),
                               headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(self._names("slots"), [])
        kv = proxy.kv_status_all()["eng"]
        self.assertIsNone(kv["held"])
        self.assertEqual((kv["sessions"], kv["saves"], kv["restores"], kv["continued"], kv["new"]), (0, 0, 0, 0, 0))

    def test_streamed_chat_counts_as_held(self):
        status, body = self._chat(self._conv("a"), stream=True)
        self.assertEqual(status, 200)
        self.assertIn(b"data:", body)
        self._chat(self._conv("b"))
        self.assertEqual(len(self._names("save")), 1)

    def test_a_refused_request_leaves_the_held_chat_in_place(self):
        self._chat(self._conv("a"))
        MockEngine.chat_status = 400
        self.assertEqual(self._chat(self._conv("b"))[0], 400)
        MockEngine.chat_status = 200
        MockEngine.calls.clear()
        self._chat(self._conv("a", "ok", "more"))
        self.assertEqual([s[0] for s in self._steps()], ["chat"])  # still held, no restore

    def test_sessions_survive_a_gateway_restart(self):
        a_len = len("You help.\n" + "a" * 2000)
        self._chat(self._conv("a"))
        self._chat(self._conv("b"))
        proxy._kv_managers.clear()
        MockEngine.calls.clear()
        self._chat(self._conv("a", "ok", "more"))
        steps = self._steps()
        self.assertEqual([s[0] for s in steps], ["restore", "chat"])  # the unknown held chat is not saved
        self.assertEqual(steps[1][1], a_len)

    def test_idle_save(self):
        proxy._KV_IDLE_POLL_S = 0.02
        self._use({"idle_save_s": 0.05})
        self._chat(self._conv("a"))
        deadline = time.time() + 5
        while time.time() < deadline and not self._names("save"):
            time.sleep(0.02)
        self.assertEqual(len(self._names("save")), 1)
        time.sleep(0.2)
        self.assertEqual(len(self._names("save")), 1)  # once, not over and over
        self._chat(self._conv("b"))
        self.assertEqual(self._names("save").count(self._names("save")[0]), 1)  # a was already saved
        self.assertTrue(proxy.kv_status_all()["eng"]["held"])

    def test_oldest_files_are_deleted_only_when_asked_and_only_our_own(self):
        foreign = os.path.join(self.slot_dir, "notes.bin")
        stray = os.path.join(self.slot_dir, "gw-not-in-the-index.bin")
        for path in (foreign, stray):
            with open(path, "wb") as f:
                f.write(b"x")
        self._use({"max_gb": 1500 / 1024 ** 3})
        for tag in "abc":
            self._chat(self._conv(tag))
        saved = self._names("save")
        self.assertEqual(len(saved), 2)
        self.assertFalse(os.path.exists(os.path.join(self.slot_dir, saved[0])))  # the oldest went
        self.assertTrue(os.path.exists(os.path.join(self.slot_dir, saved[1])))   # the held chat's stays
        self.assertTrue(os.path.exists(foreign))
        self.assertTrue(os.path.exists(stray))
        self.assertEqual(proxy.kv_status_all()["eng"]["sessions"], 1)

    def test_nothing_is_deleted_by_default(self):
        for tag in "abcd":
            self._chat(self._conv(tag))
        self.assertEqual(len(os.listdir(self.slot_dir)), 3)

    def test_finds_stratas_engine_by_model(self):
        folder = os.path.join(self._tmp.name, "instances")
        os.mkdir(folder)
        os.environ["STRATA_INSTANCES_DIR"] = folder
        now = time.time()
        files = {
            "old.json": ({"port": 1111, "api_key": "old", "model_name": "iq3_s"}, now - 100),
            "new.json": ({"port": 2222, "api_key": "new", "model_name": "iq3_s",
                          "slot_save_path": os.path.join(self._tmp.name, "s")}, now - 50),
            "other.json": ({"port": 3333, "api_key": "o", "model_name": "q8_0", "aliases": ["fast"]}, now),
            "broken.json": (None, now + 10),
            "drafting-1.json": ({"port": 4444, "api_key": "p", "model_name": "iq3_s"}, now + 20),
        }
        for name, (cfg, mtime) in files.items():
            path = os.path.join(folder, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(cfg) if cfg else "{not json")
            os.utime(path, (mtime, mtime))
        self.assertEqual(proxy._kv_find_strata_instance("iq3_s"),
                         ("127.0.0.1:2222", "new", os.path.join(self._tmp.name, "s")))
        self.assertEqual(proxy._kv_find_strata_instance("FAST")[0], "127.0.0.1:3333")
        self.assertEqual(proxy._kv_find_strata_instance("nobody")[0], "127.0.0.1:3333")  # newest
        os.environ["STRATA_INSTANCES_DIR"] = os.path.join(folder, "missing")
        self.assertIsNone(proxy._kv_find_strata_instance("iq3_s"))

    def test_strata_endpoint_uses_the_key_strata_wrote(self):
        folder = os.path.join(self._tmp.name, "instances")
        os.mkdir(folder)
        os.environ["STRATA_INSTANCES_DIR"] = folder
        with open(os.path.join(folder, "i.json"), "w", encoding="utf-8") as f:
            json.dump({"port": self.en_port, "api_key": "instance-key", "model_name": "iq3_s"}, f)
        MockEngine.api_key = "instance-key"
        self._use({"engine": "", "slot_dir": ""}, ep_type="strata")
        self._chat(self._conv("a"), model="eng:iq3_s")
        self.assertEqual(self._chat(self._conv("b"), model="eng:iq3_s")[0], 200)
        self.assertEqual(len(self._names("save")), 1)
        self.assertEqual(proxy.kv_status_all()["eng"]["errors"], 0)

    def test_no_engine_to_talk_to_only_pauses_swapping(self):
        self._use({"engine": ""})  # a custom endpoint must name its engine
        self.assertEqual(self._chat(self._conv("a"))[0], 200)
        self.assertGreater(proxy.kv_status_all()["eng"]["paused"], 0)
        self.assertEqual([c[0] for c in MockEngine.calls], ["chat"])

    def test_wrong_engine_key_only_pauses_swapping(self):
        MockEngine.api_key = "right"
        self._use({"engine_api_key": "wrong"})
        self.assertEqual(self._chat(self._conv("a"))[0], 200)
        self.assertGreater(proxy.kv_status_all()["eng"]["paused"], 0)

    def test_health_reports_sessions_without_secrets(self):
        self._use({"engine_api_key": ""})
        self._chat(self._conv("a"))
        self._chat(self._conv("b"))
        status, _, body = request(self.gw_port, "GET", "/health")
        self.assertEqual(status, 200)
        kv = json.loads(body)["kv_sessions"]["eng"]
        self.assertEqual(kv["sessions"], 1)
        self.assertEqual(kv["saves"], 1)
        self.assertEqual(kv["bytes"], 1024)
        self.assertEqual(kv["held"]["saved"], False)

    def test_each_request_says_what_the_swap_did(self):
        proxy._stats = proxy.GatewayStats()
        cline = {"X-Title": "Cline"}
        self._chat(self._conv("a"), headers=cline)
        self._chat(self._conv("a", "ok", "more"), headers=cline)
        self._chat(self._conv("b"), headers={"X-Title": "SillyTavern"})
        self._chat(self._conv("a", "ok", "more", "fine", "again"), headers=cline)
        proxy._stats.wait_idle()
        recent = proxy._stats.snapshot()["recent"]  # newest first
        self.assertEqual([r["swap"] for r in recent], ["restored", "new", "continued", "new"])
        self.assertIsNotNone(recent[0]["swap_s"])
        self.assertEqual(recent[0]["app"], "Cline")

    def _restart_gateway(self):
        for m in proxy._kv_managers.values():
            m.cfg = {**m.cfg, "idle_save_s": 0}
        proxy._kv_managers.clear()

    def test_saved_chats_show_right_after_a_restart(self):
        self._chat(self._conv("a"), headers={"X-Title": "Cline"})
        self._chat(self._conv("b"))
        saved = len(proxy._kv_managers["eng"].index)
        self.assertGreater(saved, 0)
        self._restart_gateway()
        before = len(MockEngine.calls)
        kv = proxy.stats_snapshot()["kv_sessions"]["eng"]
        self.assertEqual(kv["sessions"], saved)
        self.assertIsNone(kv["held"])
        self.assertEqual(len(MockEngine.calls), before)  # the engine is not asked
        snap = {"gateway": {}, "engines": {"eng": {"ok": True, "info": {}, "live": {}}}, "kv_sessions": {"eng": kv}}
        self.assertIn("Cline", _as_text(proxy.panel_chats(snap)))
        self.assertNotIn("turn on", _as_text(proxy.panel_chats(snap)))
        live = _as_text(proxy.render_live(snap))
        self.assertIn("none tracked yet", live)
        self.assertIn("1 chat,", live)

    def test_a_restart_still_finds_its_chats_after_being_looked_at(self):
        self._chat(self._conv("a"))
        self._chat(self._conv("b"))
        self._restart_gateway()
        proxy.kv_status_all()  # the console looks before the first request
        MockEngine.calls = []
        self._chat(self._conv("a", "ok", "more"))
        self.assertEqual(proxy.kv_status_all()["eng"]["restores"], 1)

    def test_looking_never_waits_for_a_request(self):
        self._chat(self._conv("a"))
        self._restart_gateway()
        ep = proxy.get_config("endpoints")[0]
        manager = proxy._get_kv_manager(ep, proxy._kv_config(ep))
        held, done = threading.Event(), threading.Event()

        def hold():
            with manager.lock:
                held.set()
                done.wait(10)

        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        held.wait(5)
        try:
            t0 = time.time()
            proxy.kv_status_all()
            self.assertLess(time.time() - t0, 1)
        finally:
            done.set()
            worker.join(5)

    def test_saved_chats_are_listed_with_their_app_and_counts(self):
        self._chat(self._conv("a"), headers={"X-Title": "Cline"})
        self._chat(self._conv("b"), headers={"X-Title": "SillyTavern"})
        self._chat(self._conv("a", "ok", "more"), headers={"X-Title": "Cline"})
        kv = proxy.kv_status_all()["eng"]
        self.assertEqual({c["app"] for c in kv["chats"]}, {"Cline", "SillyTavern"})
        cline = next(c for c in kv["chats"] if c["app"] == "Cline")
        self.assertEqual(cline["restores"], 1)
        self.assertEqual(kv["held"]["app"], "Cline")
        self.assertEqual((kv["new"], kv["restores"], kv["saves"]), (2, 1, 2))
        self.assertGreater(kv["restored_tokens"], 0)
        self.assertEqual(kv["written_bytes"], 2048)
        self.assertIn("restored", kv["last_event"])

    def test_engine_numbers_fill_in_a_request(self):
        proxy._stats = proxy.GatewayStats()
        MockEngine.metrics = {"engine": {"model": "m"}, "requests": [{
            "time": time.time() + 0.5, "prompt_tokens": 9000, "reused": 8000, "output_tokens": 200, "prompt_read": 1000,
            "prompt_ms": 2000.0, "decode_tok_s": 31.5, "hit_rate": 0.95, "drafts_offered": 150, "drafts_accepted": 100,
            "secret_text": "must not be kept"}]}
        self._chat(self._conv("a"))
        proxy._stats.wait_idle()
        rec = proxy._stats.snapshot()["recent"][0]
        self.assertEqual((rec["prompt"], rec["cached"], rec["out"]), (9000, 8000, 200))
        self.assertEqual((rec["tok_s"], rec["read_tok_s"], rec["expert_hit"]), (31.5, 500.0, 0.95))
        self.assertAlmostEqual(rec["saved_s"], 16.0)  # 8000 reused at the 500 t/s it just read at
        self.assertEqual(proxy._stats.snapshot()["session"]["saved_s"], rec["saved_s"])
        self.assertNotIn("must not be kept", json.dumps(proxy.stats_snapshot()))


class ResponseTapTests(unittest.TestCase):
    """Usage and speed are read out of a response as it passes."""

    @staticmethod
    def _sse(*events):
        return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"

    def _tapped(self, payload, chunk=7, streaming=True):
        tap = proxy._ResponseTap()
        for i in range(0, len(payload), chunk):  # lines split across chunks, as on the wire
            tap.feed(payload[i:i + chunk], streaming)
        tap.finish()
        return tap

    def test_openai_stream(self):
        tap = self._tapped(self._sse(
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},  # no token yet
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"reasoning_content": "hm"}}]},
            {"choices": [{"delta": {"tool_calls": [{"id": "1"}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 3,
                                      "prompt_tokens_details": {"cached_tokens": 90}, "cost": 0.5}}))
        self.assertEqual(tap.deltas, 3)
        self.assertIsNotNone(tap.first_token)
        self.assertEqual(tap.usage, {"prompt": 100, "cached": 90, "out": 3, "cost": 0.5})

    def test_llama_cpp_timings(self):
        tap = self._tapped(self._sse({"choices": [{"delta": {"content": "x"}}],
                                      "timings": {"prompt_n": 5, "cache_n": 95, "predicted_per_second": 33.3}}))
        self.assertEqual(tap.timings["cache_n"], 95)

    def test_anthropic_stream_counts_cached_tokens_into_the_prompt(self):
        tap = self._tapped(self._sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 90,
                                                            "cache_creation_input_tokens": 5, "output_tokens": 0}}},
            {"type": "content_block_delta", "delta": {"text": "hi"}},
            {"type": "message_delta", "usage": {"output_tokens": 7}}))
        self.assertEqual(tap.usage, {"prompt": 105, "cached": 90, "out": 7})
        self.assertEqual(tap.deltas, 1)

    def test_responses_api_stream(self):
        tap = self._tapped(self._sse(
            {"type": "response.output_text.delta", "delta": "hi"},
            {"type": "response.completed", "response": {"usage": {
                "input_tokens": 50, "output_tokens": 4, "input_tokens_details": {"cached_tokens": 40}}}}))
        self.assertEqual(tap.usage, {"prompt": 50, "cached": 40, "out": 4})
        self.assertEqual(tap.deltas, 1)

    def test_deepseek_cache_field(self):
        self.assertEqual(proxy._norm_usage({"prompt_tokens": 10, "prompt_cache_hit_tokens": 6})["cached"], 6)

    def test_plain_reply(self):
        body = json.dumps({"choices": [{"message": {"content": "hi"}}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 1}}).encode()
        tap = self._tapped(body, streaming=False)
        self.assertEqual(tap.usage, {"prompt": 5, "out": 1})
        self.assertIsNone(tap.first_token)  # one body: no token timing in it
        self.assertEqual(tap.bytes, len(body))

    def test_a_huge_plain_reply_is_passed_on_unread(self):
        saved = proxy._TAP_MAX_BODY
        proxy._TAP_MAX_BODY = 10
        try:
            tap = self._tapped(b'{"usage": {"prompt_tokens": 5}}', streaming=False)
        finally:
            proxy._TAP_MAX_BODY = saved
        self.assertEqual(tap.usage, {})
        self.assertEqual(tap.bytes, 31)

    def test_junk_is_ignored(self):
        tap = self._tapped(b"data: {broken\n\ndata: [1, 2]\n\n: comment\n\nevent: x\n\ndata: \n\n" + bytes(range(256)) + b"\n")
        self.assertEqual((tap.usage, tap.deltas), ({}, 0))
        self.assertEqual(self._tapped(b"\xff\xfe not json", streaming=False).usage, {})


class StatsFormatTests(unittest.TestCase):
    def test_counts(self):
        cases = {None: "—", 0: "0", 999: "999", 1500: "1.5k", 16_400_000: "16.4M", 2_000_000_000: "2.0B"}
        for value, text in cases.items():
            self.assertEqual(proxy.fmt_count(value), text)

    def test_durations(self):
        cases = {None: "—", 0.42: "420 ms", 1.2: "1.2 s", 200: "3m 20s", 56280: "15h 38m", 190800: "2d 5h"}
        for value, text in cases.items():
            self.assertEqual(proxy.fmt_secs(value), text)

    def test_bytes(self):
        self.assertEqual(proxy.fmt_bytes(5.2 * 1024 ** 3), "5.2 GB")
        self.assertEqual(proxy.fmt_bytes(10), "10 B")
        self.assertEqual(proxy.fmt_bytes(None), "—")

    def test_spark(self):
        self.assertEqual(proxy.spark([0, 50, 100]), "▁▅█")
        self.assertEqual(proxy.spark([1, None, 1]), "█ █")
        self.assertEqual(proxy.spark([0, 0]), "▁▁")
        self.assertEqual(proxy.spark([]), "")
        self.assertEqual(proxy.spark(None), "")
        self.assertEqual(len(proxy.spark(range(100), 10)), 10)
        self.assertEqual(proxy.spark([50], top=100), "▅")
        self.assertEqual(proxy.spark([500], top=100), "█")

    def test_meter_is_always_the_asked_width(self):
        for fraction in (0, 0.02, 0.5, 0.999, 1, 5, -1, None):
            self.assertEqual(len(proxy.meter(fraction, 12)), 12)
        self.assertEqual(proxy.meter(1, 4), "████")
        self.assertEqual(proxy.meter(0, 4), "░░░░")

    def test_percentile(self):
        self.assertEqual(proxy._percentile([1, 2, 3], 50), 2)
        self.assertEqual(proxy._percentile([1, 2], 50), 1.5)
        self.assertIsNone(proxy._percentile([], 50))
        self.assertEqual(proxy._percentile([None, 4], 95), 4)

    def test_names_are_safe_to_show(self):
        self.assertEqual(proxy._stats_name("gpt\x1b[31m\n-5"), "gpt[31m-5")
        self.assertEqual(proxy._stats_name(None), "unknown")
        self.assertEqual(len(proxy._stats_name("x" * 500)), 48)

    def test_app_names(self):
        self.assertEqual(proxy._app_name({"X-Title": "Cline", "User-Agent": "node"}), "Cline")
        self.assertEqual(proxy._app_name({"User-Agent": "Mozilla/5.0", "Referer": "http://pc:8090/", "Host": "pc:8090"}), "Web UI")
        self.assertEqual(proxy._app_name({"User-Agent": "Mozilla/5.0", "Referer": "http://other.lan/x", "Host": "pc:8090"}), "other.lan")
        self.assertEqual(proxy._app_name({"User-Agent": "Mozilla/5.0"}), "Browser")
        self.assertEqual(proxy._app_name({"User-Agent": "OpenAI/Python 1.2", "HTTP-Referer": "https://cline.bot/"}), "cline.bot")
        self.assertEqual(proxy._app_name({"User-Agent": "OpenAI/Python 1.2"}), "OpenAI")
        self.assertEqual(proxy._app_name({}), "unknown")


class MockChat(BaseHTTPRequestHandler):
    """An OpenAI-style upstream with usage, in JSON or SSE, or a chosen error."""

    protocol_version = "HTTP/1.1"
    status = 200
    reply_text = "hello"

    def do_GET(self):
        self._json(200, {"object": "list", "data": [{"id": "m", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
        if MockChat.status != 200:
            self._json(MockChat.status, {"error": {"message": "nope"}})
        elif body.get("stream"):
            payload = MockChat.stream_body()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self._json(200, {"choices": [{"message": {"role": "assistant", "content": MockChat.reply_text}}],
                             "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                       "prompt_tokens_details": {"cached_tokens": 75}, "cost": 0.25}})

    @staticmethod
    def stream_body():
        events = [{"choices": [{"delta": {"content": MockChat.reply_text}}]},
                  {"choices": [{"delta": {"content": "!"}}]},
                  {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 2}}]
        return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"

    _json = MockUpstream._json

    def log_message(self, *args):
        pass


class LiveStatsTests(unittest.TestCase):
    """What the gateway counts as requests pass through it."""

    SECRET = "my-private-chat-text-4711"

    @classmethod
    def setUpClass(cls):
        cls.up_srv, cls.up_t, cls.up_port = start_server(MockChat)
        cls.gw_srv, cls.gw_t, cls.gw_port = start_server(proxy.ProxyHandler)

    @classmethod
    def tearDownClass(cls):
        cls.gw_srv.shutdown(); cls.gw_srv.server_close()
        cls.up_srv.shutdown(); cls.up_srv.server_close()

    def setUp(self):
        self._saved = (proxy.CONFIG_FILE, proxy._config, proxy._stats)
        proxy._config = dict(proxy._config)
        self._tmp = tempfile.TemporaryDirectory()
        proxy.CONFIG_FILE = os.path.join(self._tmp.name, "config.json")
        proxy._stats = proxy.GatewayStats()
        MockChat.status = 200
        MockChat.reply_text = "hello"
        proxy.set_config("force_default_model", False)
        proxy.set_config("endpoints", [{"name": "up", "type": "custom", "host": f"localhost:{self.up_port}",
                                        "enabled": True, "api_key": ""}])
        proxy.set_endpoint_status("up", "online")

    def tearDown(self):
        proxy._stats.wait_idle()
        proxy.set_endpoint_status("up", "disabled")
        proxy.CONFIG_FILE, proxy._config, proxy._stats = self._saved
        self._tmp.cleanup()

    def _post(self, stream=False, model="up:m", text="hi", headers=None, path="/v1/chat/completions"):
        status, _, body = request(self.gw_port, "POST", path,
                                  body=json.dumps({"model": model, "stream": stream,
                                                   "messages": [{"role": "user", "content": text}]}),
                                  headers={"Content-Type": "application/json", **(headers or {})})
        proxy._stats.wait_idle()
        time.sleep(0.05)  # the gateway counts it just after the client has the reply
        return status, body

    def test_a_request_is_counted(self):
        self.assertEqual(self._post(headers={"X-Title": "Cline"})[0], 200)
        snap = proxy._stats.snapshot()
        session = snap["session"]
        self.assertEqual((session["requests"], session["errors"], session["prompt"], session["cached"], session["out"]),
                         (1, 0, 100, 75, 20))
        self.assertEqual(session["cost"], 0.25)
        self.assertEqual(snap["all_time"]["requests"], 1)
        self.assertEqual(snap["today"]["requests"], 1)
        self.assertEqual(snap["by_endpoint"]["up"]["requests"], 1)
        self.assertEqual(snap["by_model"]["m"]["prompt"], 100)
        self.assertEqual(snap["by_app"]["Cline"]["out"], 20)
        self.assertEqual(snap["per_minute"][-1]["requests"], 1)
        self.assertEqual(snap["active"], [])
        rec = snap["recent"][0]
        self.assertEqual((rec["status"], rec["app"], rec["client"]), (200, "Cline", "127.0.0.1"))
        self.assertFalse(rec["error"])

    def test_a_streamed_request_is_counted_and_timed(self):
        self.assertEqual(self._post(stream=True)[0], 200)
        rec = proxy._stats.snapshot()["recent"][0]
        self.assertEqual((rec["prompt"], rec["out"]), (40, 2))
        self.assertIsNotNone(rec["ttft_s"])
        self.assertTrue(rec["stream"])

    def test_replies_pass_through_byte_for_byte(self):
        for stream in (False, True):
            _, direct = request(self.up_port, "POST", "/v1/chat/completions",
                                body=json.dumps({"model": "m", "stream": stream}), headers={"Content-Type": "application/json"})[1:], None
            status, through = self._post(stream=stream)
            self.assertEqual(status, 200)
            self.assertEqual(through, MockChat.stream_body() if stream else _[1])

    def test_an_upstream_error_is_counted(self):
        MockChat.status = 429
        self.assertEqual(self._post()[0], 429)
        snap = proxy._stats.snapshot()
        self.assertEqual((snap["session"]["requests"], snap["session"]["errors"]), (1, 1))
        self.assertEqual(snap["recent"][0]["status"], 429)

    def test_an_unreachable_endpoint_is_counted_as_failed(self):
        proxy.set_config("endpoints", [{"name": "up", "type": "custom", "host": "localhost:1",
                                        "enabled": True, "api_key": ""}])
        self.assertEqual(self._post()[0], 502)
        snap = proxy._stats.snapshot()
        self.assertEqual(snap["session"]["errors"], 1)
        self.assertIsNone(snap["recent"][0]["status"])

    def test_other_requests_are_not_counted(self):
        request(self.gw_port, "GET", "/v1/models")
        request(self.gw_port, "GET", "/health")
        self.assertEqual(proxy._stats.snapshot()["session"]["requests"], 0)

    def test_embeddings_are_counted(self):
        self.assertEqual(self._post(path="/v1/embeddings")[0], 200)
        self.assertEqual(proxy._stats.snapshot()["session"]["requests"], 1)

    def test_a_client_pressing_stop_is_not_an_error(self):
        req = proxy._RequestStats({"name": "up"}, "/v1/chat/completions", "m", "1.2.3.4", "Cline", True)
        req.id, req.status, req.cancelled = 1, 200, True
        proxy._stats._record(req, None)
        snap = proxy._stats.snapshot()
        self.assertEqual(snap["session"]["errors"], 0)
        self.assertTrue(snap["recent"][0]["cancelled"])
        self.assertIn("stopped by the client", proxy._log_line(snap["recent"][0]))

    def test_no_chat_text_is_kept_anywhere(self):
        MockChat.reply_text = self.SECRET
        self._post(text=self.SECRET, headers={"X-Title": "Cline"})
        self._post(stream=True, text=self.SECRET)
        proxy._stats._dirty = True
        proxy._stats.save(force=True)
        self.assertNotIn(self.SECRET, json.dumps(proxy.stats_snapshot()))
        with open(proxy._stats._path(), encoding="utf-8") as f:
            self.assertNotIn(self.SECRET, f.read())
        lines = []
        while not proxy._log_queue.empty():
            lines.append(proxy._log_queue.get_nowait())
        self.assertTrue(any("→ up [200]" in line for line in lines))
        self.assertNotIn(self.SECRET, "\n".join(lines))

    def test_the_log_line_tells_the_story(self):
        while not proxy._log_queue.empty():
            proxy._log_queue.get_nowait()
        self._post()
        lines = []
        while not proxy._log_queue.empty():
            lines.append(proxy._log_queue.get_nowait())
        self.assertEqual(lines, ["[127.0.0.1] → up [200]  100 in (75% cached) · 20 out"])

    def test_totals_survive_a_restart_but_this_run_starts_over(self):
        self._post(headers={"X-Title": "Cline"})
        proxy._stats._dirty = True
        proxy._stats.save(force=True)
        proxy._stats = proxy.GatewayStats()
        snap = proxy._stats.snapshot()
        self.assertEqual(snap["session"]["requests"], 0)
        self.assertEqual((snap["all_time"]["requests"], snap["today"]["prompt"]), (1, 100))
        self.assertEqual(snap["all_time_by_app"]["Cline"]["requests"], 1)
        self._post()
        snap = proxy._stats.snapshot()
        self.assertEqual((snap["session"]["requests"], snap["all_time"]["requests"]), (1, 2))

    def test_a_damaged_totals_file_is_ignored(self):
        for junk in ("{not json", "[1, 2]", '{"requests": "many", "by_app": [1], "days": {"x": 5}}',
                     '{"by_app": {"Cline": {"requests": "many", "prompt": [1], "extra": 5}}}'):
            with open(proxy._stats._path(), "w", encoding="utf-8") as f:
                f.write(junk)
            proxy._stats = proxy.GatewayStats()
            self.assertEqual(proxy._stats.snapshot()["all_time"]["requests"], 0)
            self._post(headers={"X-Title": "Cline"})  # counting still works on top of it
            self.assertEqual(proxy._stats.snapshot()["all_time_by_app"]["Cline"]["requests"], 1)

    def test_a_failure_while_counting_leaves_nothing_active(self):
        ep = {"name": "up", "type": "strata"}
        req = proxy._stats.begin(ep, "POST", "/v1/chat/completions", {"model": "m"}, "1.2.3.4", {})
        req.status = 200
        proxy._stats._record = lambda *a: 1 / 0
        proxy._stats.end(req)
        self.assertTrue(proxy._stats.wait_idle())
        self.assertEqual(proxy._stats.snapshot()["active"], [])

    def test_saving_is_not_done_on_every_request(self):
        self._post()
        first = proxy._stats._last_save
        self._post()
        self.assertEqual(proxy._stats._last_save, first)

    def test_too_many_names_share_other(self):
        saved = proxy._STATS_MAX_KEYS
        proxy._STATS_MAX_KEYS = 2
        try:
            for app in ("a", "b", "c", "d"):
                self._post(headers={"X-Title": app})
        finally:
            proxy._STATS_MAX_KEYS = saved
        apps = proxy._stats.snapshot()["by_app"]
        self.assertEqual(apps["other"]["requests"], 2)
        self.assertEqual(sum(a["requests"] for a in apps.values()), 4)

    def test_old_days_are_dropped(self):
        proxy._stats._ensure_loaded()
        for i in range(40):
            proxy._stats.life["days"][f"2020-01-{i:02d}"] = proxy._new_agg()
        self._post()
        snap = proxy._stats.snapshot()
        self.assertEqual(len(proxy._stats.life["days"]), proxy._STATS_DAYS_KEPT)
        self.assertEqual(len(snap["per_day"]), 14)
        self.assertEqual(snap["per_day"][-1]["requests"], 1)

    def test_stats_endpoint(self):
        self._post()
        status, headers, body = request(self.gw_port, "GET", "/stats")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["gateway"]["session"]["requests"], 1)
        self.assertEqual(data["kv_sessions"], {})
        self.assertIn("engines", data)


class EngineProbeTests(unittest.TestCase):
    """Reading an engine's own numbers (Strata's /metrics)."""

    @classmethod
    def setUpClass(cls):
        cls.en_srv, cls.en_t, cls.en_port = start_server(MockEngine)

    @classmethod
    def tearDownClass(cls):
        cls.en_srv.shutdown(); cls.en_srv.server_close()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        MockEngine.reset(self._tmp.name)
        self.probe = proxy.EngineProbe()
        self.ep = {"name": "eng", "type": "custom", "kv_sessions": {"engine": f"127.0.0.1:{self.en_port}"}}

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_listed_fields_are_kept(self):
        view = proxy._engine_view({
            "engine": {"model": "m", "version": "1", "system_prompt": "SECRET"},
            "live": {"state": "idle", "prompt_text": "SECRET"},
            "requests": [{"prompt_tokens": 5, "messages": "SECRET"}, "junk"],
            "totals": {"requests": 1, "note": "SECRET"},
            "history": {"tok_s": [1], "last_prompt": "SECRET"},
            "conversation_cache": {"parked": 1, "last_event": "SECRET"},
            "hardware_static": {"gpu_name": "P40", "note": "SECRET"},
        })
        self.assertNotIn("SECRET", json.dumps(view))
        self.assertEqual((view["model"], view["gpu_name"]), ("m", "P40"))
        self.assertEqual(view["requests"][0]["prompt_tokens"], 5)
        self.assertEqual(view["history"]["tok_s"], [1])

    def test_odd_shapes_do_not_break_the_view(self):
        for m in ({}, {"engine": None, "live": 5, "requests": "x", "hardware": [], "history": None},):
            self.assertIn("live", proxy._engine_view(m))

    def test_reads_an_engine(self):
        MockEngine.metrics = {"engine": {"model": "m"}, "requests": [], "totals": {"prompt_tokens": 3000, "reused": 1000, "prompt_ms": 4000.0}}
        view = self.probe._poll_one(self.ep)
        self.assertTrue(view["ok"])
        self.assertEqual(self.probe.data["eng"]["model"], "m")
        self.assertEqual(self.probe.read_speed("eng"), 500.0)

    def test_a_short_read_is_no_measure_of_speed(self):
        MockEngine.metrics = {"engine": {}, "totals": {"prompt_tokens": 100, "reused": 0, "prompt_ms": 4000.0}}
        self.probe._poll_one(self.ep)
        self.assertIsNone(self.probe.read_speed("eng"))
        self.assertIsNone(self.probe.read_speed("nobody"))

    def test_an_engine_without_metrics_is_not_ok(self):
        view = self.probe._poll_one(self.ep)  # answers /metrics with something else
        self.assertFalse(view["ok"])

    def test_a_dead_engine_is_not_ok(self):
        view = self.probe._poll_one({"name": "x", "kv_sessions": {"engine": "127.0.0.1:1"}})
        self.assertFalse(view["ok"])
        self.assertIn("error", view)

    def test_an_endpoint_with_no_engine_is_skipped(self):
        self.assertIsNone(self.probe._poll_one({"name": "x", "type": "openrouter"}))

    def test_the_engines_key_is_sent(self):
        MockEngine.api_key = "k"
        MockEngine.metrics = {"engine": {}}
        self.assertFalse(self.probe._poll_one(self.ep)["ok"])
        self.ep["kv_sessions"]["engine_api_key"] = "k"
        self.assertTrue(self.probe._poll_one(self.ep)["ok"])

    def test_finds_the_row_for_a_request(self):
        now = time.time()
        MockEngine.metrics = {"engine": {}, "requests": [{"time": now + 1, "prompt_tokens": 2}, {"time": now - 100, "prompt_tokens": 1}]}
        self.assertEqual(self.probe.request_row(self.ep, "m", now)["prompt_tokens"], 2)
        self.assertIsNone(self.probe.request_row(self.ep, "m", now + 100))

    def test_polling_stops_when_nobody_looks(self):
        saved = (proxy._ENGINE_IDLE_STOP_S, proxy._ENGINE_POLL_S)
        proxy._ENGINE_IDLE_STOP_S, proxy._ENGINE_POLL_S = 0.1, 0.02
        try:
            self.probe.read()
            self.assertIsNotNone(self.probe.thread)
            deadline = time.time() + 5
            while time.time() < deadline and self.probe.thread is not None:
                time.sleep(0.02)
            self.assertIsNone(self.probe.thread)
        finally:
            proxy._ENGINE_IDLE_STOP_S, proxy._ENGINE_POLL_S = saved


def _full_snapshot(state="generating"):
    """A snapshot with everything in it, as the console would receive."""
    now = time.time()
    agg = {"requests": 91, "errors": 2, "prompt": 16_700_000, "cached": 16_400_000, "out": 70_800, "cost": 1.5,
           "saved_s": 56280.0, "last": now - 30}
    rec = {"time": now - 5, "endpoint": "strata", "model": "vendor/qwen3.8", "app": "Cline", "client": "127.0.0.1", "status": 200,
           "error": False, "cancelled": False, "stream": True, "duration_s": 12.0, "ttft_s": 1.2, "prompt": 148000, "cached": 147000,
           "out": 512, "out_estimated": False, "tok_s": 33.3, "read_tok_s": 290.0, "read": 1000, "cost": None, "bytes": 1000,
           "swap_s": 3.9, "swap": "restored", "expert_hit": 0.951, "drafts": 100, "drafts_accepted": 69, "saved_s": 500.0}
    return {
        "gateway": {
            "uptime_s": 3600, "active": [{"endpoint": "strata", "model": "m", "app": "Cline", "elapsed_s": 3.0}],
            "session": agg, "today": agg, "all_time": {**agg, "since": now - 86400}, "ttft_p50": 1.2, "ttft_p95": 22.0, "tok_s_p50": 33.1,
            "tok_s_best": 48.7, "read_tok_s_p50": 291.0, "by_endpoint": {"strata": {**agg, "ttft_p50": 1.2, "tok_s_p50": 33.0}},
            "by_model": {"vendor/qwen3.8": agg}, "by_app": {"Cline": agg}, "all_time_by_app": {"Cline": agg},
            "recent": [rec, {**rec, "status": 429, "error": True, "swap": None, "prompt": None, "cached": None, "out": None,
                             "tok_s": None, "expert_hit": None, "ttft_s": None},
                       {**rec, "cancelled": True, "swap": "new", "swap_s": None, "out_estimated": True}],
            "per_minute": [{"requests": i % 3, "in": 10, "out": 1} for i in range(60)],
            "per_day": [{"day": "2026-11-06", "requests": 5, "in": 1000, "out": 10}],
        },
        "engines": {"strata": {
            "ok": True, "at": now, "model": "qwen3.8", "gpu_name": "NVIDIA Tesla P40", "cpu_name": "Xeon",
            "info": {"version": "0.1.55", "max_context": 524288, "expert_slots": 7800, "expert_cache_mib": 15146},
            "live": {"state": state, "queued": 2, "phase": None, "prompt_read": 100000, "prompt_total": 400000, "generated": 1500,
                     "elapsed_s": 12.0, "tok_s": 34.2, "tok_s_mean": 33.0, "prefill_tok_s_mean": 283.0},
            "totals": {"prompt_tokens": 16_663_974, "reused": 16_403_888, "output_tokens": 70836, "prompt_ms": 891947.0,
                       "drafts_offered": 62879, "drafts_accepted": 43426},
            "parking": {"enabled": True, "budget_mib": 3584, "slots": 4, "parked": 3, "bytes": 1952003968, "evictions": 1},
            "hardware": {"gpu_util": 99, "gpu_mem_used": 23802281984, "gpu_mem_total": 25769803776, "gpu_temp": 91, "gpu_power": 142.0,
                         "gpu_power_limit": 250.0, "gpu_pcie_gen": 3, "gpu_pcie_width": 8, "gpu_pcie_rx_mb": 100.0, "gpu_pcie_tx_mb": 40.0,
                         "cpu": 27.7, "ram_used": 125966106624, "ram_total": 137344978944, "disk_read_mb": 100.0, "disk_write_mb": 87.0},
            "history": {"tok_s": [0, 10, 34], "gpu_util": [50, 99], "gpu_temp": [60, 91], "cpu": [10, 28]},
            "requests": [{"hit_rate": 0.951}, {"hit_rate": 0.9}, {"hit_rate": None}],
        }},
        "kv_sessions": {"strata": {
            "sessions": 3, "bytes": 5600669156, "held": {"id": "3d426048c7f4", "tokens": 148000, "saved": False, "app": "Cline"},
            "paused": 30, "last_event": "restored 3d42: 148,000 tokens in 3.0 s", "saves": 8, "restores": 5, "errors": 0, "continued": 86, "new": 3,
            "restored_tokens": 240000, "save_s": 25.0, "restore_s": 15.0, "written_bytes": 12_000_000_000,
            "chats": [{"id": "3d426048c7f4", "app": "Cline", "tokens": 148000, "bytes": 2_400_000_000, "last_used": now - 60, "restores": 5},
                      {"id": "aeb6ed2d4592", "app": "", "tokens": 33000, "bytes": 700_000_000, "last_used": now - 7200, "restores": 0}],
        }},
    }


def _as_text(renderable, width=120) -> str:
    import io
    from rich.console import Console
    out = io.StringIO()
    Console(file=out, width=width, force_terminal=False, color_system=None).print(renderable)
    return out.getvalue()


_PANELS = [p[2] for p in proxy._DASHBOARD_PANELS]


class ConsoleRenderTests(unittest.TestCase):
    """The console's panels, from a snapshot to text."""

    def setUp(self):
        self._saved = proxy._config
        proxy._config = dict(proxy._config)
        proxy.set_config("endpoints", [{"name": "strata", "type": "strata", "host": "h:1", "enabled": True}])

    def tearDown(self):
        proxy._config = self._saved

    def test_live_panel_generating(self):
        text = _as_text(proxy.render_live(_full_snapshot()))
        for expected in ("generating", "34.2 t/s", "3d4260", "148.0k/524.3k", "28%", "98.4%", "95%", "69%", "86 kept", "5 restored",
                         "3 chats, 5.2 GB", "15h 38m", "91°C", "VRAM", "91 req", "16.7M in", "2 errors", "failed"[:0]):
            self.assertIn(expected, text)

    def test_live_panel_counts_errors_in_good_english(self):
        snap = _full_snapshot()
        snap["gateway"]["today"] = {**snap["gateway"]["today"], "errors": 1}
        self.assertIn("1 error", _as_text(proxy.render_live(snap)))
        self.assertNotIn("1 errors", _as_text(proxy.render_live(snap)))

    def test_live_panel_reading_shows_progress_and_time_left(self):
        text = _as_text(proxy.render_live(_full_snapshot("reading")))
        self.assertIn("reading", text)
        self.assertIn("100.0k/400.0k", text)
        self.assertIn("283 t/s", text)
        self.assertIn("17m 40s left", text)

    def test_live_panel_says_when_the_model_is_not_loaded(self):
        self.assertIn("not loaded", _as_text(proxy.render_live(_full_snapshot("unloaded"))))
        self.assertIn("idle", _as_text(proxy.render_live(_full_snapshot("idle"))))

    def test_live_panel_shows_a_failed_last_request(self):
        snap = _full_snapshot()
        snap["gateway"]["recent"] = snap["gateway"]["recent"][1:]
        self.assertIn("failed 429", _as_text(proxy.render_live(snap)))

    def test_cache_panel(self):
        text = _as_text(proxy.panel_cache(_full_snapshot()))
        for expected in ("98.4%", "16.4M of 16.7M reused", "95.1% last", "92.5% avg", "7.8k in VRAM", "69.1%", "3 of 4 chats parked", "1 evicted",
                         "96.8% ready: 86 kept, 5 restored, 3 new", "11.2 GB written", "save 3.1 s", "restore 3.0 s", "15h 38m"):
            self.assertIn(expected, text)

    def test_hardware_panel(self):
        text = _as_text(proxy.panel_hardware(_full_snapshot()))
        for expected in ("Tesla P40", "91°C", "142 / 250 W", "22.2 / 24.0 GB", "Gen3 x8", "117.3 / 127.9 GB", "read 100"):
            self.assertIn(expected, text)
        self.assertNotIn("NVIDIA", text)

    def test_tables_show_what_they_should(self):
        snap = _full_snapshot()
        self.assertIn("Cline", _as_text(proxy.panel_chats(snap)))
        self.assertIn("●", _as_text(proxy.panel_chats(snap)))
        self.assertIn("5×", _as_text(proxy.panel_chats(snap)))
        self.assertIn("$1.50", _as_text(proxy.panel_totals(snap)))
        self.assertIn("All time since", _as_text(proxy.panel_totals(snap)))
        self.assertIn("qwen3.8", _as_text(proxy.panel_models(snap)))
        self.assertNotIn("vendor/", _as_text(proxy.panel_models(snap)))
        recent = _as_text(proxy.panel_recent(snap), 160)
        for expected in ("restored 3.9 s", "429", "stopped", "~512", "33.3 t/s", "95%"):
            self.assertIn(expected, recent)
        self.assertIn("strata", _as_text(proxy.panel_endpoints(snap)))

    def test_every_panel_survives_an_empty_gateway(self):
        snap = {"gateway": proxy.GatewayStats().snapshot(), "engines": {}, "kv_sessions": {}}
        for render in _PANELS + [proxy.render_live]:
            self.assertTrue(_as_text(render(snap)).strip(), render.__name__)
        self.assertIn("no requests yet", _as_text(proxy.panel_recent(snap)))
        self.assertIn("kv_sessions", _as_text(proxy.panel_chats(snap)))

    def test_every_panel_survives_a_silent_engine(self):
        snap = _full_snapshot()
        snap["engines"] = {"strata": {"ok": False, "at": 1.0, "error": "refused"}}
        for render in _PANELS + [proxy.render_live]:
            self.assertTrue(_as_text(render(snap)).strip(), render.__name__)
        self.assertIn("no engine metrics", _as_text(proxy.render_live(snap)))

    def test_every_panel_survives_an_engine_that_reports_little(self):
        snap = _full_snapshot()
        snap["engines"]["strata"] = {"ok": True, "at": 1.0, "model": "", "info": {}, "live": {}, "totals": {}, "parking": {},
                                     "hardware": {}, "history": {}, "requests": [], "gpu_name": "", "cpu_name": ""}
        for render in _PANELS + [proxy.render_live]:
            self.assertTrue(_as_text(render(snap)).strip(), render.__name__)

    def test_names_cannot_inject_styles(self):
        for name in ("[bold red]evil[/]", "[/oops]", "[link=http://x]y"):  # the second would raise if read as markup
            snap = _full_snapshot()
            snap["gateway"]["by_app"] = {name: snap["gateway"]["session"]}
            snap["gateway"]["by_model"] = {name: snap["gateway"]["session"]}
            snap["gateway"]["recent"][0]["app"] = name
            snap["gateway"]["recent"][0]["model"] = name
            snap["kv_sessions"]["strata"]["chats"][0]["app"] = name
            self.assertIn(name, _as_text(proxy.panel_apps(snap), 160))
            self.assertIn(name[:14], _as_text(proxy.panel_recent(snap), 200))
            self.assertIn(name[:16], _as_text(proxy.panel_chats(snap), 160))
            _as_text(proxy.panel_models(snap), 160)

    def test_only_models_lose_their_prefix(self):
        snap = _full_snapshot()
        snap["gateway"]["by_app"] = {"my/app": snap["gateway"]["session"]}
        snap["gateway"]["by_model"] = {"vendor/model/": snap["gateway"]["session"]}
        self.assertIn("my/app", _as_text(proxy.panel_apps(snap)))
        self.assertIn("model", _as_text(proxy.panel_models(snap)))
        self.assertNotIn("vendor", _as_text(proxy.panel_models(snap)))
        self.assertEqual(proxy._model_short("/"), "/")

    def test_no_secret_reaches_the_snapshot(self):
        proxy.set_config("endpoints", [{"name": "s", "type": "custom", "host": "h:1", "enabled": True,
                                        "api_key": "sk-secret-key", "kv_sessions": {"enabled": True, "engine_api_key": "other-secret"}}])
        text = json.dumps(proxy.stats_snapshot())
        self.assertNotIn("sk-secret-key", text)
        self.assertNotIn("other-secret", text)


class ConsoleScreenTests(unittest.TestCase):
    """The real screens, run headless."""

    def setUp(self):
        self._saved = (proxy._config, proxy._stats, proxy.get_models())
        proxy._config = dict(proxy._config)
        proxy._stats = proxy.GatewayStats()
        proxy.set_config("endpoints", [{"name": "st", "type": "custom", "host": "localhost:1", "enabled": True, "api_key": ""}])

    def tearDown(self):
        proxy._config, proxy._stats, models = self._saved
        proxy.set_models(models)

    def _run(self, app, script, size=(140, 40)):
        import asyncio

        async def run():
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await script(app, pilot)
                app.exit()
        asyncio.run(run())

    def _check_dashboard(self, app, pilot):
        async def go():
            self.assertTrue(app.screen.query_one("#live-stats"))
            await pilot.press("d")
            await pilot.pause()
            self.assertIsInstance(app.screen, proxy.StatsScreen)
            self.assertEqual(len(app.screen.query(".panel")), len(proxy._DASHBOARD_PANELS))
            await pilot.press("d")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, proxy.StatsScreen)
            await pilot.press("d")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, proxy.StatsScreen)
        return go()

    def test_proxy_app_has_live_stats_and_a_dashboard(self):
        self._run(proxy.ProxyTUI(), self._check_dashboard)

    def test_split_view_app_has_them_too(self):
        import queue
        import gateway_tui
        self._run(gateway_tui.GatewayApp(queue.Queue(), None), self._check_dashboard)

    def test_layouts_follow_the_window_width(self):
        seen = {}

        async def script(app, pilot):
            seen["main"] = app.screen.query_one("#overview").has_class("narrow")
            await pilot.press("d")
            await pilot.pause()
            seen["grid"] = app.screen.query_one("#stats-grid").has_class("narrow")

        self._run(proxy.ProxyTUI(), script, size=(150, 40))
        self.assertEqual(seen, {"main": False, "grid": False})
        self._run(proxy.ProxyTUI(), script, size=(90, 30))
        self.assertEqual(seen, {"main": True, "grid": True})

    def test_a_bad_number_cannot_take_the_console_down(self):
        real = proxy.stats_snapshot
        proxy.stats_snapshot = lambda: {"gateway": {"today": "garbage", "session": 5, "recent": [1]}, "engines": {"x": 5}, "kv_sessions": 7}

        async def script(app, pilot):
            await pilot.pause(0.1)
            await pilot.press("d")
            await pilot.pause(0.1)
            self.assertIsInstance(app.screen, proxy.StatsScreen)

        try:
            self._run(proxy.ProxyTUI(), script)
        finally:
            proxy.stats_snapshot = real

    def test_snapshot_is_quick_enough_for_the_screen_timer(self):
        for i in range(200):
            req = proxy._RequestStats({"name": "st"}, "/v1/chat/completions", f"m{i % 20}", "1.2.3.4", f"a{i % 20}", True)
            req.id, req.status = i, 200
            req.tap.usage = {"prompt": 100, "cached": 50, "out": 10}
            proxy._stats._record(req, None)
        t0 = time.time()
        for _ in range(20):
            proxy.render_live(proxy.stats_snapshot())
        self.assertLess((time.time() - t0) / 20, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)