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
    lock = threading.Lock()

    @classmethod
    def reset(cls, slot_dir):
        cls.held, cls.saved, cls.calls = "", {}, []
        cls.slot_dir, cls.api_key, cls.fail = slot_dir, "", {}
        cls.chat_status, cls.chat_delay, cls.file_bytes = 200, 0.0, 1024
        cls.active = cls.max_active = 0

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
        proxy.CONFIG_FILE, proxy._config, proxy._KV_IDLE_POLL_S = self._saved
        os.environ.pop("STRATA_INSTANCES_DIR", None)
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

    def _chat(self, msgs, stream=False, model="eng:m"):
        status, _, body = request(self.gw_port, "POST", "/v1/chat/completions",
                                  body=json.dumps({"model": model, "messages": msgs, "stream": stream}),
                                  headers={"Content-Type": "application/json"})
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
        self.assertEqual(proxy.kv_status_all(), {})

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


if __name__ == "__main__":
    unittest.main(verbosity=2)