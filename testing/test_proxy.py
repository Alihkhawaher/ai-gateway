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


if __name__ == "__main__":
    unittest.main(verbosity=2)