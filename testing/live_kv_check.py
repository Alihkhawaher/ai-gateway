"""
Live check of the KV session swap against a real engine.

Not part of the unit tests: it needs a loaded model and takes minutes. It starts
its own private gateway (it does not touch config.json or a running gateway) and
talks to the engine's own server, which must run with slot saving on:

  control\\python\\python.exe control\\serve\\server.py --engine strata
      --config <run config> --port 8081 --slot-save-path <folder>

Then:
  python testing/live_kv_check.py 127.0.0.1:8081
  python testing/live_kv_check.py 127.0.0.1:18500 --api-key KEY --model NAME

Every prompt is made-up filler text, so no real chat is read or saved. It leaves
gw-*.bin files in the engine's slot folder; delete them when you are done.

A switch back to a chat must come back with almost all of it reused.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proxy


def filler(seed: int, sentences: int) -> str:
    rng = random.Random(seed)
    return "".join(
        f"Record {i}: unit {rng.randint(100, 9999)} logged value {rng.randint(10, 99999)} on day "
        f"{rng.randint(1, 365)}; reviewer {rng.randint(1, 500)} signed entry {rng.randint(1000, 99999)}.\n"
        for i in range(1, sentences + 1)
    )


def chat(port, model, messages, stream=False):
    """-> (seconds, prompt tokens, reused tokens) for one request through the gateway."""
    body = {"model": model, "messages": messages, "max_tokens": 8, "temperature": 0, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    conn = HTTPConnection("127.0.0.1", port, timeout=1800)
    t0 = time.time()
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
    finally:
        conn.close()
    took = time.time() - t0
    if resp.status != 200:
        raise SystemExit(f"HTTP {resp.status}: {raw[:300]}")
    usage = None
    if stream:
        for line in raw.splitlines():
            if line.startswith("data: ") and '"usage"' in line:
                try:
                    usage = json.loads(line[6:]).get("usage") or usage
                except ValueError:
                    pass
    else:
        usage = json.loads(raw).get("usage")
    usage = usage or {}
    return took, usage.get("prompt_tokens"), (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


def drain_log():
    lines = []
    while not proxy._log_queue.empty():
        line = proxy._log_queue.get_nowait()
        if line.startswith("[kv"):
            lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine", help="host:port of the engine's own server")
    ap.add_argument("--api-key", default="", help="the engine's API key, if it has one")
    ap.add_argument("--model", default="local")
    ap.add_argument("--sentences", type=int, default=800, help="size of each chat (800 is about 33k tokens)")
    args = ap.parse_args()

    tmp = tempfile.TemporaryDirectory()
    proxy.CONFIG_FILE = os.path.join(tmp.name, "config.json")
    proxy._config = dict(proxy._config)
    proxy.set_config("force_default_model", False)
    proxy.set_config("endpoints", [{
        "name": "live", "type": "custom", "host": args.engine, "enabled": True, "api_key": args.api_key,
        "timeout": 1800,
        "kv_sessions": {"enabled": True, "engine": args.engine, "engine_api_key": args.api_key,
                        "min_tokens": 1000, "idle_save_s": 0},
    }])
    proxy.set_endpoint_status("live", "online")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), proxy.ProxyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    model = f"live:{args.model}"

    a = [{"role": "user", "content": filler(11, args.sentences) + "\nReply with the single word OK."}]
    b = [{"role": "user", "content": filler(22, args.sentences) + "\nReply with the single word OK."}]
    follow = [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "Say OK again."}]

    steps = [
        ("A: first read", a, False, False),
        ("B: first read, A saved", b, False, False),
        ("A again: B saved, A restored", a + follow, True, False),
        ("B again, streamed: A saved, B restored", b + follow, True, True),
        ("B continues: no swap", b + follow + follow, True, False),
    ]
    failed = 0
    for label, msgs, expect_reuse, stream in steps:
        took, prompt, reused = chat(port, model, msgs, stream)
        manager = proxy._kv_managers.get("live")
        if manager:
            with manager.lock:  # its bookkeeping ends just after the reply
                pass
        print(f"{label}: {took:.1f} s, prompt {prompt}, reused {reused}")
        for line in drain_log():
            print("   ", line)
        if expect_reuse and prompt is not None and (reused is None or reused < 0.95 * prompt):
            failed += 1
            print("    FAILED: expected almost the whole prompt to be reused")

    print("status:", json.dumps(proxy.kv_status_all()["live"]))
    srv.shutdown()
    tmp.cleanup()
    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
