"""The play app's HTTP server: the table page and a small JSON API, stdlib only.

* ``GET /``: the table (``static/index.html``); ``GET /static/<file>``: its
  script and style.
* ``GET /api/state?since=<version>``: the session's public view
  (:meth:`PlaySession.snapshot`). With ``since``, a long poll that answers once
  there is a newer version, or after 20 s.
* ``POST /api/action``: ``{"kind": "fold" | "check" | "call" | "raise" |
  "allin", "amount": <raise to>}``.
* ``POST /api/next``: deal the next hand. ``POST /api/stop``: end the session.

A turn-based game needs no websocket: the long poll delivers every change at
once. The server listens on 127.0.0.1 by default and answers only requests
whose ``Host`` header names this machine, so another web page cannot drive it
through DNS rebinding; POST bodies must be JSON, which a plain cross-site form
cannot send.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

STATIC = Path(__file__).resolve().with_name("static")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _finite(obj: Any) -> Any:
    """JSON has no NaN: replace non-finite floats with ``None``."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def make_handler(session: Any, allowed_hosts: frozenset[str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "pokerbot-play"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # the long poll would flood the console

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").strip().lower()
            if host.startswith("["):  # [::1]:8765
                host = host[1:].split("]", 1)[0]
            else:
                host = host.rsplit(":", 1)[0]
            return host in allowed_hosts

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                pass  # the page went away (a reload drops its pending long poll)

        def _json(self, code: int, obj: Any) -> None:
            body = json.dumps(_finite(obj), allow_nan=False).encode("utf-8")
            self._send(code, body, "application/json")

        def _file(self, name: str) -> None:
            root = STATIC.resolve()
            path = (root / name).resolve()
            if root not in path.parents or not path.is_file():
                self._json(404, {"error": "not found"})
                return
            ctype = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
            self._send(200, path.read_bytes(), ctype)

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                self._json(403, {"error": "forbidden host"})
                return
            url = urlsplit(self.path)
            if url.path in ("/", "/index.html"):
                self._file("index.html")
            elif url.path.startswith("/static/"):
                self._file(url.path[len("/static/") :])
            elif url.path == "/api/state":
                query = parse_qs(url.query)
                try:
                    since = int(query["since"][0]) if "since" in query else None
                except ValueError:
                    since = None
                self._json(200, session.snapshot(since=since, timeout=20.0))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                self._json(403, {"error": "forbidden host"})
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._json(415, {"ok": False, "error": "send JSON"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            except (ValueError, json.JSONDecodeError):
                body = None
            if not isinstance(body, dict):
                self._json(400, {"ok": False, "error": "send a JSON object"})
                return
            path = urlsplit(self.path).path
            if path == "/api/action":
                ok, err = session.submit_action(body.get("kind"), body.get("amount"))
            elif path == "/api/next":
                ok, err = session.request_next()
            elif path == "/api/stop":
                session.stop()
                ok, err = True, None
            else:
                self._json(404, {"ok": False, "error": "not found"})
                return
            self._json(200 if ok else 409, {"ok": ok, "error": err})

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64  # a page load opens several connections next to the long polls


class PlayServer:
    """``ThreadingHTTPServer`` around a :class:`~.session.PlaySession`. Port 0
    picks a free port."""

    def __init__(self, session: Any, host: str = "127.0.0.1", port: int = 8765) -> None:
        allowed = LOCAL_HOSTS if host in LOCAL_HOSTS else LOCAL_HOSTS | {host.lower()}
        self.httpd = _Server((host, port), make_handler(session, frozenset(allowed)))
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        host = str(self.httpd.server_address[0])
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        return f"http://{host}:{self.port}/"

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.25}, daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
