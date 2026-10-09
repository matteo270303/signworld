"""Fixture: a local HTTP server with Range support."""

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@dataclass
class FileServer:
    """What the test server serves, and every (path, Range header) it was asked for."""

    base_url: str = ""
    files: dict[str, bytes] = field(default_factory=dict)
    truncate_once: set[str] = field(default_factory=set)
    requests: list[tuple[str, str | None]] = field(default_factory=list)

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"


def _handler(state: FileServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            byte_range = self.headers.get("Range")
            state.requests.append((self.path, byte_range))
            body = state.files.get(self.path)
            if body is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            start = int(byte_range.removeprefix("bytes=").split("-")[0]) if byte_range else 0
            if byte_range and start >= len(body):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.end_headers()
                return
            payload = body[start:]
            self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
            if byte_range:
                self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.path in state.truncate_once:
                state.truncate_once.discard(self.path)
                self.wfile.write(payload[: len(payload) // 2])
                self.close_connection = True
                return
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


@pytest.fixture
def file_server() -> Iterator[FileServer]:
    state = FileServer()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()
