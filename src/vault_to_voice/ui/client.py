from collections.abc import Iterator
from typing import Any

import httpx

from vault_to_voice.api.events import parse_sse

Event = tuple[str, Any]


class ApiError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class ApiClient:
    """Thin HTTP client for the vault-to-voice API. `http` is an httpx-style client
    (httpx.Client in the app, FastAPI's TestClient in tests)."""

    def __init__(self, http):
        self.http = http

    @classmethod
    def from_url(cls, base_url: str, *, timeout: float = 1800.0) -> "ApiClient":
        # Generous read timeout: setup actions on a local (Ollama) model can take many minutes.
        return cls(httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=10.0)))

    @staticmethod
    def _check(response) -> None:
        if response.status_code < 400:
            return
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(response.status_code, str(detail))

    def _json(self, method: str, url: str, **kwargs) -> Any:
        response = self.http.request(method, url, **kwargs)
        self._check(response)
        return response.json()

    def _events(self, method: str, url: str, **kwargs) -> Iterator[Event]:
        with self.http.stream(method, url, **kwargs) as response:
            if response.status_code >= 400:
                response.read()
                self._check(response)
            yield from parse_sse(response.iter_lines())

    def health(self) -> dict:
        return self._json("GET", "/health")

    def ideas(self) -> list[dict]:
        return self._json("GET", "/ideas")

    def threads(self) -> list[dict]:
        return self._json("GET", "/threads")

    def thread(self, thread_id: str) -> dict:
        return self._json("GET", f"/threads/{thread_id}")

    def post(self, thread_id: str) -> dict:
        return self._json("GET", f"/threads/{thread_id}/post")

    def start(self, idea_id: str) -> Iterator[Event]:
        return self._events("POST", "/threads", json={"idea_id": idea_id})

    def resume(self, thread_id: str, payload: str | dict) -> Iterator[Event]:
        return self._events("POST", f"/threads/{thread_id}/resume", json={"payload": payload})

    def retry(self, thread_id: str) -> Iterator[Event]:
        return self._events("GET", f"/threads/{thread_id}/stream")

    # ---- setup actions -------------------------------------------------------------------
    def config(self) -> dict:
        return self._json("GET", "/config")

    def ingest(self, *, no_score: bool = False) -> dict:
        return self._json("POST", "/ingest", json={"no_score": no_score})

    def build_style(self) -> dict:
        return self._json("POST", "/style")

    def detect_ideas(self) -> list[dict]:
        return self._json("POST", "/ideas/detect")
