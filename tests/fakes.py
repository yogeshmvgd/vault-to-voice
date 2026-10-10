from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

Response = BaseModel | list[BaseModel] | Callable[[str], BaseModel]


class FakeLLMClient:
    """Scripted LLMClient keyed by output schema."""

    def __init__(self, responses: dict[type, Response]):
        self.responses: dict[type, Any] = dict(responses)
        self.calls: list[tuple[type, str]] = []

    def structured(self, prompt: str, schema: type, *, system: str | None = None):
        self.calls.append((schema, prompt))
        if schema not in self.responses:
            raise KeyError(f"FakeLLMClient has no response for {schema.__name__}")
        r = self.responses[schema]
        if isinstance(r, list):
            return r.pop(0)
        if callable(r) and not isinstance(r, BaseModel):
            return r(prompt)
        return r
