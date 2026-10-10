import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field

from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from vault_to_voice.config import Settings
from vault_to_voice.index.embedder import Embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LLMClient
from vault_to_voice.publish import Publisher

# Retry LLM-calling nodes on transient errors (exponential backoff, LangGraph defaults).
LLM_RETRY = RetryPolicy(max_attempts=3)


@dataclass
class GraphDeps:
    """Everything the post graph's nodes need, injected once at build time."""

    llm: LLMClient
    manifest: Manifest
    index: FaissStore
    embedder: Embedder
    settings: Settings
    store: BaseStore
    publisher: Publisher | None = None
    today: Callable[[], dt.date] = field(default=dt.date.today)
