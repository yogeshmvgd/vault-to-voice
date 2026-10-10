import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from vault_to_voice.graph import state as s
from vault_to_voice.ideas.detector import Idea

# Every pydantic model that can sit in PostState. LangGraph stores them as msgpack "ext" types
# and, when reading a checkpoint back, only rebuilds types on this allowlist (any other type
# logs "Deserializing unregistered type" today and will be blocked by future versions).
# tests/test_checkpoint.py fails if a model reachable from PostState is missing here.
CHECKPOINT_TYPES: tuple[type, ...] = (
    Idea,
    s.SourceChunk,
    s.NoteContext,
    s.Question,
    s.QA,
    s.Section,
    s.Outline,
    s.Sentence,
    s.Paragraph,
    s.SectionDraft,
    s.Flag,
    s.CitationReport,
)


def checkpoint_serde() -> JsonPlusSerializer:
    """Checkpoint serializer that rebuilds exactly our state types (strict allowlist)."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def memory_checkpointer() -> InMemorySaver:
    """In-process checkpointer (eval runs) with the same type allowlist."""
    return InMemorySaver(serde=checkpoint_serde())


def make_checkpointer(path: Path) -> SqliteSaver:
    """Short-term memory: one checkpoint history per thread (thread_id = idea_id)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn, serde=checkpoint_serde())


def make_store(path: Path) -> SqliteStore:
    """Long-term memory shared by all threads (style profile, past interview answers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # SqliteStore manages its own transactions, so the connection must be in autocommit mode.
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    return store
