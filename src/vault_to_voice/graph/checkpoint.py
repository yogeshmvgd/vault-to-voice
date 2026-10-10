import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


def make_checkpointer(path: Path) -> SqliteSaver:
    """Short-term memory: one checkpoint history per thread (thread_id = idea_id)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def make_store(path: Path) -> SqliteStore:
    """Long-term memory shared by all threads (style profile, past interview answers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # SqliteStore manages its own transactions, so the connection must be in autocommit mode.
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    return store
