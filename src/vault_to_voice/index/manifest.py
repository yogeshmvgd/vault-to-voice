import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from vault_to_voice.vault.models import Note

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
  note_id TEXT PRIMARY KEY, path TEXT NOT NULL, title TEXT NOT NULL, type TEXT NOT NULL,
  date TEXT, hash TEXT NOT NULL, opinion_score REAL, highlights TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, note_id TEXT NOT NULL,
  para_idx INTEGER NOT NULL, text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_note ON chunks(note_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ideas (
  idea_id TEXT PRIMARY KEY, data TEXT NOT NULL, status TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class NoteRow:
    note_id: str
    path: str
    title: str
    type: str
    date: str | None
    hash: str
    opinion_score: float | None
    highlights: list[str]


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    note_id: str
    para_idx: int
    text: str


@dataclass
class Diff:
    new: list[Note] = field(default_factory=list)
    changed: list[Note] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[Note] = field(default_factory=list)


class Manifest:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def diff(self, notes: Iterable[Note]) -> Diff:
        known = dict(self.conn.execute("SELECT note_id, hash FROM notes"))
        d = Diff()
        seen = set()
        for n in notes:
            seen.add(n.id)
            h = known.get(n.id)
            if h is None:
                d.new.append(n)
            elif h != n.hash:
                d.changed.append(n)
            else:
                d.unchanged.append(n)
        d.deleted = sorted(set(known) - seen)
        return d

    def upsert_note(self, note: Note) -> None:
        self.conn.execute(
            """INSERT INTO notes (note_id, path, title, type, date, hash, opinion_score, highlights)
               VALUES (?, ?, ?, ?, ?, ?, NULL, '[]')
               ON CONFLICT(note_id) DO UPDATE SET path=excluded.path, title=excluded.title,
                 type=excluded.type, date=excluded.date, hash=excluded.hash,
                 opinion_score=NULL, highlights='[]'""",
            (
                note.id,
                note.path,
                note.title,
                note.type,
                note.date.isoformat() if note.date else None,
                note.hash,
            ),
        )
        self.conn.commit()

    def delete_note(self, note_id: str) -> list[int]:
        ids = self.chunk_ids(note_id)
        self.conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
        self.conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
        self.conn.commit()
        return ids

    def add_chunks(self, note_id: str, texts: list[str]) -> list[int]:
        ids = []
        for i, text in enumerate(texts):
            cur = self.conn.execute(
                "INSERT INTO chunks (note_id, para_idx, text) VALUES (?, ?, ?)", (note_id, i, text)
            )
            ids.append(int(cur.lastrowid))
        self.conn.commit()
        return ids

    def chunk_ids(self, note_id: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE note_id = ? ORDER BY para_idx", (note_id,)
        )
        return [r[0] for r in rows]

    def get_chunks(self, chunk_ids: list[int]) -> list[Chunk]:
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, note_id, para_idx, text FROM chunks WHERE chunk_id IN ({marks})",
            chunk_ids,
        )
        by_id = {r[0]: Chunk(*r) for r in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]

    def chunks_for_notes(self, note_ids: list[str]) -> list[Chunk]:
        if not note_ids:
            return []
        marks = ",".join("?" * len(note_ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, note_id, para_idx, text FROM chunks WHERE note_id IN ({marks}) "
            "ORDER BY note_id, para_idx",
            note_ids,
        )
        return [Chunk(*r) for r in rows]

    def chunk_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def _row(self, r: tuple) -> NoteRow:
        return NoteRow(*r[:7], highlights=json.loads(r[7]))

    def notes(self) -> list[NoteRow]:
        rows = self.conn.execute(
            "SELECT note_id, path, title, type, date, hash, opinion_score, highlights "
            "FROM notes ORDER BY note_id"
        )
        return [self._row(r) for r in rows]

    def get_note(self, note_id: str) -> NoteRow | None:
        r = self.conn.execute(
            "SELECT note_id, path, title, type, date, hash, opinion_score, highlights "
            "FROM notes WHERE note_id = ?",
            (note_id,),
        ).fetchone()
        return self._row(r) if r else None

    def set_opinion(self, note_id: str, score: float, highlights: list[str]) -> None:
        self.conn.execute(
            "UPDATE notes SET opinion_score = ?, highlights = ? WHERE note_id = ?",
            (score, json.dumps(highlights), note_id),
        )
        self.conn.commit()

    def notes_needing_opinion(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT note_id FROM notes WHERE opinion_score IS NULL ORDER BY note_id"
        )
        return [r[0] for r in rows]

    def get_meta(self, key: str) -> str | None:
        r = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def clear_index(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM notes")
        self.conn.commit()

    def save_ideas(self, ideas: list) -> None:
        keep = {i.idea_id for i in ideas}
        for (idea_id,) in self.conn.execute(
            "SELECT idea_id FROM ideas WHERE status = 'new'"
        ).fetchall():
            if idea_id not in keep:
                self.conn.execute("DELETE FROM ideas WHERE idea_id = ?", (idea_id,))
        for idea in ideas:
            self.conn.execute(
                "INSERT INTO ideas (idea_id, data, status) VALUES (?, ?, ?) "
                "ON CONFLICT(idea_id) DO UPDATE SET data = excluded.data, status = excluded.status",
                (idea.idea_id, idea.model_dump_json(), idea.status),
            )
        self.conn.commit()

    def list_ideas(self) -> list:
        from vault_to_voice.ideas.detector import Idea

        rows = self.conn.execute("SELECT data FROM ideas ORDER BY idea_id")
        return [Idea.model_validate_json(r[0]) for r in rows]
