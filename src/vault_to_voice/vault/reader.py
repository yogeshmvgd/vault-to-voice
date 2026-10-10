import datetime as dt
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, get_args

import frontmatter
import structlog

from vault_to_voice.vault.models import Note, NoteType

log = structlog.get_logger()


class VaultReader(Protocol):
    def iter_notes(self) -> Iterator[Note]: ...


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _path_id(rel: Path) -> str:
    """Slug of the vault-relative path; hash fallback when the slug is empty (non-ASCII)."""
    posix = rel.with_suffix("").as_posix()
    return slugify(posix) or "note-" + hashlib.sha1(posix.encode()).hexdigest()[:10]


def _parse_date(value: object) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


class FilesystemVaultReader:
    def __init__(self, root: Path):
        self.root = Path(root)

    def iter_notes(self) -> Iterator[Note]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Vault not found: {self.root}")
        seen: dict[str, str] = {}
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            note = self._parse(path, rel)
            if note is None:
                continue
            if note.id in seen:
                new_id = _path_id(rel)
                n = 2
                while new_id in seen:
                    new_id = f"{_path_id(rel)}-{n}"
                    n += 1
                log.warning(
                    "duplicate_note_id",
                    id=note.id,
                    first=seen[note.id],
                    second=rel.as_posix(),
                    using=new_id,
                )
                note = note.model_copy(update={"id": new_id})
            seen[note.id] = rel.as_posix()
            yield note

    def _parse(self, path: Path, rel: Path) -> Note | None:
        raw = path.read_bytes().decode("utf-8", errors="replace")
        try:
            post = frontmatter.loads(raw)
        except Exception as exc:  # malformed YAML in a user's vault must not crash ingest
            log.warning("bad_frontmatter", path=rel.as_posix(), error=str(exc))
            return None
        body = post.content.strip()
        if not body:
            return None
        meta = post.metadata
        note_type = str(meta.get("type", "other")).lower()
        if note_type not in get_args(NoteType):
            note_type = "other"
        return Note(
            id=str(meta.get("id") or _path_id(rel)),
            title=str(meta.get("title") or path.stem),
            type=note_type,
            date=_parse_date(meta.get("date")),
            body=body,
            path=rel.as_posix(),
            hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )
