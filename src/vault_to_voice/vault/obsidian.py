import datetime as dt
import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import frontmatter
import structlog

from vault_to_voice.vault.links import find_links, render_links, work_id_for
from vault_to_voice.vault.models import Note, NoteType
from vault_to_voice.vault.reader import _parse_date, _path_id

log = structlog.get_logger()

DEFAULT_CATEGORIES = {"games": "game", "books": "book", "movies": "movie"}
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
_NESTED_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")


@dataclass
class ReadStats:
    daily_files: int = 0
    entries: int = 0
    pages: int = 0
    unlinked_skipped: int = 0
    empty_skipped: int = 0


def split_entries(body: str, categories: dict[str, str]) -> list[tuple[str, str]]:
    """(note_type, raw entry text) per top-level bullet inside a category section."""
    entries: list[tuple[str, str]] = []
    current: str | None = None
    buf: list[str] | None = None
    blank_seen = False

    def flush() -> None:
        if buf is not None and current is not None:
            entries.append((current, "\n".join(buf).strip()))

    for line in body.splitlines():
        heading = _HEADING.match(line)
        if heading:
            flush()
            buf = None
            current = categories.get(heading.group(2).strip().lower())
            continue
        if current is None:
            continue
        bullet = _BULLET.match(line)
        if bullet:
            flush()
            buf, blank_seen = [bullet.group(1).strip()], False
        elif buf is None:
            continue
        elif not line.strip():
            blank_seen = True
        elif line[0] in " \t":
            buf.append(_NESTED_MARKER.sub("", line.strip()))
            blank_seen = False
        elif blank_seen:  # an unindented paragraph after a blank line ends the list
            flush()
            buf = None
        else:  # lazy continuation line
            buf.append(line.strip())
    flush()
    return entries


class ObsidianVaultReader:
    def __init__(
        self,
        root: Path,
        *,
        daily_folder: str = "Daily",
        date_format: str = "%d-%m-%Y",
        categories: dict[str, str] | None = None,
    ):
        self.root = Path(root)
        self.daily_folder = daily_folder.lower()
        self.date_format = date_format
        self.categories = categories or DEFAULT_CATEGORIES
        self.stats = ReadStats()

    def iter_notes(self) -> Iterator[Note]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Vault not found: {self.root}")
        self.stats = ReadStats()
        paths = [
            p
            for p in sorted(self.root.rglob("*.md"))
            if not any(part.startswith(".") for part in p.relative_to(self.root).parts)
        ]
        titles = {p.stem.lower(): p.stem for p in paths}
        seen: set[str] = set()
        for path in paths:
            rel = path.relative_to(self.root)
            day = self._daily_date(rel)
            notes = self._daily(path, rel, day, titles) if day else self._page(path, rel)
            for note in notes:
                if note.id in seen:  # e.g. two pages with the same name in different folders
                    note = note.model_copy(update={"id": _path_id(rel)})
                seen.add(note.id)
                yield note

    def _daily_date(self, rel: Path) -> dt.date | None:
        if len(rel.parts) < 2 or rel.parts[0].lower() != self.daily_folder:
            return None
        try:
            return dt.datetime.strptime(rel.stem, self.date_format).date()
        except ValueError:
            log.warning("daily_note_bad_name", path=rel.as_posix(), format=self.date_format)
            return None

    def _load(self, path: Path, rel: Path) -> tuple[str, frontmatter.Post] | None:
        raw = path.read_bytes().decode("utf-8", errors="replace")
        try:
            return raw, frontmatter.loads(raw)
        except Exception as exc:  # malformed YAML must not crash ingest
            log.warning("bad_frontmatter", path=rel.as_posix(), error=str(exc))
            return None

    def _daily(self, path: Path, rel: Path, day: dt.date, titles: dict[str, str]) -> list[Note]:
        loaded = self._load(path, rel)
        if loaded is None:
            return []
        self.stats.daily_files += 1
        notes: list[Note] = []
        per_work: dict[str, int] = {}
        for note_type, text in split_entries(loaded[1].content, self.categories):
            targets = find_links(text)
            if not targets:
                self.stats.unlinked_skipped += 1
                continue
            body = render_links(text).strip()
            if not re.search(r"\w", render_links(re.sub(r"\[\[[^\]]*\]\]", "", text))):
                self.stats.empty_skipped += 1
                continue
            work_id = work_id_for(targets[0])
            per_work[work_id] = per_work.get(work_id, 0) + 1
            suffix = "" if per_work[work_id] == 1 else f"-{per_work[work_id]}"
            notes.append(
                Note(
                    id=f"{work_id}@{day.isoformat()}{suffix}",
                    title=titles.get(targets[0].lower(), targets[0]),
                    type=note_type,
                    date=day,
                    body=body,
                    path=rel.as_posix(),
                    hash=hashlib.sha256(f"{note_type}\n{body}".encode()).hexdigest(),
                    work_id=work_id,
                    links=tuple(dict.fromkeys(work_id_for(t) for t in targets)),
                )
            )
        self.stats.entries += len(notes)
        return notes

    def _page(self, path: Path, rel: Path) -> list[Note]:
        loaded = self._load(path, rel)
        if loaded is None:
            return []
        raw, post = loaded
        content = post.content.strip()
        if not any(ln.strip() and not _HEADING.match(ln) for ln in content.splitlines()):
            self.stats.empty_skipped += 1
            return []
        meta = post.metadata
        note_type = str(meta.get("type", "")).lower()
        if note_type not in get_args(NoteType):
            folder = rel.parts[0].lower() if len(rel.parts) > 1 else ""
            note_type = self.categories.get(folder, "other")
        work_id = work_id_for(path.stem)
        self.stats.pages += 1
        return [
            Note(
                id=str(meta.get("id") or work_id),
                title=str(meta.get("title") or path.stem),
                type=note_type,
                date=_parse_date(meta.get("date")),
                body=render_links(content).strip(),
                path=rel.as_posix(),
                hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                work_id=work_id,
                links=tuple(dict.fromkeys(work_id_for(t) for t in find_links(content))),
            )
        ]
