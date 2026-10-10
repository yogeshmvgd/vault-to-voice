import datetime as dt
import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class RenderedPost(BaseModel):
    slug: str
    date: dt.date
    markdown: str
    sources: dict


class Publisher(Protocol):
    def publish(self, post: RenderedPost) -> Path: ...


class OutboxPublisher:
    """Writes posts to a local folder. Later publishers (dev.to, Hashnode) share the protocol."""

    def __init__(self, outbox_dir: Path):
        self.outbox_dir = Path(outbox_dir)

    def publish(self, post: RenderedPost) -> Path:
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{post.date.isoformat()}-{post.slug}"
        path = self.outbox_dir / f"{stem}.md"
        n = 2
        while path.exists():
            path = self.outbox_dir / f"{stem}-{n}.md"
            n += 1
        path.write_text(post.markdown, encoding="utf-8")
        path.with_suffix(".sources.json").write_text(
            json.dumps(post.sources, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path
