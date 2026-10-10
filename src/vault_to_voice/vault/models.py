import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict

NoteType = Literal["game", "book", "movie", "other"]


class Note(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    type: NoteType = "other"
    date: dt.date | None = None
    body: str
    path: str
    hash: str
    work_id: str | None = None  # the game/book/movie this note is about (Obsidian format)
    links: tuple[str, ...] = ()  # work_ids of every work linked in the note
