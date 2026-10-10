from pydantic import BaseModel

from vault_to_voice.guardrails import DATA_INSTRUCTION, scrub_pii, wrap_note
from vault_to_voice.index.manifest import NoteRow
from vault_to_voice.llm import LLMClient

LABEL_PROMPT = (
    "These notes were grouped by theme. Propose one blog post idea that builds on the "
    "author's opinions.\n"
    "title: catchy, at most 10 words. angle: one sentence stating the argument the post makes. "
    "theme: 2-4 words.\n\n"
)


WORK_PROMPT = (
    "These notes are all about one work (a single game, book or movie), written over time. "
    "Propose one blog post about the author's experience with it, built on their opinions.\n"
    "title: catchy, at most 10 words. angle: one sentence stating the argument the post makes. "
    "theme: 2-4 words.\n\n"
)


class ClusterLabel(BaseModel):
    title: str
    angle: str
    theme: str


def label_cluster(llm: LLMClient, rows: list[NoteRow], *, kind: str = "theme") -> ClusterLabel:
    parts = [
        wrap_note(
            r.note_id,
            scrub_pii(f"{r.title} ({r.type})\nOpinions: " + "; ".join(r.highlights or ["(none)"])),
        )
        for r in rows
    ]
    prompt = (WORK_PROMPT if kind == "work" else LABEL_PROMPT) + "\n".join(parts)
    return llm.structured(prompt, ClusterLabel, system=DATA_INSTRUCTION)
