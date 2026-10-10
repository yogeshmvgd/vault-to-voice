from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from vault_to_voice.config import Settings
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.state import (
    CoverageCheck,
    Outline,
    Paragraph,
    Question,
    Section,
    SectionText,
    Sentence,
)
from vault_to_voice.ideas.detector import Idea, make_idea_id
from vault_to_voice.index.embedder import HashEmbedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.ingest import ingest
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.publish import OutboxPublisher
from vault_to_voice.vault.reader import FilesystemVaultReader

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


NOTES = {
    "n0": "Boss fights feel fair when every attack has a clear tell.\n\n"
    "I died twenty times and learned the pattern.",
    "n1": "Fair difficulty means every death teaches you something.\n\n"
    "The tide mechanic shows the danger before it arrives.",
    "n2": "Cheap bosses spam attacks with no wind up.\n\n"
    "I quit the game when the camera lost lock on.",
    "n3": "Hard games respect the player when the rules are honest.\n\n"
    "Patience beats button mashing in the late game.",
    "x0": "Baking bread on a rainy afternoon is calming.\n\nThe oven timer never rushes me.",
}


def make_deps(tmp_path: Path, llm, *, dim: int = 256, **settings_overrides) -> GraphDeps:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        outbox_dir=tmp_path / "outbox",
        embedding_provider="hash",
        **settings_overrides,
    )
    return GraphDeps(
        llm=llm,
        manifest=Manifest(":memory:"),
        index=FaissStore(tmp_path / "f.index", dim),
        embedder=HashEmbedder(dim),
        settings=settings,
        store=InMemoryStore(),
        publisher=OutboxPublisher(tmp_path / "outbox"),
    )


def seed_idea(deps: GraphDeps, vault_dir: Path, note_ids=("n0", "n1", "n2", "n3")) -> Idea:
    for nid, body in NOTES.items():
        (vault_dir / f"{nid}.md").write_text(
            f"---\nid: {nid}\ndate: 2026-01-01\ntype: game\ntitle: Note {nid}\n---\n{body}\n",
            encoding="utf-8",
        )
    ingest(FilesystemVaultReader(vault_dir), deps.manifest, deps.index, deps.embedder)
    ids = sorted(note_ids)
    idea = Idea(
        idea_id=make_idea_id(ids),
        title="Fair difficulty",
        angle="Hard is fine if it is honest.",
        theme="difficulty",
        note_ids=ids,
        opinion_score=0.8,
    )
    deps.manifest.save_ideas([idea])
    return idea


GOOD_SENTENCE = NOTES["n0"].split("\n\n")[0]


def good_section(_prompt) -> SectionText:
    return SectionText(
        paragraphs=[
            Paragraph(
                sentences=[
                    Sentence(text=GOOD_SENTENCE, sources=["n0"]),
                    Sentence(text="That is the deal.", kind="connective"),
                ]
            )
        ]
    )


def graph_responses(section_fn=good_section) -> dict:
    return {
        Question: lambda _: Question(question="What made it fair?", grounding_note_ids=["n0"]),
        CoverageCheck: lambda _: CoverageCheck(enough=True),
        Outline: lambda _: Outline(
            title="Fair Bosses",
            sections=[
                Section(
                    id="a", heading="Tells", point="Tells make it fair.", planned_sources=["n0"]
                ),
                Section(
                    id="b", heading="Learning", point="Death teaches.", planned_sources=["qa:1"]
                ),
            ],
        ),
        SectionText: section_fn,
    }
