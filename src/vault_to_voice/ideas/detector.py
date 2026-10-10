import hashlib
from statistics import mean
from typing import Literal

from pydantic import BaseModel

from vault_to_voice.ideas.cluster import cluster, note_embeddings
from vault_to_voice.ideas.labeler import label_cluster
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LLMClient


class Idea(BaseModel):
    idea_id: str
    title: str
    angle: str
    theme: str
    note_ids: list[str]
    opinion_score: float
    status: Literal["new", "done"] = "new"


def make_idea_id(note_ids: list[str]) -> str:
    return "idea_" + hashlib.sha1(",".join(sorted(note_ids)).encode()).hexdigest()[:10]


def detect_ideas(
    manifest: Manifest,
    store: FaissStore,
    llm: LLMClient,
    *,
    min_notes: int,
    min_opinionated: int,
    opinion_threshold: float,
    seed: int,
) -> list[Idea]:
    note_ids, vectors = note_embeddings(manifest, store)
    if not note_ids:
        manifest.save_ideas([])
        return []
    labels = cluster(vectors, seed=seed)
    rows = {r.note_id: r for r in manifest.notes()}
    existing = {i.idea_id: i for i in manifest.list_ideas()}

    ideas: list[Idea] = []
    for label in sorted(set(labels.tolist())):
        members = sorted(n for n, lab in zip(note_ids, labels, strict=True) if lab == label)
        scores = [rows[n].opinion_score or 0.0 for n in members]
        opinionated = sum(score >= opinion_threshold for score in scores)
        if len(members) < min_notes or opinionated < min_opinionated:
            continue
        idea_id = make_idea_id(members)
        if idea_id in existing:
            ideas.append(existing[idea_id])
            continue
        lab = label_cluster(llm, [rows[n] for n in members])
        ideas.append(
            Idea(
                idea_id=idea_id,
                title=lab.title,
                angle=lab.angle,
                theme=lab.theme,
                note_ids=members,
                opinion_score=round(mean(scores), 3),
            )
        )
    manifest.save_ideas(ideas)
    return ideas
