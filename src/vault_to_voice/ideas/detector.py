import hashlib
from collections import defaultdict
from statistics import mean
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from vault_to_voice.ideas.cluster import cluster, note_embeddings
from vault_to_voice.ideas.labeler import label_cluster
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LLMClient


class IndexOutOfSyncError(RuntimeError):
    pass


class Idea(BaseModel):
    idea_id: str
    title: str
    angle: str
    theme: str
    note_ids: list[str]
    opinion_score: float
    status: Literal["new", "done"] = "new"
    kind: Literal["work", "theme"] = "theme"
    work_ids: list[str] = Field(default_factory=list)


def make_idea_id(ids: list[str]) -> str:
    return "idea_" + hashlib.sha1(",".join(sorted(ids)).encode()).hexdigest()[:10]


def work_idea_id(work_id: str) -> str:
    return f"idea_work_{work_id}"


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
    if len(store) != manifest.chunk_count() or manifest.get_meta("dirty") == "1":
        raise IndexOutOfSyncError(
            "Vector index is missing or out of sync with the manifest "
            "(e.g. EMBEDDING_PROVIDER changed). Run `v2v ingest` first."
        )
    note_ids, vectors = note_embeddings(manifest, store)
    if not note_ids:
        manifest.save_ideas([])
        return []
    rows = {r.note_id: r for r in manifest.notes()}
    existing = {i.idea_id: i for i in manifest.list_ideas()}
    vector_of = dict(zip(note_ids, vectors, strict=True))

    groups: dict[str, list[str]] = defaultdict(list)
    for nid in note_ids:
        groups[rows[nid].work_id or nid].append(nid)

    def qualifies(members: list[str]) -> bool:
        scores = [rows[n].opinion_score or 0.0 for n in members]
        return len(members) >= min_notes and (
            sum(s >= opinion_threshold for s in scores) >= min_opinionated
        )

    def make(idea_id: str, kind: str, works: list[str], members: list[str]) -> Idea:
        score = round(mean(rows[n].opinion_score or 0.0 for n in members), 3)
        if idea_id in existing:  # keep title/angle/status; refresh membership
            return existing[idea_id].model_copy(
                update={"note_ids": members, "work_ids": works, "opinion_score": score}
            )
        lab = label_cluster(llm, [rows[n] for n in members], kind=kind)
        return Idea(
            idea_id=idea_id,
            title=lab.title,
            angle=lab.angle,
            theme=lab.theme,
            note_ids=members,
            opinion_score=score,
            kind=kind,
            work_ids=works,
        )

    ideas: list[Idea] = []
    for work, members in sorted(groups.items()):
        members = sorted(members)
        if qualifies(members):
            ideas.append(make(work_idea_id(work), "work", [work], members))

    works = sorted(groups)
    if len(works) // 3 < 2:  # too few works to split into >= 2 clusters: no catch-all theme
        manifest.save_ideas(ideas)
        return ideas
    work_vectors = np.vstack([np.mean([vector_of[n] for n in groups[w]], axis=0) for w in works])
    norms = np.linalg.norm(work_vectors, axis=1, keepdims=True)
    work_vectors = work_vectors / np.where(norms == 0, 1.0, norms)
    labels = cluster(work_vectors.astype(np.float32), seed=seed)
    for label in sorted(set(labels.tolist())):
        members_works = [w for w, lab in zip(works, labels, strict=True) if lab == label]
        if len(members_works) < 2:
            continue
        members = sorted(n for w in members_works for n in groups[w])
        if qualifies(members):
            ideas.append(make(make_idea_id(members_works), "theme", members_works, members))

    manifest.save_ideas(ideas)
    return ideas
