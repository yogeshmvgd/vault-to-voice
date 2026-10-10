import numpy as np
import structlog

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.memory import get_style, past_answers
from vault_to_voice.graph.state import NoteContext, PostState, SourceChunk

log = structlog.get_logger()


def load_context(state: PostState, deps: GraphDeps) -> dict:
    idea = state["idea"]
    notes = _note_contexts(deps, idea.note_ids, related=False)
    if not notes:
        raise ValueError(
            f"None of the notes for {idea.idea_id} are indexed any more. "
            "Run `v2v ingest` and `v2v ideas` again."
        )
    # Chronological order (undated work pages first) so drafts can follow time.
    notes.sort(key=lambda n: (n.date or "", n.note_id))
    related = _related_note_ids(deps, [n.note_id for n in notes])
    notes += _note_contexts(deps, related, related=True)
    return {
        "notes": notes,
        "style": get_style(deps.store),
        "past_answers": past_answers(deps.store, exclude_idea=idea.idea_id),
        "qa": [],
        "user_rounds": {"outline": 0, "draft": 0},
        "auto_revised": False,
    }


def _note_contexts(deps: GraphDeps, note_ids: list[str], *, related: bool) -> list[NoteContext]:
    out = []
    for nid in note_ids:
        row = deps.manifest.get_note(nid)
        chunk_ids = deps.manifest.chunk_ids(nid)
        if row is None or not chunk_ids:
            log.warning("context_note_missing", note_id=nid)
            continue
        chunks = [
            SourceChunk(chunk_id=c.chunk_id, text=c.text)
            for c in deps.manifest.get_chunks(chunk_ids)
        ]
        out.append(
            NoteContext(
                note_id=nid,
                title=row.title,
                type=row.type,
                date=row.date,
                highlights=row.highlights,
                chunks=chunks,
                related=related,
                work_id=row.work_id,
            )
        )
    return out


def _related_note_ids(deps: GraphDeps, note_ids: list[str]) -> list[str]:
    """Extra notes close to the idea's centroid in embedding space (RAG beyond the cluster)."""
    k = deps.settings.related_notes
    chunk_ids = [c for nid in note_ids for c in deps.manifest.chunk_ids(nid)]
    if k <= 0 or not chunk_ids or len(deps.index) == 0:
        return []
    try:
        vectors = deps.index.get_vectors(chunk_ids)
    except RuntimeError:  # index out of sync with the manifest
        log.warning("context_index_out_of_sync")
        return []
    query = vectors.mean(axis=0)
    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    hits = deps.index.search(query / norm, k=min(len(deps.index), 50))
    owner = {c.chunk_id: c.note_id for c in deps.manifest.get_chunks([i for i, _ in hits])}
    found: list[str] = []
    for chunk_id, score in hits:
        nid = owner.get(chunk_id)
        if nid is None or nid in note_ids or nid in found:
            continue
        if score < deps.settings.related_min_score:
            break  # hits are sorted by score
        found.append(nid)
        if len(found) == k:
            break
    return found
