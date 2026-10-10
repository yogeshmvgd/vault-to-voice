from dataclasses import dataclass

import structlog

from vault_to_voice.guardrails import scrub_pii
from vault_to_voice.ideas.opinion import OpinionScorer
from vault_to_voice.index.chunking import split_paragraphs
from vault_to_voice.index.embedder import Embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.vault.reader import VaultReader

log = structlog.get_logger()


@dataclass
class IngestReport:
    new: int
    changed: int
    deleted: int
    unchanged: int
    rebuilt: bool
    scored: int = 0


def ingest(
    reader: VaultReader,
    manifest: Manifest,
    store: FaissStore,
    embedder: Embedder,
    scorer: OpinionScorer | None = None,
) -> IngestReport:
    notes = list(reader.iter_notes())
    signature = f"{embedder.name}:{embedder.dim}"

    rebuilt = False
    out_of_sync = len(store) != manifest.chunk_count()
    if manifest.get_meta("embedder") != signature or out_of_sync:
        rebuilt = manifest.chunk_count() > 0 or len(store) > 0
        if rebuilt:
            log.info("index_rebuild", signature=signature, out_of_sync=out_of_sync)
        manifest.clear_index()
        store.reset()
        manifest.set_meta("embedder", signature)

    diff = manifest.diff(notes)
    for note_id in diff.deleted:
        store.remove(manifest.delete_note(note_id))
    for note in diff.new + diff.changed:
        store.remove(manifest.delete_note(note.id))  # also clears orphans from a crashed run
        paragraphs = split_paragraphs(note.body) or [note.title]
        ids = manifest.add_chunks(note.id, paragraphs)
        store.add(ids, embedder.embed([scrub_pii(p) for p in paragraphs]))
        manifest.upsert_note(note)
    store.save()

    scored = 0
    if scorer is not None:
        by_id = {n.id: n for n in notes}
        for note_id in manifest.notes_needing_opinion():
            result = scorer.score(by_id[note_id])
            manifest.set_opinion(note_id, result.score, result.highlights)
            scored += 1

    return IngestReport(
        new=len(diff.new),
        changed=len(diff.changed),
        deleted=len(diff.deleted),
        unchanged=len(diff.unchanged),
        rebuilt=rebuilt,
        scored=scored,
    )
