"""Setup actions shared by the CLI and the API: index the vault, build the style profile,
detect ideas. Callers pass the dependencies, so both use exactly one code path."""

from dataclasses import dataclass
from typing import Any

from langgraph.store.base import BaseStore

from vault_to_voice.config import Settings
from vault_to_voice.graph.memory import put_style
from vault_to_voice.ideas.detector import Idea, detect_ideas
from vault_to_voice.ideas.opinion import OpinionScorer
from vault_to_voice.index.embedder import Embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.ingest import IngestReport, ingest
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LLMClient
from vault_to_voice.style.profile import StyleProfile, build_style_profile, save_profile
from vault_to_voice.vault.factory import make_reader


class NoNotesError(RuntimeError):
    pass


@dataclass
class IngestResult:
    report: IngestReport
    stats: Any = None  # ReadStats for the Obsidian reader, None for flat vaults


def ingest_vault(
    settings: Settings,
    manifest: Manifest,
    index: FaissStore,
    embedder: Embedder,
    llm: LLMClient | None,
) -> IngestResult:
    """Index new/changed notes; score them for opinion unless `llm` is None."""
    reader = make_reader(settings)
    scorer = OpinionScorer(llm) if llm is not None else None
    report = ingest(reader, manifest, index, embedder, scorer)
    return IngestResult(report, getattr(reader, "stats", None))


def build_style(settings: Settings, llm: LLMClient, store: BaseStore) -> StyleProfile:
    """Build the style profile, save it to STYLE_PROFILE_PATH and to the Store."""
    notes = list(make_reader(settings).iter_notes())
    if not notes:
        raise NoNotesError("no notes found")
    profile = save_profile(
        build_style_profile(notes, llm, seed=settings.seed), settings.style_profile_path
    )
    put_style(store, profile)
    return profile


def find_ideas(
    settings: Settings, manifest: Manifest, index: FaissStore, llm: LLMClient
) -> list[Idea]:
    """Cluster indexed notes into ideas. Raises IndexOutOfSyncError if ingest is needed."""
    return detect_ideas(
        manifest,
        index,
        llm,
        min_notes=settings.min_notes,
        min_opinionated=settings.min_opinionated,
        opinion_threshold=settings.opinion_threshold,
        seed=settings.seed,
    )
