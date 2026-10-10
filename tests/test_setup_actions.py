import pytest
from langgraph.store.memory import InMemoryStore

from tests.conftest import note_text
from tests.test_cli import fake_llm
from vault_to_voice import setup_actions as sa
from vault_to_voice.config import Settings
from vault_to_voice.graph.memory import get_style
from vault_to_voice.ideas.detector import IndexOutOfSyncError
from vault_to_voice.index.embedder import HashEmbedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest


@pytest.fixture
def parts(tmp_path, vault_dir):
    settings = Settings(
        _env_file=None,
        vault_path=vault_dir,
        vault_format="flat",
        data_dir=tmp_path / "data",
        embedding_provider="hash",
        min_notes=2,
        min_opinionated=2,
    )
    embedder = HashEmbedder(256)
    return settings, Manifest(":memory:"), FaissStore(tmp_path / "f.index", 256), embedder


def write_notes(write_note, n=6):
    for i in range(n):
        write_note(f"n{i}.md", note_text(f"n{i}", f"boss fight fair difficulty note {i}"))


def test_ingest_counts_and_scores(parts, write_note):
    settings, manifest, index, embedder = parts
    write_notes(write_note)
    result = sa.ingest_vault(settings, manifest, index, embedder, fake_llm())
    assert result.report.new == 6 and result.report.scored == 6 and result.stats is None
    again = sa.ingest_vault(settings, manifest, index, embedder, None)
    assert again.report.unchanged == 6 and again.report.scored == 0


def test_ingest_obsidian_reports_stats(parts, write_note):
    settings, manifest, index, embedder = parts
    settings = settings.model_copy(update={"vault_format": "obsidian"})
    write_note("Daily/10-10-2026.md", "## Games\n- Played [[Hades]], loved it.\n- Too tired.\n")
    write_note("Games/Hades.md", "Best roguelike dialogue I know.\n")
    result = sa.ingest_vault(settings, manifest, index, embedder, None)
    assert result.report.new == 2
    assert result.stats.entries == 1 and result.stats.unlinked_skipped == 1


def test_build_style_saves_file_and_store(parts, write_note):
    settings, *_ = parts
    write_notes(write_note, 1)
    store = InMemoryStore()
    profile = sa.build_style(settings, fake_llm(), store)
    assert profile.version == 1 and profile.tone.tone == ["wry"]
    assert settings.style_profile_path.exists()
    assert get_style(store)["version"] == 1


def test_build_style_empty_vault(parts):
    settings, *_ = parts
    with pytest.raises(sa.NoNotesError):
        sa.build_style(settings, fake_llm(), InMemoryStore())


def test_find_ideas_and_out_of_sync(parts, write_note):
    settings, manifest, index, embedder = parts
    write_notes(write_note)
    sa.ingest_vault(settings, manifest, index, embedder, fake_llm())
    ideas = sa.find_ideas(settings, manifest, index, fake_llm())
    assert ideas and manifest.list_ideas()
    manifest.set_meta("dirty", "1")
    with pytest.raises(IndexOutOfSyncError):
        sa.find_ideas(settings, manifest, index, fake_llm())
