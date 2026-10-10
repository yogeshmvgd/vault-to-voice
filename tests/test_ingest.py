from tests.conftest import note_text
from tests.fakes import FakeLLMClient
from vault_to_voice.ideas.opinion import OpinionScore, OpinionScorer
from vault_to_voice.index.embedder import HashEmbedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.ingest import ingest
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.vault.reader import FilesystemVaultReader


class CountingEmbedder(HashEmbedder):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return super().embed(texts)


def setup(tmp_path, vault_dir, dim=64):
    m = Manifest(tmp_path / "m.db")
    s = FaissStore(tmp_path / "f.index", dim)
    return FilesystemVaultReader(vault_dir), m, s


def test_first_run_indexes_everything(tmp_path, vault_dir, write_note):
    write_note("a.md", note_text("a", "p one\n\np two"))
    write_note("b.md", note_text("b", "solo"))
    reader, m, s = setup(tmp_path, vault_dir)
    r = ingest(reader, m, s, HashEmbedder())
    assert (r.new, r.changed, r.deleted, r.unchanged, r.rebuilt) == (2, 0, 0, 0, False)
    assert len(s) == m.chunk_count() == 3


def test_second_run_skips_unchanged(tmp_path, vault_dir, write_note):
    write_note("a.md", note_text("a", "text"))
    reader, m, s = setup(tmp_path, vault_dir)
    ingest(reader, m, s, HashEmbedder())
    emb = CountingEmbedder()
    r = ingest(reader, m, s, emb)
    assert r.unchanged == 1 and emb.calls == 0


def test_changed_note_replaces_chunks(tmp_path, vault_dir, write_note):
    p = write_note("a.md", note_text("a", "old"))
    reader, m, s = setup(tmp_path, vault_dir)
    ingest(reader, m, s, HashEmbedder())
    old_ids = m.chunk_ids("a")
    p.write_text(note_text("a", "new one\n\nnew two"), encoding="utf-8")
    r = ingest(reader, m, s, HashEmbedder())
    assert r.changed == 1
    assert not set(old_ids) & set(m.chunk_ids("a"))
    assert len(s) == m.chunk_count() == 2


def test_deleted_note_removed(tmp_path, vault_dir, write_note):
    p = write_note("a.md", note_text("a", "x"))
    write_note("b.md", note_text("b", "y"))
    reader, m, s = setup(tmp_path, vault_dir)
    ingest(reader, m, s, HashEmbedder())
    p.unlink()
    r = ingest(reader, m, s, HashEmbedder())
    assert r.deleted == 1 and len(s) == m.chunk_count() == 1


def test_provider_switch_forces_rebuild(tmp_path, vault_dir, write_note):
    write_note("a.md", note_text("a", "x"))
    reader, m, s = setup(tmp_path, vault_dir, dim=64)
    ingest(reader, m, s, HashEmbedder(64))
    s32 = FaissStore(tmp_path / "f.index", 32)
    r = ingest(reader, m, s32, HashEmbedder(32))
    assert r.rebuilt and r.new == 1 and len(s32) == 1
    assert m.get_meta("embedder") == "hash:32"


def test_interrupted_run_detected_and_rebuilt(tmp_path, vault_dir, write_note):
    write_note("a.md", note_text("a", "x\n\ny"))
    reader, m, s = setup(tmp_path, vault_dir)
    ingest(reader, m, s, HashEmbedder())
    (tmp_path / "f.index").unlink()  # simulate crash before faiss.write_index
    fresh = FaissStore(tmp_path / "f.index", 64)
    r = ingest(reader, m, fresh, HashEmbedder())
    assert r.rebuilt and len(fresh) == m.chunk_count() == 2


def test_opinion_scored_once_and_rescored_on_change(tmp_path, vault_dir, write_note):
    p = write_note("a.md", note_text("a", "Loved it"))
    reader, m, s = setup(tmp_path, vault_dir)
    fake = FakeLLMClient({OpinionScore: lambda _: OpinionScore(score=0.7, highlights=["loved"])})
    scorer = OpinionScorer(fake)
    assert ingest(reader, m, s, HashEmbedder(), scorer).scored == 1
    assert ingest(reader, m, s, HashEmbedder(), scorer).scored == 0
    p.write_text(note_text("a", "Hated it"), encoding="utf-8")
    assert ingest(reader, m, s, HashEmbedder(), scorer).scored == 1
    assert m.get_note("a").opinion_score == 0.7


def test_dirty_flag_forces_rebuild_even_when_counts_match(tmp_path, vault_dir, write_note):
    write_note("a.md", note_text("a", "x"))
    reader, m, s = setup(tmp_path, vault_dir)
    ingest(reader, m, s, HashEmbedder())
    m.set_meta("dirty", "1")  # simulate crash after SQLite writes, before store.save()
    r = ingest(reader, m, s, HashEmbedder())
    assert r.rebuilt and m.get_meta("dirty") == "0"
