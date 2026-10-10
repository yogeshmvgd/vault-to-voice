from vault_to_voice.index.manifest import Manifest
from vault_to_voice.vault.models import Note


def mk(id_: str, h: str = "h1") -> Note:
    return Note(id=id_, title=id_, type="game", body="b", path=f"{id_}.md", hash=h)


def test_diff_classifies():
    m = Manifest(":memory:")
    for n in (mk("keep"), mk("change"), mk("gone")):
        m.upsert_note(n)
    d = m.diff([mk("keep"), mk("change", "h2"), mk("new")])
    assert [n.id for n in d.new] == ["new"]
    assert [n.id for n in d.changed] == ["change"]
    assert [n.id for n in d.unchanged] == ["keep"]
    assert d.deleted == ["gone"]


def test_chunks_roundtrip_and_delete():
    m = Manifest(":memory:")
    m.upsert_note(mk("a"))
    ids = m.add_chunks("a", ["p0", "p1"])
    assert ids[1] > ids[0] and m.chunk_count() == 2
    assert [c.text for c in m.get_chunks(ids)] == ["p0", "p1"]
    assert m.delete_note("a") == ids
    assert m.chunk_count() == 0 and m.get_note("a") is None


def test_chunk_ids_never_reused():
    m = Manifest(":memory:")
    first = m.add_chunks("a", ["x"])
    m.delete_note("a")
    assert m.add_chunks("a", ["y"])[0] > first[0]


def test_opinion_lifecycle():
    m = Manifest(":memory:")
    m.upsert_note(mk("a"))
    assert m.notes_needing_opinion() == ["a"]
    m.set_opinion("a", 0.8, ["great boss design"])
    row = m.get_note("a")
    assert (row.opinion_score, row.highlights) == (0.8, ["great boss design"])
    assert m.notes_needing_opinion() == []
    m.upsert_note(mk("a", "h2"))  # content changed -> must be rescored
    assert m.notes_needing_opinion() == ["a"]


def test_meta_and_persistence(tmp_path):
    path = tmp_path / "sub" / "m.db"
    m = Manifest(path)
    m.set_meta("embedder", "hash:64")
    m.upsert_note(mk("a"))
    m.close()
    m2 = Manifest(path)
    assert m2.get_meta("embedder") == "hash:64"
    assert [r.note_id for r in m2.notes()] == ["a"]


def test_clear_index():
    m = Manifest(":memory:")
    m.upsert_note(mk("a"))
    m.add_chunks("a", ["x"])
    m.clear_index()
    assert m.notes() == [] and m.chunk_count() == 0
