import numpy as np

from vault_to_voice.ideas.cluster import cluster, note_embeddings
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.vault.models import Note


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(8)[:3]
    pts = np.vstack([c + rng.normal(0, 0.01, (6, 8)) for c in centers])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def test_finds_three_blobs_deterministically():
    labels = cluster(blobs(), seed=42)
    assert len(set(labels)) == 3
    assert all(len(set(labels[i * 6 : i * 6 + 6])) == 1 for i in range(3))
    assert (cluster(blobs(), seed=42) == labels).all()


def test_small_inputs_single_cluster():
    assert cluster(np.zeros((0, 8)), seed=1).tolist() == []
    assert cluster(blobs()[:5], seed=1).tolist() == [0] * 5


def test_note_embeddings_mean_and_normalised(tmp_path):
    m = Manifest(":memory:")
    s = FaissStore(tmp_path / "f.index", 2)
    m.upsert_note(Note(id="a", title="a", body="b", path="a.md", hash="h"))
    ids = m.add_chunks("a", ["x", "y"])
    s.add(ids, np.array([[1, 0], [0, 1]], dtype=np.float32))
    m.upsert_note(Note(id="empty", title="e", body="b", path="e.md", hash="h"))
    note_ids, vecs = note_embeddings(m, s)
    assert note_ids == ["a"]
    assert np.allclose(vecs[0], [0.7071, 0.7071], atol=1e-3)


def test_note_embeddings_empty(tmp_path):
    ids, vecs = note_embeddings(Manifest(":memory:"), FaissStore(tmp_path / "f.index", 4))
    assert ids == [] and vecs.shape == (0, 4)
