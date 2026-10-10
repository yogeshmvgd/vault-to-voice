import numpy as np

from vault_to_voice.index.faiss_store import FaissStore


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_add_search_remove(tmp_path):
    s = FaissStore(tmp_path / "f.index", 3)
    s.add([10, 20], np.vstack([unit(1, 0, 0), unit(0, 1, 0)]))
    assert len(s) == 2
    assert s.search(unit(1, 0.1, 0), k=5)[0][0] == 10
    s.remove([10])
    assert [i for i, _ in s.search(unit(1, 0, 0), k=5)] == [20]


def test_search_empty_and_k_larger_than_index(tmp_path):
    s = FaissStore(tmp_path / "f.index", 3)
    assert s.search(unit(1, 0, 0), k=3) == []
    s.add([1], unit(1, 0, 0).reshape(1, -1))
    assert len(s.search(unit(1, 0, 0), k=10)) == 1


def test_persist_and_reload(tmp_path):
    p = tmp_path / "d" / "f.index"
    s = FaissStore(p, 3)
    s.add([7], unit(0, 0, 1).reshape(1, -1))
    s.save()
    s2 = FaissStore(p, 3)
    assert len(s2) == 1
    assert np.allclose(s2.get_vectors([7])[0], unit(0, 0, 1))


def test_dim_mismatch_on_load_gives_empty(tmp_path):
    p = tmp_path / "f.index"
    s = FaissStore(p, 3)
    s.add([1], unit(1, 0, 0).reshape(1, -1))
    s.save()
    assert len(FaissStore(p, 5)) == 0


def test_reset(tmp_path):
    s = FaissStore(tmp_path / "f.index", 3)
    s.add([1], unit(1, 0, 0).reshape(1, -1))
    s.reset()
    assert len(s) == 0
