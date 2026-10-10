import numpy as np
import pytest

from vault_to_voice.config import Settings
from vault_to_voice.index.embedder import HashEmbedder, LocalEmbedder, make_embedder


def test_hash_embedder_shape_norm_determinism():
    e = HashEmbedder(dim=32)
    v = e.embed(["the boss fight was fair", "the boss fight was fair"])
    assert v.shape == (2, 32) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[0], v[1])


def test_hash_embedder_similarity_tracks_word_overlap():
    e = HashEmbedder(dim=256)
    a, b, c = e.embed(["fair boss fight design", "boss fight design was fair", "baking bread"])
    assert a @ b > a @ c


def test_empty_input():
    assert HashEmbedder(dim=8).embed([]).shape == (0, 8)


def test_factory_hash():
    e = make_embedder(Settings(_env_file=None, embedding_provider="hash"))
    assert e.name == "hash" and e.dim == 64


@pytest.mark.integration
def test_local_minilm():
    e = LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    v = e.embed(["hello world"])
    assert e.dim == 384 and v.shape == (1, 384)
