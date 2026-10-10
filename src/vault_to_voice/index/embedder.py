import hashlib
import re
from typing import Protocol

import numpy as np

from vault_to_voice.config import Settings

OPENAI_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for offline tests."""

    name = "hash"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9']+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        return _normalise(out)


class LocalEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class OpenAIEmbedder:
    def __init__(self, model: str, batch_size: int = 100):
        from openai import OpenAI

        self.name = model
        self.dim = OPENAI_DIMS[model]
        self._client = OpenAI()
        self._batch = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            resp = self._client.embeddings.create(model=self.name, input=texts[i : i + self._batch])
            rows.extend(d.embedding for d in resp.data)
        return _normalise(np.array(rows))


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder()
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.openai_embedding_model)
    return LocalEmbedder(settings.local_embedding_model)
