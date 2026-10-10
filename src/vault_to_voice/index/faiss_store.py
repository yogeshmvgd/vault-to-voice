from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    """Cosine search over L2-normalised vectors keyed by SQLite chunk ids."""

    def __init__(self, path: Path, dim: int):
        self.path = Path(path)
        self.dim = dim
        self.index = self._load_or_new()

    def _new(self) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def _load_or_new(self) -> faiss.Index:
        if self.path.exists():
            index = faiss.read_index(str(self.path))
            if index.d == self.dim:
                return index
        return self._new()

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids:
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.shape != (len(ids), self.dim):
            raise ValueError(f"expected shape {(len(ids), self.dim)}, got {vecs.shape}")
        self.index.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))

    def remove(self, ids: list[int]) -> None:
        if ids:
            self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(np.asarray(query).reshape(1, -1), dtype=np.float32)
        scores, ids = self.index.search(q, min(k, int(self.index.ntotal)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def get_vectors(self, ids: list[int]) -> np.ndarray:
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.index.reconstruct(int(i)) for i in ids])

    def reset(self) -> None:
        self.index = self._new()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path))
