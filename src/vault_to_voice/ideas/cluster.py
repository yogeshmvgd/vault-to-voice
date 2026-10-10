import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest


def note_embeddings(manifest: Manifest, store: FaissStore) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for row in manifest.notes():
        chunk_ids = manifest.chunk_ids(row.note_id)
        if not chunk_ids:
            continue
        v = store.get_vectors(chunk_ids).mean(axis=0)
        norm = np.linalg.norm(v)
        vecs.append(v / norm if norm else v)
        ids.append(row.note_id)
    if not vecs:
        return [], np.zeros((0, store.dim), dtype=np.float32)
    return ids, np.vstack(vecs).astype(np.float32)


def cluster(vectors: np.ndarray, *, seed: int, k_max: int = 10) -> np.ndarray:
    n = len(vectors)
    upper = min(k_max, n // 3)
    if upper < 2:
        return np.zeros(n, dtype=int)
    best_labels, best_score = None, -1.0
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_labels, best_score = labels, score
    return best_labels if best_labels is not None else np.zeros(n, dtype=int)
