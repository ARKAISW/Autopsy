# fingerprint/embedding.py

import numpy as np
from sklearn.decomposition import PCA


def build_umap_embedding(crisis_fingerprints: dict, live_vector: np.ndarray | None = None):
    """
    Projects crisis fingerprints into 2D using PCA.
    Deterministic — same result every run.
    Name kept as build_umap_embedding for compatibility with existing imports.
    """
    keys = list(crisis_fingerprints.keys())
    vectors = np.array([crisis_fingerprints[k] for k in keys])

    all_vectors = vectors
    if live_vector is not None:
        all_vectors = np.vstack([vectors, live_vector.reshape(1, -1)])

    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_vectors)

    crisis_coords = {keys[i]: tuple(coords_2d[i]) for i in range(len(keys))}
    live_coords = None
    if live_vector is not None:
        live_coords = tuple(coords_2d[-1])

    return crisis_coords, live_coords
