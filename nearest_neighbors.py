from cuml.neighbors import NearestNeighbors
import cupy as cp
import numpy as np


def precompute_nearest_neighbors(
    dataset, n_neighbors, algorithm="brute", metric="cosine", verbose=False
):
    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1, algorithm=algorithm, metric=metric, verbose=verbose
    )  # +1 to account for self-match
    model.fit(cp.asarray(dataset))
    distances, indices = model.kneighbors()

    final_indices: cp.ndarray = indices[:, 1:]  # remove self match
    final_distances: cp.ndarray = distances[:, 1:]  # remove self match

    final_indices = cp.asnumpy(final_indices)
    final_distances = cp.asnumpy(final_distances)

    return final_indices, final_distances


def convert_knn_to_pairs(knn_indices: np.ndarray, n_neighbors: int) -> np.ndarray:
    neighbors = knn_indices[:, 1 : n_neighbors + 1]  # remove self match

    i_indices = np.repeat(np.arange(knn_indices.shape[0]), n_neighbors) # source indices
    j_indices = neighbors.flatten() # target indices

    pair_neighbors = np.stack([i_indices, j_indices], axis=1)
    return pair_neighbors.astype(np.int32)
