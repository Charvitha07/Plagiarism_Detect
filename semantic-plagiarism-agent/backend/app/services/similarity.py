"""
Pure-numpy cosine similarity utilities.

This is the exact-math fallback/reference implementation used directly
for small corpora and for testing; Phase 2 adds a FAISS-backed index for
scale, but FAISS's inner-product search over normalized vectors returns
the same cosine similarity values computed here — these functions double
as the correctness oracle for the FAISS layer.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between every row of `a` and every row
    of `b`. Returns an (a.shape[0], b.shape[0]) array in [-1, 1]."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = _row_normalize(a.astype(np.float32))
    b_norm = _row_normalize(b.astype(np.float32))
    return a_norm @ b_norm.T


def top_k_matches(query_vector: np.ndarray, corpus_matrix: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    """Returns up to `k` (index, cosine_similarity) pairs from
    `corpus_matrix`, sorted by similarity descending."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if corpus_matrix.shape[0] == 0:
        return []

    sims = cosine_similarity_matrix(query_vector.reshape(1, -1), corpus_matrix)[0]
    k = min(k, sims.shape[0])
    top_indices = np.argpartition(-sims, k - 1)[:k]
    top_indices = top_indices[np.argsort(-sims[top_indices])]
    return [(int(i), float(sims[i])) for i in top_indices]
