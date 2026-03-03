"""Multi-metric similarity computation service for the Semantic Scale Network."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist, cityblock, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

MetricType = Literal["cosine", "euclidean", "manhattan", "dot"]

logger = logging.getLogger(__name__)

_SUPPORTED_METRICS: set[str] = {"cosine", "euclidean", "manhattan", "dot"}


def _validate_metric(metric: str) -> None:
    """Validate that the metric is supported."""
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"metric must be one of {_SUPPORTED_METRICS}, got {metric!r}"
        )


def _distance_to_similarity(distance: float, metric: str) -> float:
    """Convert distance to similarity (higher = more similar)."""
    if metric == "euclidean":
        return float(1.0 / (1.0 + distance))
    if metric == "manhattan":
        return float(1.0 / (1.0 + distance))
    return float(distance)


def compute_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Compute similarity between two vectors using the specified metric.

    Args:
        vec_a: First vector (1D array).
        vec_b: Second vector (1D array).
        metric: One of "cosine", "euclidean", "manhattan", "dot".

    Returns:
        Similarity score. Higher values indicate greater similarity.
        For cosine/dot: range [-1, 1] (typically [0, 1] for text embeddings).
        For euclidean/manhattan: range (0, 1] via 1/(1+distance).

    Raises:
        ValueError: If metric is not supported or vectors have incompatible shapes.
    """
    _validate_metric(metric)
    vec_a = np.asarray(vec_a, dtype=np.float64).flatten()
    vec_b = np.asarray(vec_b, dtype=np.float64).flatten()
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"vec_a shape {vec_a.shape} != vec_b shape {vec_b.shape}"
        )

    if metric == "cosine":
        sim = cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0, 0]
        return float(np.clip(sim, -1.0, 1.0))

    if metric == "dot":
        vec_a_norm = normalize(vec_a.reshape(1, -1), norm="l2")[0]
        vec_b_norm = normalize(vec_b.reshape(1, -1), norm="l2")[0]
        return float(np.dot(vec_a_norm, vec_b_norm))

    if metric == "euclidean":
        d = euclidean(vec_a, vec_b)
        return _distance_to_similarity(d, metric)

    if metric == "manhattan":
        d = cityblock(vec_a, vec_b)
        return _distance_to_similarity(d, metric)

    raise ValueError(f"Unsupported metric: {metric}")


def compute_pairwise_similarity(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute pairwise similarity matrix between rows of two matrices.

    Args:
        matrix_a: Matrix of shape (n_a, dim).
        matrix_b: Matrix of shape (n_b, dim).
        metric: One of "cosine", "euclidean", "manhattan", "dot".

    Returns:
        Similarity matrix of shape (n_a, n_b). Entry [i, j] = similarity between
        matrix_a[i] and matrix_b[j].

    Raises:
        ValueError: If metric is not supported or column dimensions differ.
    """
    _validate_metric(metric)
    matrix_a = np.asarray(matrix_a, dtype=np.float64)
    matrix_b = np.asarray(matrix_b, dtype=np.float64)
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        raise ValueError("matrix_a and matrix_b must be 2D arrays")
    if matrix_a.shape[1] != matrix_b.shape[1]:
        raise ValueError(
            f"Column dimension mismatch: {matrix_a.shape[1]} vs {matrix_b.shape[1]}"
        )

    if metric == "cosine":
        sim = cosine_similarity(matrix_a, matrix_b)
        return np.clip(sim, -1.0, 1.0).astype(np.float64)

    if metric == "dot":
        matrix_a_norm = normalize(matrix_a, norm="l2")
        matrix_b_norm = normalize(matrix_b, norm="l2")
        return np.dot(matrix_a_norm, matrix_b_norm.T).astype(np.float64)

    if metric in ("euclidean", "manhattan"):
        scipy_metric = "euclidean" if metric == "euclidean" else "cityblock"
        dist = cdist(matrix_a, matrix_b, metric=scipy_metric)
        return (1.0 / (1.0 + dist)).astype(np.float64)

    raise ValueError(f"Unsupported metric: {metric}")


def find_top_n_neighbors(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    corpus_ids: list[str],
    top_n: int = 5,
    metric: str = "cosine",
) -> list[dict]:
    """Find top-N most similar items in the corpus.

    Args:
        query_vec: Query vector (1D).
        corpus_matrix: Corpus embeddings, shape (n_items, dim).
        corpus_ids: List of item IDs corresponding to corpus_matrix rows.
        top_n: Number of neighbors to return.
        metric: Similarity metric.

    Returns:
        List of dicts with keys: id, similarity, rank (1-based).
        Sorted by similarity descending.

    Raises:
        ValueError: If corpus_ids length does not match corpus_matrix rows.
    """
    _validate_metric(metric)
    corpus_matrix = np.asarray(corpus_matrix, dtype=np.float64)
    if len(corpus_ids) != corpus_matrix.shape[0]:
        raise ValueError(
            f"corpus_ids length ({len(corpus_ids)}) != corpus_matrix rows ({corpus_matrix.shape[0]})"
        )

    query = np.asarray(query_vec, dtype=np.float64).flatten().reshape(1, -1)
    sim_matrix = compute_pairwise_similarity(query, corpus_matrix, metric)
    sims = sim_matrix[0]

    top_indices = np.argsort(sims)[::-1][:top_n]
    return [
        {
            "id": corpus_ids[i],
            "similarity": float(sims[i]),
            "rank": r,
        }
        for r, i in enumerate(top_indices, start=1)
    ]


def compute_cross_similarity_matrix(
    items_a: list[dict],
    items_b: list[dict],
    embeddings: dict[str, np.ndarray],
    metric: str = "cosine",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute item-level cross-similarity matrix between two item sets.

    Args:
        items_a: List of item dicts with an 'id' key (or similar identifier).
        items_b: List of item dicts with an 'id' key.
        embeddings: Dict mapping item ID to embedding vector.
        metric: Similarity metric.

    Returns:
        Tuple of (matrix, row_labels, col_labels).
        matrix[i, j] = similarity between items_a[i] and items_b[j].

    Raises:
        KeyError: If any item ID is missing from embeddings.
    """
    _validate_metric(metric)

    def _get_id(item: dict) -> str:
        for key in ("id", "item_id", "construct_id"):
            if key in item:
                return str(item[key])
        raise KeyError(f"Item dict must have 'id', 'item_id', or 'construct_id': {item}")

    row_ids = [_get_id(it) for it in items_a]
    col_ids = [_get_id(it) for it in items_b]

    missing = set(row_ids + col_ids) - set(embeddings.keys())
    if missing:
        raise KeyError(f"Missing embeddings for: {missing}")

    matrix_a = np.array([embeddings[i] for i in row_ids], dtype=np.float64)
    matrix_b = np.array([embeddings[i] for i in col_ids], dtype=np.float64)

    sim_matrix = compute_pairwise_similarity(matrix_a, matrix_b, metric)
    return sim_matrix, row_ids, col_ids


def find_scale_neighbors(
    scale_embedding: np.ndarray,
    construct_embeddings: dict[str, np.ndarray],
    top_n: int = 5,
    metric: str = "cosine",
) -> list[dict]:
    """Find top-N most similar constructs/scales to a given scale embedding.

    Args:
        scale_embedding: Embedding of the query scale (1D).
        construct_embeddings: Dict mapping construct/scale ID to embedding.
        top_n: Number of neighbors to return.
        metric: Similarity metric.

    Returns:
        List of dicts with keys: id, similarity, rank (1-based).
    """
    _validate_metric(metric)
    if not construct_embeddings:
        return []

    ids = list(construct_embeddings.keys())
    matrix = np.array([construct_embeddings[i] for i in ids], dtype=np.float64)
    return find_top_n_neighbors(
        scale_embedding, matrix, ids, top_n=top_n, metric=metric
    )


def compute_similarity_distribution(embeddings: np.ndarray) -> dict:
    """Compute summary statistics of pairwise similarities across corpus.

    Uses cosine similarity by default (most common for embeddings).
    Only upper triangle (excluding diagonal) is used to avoid duplicates.

    Args:
        embeddings: Matrix of shape (n_items, dim).

    Returns:
        Dict with keys: mean, std, min, max, median, q25, q75, n_pairs.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D")

    n = embeddings.shape[0]
    if n < 2:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "n_pairs": 0,
        }

    sim_matrix = cosine_similarity(embeddings)
    triu_indices = np.triu_indices(n, k=1)
    sims = sim_matrix[triu_indices]

    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "median": float(np.median(sims)),
        "q25": float(np.percentile(sims, 25)),
        "q75": float(np.percentile(sims, 75)),
        "n_pairs": int(len(sims)),
    }
