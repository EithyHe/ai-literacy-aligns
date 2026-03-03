"""Dimensionality reduction / decomposition service (PRD FR-2.2.1).

Supports PCA, Factor Analysis, t-SNE, UMAP, MDS, ICA, and Diffusion Maps.
"""

from __future__ import annotations

import inspect
import logging
from typing import Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS as SklearnMDS

logger = logging.getLogger(__name__)

MethodType = Literal["pca", "factor_analysis", "tsne", "umap", "mds", "ica", "diffusion_maps"]

_ALL_METHODS: list[str] = [
    "pca",
    "factor_analysis",
    "tsne",
    "umap",
    "mds",
    "ica",
    "diffusion_maps",
]


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Validate and ensure 2D float array."""
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {X.shape}")
    if X.size == 0:
        raise ValueError("embeddings cannot be empty")
    return X


def _clamp_components(n_samples: int, n_features: int, n_components: int) -> int:
    """Clamp n_components to valid range for the data."""
    max_comp = min(n_samples - 1, n_features)
    if n_components > max_comp:
        logger.warning(
            f"n_components={n_components} exceeds max ({max_comp}) for "
            f"n_samples={n_samples}, n_features={n_features}. Using {max_comp}."
        )
        return max(1, max_comp)
    return n_components


def run_pca(embeddings: np.ndarray, n_components: int = 2) -> dict:
    """PCA with variance explained, loadings, scores.

    Returns dict with coords, explained_variance, loadings, method, params.
    """
    X = _validate_embeddings(embeddings)
    n_samples, n_features = X.shape
    n_comp = _clamp_components(n_samples, n_features, n_components)

    pca = SklearnPCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X)

    return {
        "coords": coords,
        "explained_variance": pca.explained_variance_ratio_,
        "loadings": pca.components_.T,
        "method": "pca",
        "params": {"n_components": n_comp},
    }


def run_factor_analysis(embeddings: np.ndarray, n_components: int = 2) -> dict:
    """Factor Analysis with loadings.

    Returns dict with coords, explained_variance (None), loadings, method, params.
    """
    X = _validate_embeddings(embeddings)
    n_samples, n_features = X.shape
    n_comp = _clamp_components(n_samples, n_features, n_components)

    fa = FactorAnalysis(n_components=n_comp, random_state=42, max_iter=1000)
    coords = fa.fit_transform(X)

    return {
        "coords": coords,
        "explained_variance": None,
        "loadings": fa.components_.T,
        "method": "factor_analysis",
        "params": {"n_components": n_comp},
    }


def run_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
) -> dict:
    """t-SNE dimensionality reduction.

    Returns dict with coords, explained_variance (None), loadings (None), method, params.
    """
    X = _validate_embeddings(embeddings)
    n_samples, n_features = X.shape
    n_comp = min(n_components, 3)
    if n_comp != n_components:
        logger.warning(f"t-SNE supports at most 3 components, using {n_comp}.")

    perplexity = min(perplexity, (n_samples - 1) / 3)
    if perplexity < 5:
        perplexity = 5.0
        logger.warning(f"Perplexity too low for n_samples={n_samples}, using 5.0.")

    tsne = TSNE(
        n_components=n_comp,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        n_iter=1000,
    )
    coords = tsne.fit_transform(X)

    return {
        "coords": coords,
        "explained_variance": None,
        "loadings": None,
        "method": "tsne",
        "params": {"n_components": n_comp, "perplexity": perplexity},
    }


def run_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> dict:
    """UMAP dimensionality reduction.

    Returns dict with coords, explained_variance (None), loadings (None), method, params.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP. Install with: pip install umap-learn"
        )

    X = _validate_embeddings(embeddings)
    n_samples, n_features = X.shape
    n_comp = _clamp_components(n_samples, n_features, n_components)

    n_neighbors = min(n_neighbors, n_samples - 1)
    if n_neighbors < 2:
        n_neighbors = 2
        logger.warning(f"n_neighbors too low for n_samples={n_samples}, using 2.")

    reducer = umap.UMAP(
        n_components=n_comp,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(X)

    return {
        "coords": coords,
        "explained_variance": None,
        "loadings": None,
        "method": "umap",
        "params": {
            "n_components": n_comp,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
    }


def run_mds(embeddings: np.ndarray, n_components: int = 2) -> dict:
    """Multidimensional Scaling (metric MDS).

    Returns dict with coords, explained_variance (None), loadings (None), method, params.
    """
    X = _validate_embeddings(embeddings)
    n_samples, n_features = X.shape
    n_comp = _clamp_components(n_samples, n_features, n_components)

    mds = SklearnMDS(
        n_components=n_comp,
        metric=True,
        random_state=42,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(X)

    return {
        "coords": coords,
        "explained_variance": None,
        "loadings": None,
        "method": "mds",
        "params": {"n_components": n_comp},
    }


def run_ica(embeddings: np.ndarray, n_components: int = 2) -> dict:
    """Independent Component Analysis.

    Returns dict with coords, explained_variance (None), loadings (mixing matrix), method, params.
    """
    X = _validate_embeddings(embeddings)
    n_samples, n_features = X.shape
    n_comp = _clamp_components(n_samples, n_features, n_components)

    ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
    coords = ica.fit_transform(X)
    loadings = ica.components_.T

    return {
        "coords": coords,
        "explained_variance": None,
        "loadings": loadings,
        "method": "ica",
        "params": {"n_components": n_comp},
    }


def run_diffusion_maps(
    embeddings: np.ndarray,
    n_components: int = 2,
    epsilon: float | None = None,
    alpha: float = 0.5,
) -> dict:
    """Diffusion Maps: manifold learning via diffusion process.

    Uses Gaussian kernel affinity and Markov normalization.
    """
    X = _validate_embeddings(embeddings)
    n_samples, n_features = X.shape
    n_comp = _clamp_components(n_samples, n_features, n_components)

    D_sq = squareform(pdist(X, "sqeuclidean"))

    if epsilon is None:
        med = np.median(np.sqrt(D_sq + np.finfo(float).eps))
        epsilon = max(med**2 * 0.5, 1e-10)
        logger.debug(f"Diffusion maps: auto epsilon={epsilon:.6f}")

    W = np.exp(-D_sq / (2 * epsilon))
    d = np.sum(W, axis=1) ** alpha
    d_inv = np.where(d > 0, 1.0 / d, 0)
    W_norm = W * d_inv[:, np.newaxis] * d_inv[np.newaxis, :]
    row_sum = np.sum(W_norm, axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1)
    P = W_norm / row_sum

    eigenvalues, eigenvectors = np.linalg.eigh(P)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    n_use = min(n_comp + 1, len(eigenvalues))
    coords = eigenvectors[:, 1:n_use] * eigenvalues[1:n_use]
    coords = coords[:, :n_comp]

    return {
        "coords": coords,
        "explained_variance": None,
        "loadings": None,
        "method": "diffusion_maps",
        "params": {"n_components": n_comp, "epsilon": float(epsilon), "alpha": alpha},
    }


def run_decomposition(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    **kwargs,
) -> dict:
    """Run dimensionality reduction. Returns dict with keys:

    - 'coords': np.ndarray (n_samples, n_components) - the reduced coordinates
    - 'explained_variance': np.ndarray or None (for methods that support it)
    - 'loadings': np.ndarray or None (for PCA/FA)
    - 'method': str
    - 'params': dict of parameters used
    """
    method_lower = method.lower().replace("-", "_").replace(" ", "_")
    if method_lower not in _ALL_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {_ALL_METHODS}")

    dispatcher = {
        "pca": run_pca,
        "factor_analysis": run_factor_analysis,
        "tsne": run_tsne,
        "umap": run_umap,
        "mds": run_mds,
        "ica": run_ica,
        "diffusion_maps": run_diffusion_maps,
    }

    fn = dispatcher[method_lower]
    all_params: dict = {"n_components": n_components, **kwargs}
    sig = inspect.signature(fn)
    valid_params = {k: v for k, v in all_params.items() if k in sig.parameters}
    return fn(embeddings, **valid_params)


def suggest_n_components(
    embeddings: np.ndarray,
    variance_threshold: float = 0.95,
) -> int:
    """Suggest optimal number of PCA components using cumulative variance threshold."""
    X = _validate_embeddings(embeddings)
    n_samples, n_features = X.shape
    n_max = min(n_samples - 1, n_features)
    if n_max < 1:
        return 1

    pca = SklearnPCA(n_components=n_max, random_state=42)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    idx = np.searchsorted(cumvar, variance_threshold, side="left")
    n_comp = int(idx) + 1
    return max(1, min(n_comp, n_max))


def compare_methods(
    embeddings: np.ndarray,
    methods: list[str] | None = None,
    n_components: int = 2,
) -> dict[str, dict]:
    """Run multiple methods and return results for comparison."""
    X = _validate_embeddings(embeddings)
    methods_to_run = methods or _ALL_METHODS.copy()

    results: dict[str, dict] = {}
    for m in methods_to_run:
        try:
            results[m] = run_decomposition(X, method=m, n_components=n_components)
        except Exception as e:
            logger.warning(f"Method '{m}' failed: {e}")
            results[m] = {"error": str(e), "coords": None}

    return results


def compute_procrustes(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute Procrustes disparity between two sets of coordinates.

    Disparity is the sum of squared residuals after optimal alignment
    (translation, rotation, scaling). Lower = more similar.
    """
    A = np.asarray(coords_a, dtype=np.float64)
    B = np.asarray(coords_b, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: coords_a {A.shape} vs coords_b {B.shape}")
    if A.size == 0:
        return 0.0

    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)
    R, _ = orthogonal_procrustes(A_centered, B_centered)
    B_aligned = B_centered @ R
    disparity = np.sum((A_centered - B_aligned) ** 2)
    return float(disparity)


def get_top_loadings(
    loadings: np.ndarray,
    feature_names: list[str],
    n_top: int = 10,
) -> list[dict]:
    """Get top-N features by absolute loading for each component.

    Returns list of dicts: [{"component": 0, "features": [{"name", "loading", "rank"}, ...]}, ...]
    """
    loadings_arr = np.asarray(loadings)
    if loadings_arr.ndim != 2:
        raise ValueError(
            f"loadings must be 2D (n_features, n_components), got shape {loadings_arr.shape}"
        )

    n_features, n_components = loadings_arr.shape
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match loadings rows ({n_features})"
        )

    result: list[dict] = []
    for c in range(n_components):
        col = loadings_arr[:, c]
        abs_load = np.abs(col)
        top_idx = np.argsort(abs_load)[::-1][:n_top]

        features = [
            {"name": feature_names[i], "loading": float(col[i]), "rank": r + 1}
            for r, i in enumerate(top_idx)
        ]
        result.append({"component": c, "features": features})

    return result
