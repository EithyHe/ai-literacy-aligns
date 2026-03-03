"""Spatial density analysis service (PRD FR-3.1.1).

Estimates point density via KDE, k-NN, LOF, DBSCAN, and GMM.
Computes redundancy risk and per-group density statistics.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DensityMethod = Literal["kde", "knn", "lof", "dbscan", "gmm", "voronoi"]

_ALL_METHODS: list[str] = ["kde", "knn", "lof", "dbscan", "gmm", "voronoi"]


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    """Validate and ensure 2D float array."""
    X = np.asarray(coords, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"coords must be 2D, got shape {X.shape}")
    if X.size == 0:
        raise ValueError("coords cannot be empty")
    return X


def estimate_kde(
    coords: np.ndarray,
    bandwidth: float | str = "scott",
) -> np.ndarray:
    """Kernel Density Estimation.

    Returns density at each point (higher = denser).
    """
    X = _validate_coords(coords)
    n_samples, n_features = X.shape

    if bandwidth == "scott" and n_samples < 2:
        logger.warning("Too few samples for Scott bandwidth; using fixed 0.5")
        bandwidth = 0.5

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(X)
    log_density = kde.score_samples(X)
    density = np.exp(log_density)
    return density.astype(np.float64)


def estimate_knn_density(coords: np.ndarray, k: int = 5) -> np.ndarray:
    """k-NN density: inverse of distance to k-th nearest neighbor.

    Higher value = denser (closer neighbors).
    """
    X = _validate_coords(coords)
    n_samples = X.shape[0]

    k = min(k, n_samples - 1)
    if k < 1:
        k = 1
        logger.warning("k-NN: k must be at least 1, using k=1 (all points same distance)")

    tree = KDTree(X)
    # k+1 because the first neighbor is the point itself (distance 0)
    k_actual = min(k + 1, n_samples)
    dists, _ = tree.query(X, k=k_actual, workers=-1)

    # Use k-th neighbor (index k, 0-based) or last column if fewer neighbors
    kth_dist = dists[:, -1] if dists.shape[1] > 1 else dists[:, 0]
    kth_dist = np.maximum(kth_dist, 1e-10)
    density = 1.0 / kth_dist
    return density.astype(np.float64)


def estimate_lof(coords: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    """Local Outlier Factor scores.

    Returns LOF values: higher = more outlier-like (lower local density).
    For consistency with 'higher = denser', we return negative LOF so that
    higher values indicate denser regions.
    """
    X = _validate_coords(coords)
    n_samples = X.shape[0]

    n_neighbors = min(n_neighbors, n_samples - 1)
    if n_neighbors < 2:
        n_neighbors = 2
        logger.warning(
            f"LOF n_neighbors too low for n_samples={n_samples}, using 2"
        )

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof.fit(X)
    scores = lof.negative_outlier_factor_
    # negative_outlier_factor_: more negative = more outlier
    # Invert: -scores so higher = less outlier = denser
    density = -scores.astype(np.float64)
    return density


def estimate_dbscan_density(coords: np.ndarray, eps: float = 0.5) -> np.ndarray:
    """DBSCAN-based density: count of points within eps-neighborhood.

    Higher count = denser. Uses sklearn's DBSCAN to get core point counts.
    """
    X = _validate_coords(coords)
    n_samples, n_features = X.shape

    tree = KDTree(X)
    counts = tree.query_ball_point(X, r=eps, return_length=True)
    density = np.array(counts, dtype=np.float64)
    return density


def estimate_gmm_density(coords: np.ndarray, n_components: int = 5) -> np.ndarray:
    """Gaussian Mixture Model log-likelihood at each point.

    Higher log-likelihood = denser (better fit to the mixture).
    We return exp(log_likelihood) scaled for interpretability.
    """
    X = _validate_coords(coords)
    n_samples, n_features = X.shape

    n_components = min(n_components, n_samples // 2)
    if n_components < 1:
        n_components = 1
        logger.warning(
            f"GMM: n_components reduced to 1 for n_samples={n_samples}"
        )

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42,
        covariance_type="full",
        max_iter=200,
    )
    gmm.fit(X)
    log_lik = gmm.score_samples(X)
    density = np.exp(log_lik).astype(np.float64)
    return density


def estimate_voronoi_density(coords: np.ndarray) -> np.ndarray:
    """Voronoi cell inverse-area density (2D only).

    Uses scipy.spatial.Voronoi. Density at each point = 1 / (Voronoi cell area).
    Points at the convex hull get infinite/unbounded cells; we cap by using
    the maximum finite area. For >2D, uses first two dimensions.
    """
    X = _validate_coords(coords)
    n_samples, n_features = X.shape

    if n_samples < 3:
        return np.ones(n_samples, dtype=np.float64)

    from scipy.spatial import Voronoi

    # Use 2D for Voronoi (cell area is defined in 2D)
    X2 = X[:, :2].astype(np.float64)
    vor = Voronoi(X2)

    # Per-point density = 1 / cell area (finite regions only)
    density = np.full(n_samples, np.nan, dtype=np.float64)
    for i in range(n_samples):
        reg_idx = vor.point_region[i]
        region = vor.regions[reg_idx]
        if -1 in region or len(region) < 3:
            continue
        try:
            verts = np.array([vor.vertices[j] for j in region])
            from scipy.spatial import ConvexHull
            hull = ConvexHull(verts)
            area = hull.volume  # in 2D, volume is area
            if area > 1e-20:
                density[i] = 1.0 / area
        except Exception:
            continue

    # Replace nan/inf with max finite density so all points have a value
    valid = np.isfinite(density) & (density > 0)
    if np.any(valid):
        max_d = np.nanmax(np.where(valid, density, 0))
        density = np.where(np.isfinite(density) & (density > 0), density, max_d)
    else:
        density = np.ones(n_samples, dtype=np.float64)
    return density.astype(np.float64)


def estimate_density(coords: np.ndarray, method: str = "kde", **kwargs) -> np.ndarray:
    """Estimate density at each point. Returns array of density values (higher = denser)."""
    method_lower = method.lower()
    if method_lower not in _ALL_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {_ALL_METHODS}"
        )

    dispatcher = {
        "kde": estimate_kde,
        "knn": estimate_knn_density,
        "lof": estimate_lof,
        "dbscan": estimate_dbscan_density,
        "gmm": estimate_gmm_density,
        "voronoi": estimate_voronoi_density,
    }
    fn = dispatcher[method_lower]
    return fn(coords, **kwargs)


def compute_redundancy_risk(density: np.ndarray) -> np.ndarray:
    """Compute redundancy risk index (0-1) based on density percentile.

    Higher density -> higher redundancy risk. Uses percentile rank.
    """
    d = np.asarray(density, dtype=np.float64)
    if d.size == 0:
        return d

    from scipy.stats import rankdata
    ranks = rankdata(d, method="average")
    pct = (ranks - 1) / (len(d) - 1) if len(d) > 1 else np.zeros_like(d)
    return pct.astype(np.float64)


def compare_density_methods(
    coords: np.ndarray,
    methods: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run multiple density methods and return results for comparison."""
    X = _validate_coords(coords)
    methods_to_run = methods or _ALL_METHODS.copy()

    results: dict[str, np.ndarray] = {}
    for m in methods_to_run:
        try:
            results[m] = estimate_density(X, method=m)
        except Exception as e:
            logger.warning(f"Density method '{m}' failed: {e}")
            results[m] = np.full(X.shape[0], np.nan)

    return results


def density_by_group(
    coords: np.ndarray,
    density: np.ndarray,
    groups: dict[str, list[int]],
) -> dict[str, dict]:
    """Compute density statistics per group (e.g., per domain).

    Returns {group: {mean, std, median, coverage_area}}.
    coverage_area: approximate convex hull area (2D) or volume proxy (higher dim).
    """
    X = _validate_coords(coords)
    d = np.asarray(density, dtype=np.float64)
    if len(d) != X.shape[0]:
        raise ValueError("density length must match coords rows")

    result: dict[str, dict] = {}
    for group_name, indices in groups.items():
        valid = [i for i in indices if 0 <= i < X.shape[0]]
        if not valid:
            result[group_name] = {
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "coverage_area": np.nan,
            }
            continue

        sub_d = d[valid]
        sub_coords = X[valid]

        mean_d = float(np.mean(sub_d))
        std_d = float(np.std(sub_d)) if len(sub_d) > 1 else 0.0
        median_d = float(np.median(sub_d))

        if sub_coords.shape[1] >= 2 and len(valid) >= 3:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(sub_coords[:, :2])
                coverage = float(hull.volume)
            except Exception:
                coverage = float(np.max(sub_coords[:, 0]) - np.min(sub_coords[:, 0]) + 1e-10)
        else:
            coverage = np.nan

        result[group_name] = {
            "mean": mean_d,
            "std": std_d,
            "median": median_d,
            "coverage_area": coverage,
        }

    return result


def intra_domain_density(
    embeddings: dict[str, np.ndarray],
    groups: dict[str, list[str]],
    item_counts: dict[str, int],
) -> dict[str, dict]:
    """Compute intra-domain density metrics in the original high-dim embedding space.

    For each domain, computes pairwise cosine similarity among its constructs
    and derives dispersion statistics.

    Args:
        embeddings: {construct_id: L2-normalized embedding vector}.
        groups: {domain_name: [construct_id, ...]}.
        item_counts: {construct_id: number_of_items}.

    Returns:
        {domain_name: {construct_count, item_count, mean_pairwise_cosine_sim,
         std_pairwise_cosine_sim, min_pairwise_cosine_sim, max_pairwise_cosine_sim}}.
    """
    result: dict[str, dict] = {}

    for domain_name, construct_ids in groups.items():
        valid_ids = [cid for cid in construct_ids if cid in embeddings]
        n = len(valid_ids)
        total_items = sum(item_counts.get(cid, 0) for cid in construct_ids)

        if n < 2:
            result[domain_name] = {
                "construct_count": n,
                "item_count": total_items,
                "mean_pairwise_cosine_sim": np.nan,
                "std_pairwise_cosine_sim": np.nan,
                "min_pairwise_cosine_sim": np.nan,
                "max_pairwise_cosine_sim": np.nan,
            }
            continue

        vecs = np.array([embeddings[cid] for cid in valid_ids], dtype=np.float64)
        sim_matrix = cosine_similarity(vecs)

        triu_idx = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_idx]

        result[domain_name] = {
            "construct_count": n,
            "item_count": total_items,
            "mean_pairwise_cosine_sim": float(np.mean(pairwise_sims)),
            "std_pairwise_cosine_sim": float(np.std(pairwise_sims)),
            "min_pairwise_cosine_sim": float(np.min(pairwise_sims)),
            "max_pairwise_cosine_sim": float(np.max(pairwise_sims)),
        }

    return result
