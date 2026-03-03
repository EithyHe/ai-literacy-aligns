"""Construct exploration service (PRD FR-4.1.1–4.1.6, FR-2.1.1–2.1.3).

Provides semantic exploration of constructs, hierarchical lookup (domain/construct/semantic),
neighborhood summaries, and iterative exploration support.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D for similarity computations."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine similarity consistency."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return embeddings / norms


def explore_construct(
    query_text: str,
    encode_fn: Callable[[list[str]], np.ndarray],
    construct_embeddings: dict[str, np.ndarray],
    construct_metadata: dict[str, dict],
    top_n: int = 10,
) -> dict:
    """Explore constructs semantically similar to a natural language description.

    Args:
        query_text: Natural language description of the construct to explore.
        encode_fn: Function that takes list[str] and returns np.ndarray of embeddings.
        construct_embeddings: {construct_id: embedding_vector}
        construct_metadata: {construct_id: {name, framework, domain, item_count, ...}}
        top_n: Number of neighbors to return.

    Returns:
        dict with keys:
            - query_embedding: np.ndarray
            - neighbors: [{construct_id, name, framework, domain, similarity, item_count}, ...]
            - density_percentile: float (how crowded the area is, 0–100)
            - novelty_assessment: 'high_novelty' | 'moderate' | 'low_novelty'
    """
    if not construct_embeddings:
        return {
            "query_embedding": np.array([]),
            "neighbors": [],
            "density_percentile": 0.0,
            "novelty_assessment": "high_novelty",
        }

    try:
        query_emb = encode_fn([query_text.strip() or " "])
        query_emb = _ensure_2d(query_emb)
        query_emb = _normalize(query_emb)
    except Exception as e:
        logger.warning("Failed to encode query: %s", e)
        return {
            "query_embedding": np.array([]),
            "neighbors": [],
            "density_percentile": 0.0,
            "novelty_assessment": "moderate",
        }

    ids = list(construct_embeddings.keys())
    emb_matrix = np.vstack([construct_embeddings[cid] for cid in ids])
    emb_matrix = _normalize(emb_matrix)

    sims = cosine_similarity(query_emb, emb_matrix)[0]
    top_indices = np.argsort(sims)[::-1][:top_n]

    neighbors = []
    for idx in top_indices:
        cid = ids[idx]
        meta = construct_metadata.get(cid, {})
        neighbors.append(
            {
                "construct_id": cid,
                "name": meta.get("name", cid),
                "framework": meta.get("framework", ""),
                "domain": meta.get("domain", ""),
                "similarity": float(sims[idx]),
                "item_count": meta.get("item_count", 0),
            }
        )

    # Density: mean distance from query to all constructs (inverted: lower distance = more crowded)
    distances = 1.0 - sims
    mean_dist = float(np.mean(distances))
    # Percentile: compare to distribution of pairwise distances in corpus
    if len(ids) >= 2:
        pair_dists = []
        emb_arr = np.array([construct_embeddings[cid] for cid in ids])
        emb_arr = _normalize(emb_arr)
        sim_matrix = cosine_similarity(emb_arr, emb_arr)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair_dists.append(1.0 - sim_matrix[i, j])
        pair_dists = np.array(pair_dists)
        density_percentile = float(np.mean(pair_dists <= mean_dist) * 100)
    else:
        density_percentile = 50.0

    # Novelty: based on nearest neighbor distance (higher = more novel)
    max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
    nearest_dist = 1.0 - max_sim
    if nearest_dist >= 0.3:
        novelty_assessment = "high_novelty"
    elif nearest_dist >= 0.15:
        novelty_assessment = "moderate"
    else:
        novelty_assessment = "low_novelty"

    return {
        "query_embedding": query_emb[0],
        "neighbors": neighbors,
        "density_percentile": density_percentile,
        "novelty_assessment": novelty_assessment,
    }


def hierarchical_lookup(
    query: str,
    db_search_fn: Callable[[str, str], list[dict]],
    encode_fn: Callable[[list[str]], np.ndarray],
    construct_embeddings: dict[str, np.ndarray],
    construct_metadata: dict[str, dict] | None = None,
) -> dict:
    """Three-level fallback lookup (PRD 4.2):

    1. Exact match on Domain name
    2. Exact match on Construct name
    3. Semantic search via embeddings

    Args:
        query: User input (could be domain name, construct name, or free text).
        db_search_fn: Function to search DB by name. Signature: (table, name) -> list[dict].
            Tables: 'domains', 'constructs'. Returns list of matching entities.
        encode_fn: Function to encode text to embedding.
        construct_embeddings: All construct embeddings.
        construct_metadata: Optional {construct_id: metadata} for domain/construct info.

    Returns:
        dict with keys:
            - match_type: 'domain' | 'construct' | 'semantic'
            - matched_entity: dict (the matched domain/construct info)
            - related_constructs: list[dict] (constructs in scope for analysis)
    """
    query_clean = (query or "").strip()
    construct_metadata = construct_metadata or {}

    # 1. Exact match on Domain name
    try:
        domain_matches = db_search_fn("domains", query_clean)
        for d in domain_matches:
            if (d.get("name") or "").strip().lower() == query_clean.lower():
                domain_id = d.get("domain_id")
                if domain_id:
                    related = _get_constructs_for_domain(domain_id, db_search_fn, construct_embeddings, construct_metadata)
                    return {
                        "match_type": "domain",
                        "matched_entity": dict(d),
                        "related_constructs": related,
                    }
    except Exception as e:
        logger.debug("Domain search failed: %s", e)

    # 2. Exact match on Construct name
    try:
        construct_matches = db_search_fn("constructs", query_clean)
        for c in construct_matches:
            if (c.get("name") or "").strip().lower() == query_clean.lower():
                cid = c.get("construct_id")
                if cid and cid in construct_embeddings:
                    meta = construct_metadata.get(cid, {})
                    related = [{
                        "construct_id": cid,
                        "name": c.get("name", cid),
                        "framework": meta.get("framework", ""),
                        "domain": meta.get("domain", ""),
                        "item_count": meta.get("item_count", c.get("item_count", 0)),
                    }]
                    return {
                        "match_type": "construct",
                        "matched_entity": dict(c),
                        "related_constructs": related,
                    }
    except Exception as e:
        logger.debug("Construct search failed: %s", e)

    # 3. Semantic search
    if not construct_embeddings or not query_clean:
        return {
            "match_type": "semantic",
            "matched_entity": {},
            "related_constructs": [],
        }

    try:
        query_emb = encode_fn([query_clean])
        query_emb = _ensure_2d(query_emb)
        query_emb = _normalize(query_emb)

        ids = list(construct_embeddings.keys())
        emb_matrix = np.vstack([construct_embeddings[cid] for cid in ids])
        emb_matrix = _normalize(emb_matrix)

        sims = cosine_similarity(query_emb, emb_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:10]

        related = []
        for idx in top_indices:
            cid = ids[idx]
            meta = construct_metadata.get(cid, {})
            related.append({
                "construct_id": cid,
                "name": meta.get("name", cid),
                "framework": meta.get("framework", ""),
                "domain": meta.get("domain", ""),
                "similarity": float(sims[idx]),
                "item_count": meta.get("item_count", 0),
            })

        return {
            "match_type": "semantic",
            "matched_entity": {"query": query_clean, "top_similarity": float(sims[top_indices[0]]) if len(top_indices) > 0 else 0},
            "related_constructs": related,
        }
    except Exception as e:
        logger.warning("Semantic search failed: %s", e)
        return {
            "match_type": "semantic",
            "matched_entity": {},
            "related_constructs": [],
        }


def _get_constructs_for_domain(
    domain_id: str,
    db_search_fn: Callable,
    construct_embeddings: dict[str, np.ndarray],
    construct_metadata: dict[str, dict],
) -> list[dict]:
    """Get constructs linked to a domain. Uses schema if available."""
    try:
        from ssn.db.schema import get_constructs_by_domain

        constructs = get_constructs_by_domain(domain_id)
    except ImportError:
        return []
    except Exception as e:
        logger.debug("get_constructs_by_domain failed: %s", e)
        return []

    result = []
    for c in constructs:
        cid = c.get("construct_id")
        if cid and cid in construct_embeddings:
            meta = construct_metadata.get(cid, {})
            result.append({
                "construct_id": cid,
                "name": c.get("name", cid),
                "framework": meta.get("framework", ""),
                "domain": meta.get("domain", domain_id),
                "item_count": meta.get("item_count", c.get("item_count", 0)),
            })
    return result


def nearest_constructs_for_id(
    construct_id: str,
    construct_embeddings: dict[str, np.ndarray],
    construct_metadata: dict[str, dict],
    top_n: int = 15,
) -> list[dict]:
    """Return nearest other constructs by embedding similarity (excludes the given construct).

    Used when user has exact-matched a construct; show semantically nearest others.
    """
    if construct_id not in construct_embeddings or not construct_embeddings:
        return []
    ids = [cid for cid in construct_embeddings if cid != construct_id]
    if not ids:
        return []
    ref_emb = _ensure_2d(construct_embeddings[construct_id])
    ref_emb = _normalize(ref_emb)
    emb_matrix = np.vstack([construct_embeddings[cid] for cid in ids])
    emb_matrix = _normalize(emb_matrix)
    sims = cosine_similarity(ref_emb, emb_matrix)[0]
    top_indices = np.argsort(sims)[::-1][:top_n]
    neighbors = []
    for idx in top_indices:
        cid = ids[idx]
        meta = construct_metadata.get(cid, {})
        neighbors.append({
            "construct_id": cid,
            "name": meta.get("name", cid),
            "framework": meta.get("framework", ""),
            "domain": meta.get("domain", ""),
            "similarity": float(sims[idx]),
            "item_count": meta.get("item_count", 0),
        })
    return neighbors


def compute_neighborhood_summary(
    query_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    all_ids: list[str],
    k: int = 10,
) -> dict:
    """Compute neighborhood summary report (PRD FR-4.1.4).

    Args:
        query_embedding: 1D embedding of the query.
        all_embeddings: 2D array of corpus embeddings.
        all_ids: List of IDs aligned with all_embeddings.
        k: Number of nearest neighbors to consider.

    Returns:
        dict with keys:
            - nearest_distance: float
            - mean_distance_top_k: float
            - distance_percentile: float (percentile rank vs corpus)
            - density_assessment: str
            - novelty_score: float (0–1, higher = more novel)
    """
    if all_embeddings.size == 0:
        return {
            "nearest_distance": 1.0,
            "mean_distance_top_k": 1.0,
            "distance_percentile": 100.0,
            "density_assessment": "empty_corpus",
            "novelty_score": 1.0,
        }

    query_2d = _ensure_2d(query_embedding)
    corpus_norm = _normalize(np.asarray(all_embeddings, dtype=np.float32))

    sims = cosine_similarity(query_2d, corpus_norm)[0]
    distances = 1.0 - sims

    top_k = min(k, len(distances))
    top_indices = np.argsort(distances)[:top_k]

    nearest_dist = float(distances[top_indices[0]]) if top_k > 0 else 1.0
    mean_dist_top_k = float(np.mean(distances[top_indices])) if top_k > 0 else 1.0

    # Distance percentile: compare query's nearest distance to corpus self-distances
    n = corpus_norm.shape[0]
    if n >= 2:
        corpus_sims = cosine_similarity(corpus_norm, corpus_norm)
        np.fill_diagonal(corpus_sims, -np.inf)
        max_sim_per_row = np.max(corpus_sims, axis=1)
        corpus_nn_distances = 1.0 - max_sim_per_row
        distance_percentile = float(np.mean(corpus_nn_distances <= nearest_dist) * 100)
    else:
        distance_percentile = 50.0

    # Density assessment
    if mean_dist_top_k >= 0.4:
        density_assessment = "sparse"
    elif mean_dist_top_k >= 0.2:
        density_assessment = "moderate"
    else:
        density_assessment = "dense"

    # Novelty: 1 - max_similarity (higher distance = more novel), scaled to 0–1
    max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
    novelty_score = 1.0 - max_sim

    return {
        "nearest_distance": nearest_dist,
        "mean_distance_top_k": mean_dist_top_k,
        "distance_percentile": distance_percentile,
        "density_assessment": density_assessment,
        "novelty_score": novelty_score,
    }


def iterative_explore(
    descriptions: list[str],
    encode_fn: Callable[[list[str]], np.ndarray],
    construct_embeddings: dict[str, np.ndarray],
    construct_metadata: dict[str, dict] | None = None,
    top_n: int = 10,
) -> list[dict]:
    """Support iterative exploration with multiple descriptions.

    Returns list of exploration results, one per description, with position change
    tracking for constructs across iterations.

    Args:
        descriptions: List of natural language descriptions to explore.
        encode_fn: Function to encode text to embeddings.
        construct_embeddings: {construct_id: embedding}
        construct_metadata: Optional {construct_id: metadata}
        top_n: Number of neighbors per exploration.

    Returns:
        List of exploration result dicts (same structure as explore_construct),
        with an optional 'position_changes' key comparing to previous result.
    """
    construct_metadata = construct_metadata or {}
    results = []

    for i, desc in enumerate(descriptions):
        res = explore_construct(
            desc,
            encode_fn,
            construct_embeddings,
            construct_metadata,
            top_n=top_n,
        )
        res["description"] = desc
        res["iteration"] = i

        if i > 0 and results:
            prev_neighbors = {n["construct_id"]: idx for idx, n in enumerate(results[-1]["neighbors"])}
            position_changes = []
            for idx, n in enumerate(res["neighbors"]):
                cid = n["construct_id"]
                prev_rank = prev_neighbors.get(cid)
                if prev_rank is not None and prev_rank != idx:
                    position_changes.append({
                        "construct_id": cid,
                        "name": n.get("name", cid),
                        "previous_rank": prev_rank + 1,
                        "current_rank": idx + 1,
                        "change": idx - prev_rank,
                    })
            res["position_changes"] = position_changes

        results.append(res)

    return results
