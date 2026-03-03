"""Item-level redundancy diagnosis service (PRD FR-4.2.1–4.2.5).

Provides functions to diagnose item redundancy against a corpus, find redundant pairs
within scales, compute uniqueness metrics, and generate LLM-based modification suggestions.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

THRESHOLDS = {"high": 0.85, "medium": 0.65}


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D for cosine_similarity."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine similarity consistency."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return embeddings / norms


def diagnose_item(
    item_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: list[str],
    corpus_texts: list[str],
    top_k: int = 3,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Diagnose a single item against the corpus.

    Args:
        item_embedding: 1D embedding vector of the item.
        corpus_embeddings: 2D array of corpus embeddings (n_samples, n_features).
        corpus_ids: List of corpus item IDs aligned with corpus_embeddings.
        corpus_texts: List of corpus item texts aligned with corpus_embeddings.
        top_k: Number of top similar matches to return.
        thresholds: Optional dict with 'high' and 'medium' keys. Defaults to THRESHOLDS.

    Returns:
        dict with keys:
            - top_matches: [{id, text, similarity, rank}, ...]
            - risk_level: 'high' | 'medium' | 'low'
            - max_similarity: float
            - risk_color: '#FF4444' | '#FFAA00' | '#44BB44'
    """
    thresholds = thresholds or THRESHOLDS
    high_thresh = thresholds.get("high", THRESHOLDS["high"])
    medium_thresh = thresholds.get("medium", THRESHOLDS["medium"])

    if corpus_embeddings.size == 0 or len(corpus_ids) == 0:
        return {
            "top_matches": [],
            "risk_level": "low",
            "max_similarity": 0.0,
            "risk_color": "#44BB44",
        }

    item_2d = _ensure_2d(item_embedding)
    corpus_norm = _normalize(np.asarray(corpus_embeddings, dtype=np.float32))

    sims = cosine_similarity(item_2d, corpus_norm)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    top_matches = []
    for rank, idx in enumerate(top_indices, start=1):
        top_matches.append(
            {
                "id": corpus_ids[idx],
                "text": corpus_texts[idx] if idx < len(corpus_texts) else "",
                "similarity": float(sims[idx]),
                "rank": rank,
            }
        )

    max_sim = float(np.max(sims))
    if max_sim >= high_thresh:
        risk_level = "high"
        risk_color = "#FF4444"
    elif max_sim >= medium_thresh:
        risk_level = "medium"
        risk_color = "#FFAA00"
    else:
        risk_level = "low"
        risk_color = "#44BB44"

    return {
        "top_matches": top_matches,
        "risk_level": risk_level,
        "max_similarity": max_sim,
        "risk_color": risk_color,
    }


def diagnose_scale(
    item_embeddings: np.ndarray,
    item_texts: list[str],
    corpus_embeddings: np.ndarray,
    corpus_ids: list[str],
    corpus_texts: list[str],
    top_k: int = 3,
    thresholds: dict | None = None,
) -> dict:
    """Diagnose all items in a scale against the corpus.

    Args:
        item_embeddings: 2D array of item embeddings (n_items, n_features).
        item_texts: List of item texts aligned with item_embeddings.
        corpus_embeddings: 2D array of corpus embeddings.
        corpus_ids: List of corpus item IDs.
        corpus_texts: List of corpus item texts.
        top_k: Number of top matches per item.
        thresholds: Optional similarity thresholds dict.

    Returns:
        dict with keys:
            - items: [diagnosis_dict_per_item, ...]
            - summary: {total, high_risk, medium_risk, low_risk, overall_risk_score}
    """
    thresholds = thresholds or THRESHOLDS
    items_result = []
    high_count = medium_count = low_count = 0

    n_items = len(item_embeddings)
    if n_items == 0:
        return {
            "items": [],
            "summary": {
                "total": 0,
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "overall_risk_score": 0.0,
            },
        }

    for i in range(n_items):
        emb = item_embeddings[i]
        text = item_texts[i] if i < len(item_texts) else ""
        diag = diagnose_item(
            emb, corpus_embeddings, corpus_ids, corpus_texts, top_k=top_k, thresholds=thresholds
        )
        diag["item_text"] = text
        diag["item_index"] = i
        items_result.append(diag)

        if diag["risk_level"] == "high":
            high_count += 1
        elif diag["risk_level"] == "medium":
            medium_count += 1
        else:
            low_count += 1

    # overall_risk_score: 0–1, higher = more risky
    max_sims = [d["max_similarity"] for d in items_result]
    overall_risk_score = float(np.mean(max_sims)) if max_sims else 0.0

    return {
        "items": items_result,
        "summary": {
            "total": n_items,
            "high_risk": high_count,
            "medium_risk": medium_count,
            "low_risk": low_count,
            "overall_risk_score": overall_risk_score,
        },
    }


def generate_modification_suggestion(
    item_text: str,
    similar_items: list[dict],
    construct_description: str = "",
) -> str:
    """Generate LLM-based modification suggestion for a redundant item.

    Uses OpenAI API (gpt-4o) to analyze overlap with similar items and suggest
    modifications. Falls back to a non-LLM description if API key is unavailable.

    Args:
        item_text: The redundant item text.
        similar_items: List of dicts with 'text' and optionally 'similarity', 'id'.
        construct_description: Optional construct context for the suggestion.

    Returns:
        Suggestion text string.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not api_key.strip():
        logger.warning("OPENAI_API_KEY not set; returning fallback suggestion")
        return _fallback_modification_suggestion(item_text, similar_items, construct_description)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
    except ImportError as e:
        logger.warning("openai package not available: %s", e)
        return _fallback_modification_suggestion(item_text, similar_items, construct_description)

    similar_texts = []
    for s in similar_items[:5]:
        t = s.get("text", s.get("item_text", ""))
        sim = s.get("similarity", 0)
        if t:
            similar_texts.append(f"- {t} (similarity: {sim:.2f})" if sim else f"- {t}")

    prompt = f"""You are an expert in psychological scale development. An item in a scale appears redundant (too similar) to existing items in the corpus.

**Item to modify:**
{item_text}

**Similar items in the corpus:**
{chr(10).join(similar_texts) if similar_texts else "(none provided)"}
"""
    if construct_description:
        prompt += f"\n**Construct context:** {construct_description}\n"

    prompt += """
Provide a concise modification suggestion (2–4 sentences) to make this item more distinct while preserving its intended measurement of the construct. Focus on:
1. What semantic overlap exists with the similar items
2. Specific wording changes or alternative phrasings
3. How to retain construct validity while reducing redundancy
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        suggestion = response.choices[0].message.content or ""
        return suggestion.strip() if suggestion else _fallback_modification_suggestion(
            item_text, similar_items, construct_description
        )
    except Exception as e:
        logger.warning("OpenAI API call failed: %s", e)
        return _fallback_modification_suggestion(item_text, similar_items, construct_description)


def _fallback_modification_suggestion(
    item_text: str,
    similar_items: list[dict],
    construct_description: str = "",
) -> str:
    """Fallback suggestion when LLM is unavailable."""
    lines = [
        f"The item \"{item_text}\" overlaps semantically with the following items:",
        "",
    ]
    for i, s in enumerate(similar_items[:5], 1):
        t = s.get("text", s.get("item_text", ""))
        sim = s.get("similarity", 0)
        if t:
            lines.append(f"  {i}. {t} (similarity: {sim:.2f})" if sim else f"  {i}. {t}")
    lines.extend([
        "",
        "Consider rephrasing to emphasize a distinct aspect of the construct, using different vocabulary, or targeting a different behavioral manifestation. Without LLM analysis, run with OPENAI_API_KEY set for detailed suggestions.",
    ])
    return "\n".join(lines)


def find_redundant_pairs(
    embeddings: np.ndarray,
    labels: list[str],
    threshold: float = 0.90,
) -> list[dict]:
    """Find pairs of items within the same scale that are too similar.

    Args:
        embeddings: 2D array of item embeddings (n_items, n_features).
        labels: List of item identifiers (e.g., item_id or text) aligned with embeddings.
        threshold: Similarity threshold above which a pair is considered redundant.

    Returns:
        List of dicts: [{item_a, item_b, similarity}, ...], sorted by similarity descending.
    """
    if embeddings.size == 0 or len(labels) < 2:
        return []

    emb_norm = _normalize(np.asarray(embeddings, dtype=np.float32))
    sim_matrix = cosine_similarity(emb_norm, emb_norm)

    pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                pairs.append(
                    {
                        "item_a": labels[i],
                        "item_b": labels[j],
                        "similarity": sim,
                    }
                )

    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs


def compute_item_uniqueness(
    item_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
) -> dict:
    """Compute uniqueness metrics for an item: percentile rank of nearest-neighbor distance vs corpus distribution.

    Uses cosine distance (1 - cosine_similarity). Higher percentile = more unique
    (farther from nearest neighbor relative to the corpus distribution).

    Args:
        item_embedding: 1D embedding of the item.
        corpus_embeddings: 2D array of corpus embeddings.

    Returns:
        dict with keys:
            - nearest_distance: float (1 - max cosine similarity)
            - percentile_rank: float 0–100 (higher = more unique)
            - mean_corpus_nn_distance: float (mean nearest-neighbor distance in corpus)
    """
    if corpus_embeddings.size == 0:
        return {
            "nearest_distance": 1.0,
            "percentile_rank": 100.0,
            "mean_corpus_nn_distance": 0.0,
        }

    item_2d = _ensure_2d(item_embedding)
    corpus_norm = _normalize(np.asarray(corpus_embeddings, dtype=np.float32))

    # Distance from item to each corpus vector (1 - cosine_sim)
    sims = cosine_similarity(item_2d, corpus_norm)[0]
    distances = 1.0 - sims
    nearest_dist = float(np.min(distances))

    # Corpus self-distances: for each corpus vector, distance to its nearest neighbor (excluding self)
    n_corpus = corpus_norm.shape[0]
    corpus_sims = cosine_similarity(corpus_norm, corpus_norm)
    np.fill_diagonal(corpus_sims, -np.inf)
    max_sim_per_row = np.max(corpus_sims, axis=1)
    corpus_nn_distances = 1.0 - max_sim_per_row
    mean_corpus_nn = float(np.mean(corpus_nn_distances))

    # Percentile: what fraction of corpus items have nn_distance <= nearest_dist
    percentile = float(np.mean(corpus_nn_distances <= nearest_dist) * 100)

    return {
        "nearest_distance": nearest_dist,
        "percentile_rank": percentile,
        "mean_corpus_nn_distance": mean_corpus_nn,
    }
