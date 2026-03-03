"""Shared helper for computing user construct neighborhood.

Used by both the Construct Network and Density Analysis pages in Focused mode.
"""

from __future__ import annotations

import logging

import numpy as np

from ssn.services.embedding_service import encode_scale, get_construct_embeddings
from ssn.services.similarity_service import find_scale_neighbors

logger = logging.getLogger(__name__)

USER_NODE_ID = "__user__"


def get_user_neighborhood(
    user_items: list[str],
    top_k: int = 20,
) -> dict:
    """Encode user scale items and find the top-K most similar corpus constructs.

    Returns:
        {
            "user_embedding": np.ndarray of shape (dim,),
            "neighbors": list[dict] with keys id, similarity, rank,
            "sub_embeddings": dict[str, np.ndarray] containing top-K corpus
                construct embeddings plus the user vector keyed as USER_NODE_ID,
        }
    Raises ValueError if user_items is empty or encoding fails.
    """
    valid = [t for t in user_items if t and str(t).strip()]
    if not valid:
        raise ValueError("No valid item texts provided.")

    user_vec = encode_scale(valid, method="mean")
    if np.linalg.norm(user_vec) == 0:
        raise ValueError("Encoding produced a zero vector. Check item texts.")

    all_embs = get_construct_embeddings()
    if not all_embs:
        raise ValueError("No corpus construct embeddings available.")

    neighbors = find_scale_neighbors(user_vec, all_embs, top_n=top_k)

    sub_embeddings: dict[str, np.ndarray] = {USER_NODE_ID: user_vec}
    for n in neighbors:
        cid = n["id"]
        if cid in all_embs:
            sub_embeddings[cid] = all_embs[cid]

    logger.info(
        "User neighborhood: %d neighbors, sub_embeddings size %d",
        len(neighbors),
        len(sub_embeddings),
    )
    return {
        "user_embedding": user_vec,
        "neighbors": neighbors,
        "sub_embeddings": sub_embeddings,
    }
