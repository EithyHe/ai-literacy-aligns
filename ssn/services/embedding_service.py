"""Embedding service for the Semantic Scale Network.

Loads pre-computed MPNet embeddings, builds FAISS indices for fast similarity search,
and supports encoding new text via sentence-transformers or OpenAI API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ssn.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    IPIP_EMBEDDINGS_NPY,
    IPIP_ITEM_IDS_CSV,
    OPENAI_API_KEY,
)

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)

# Sentence-transformers model for IPIP/personality items (matches pre-computed embeddings)
MPNET_MODEL_NAME = "dwulff/mpnet-personality"



_item_embeddings_cache: tuple[np.ndarray, list[str]] | None = None


def load_item_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load item embeddings and their IDs. Returns (embeddings_matrix, item_id_list).

    Embeddings are loaded from the pre-computed .npy file; item IDs from the CSV.
    Row order is aligned: embeddings[i] corresponds to item_id_list[i].
    Results are cached for the session.
    """
    global _item_embeddings_cache
    if _item_embeddings_cache is not None:
        return _item_embeddings_cache

    import pandas as pd

    if not IPIP_EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {IPIP_EMBEDDINGS_NPY}. "
            "Run the embedding pipeline to generate it."
        )
    if not IPIP_ITEM_IDS_CSV.exists():
        raise FileNotFoundError(
            f"Item IDs file not found: {IPIP_ITEM_IDS_CSV}. "
            "Run the embedding pipeline to generate it."
        )

    embeddings = np.load(IPIP_EMBEDDINGS_NPY)
    df = pd.read_csv(IPIP_ITEM_IDS_CSV)

    if "item_id" not in df.columns:
        raise ValueError(
            f"Expected 'item_id' column in {IPIP_ITEM_IDS_CSV}. "
            f"Found columns: {list(df.columns)}"
        )

    item_ids = df["item_id"].astype(str).tolist()

    if len(embeddings) != len(item_ids):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match item ID count ({len(item_ids)}). "
            "Ensure embeddings and item IDs are from the same pipeline run."
        )

    if embeddings.shape[1] != EMBEDDING_DIM:
        logger.warning(
            f"Embedding dimension ({embeddings.shape[1]}) differs from config EMBEDDING_DIM ({EMBEDDING_DIM}). "
            "Using actual dimension."
        )

    logger.info(f"Loaded {len(item_ids)} item embeddings from {IPIP_EMBEDDINGS_NPY}")
    _item_embeddings_cache = (embeddings, item_ids)
    return _item_embeddings_cache


def get_construct_embeddings() -> dict[str, np.ndarray]:
    """Return {construct_id: mean_embedding_vector} by averaging item embeddings.

    Uses the DB schema to map items to constructs. Only items with embeddings
    in the pre-computed set are included. Constructs with no such items are omitted.
    """
    from ssn.db.schema import get_all_items

    embeddings, item_ids = load_item_embeddings()
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    all_items = get_all_items()
    construct_to_item_ids: dict[str, list[str]] = {}
    for item in all_items:
        cid = item.get("construct_id")
        iid = item.get("item_id")
        if not cid or not iid:
            continue
        if cid not in construct_to_item_ids:
            construct_to_item_ids[cid] = []
        construct_to_item_ids[cid].append(str(iid))

    result: dict[str, np.ndarray] = {}
    for construct_id, ids in construct_to_item_ids.items():
        indices = [item_id_to_idx[iid] for iid in ids if iid in item_id_to_idx]
        if not indices:
            logger.debug(f"Construct {construct_id} has no items with embeddings, skipping")
            continue
        vecs = embeddings[indices]
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        # L2-normalize for cosine similarity consistency
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        result[construct_id] = mean_vec

    logger.info(f"Computed embeddings for {len(result)} constructs")
    return result


def get_domain_embeddings() -> dict[str, np.ndarray]:
    """Return {domain_id: mean_embedding_vector} by averaging construct embeddings.

    Uses the DB schema to map constructs to domains via domain_construct_map.
    Domains with no constructs (or no construct embeddings) are omitted.
    """
    from ssn.db.schema import get_all_domains, get_constructs_by_domain

    construct_embs = get_construct_embeddings()
    domains = get_all_domains()

    result: dict[str, np.ndarray] = {}
    for d in domains:
        domain_id = d.get("domain_id")
        if not domain_id:
            continue
        constructs = get_constructs_by_domain(domain_id)
        vecs = [
            construct_embs[c["construct_id"]]
            for c in constructs
            if c.get("construct_id") in construct_embs
        ]
        if not vecs:
            logger.debug(f"Domain {domain_id} has no construct embeddings, skipping")
            continue
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        result[domain_id] = mean_vec

    logger.info(f"Computed embeddings for {len(result)} domains")
    return result


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """Build a FAISS IndexFlatIP (inner product / cosine) index.

    Embeddings are L2-normalized before indexing so that inner product equals
    cosine similarity. Expects float32 embeddings of shape (n, dim).
    """
    import faiss

    if embeddings.size == 0:
        raise ValueError("Cannot build FAISS index from empty embeddings")

    emb = np.ascontiguousarray(embeddings.astype(np.float32))
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    logger.info(f"Built FAISS IndexFlatIP with {index.ntotal} vectors, dim={dim}")
    return index


def load_or_build_faiss_index() -> tuple["faiss.Index", list[str]]:
    """Load cached FAISS index or build from item embeddings.

    Returns (index, item_id_list). The item_id_list order matches the index vectors.
    """
    import faiss

    embeddings, item_ids = load_item_embeddings()

    if FAISS_INDEX_PATH.exists():
        try:
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            if index.ntotal == len(item_ids):
                logger.info(f"Loaded FAISS index from {FAISS_INDEX_PATH} ({index.ntotal} vectors)")
                return index, item_ids
            logger.warning(
                f"Cached index has {index.ntotal} vectors but item_ids has {len(item_ids)}. Rebuilding."
            )
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}. Rebuilding.")

    index = build_faiss_index(embeddings)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")
    return index, item_ids


# Module-level cache for index and item_ids (avoids repeated disk reads)
_faiss_index: "faiss.Index | None" = None
_faiss_item_ids: list[str] | None = None


def _get_faiss_index_and_ids() -> tuple["faiss.Index", list[str]]:
    """Return cached (index, item_ids) or load/build."""
    global _faiss_index, _faiss_item_ids
    if _faiss_index is not None and _faiss_item_ids is not None:
        return _faiss_index, _faiss_item_ids
    _faiss_index, _faiss_item_ids = load_or_build_faiss_index()
    return _faiss_index, _faiss_item_ids


def search_similar(query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
    """Search FAISS index for most similar items. Returns [(item_id, similarity_score), ...].

    Similarity is cosine similarity (inner product of L2-normalized vectors).
    query_embedding must be 1D and match the index dimension (768 for MPNet).
    """
    import faiss

    index, item_ids = _get_faiss_index_and_ids()
    dim = index.d

    q = np.asarray(query_embedding, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.shape[1] != dim:
        raise ValueError(
            f"Query embedding dimension ({q.shape[1]}) does not match index dimension ({dim}). "
            "Use MPNet (768-dim) for compatibility with the FAISS index."
        )

    faiss.normalize_L2(q)
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []

    scores, indices = index.search(q, k)
    results: list[tuple[str, float]] = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            results.append((item_ids[idx], float(scores[0][i])))
    return results


def encode_texts(texts: list[str], use_openai: bool = False) -> np.ndarray:
    """Encode text(s) to embedding vectors using MPNet or OpenAI.

    Args:
        texts: List of strings to encode. Empty strings are replaced with zero vectors.
        use_openai: If True, use OpenAI API (requires OPENAI_API_KEY). Produces 1536-dim
            vectors - not compatible with search_similar/FAISS. Use only when MPNet unavailable.

    Returns:
        np.ndarray of shape (len(texts), dim). Dim is 768 for MPNet, 1536 for OpenAI.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    if use_openai:
        return _encode_openai(texts)

    return _encode_mpnet(texts)


def _encode_mpnet(texts: list[str]) -> np.ndarray:
    """Encode texts with sentence-transformers (MPNet)."""
    from sentence_transformers import SentenceTransformer

    valid_indices = [i for i, t in enumerate(texts) if t and str(t).strip()]
    valid_texts = [str(texts[i]).strip() for i in valid_indices]

    if not valid_texts:
        return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    model = SentenceTransformer(MPNET_MODEL_NAME)
    emb = model.encode(
        valid_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    for j, i in enumerate(valid_indices):
        out[i] = emb[j]
    return out


def _encode_openai(texts: list[str]) -> np.ndarray:
    """Encode texts with OpenAI API. Returns 1536-dim vectors."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for OpenAI embeddings. pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set. Set the environment variable or use use_openai=False for MPNet."
        )

    valid_indices = [i for i, t in enumerate(texts) if t and str(t).strip()]
    valid_texts = [str(texts[i]).strip() for i in valid_indices]

    if not valid_texts:
        return np.zeros((len(texts), 1536), dtype=np.float32)

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=valid_texts,
    )
    data = sorted(resp.data, key=lambda x: x.index)
    emb = np.array([d.embedding for d in data], dtype=np.float32)

    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    for j, i in enumerate(valid_indices):
        out[i] = emb[j]
    return out


def encode_scale(item_texts: list[str], method: str = "mean") -> np.ndarray:
    """Encode a full scale (list of items) to a single vector. Methods: mean, concat_encode.

    Args:
        item_texts: List of item text strings.
        method: 'mean' - average embeddings of each item. 'concat_encode' - concatenate
            all texts (with newlines) and encode as one string.

    Returns:
        Single embedding vector of shape (dim,).
    """
    if not item_texts:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    valid = [t for t in item_texts if t and str(t).strip()]
    if not valid:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if method == "mean":
        emb = encode_texts(valid, use_openai=False)
        vec = np.mean(emb, axis=0).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    if method == "concat_encode":
        combined = "\n".join(valid)
        emb = encode_texts([combined], use_openai=False)
        return emb[0]

    raise ValueError(f"Unknown method: {method}. Use 'mean' or 'concat_encode'.")
