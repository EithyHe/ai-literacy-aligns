"""Unified computation helpers for SSN visualization pages (UMAP / HDBSCAN / diagnostics)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from ssn.db.schema import get_all_constructs, get_all_items
from ssn.services.embedding_service import get_construct_embeddings, load_item_embeddings


JINGLE_PRIMARY_THRESHOLD = 0.60
JINGLE_SECONDARY_THRESHOLD = 0.20


def _construct_lookup() -> dict[str, dict[str, Any]]:
    constructs = get_all_constructs()
    return {str(c.get("construct_id")): dict(c) for c in constructs if c.get("construct_id")}


def get_item_embedding_table() -> tuple[pd.DataFrame, np.ndarray]:
    """Return item metadata table and aligned embedding matrix."""
    embeddings, item_ids = load_item_embeddings()
    if embeddings.size == 0:
        empty = pd.DataFrame(
            columns=[
                "id",
                "item_id",
                "construct_id",
                "construct_label",
                "instrument",
                "text",
                "framework_id",
                "level",
            ]
        )
        return empty, np.zeros((0, 0), dtype=np.float32)

    item_id_to_idx = {str(iid): idx for idx, iid in enumerate(item_ids)}
    construct_by_id = _construct_lookup()

    rows: list[dict[str, Any]] = []
    for item in get_all_items():
        item_id = str(item.get("item_id") or "")
        if not item_id or item_id not in item_id_to_idx:
            continue
        construct_id = str(item.get("construct_id") or "")
        construct = construct_by_id.get(construct_id, {})
        rows.append(
            {
                "id": item_id,
                "item_id": item_id,
                "construct_id": construct_id,
                "construct_label": construct.get("name", construct_id),
                "instrument": str(item.get("instrument") or ""),
                "text": str(item.get("text") or ""),
                "framework_id": str(item.get("framework_id") or construct.get("framework_id") or ""),
                "level": "item",
                "_emb_idx": int(item_id_to_idx[item_id]),
            }
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "id",
                "item_id",
                "construct_id",
                "construct_label",
                "instrument",
                "text",
                "framework_id",
                "level",
            ]
        )
        return empty, np.zeros((0, embeddings.shape[1]), dtype=np.float32)

    df = pd.DataFrame(rows).sort_values("_emb_idx").reset_index(drop=True)
    matrix = embeddings[df["_emb_idx"].to_numpy(dtype=int)]
    df = df.drop(columns=["_emb_idx"])
    return df, matrix


def get_construct_embedding_table() -> tuple[pd.DataFrame, np.ndarray]:
    """Return construct metadata table and aligned construct embedding matrix."""
    construct_embs = get_construct_embeddings()
    construct_by_id = _construct_lookup()

    if not construct_embs:
        empty = pd.DataFrame(
            columns=[
                "id",
                "construct_id",
                "construct_label",
                "framework_id",
                "item_count",
                "instrument",
                "text",
                "level",
            ]
        )
        return empty, np.zeros((0, 0), dtype=np.float32)

    # Instrument summary: dominant instrument per construct.
    item_df = pd.DataFrame(get_all_items())
    dominant_instrument: dict[str, str] = {}
    if not item_df.empty and "construct_id" in item_df.columns and "instrument" in item_df.columns:
        for cid, sub in item_df.groupby(item_df["construct_id"].astype(str)):
            freq = sub["instrument"].fillna("").astype(str).value_counts()
            dominant_instrument[str(cid)] = str(freq.index[0]) if len(freq) else ""

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for cid, vec in construct_embs.items():
        cid_str = str(cid)
        c_meta = construct_by_id.get(cid_str, {})
        rows.append(
            {
                "id": cid_str,
                "construct_id": cid_str,
                "construct_label": str(c_meta.get("name") or cid_str),
                "framework_id": str(c_meta.get("framework_id") or ""),
                "item_count": int(c_meta.get("item_count") or 0),
                "instrument": dominant_instrument.get(cid_str, ""),
                "text": str(c_meta.get("description") or ""),
                "level": "construct",
            }
        )
        vectors.append(np.asarray(vec, dtype=np.float32))

    df = pd.DataFrame(rows)
    matrix = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
    return df, matrix


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    if filters is None or df.empty:
        return df

    out = df.copy()
    construct_ids = filters.get("construct_ids")
    if construct_ids:
        keep = {str(x) for x in construct_ids}
        out = out[out["construct_id"].astype(str).isin(keep)]

    instruments = filters.get("instruments")
    if instruments:
        keep = {str(x) for x in instruments}
        out = out[out["instrument"].astype(str).isin(keep)]

    framework_ids = filters.get("framework_ids")
    if framework_ids:
        keep = {str(x) for x in framework_ids}
        out = out[out["framework_id"].astype(str).isin(keep)]

    return out.reset_index(drop=True)


def compute_umap(
    level: str,
    filters: dict[str, Any] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute UMAP for item or construct level and return a unified dataframe."""
    level_norm = (level or "item").lower()
    if level_norm not in {"item", "construct"}:
        raise ValueError("level must be 'item' or 'construct'")

    if level_norm == "item":
        meta, matrix = get_item_embedding_table()
    else:
        meta, matrix = get_construct_embedding_table()

    if meta.empty:
        return meta.assign(x=pd.Series(dtype=float), y=pd.Series(dtype=float))

    filtered = _apply_filters(meta, filters)
    if filtered.empty:
        return filtered.assign(x=pd.Series(dtype=float), y=pd.Series(dtype=float))

    if level_norm == "item":
        full_index = pd.Index(meta["id"].astype(str))
        idx = full_index.get_indexer(filtered["id"].astype(str))
    else:
        full_index = pd.Index(meta["construct_id"].astype(str))
        idx = full_index.get_indexer(filtered["construct_id"].astype(str))

    valid_mask = idx >= 0
    filtered = filtered.loc[valid_mask].reset_index(drop=True)
    idx = idx[valid_mask]
    emb = matrix[idx]
    emb = np.asarray(emb, dtype=np.float32)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    row_scale = np.max(np.abs(emb), axis=1, keepdims=True)
    row_scale = np.where(row_scale <= 1e-12, 1.0, row_scale)
    emb = emb / row_scale

    n = emb.shape[0]
    if n == 0:
        return filtered.assign(x=pd.Series(dtype=float), y=pd.Series(dtype=float))
    if n == 1:
        return filtered.assign(x=[0.0], y=[0.0])

    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn is required for compute_umap") from e

    k = int(max(2, min(n_neighbors, n - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=k,
        min_dist=float(min_dist),
        random_state=int(seed),
    )
    coords = reducer.fit_transform(emb)

    out = filtered.copy()
    out["x"] = coords[:, 0].astype(float)
    out["y"] = coords[:, 1].astype(float)
    return out


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
) -> dict[str, Any]:
    """Run HDBSCAN and return labels and quality metrics."""
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError("hdbscan is required. Install with `pip install hdbscan`.") from e

    X = np.asarray(embeddings, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    row_scale = np.max(np.abs(X), axis=1, keepdims=True)
    row_scale = np.where(row_scale <= 1e-12, 1.0, row_scale)
    X = X / row_scale
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={X.shape}")

    n = X.shape[0]
    if n == 0:
        return {"labels": np.array([], dtype=int), "silhouette": None, "n_clusters": 0}
    if n < 3:
        labels = np.zeros(n, dtype=int)
        return {"labels": labels, "silhouette": None, "n_clusters": 1}

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min(int(min_cluster_size), n)),
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(X)

    non_noise = labels != -1
    unique_clusters = sorted({int(x) for x in labels if int(x) != -1})
    n_clusters = len(unique_clusters)

    sil = None
    if non_noise.sum() >= 3 and n_clusters >= 2:
        try:
            sil = float(silhouette_score(X[non_noise], labels[non_noise]))
        except Exception:
            sil = None

    return {"labels": labels.astype(int), "silhouette": sil, "n_clusters": n_clusters}


def _safe_cluster_int(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    if text.strip() == "":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    m = re.search(r"-?\d+", text)
    return int(m.group(0)) if m else None


def _build_cluster_top_items(
    umap_df: pd.DataFrame,
    labels: np.ndarray,
    embeddings: np.ndarray,
    max_items_per_cluster: int = 10,
) -> pd.DataFrame:
    if umap_df.empty or len(labels) == 0:
        return pd.DataFrame(columns=["cluster", "centrality", "item_id", "text", "construct_reported"])

    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != len(labels) or X.shape[0] != len(umap_df):
        return pd.DataFrame(columns=["cluster", "centrality", "item_id", "text", "construct_reported"])

    # Stabilize numerics for pathological vectors (NaN/Inf/extreme magnitudes).
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    row_scale = np.max(np.abs(X), axis=1, keepdims=True)
    row_scale = np.where(row_scale <= 1e-12, 1.0, row_scale)
    X_scaled = X / row_scale

    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    Xn = X_scaled / norms

    rows: list[dict[str, Any]] = []
    unique_clusters = sorted({int(c) for c in labels if int(c) != -1})
    for cid in unique_clusters:
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue

        sub = Xn[idx]
        centroid = sub.mean(axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm <= 1e-12:
            scores = np.zeros(sub.shape[0], dtype=float)
        else:
            direction = centroid / c_norm
            scores = np.dot(sub, direction).astype(float)
            scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)

        rank = np.argsort(-scores)[: max(1, int(max_items_per_cluster))]
        for ridx in rank:
            row_idx = int(idx[int(ridx)])
            record = umap_df.iloc[row_idx]
            rows.append(
                {
                    "cluster": int(cid),
                    "centrality": float(scores[int(ridx)]),
                    "item_id": str(record.get("id", "")),
                    "text": str(record.get("text", "")),
                    "construct_reported": str(record.get("construct_label", "")),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["cluster", "centrality", "item_id", "text", "construct_reported"])
    return pd.DataFrame(rows).sort_values(["cluster", "centrality"], ascending=[True, False]).reset_index(drop=True)


@lru_cache(maxsize=24)
def _llm_interpret_clusters_cached(
    top_items_payload_json: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> tuple[tuple[int, str, str], ...]:
    from ssn.services.llm_interpret_graph import run_ssn_leiden_domain_interpretation

    rows = json.loads(top_items_payload_json)
    top_items_df = pd.DataFrame(rows)
    if top_items_df.empty:
        return tuple()

    result = run_ssn_leiden_domain_interpretation(
        top_items_df=top_items_df,
        out_csv=None,
        model=model,
        temperature=float(temperature),
    )
    if result is None or result.empty:
        return tuple()

    out: list[tuple[int, str, str]] = []
    for _, row in result.iterrows():
        cid = _safe_cluster_int(row.get("cluster"))
        if cid is None:
            continue
        label = str(row.get("label") or "").strip()
        rationale = str(row.get("rationale") or "").strip()
        out.append((int(cid), label, rationale))
    return tuple(out)


def llm_interpret_cluster_labels(
    umap_df: pd.DataFrame,
    labels: np.ndarray,
    embeddings: np.ndarray,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_items_per_cluster: int = 10,
) -> dict[int, dict[str, str]]:
    """Generate per-cluster LLM labels for HDBSCAN clusters.

    Returns:
        Dict[int, Dict[str, str]] mapping cluster_id ->
        {"label": "...", "rationale": "..."}.
    """
    arr = np.asarray(labels, dtype=int)
    unique_clusters = sorted({int(c) for c in arr if int(c) != -1})
    if not unique_clusters:
        return {}

    top_items_df = _build_cluster_top_items(
        umap_df=umap_df,
        labels=arr,
        embeddings=embeddings,
        max_items_per_cluster=max_items_per_cluster,
    )
    if top_items_df.empty:
        return {cid: {"label": f"Cluster {cid}", "rationale": ""} for cid in unique_clusters}

    payload_json = top_items_df.to_json(orient="records", force_ascii=False)
    try:
        rows = _llm_interpret_clusters_cached(
            payload_json,
            model=model,
            temperature=float(temperature),
        )
    except Exception:
        rows = tuple()

    mapping: dict[int, dict[str, str]] = {
        int(cid): {"label": f"Cluster {int(cid)}", "rationale": ""}
        for cid in unique_clusters
    }
    for cid, label, rationale in rows:
        if cid not in mapping:
            continue
        clean_label = label if label else f"Cluster {cid}"
        mapping[cid] = {"label": clean_label, "rationale": rationale}
    return mapping


def build_cluster_diagnostics(
    construct_labels: Sequence[str],
    cluster_labels: Sequence[int],
) -> dict[str, Any]:
    """Build contingency tables, Sankey links, and jingle-jangle flags."""
    if len(construct_labels) != len(cluster_labels):
        raise ValueError("construct_labels and cluster_labels must have same length")

    df = pd.DataFrame(
        {
            "construct_label": [str(x) for x in construct_labels],
            "cluster_label": [int(x) for x in cluster_labels],
        }
    )
    if df.empty:
        empty = pd.DataFrame()
        return {
            "contingency": empty,
            "sankey_links": [],
            "construct_summary": empty,
            "cluster_summary": empty,
            "jingle_flags": set(),
            "jangle_flags": set(),
        }

    df["cluster_name"] = df["cluster_label"].apply(lambda x: "Noise" if x == -1 else f"Cluster {x}")

    contingency = pd.crosstab(df["construct_label"], df["cluster_name"]).sort_index(axis=0).sort_index(axis=1)
    row_ratio = contingency.div(contingency.sum(axis=1), axis=0).fillna(0.0)
    col_ratio = contingency.div(contingency.sum(axis=0), axis=1).fillna(0.0)

    jingle_flags: set[str] = set()
    for construct_name, row in row_ratio.iterrows():
        sorted_ratio = row.sort_values(ascending=False)
        primary = float(sorted_ratio.iloc[0]) if len(sorted_ratio) > 0 else 0.0
        secondary = float(sorted_ratio.iloc[1]) if len(sorted_ratio) > 1 else 0.0
        if primary < JINGLE_PRIMARY_THRESHOLD and secondary >= JINGLE_SECONDARY_THRESHOLD:
            jingle_flags.add(str(construct_name))

    jangle_flags: set[str] = set()
    for cluster_name in col_ratio.columns:
        if cluster_name == "Noise":
            continue
        col = col_ratio[cluster_name].sort_values(ascending=False)
        primary = float(col.iloc[0]) if len(col) > 0 else 0.0
        secondary = float(col.iloc[1]) if len(col) > 1 else 0.0
        if primary < JINGLE_PRIMARY_THRESHOLD and secondary >= JINGLE_SECONDARY_THRESHOLD:
            jangle_flags.add(str(cluster_name))

    links: list[dict[str, Any]] = []
    for source in contingency.index:
        for target in contingency.columns:
            value = int(contingency.at[source, target])
            if value <= 0:
                continue
            links.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "value": value,
                    "is_jingle": str(source) in jingle_flags,
                    "is_jangle": str(target) in jangle_flags,
                }
            )

    construct_summary_rows: list[dict[str, Any]] = []
    for source, row in contingency.iterrows():
        total = int(row.sum())
        dominant_cluster = str(row.idxmax()) if total > 0 else ""
        dominance = float(row.max() / total) if total > 0 else 0.0
        construct_summary_rows.append(
            {
                "construct_label": str(source),
                "n_items": total,
                "dominant_cluster": dominant_cluster,
                "dominance": dominance,
                "is_jingle": str(source) in jingle_flags,
            }
        )

    cluster_summary_rows: list[dict[str, Any]] = []
    for target in contingency.columns:
        col = contingency[target]
        total = int(col.sum())
        dominant_construct = str(col.idxmax()) if total > 0 else ""
        purity = float(col.max() / total) if total > 0 else 0.0
        cluster_summary_rows.append(
            {
                "cluster": str(target),
                "n_items": total,
                "dominant_construct": dominant_construct,
                "purity": purity,
                "is_jangle": str(target) in jangle_flags,
            }
        )

    return {
        "contingency": contingency,
        "sankey_links": links,
        "construct_summary": pd.DataFrame(construct_summary_rows).sort_values(
            ["is_jingle", "n_items"], ascending=[False, False]
        ),
        "cluster_summary": pd.DataFrame(cluster_summary_rows).sort_values(
            ["is_jangle", "n_items"], ascending=[False, False]
        ),
        "jingle_flags": jingle_flags,
        "jangle_flags": jangle_flags,
    }
