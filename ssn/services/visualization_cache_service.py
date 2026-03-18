"""Offline cache builder/loader for visualization workspace (04 + 06).

This service moves expensive computation (UMAP, HDBSCAN, network building,
LLM cluster naming) offline into disk artifacts, so Streamlit pages can
focus on lightweight rendering only.
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ssn.config import PROCESSED_DIR
from ssn.services.network_service import build_construct_network
from ssn.services.visualization_service import (
    compute_umap,
    get_construct_embedding_table,
    get_item_embedding_table,
    llm_interpret_cluster_labels,
    run_hdbscan,
)

logger = logging.getLogger(__name__)

# Keep offline build logs readable; these warnings are noisy but non-fatal here.
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
    module=r"sklearn\\.utils\\.extmath",
)
warnings.filterwarnings(
    "ignore",
    message=".*n_jobs value 1 overridden to 1 by setting random_state.*",
    category=UserWarning,
    module=r"umap\\.umap_",
)

VIS_CACHE_DIR = PROCESSED_DIR / "visualization_cache"
VIS_CACHE_MANIFEST = VIS_CACHE_DIR / "manifest.json"
VIS_CACHE_FILES = {
    "item_points": VIS_CACHE_DIR / "item_points.csv",
    "construct_points": VIS_CACHE_DIR / "construct_points.csv",
    "network_full_nodes": VIS_CACHE_DIR / "network_full_nodes.json",
    "network_full_edges": VIS_CACHE_DIR / "network_full_edges.json",
    "network_backbone_nodes": VIS_CACHE_DIR / "network_backbone_nodes.json",
    "network_backbone_edges": VIS_CACHE_DIR / "network_backbone_edges.json",
}

_cached_bundle: dict[str, Any] | None = None
_cached_signature: tuple[tuple[str, int, int], ...] | None = None

_PALETTE = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
]


def _signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...] | None:
    rows: list[tuple[str, int, int]] = []
    for p in paths:
        if not p.exists():
            return None
        st = p.stat()
        rows.append((p.name, int(st.st_mtime_ns), int(st.st_size)))
    return tuple(rows)


def clear_visualization_cache_memory() -> None:
    """Clear in-memory cache only (does not delete disk artifacts)."""
    global _cached_bundle, _cached_signature
    _cached_bundle = None
    _cached_signature = None


def _normalize_positions(pos: dict[Any, np.ndarray], scale: float = 420.0) -> dict[Any, tuple[float, float]]:
    if not pos:
        return {}
    keys = list(pos.keys())
    arr = np.array([pos[k] for k in keys], dtype=float)
    center = arr.mean(axis=0)
    arr = arr - center
    spread_x = float(np.max(arr[:, 0]) - np.min(arr[:, 0])) if arr.shape[0] > 0 else 1.0
    spread_y = float(np.max(arr[:, 1]) - np.min(arr[:, 1])) if arr.shape[0] > 0 else 1.0
    denom = max(spread_x, spread_y, 1e-6)
    arr = arr * (scale / denom)
    return {k: (float(arr[i, 0]), float(arr[i, 1])) for i, k in enumerate(keys)}


def _compute_layout_positions(G: nx.Graph) -> dict[str, tuple[float, float]]:
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_nodes() == 1:
        only = next(iter(G.nodes()))
        return {str(only): (0.0, 0.0)}

    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    components.sort(key=lambda sg: sg.number_of_nodes(), reverse=True)

    packed: dict[str, tuple[float, float]] = {}
    cols = int(max(1, np.ceil(np.sqrt(len(components)))))
    step = 940.0
    gap = 220.0

    for idx, sg in enumerate(components):
        n = sg.number_of_nodes()
        if n == 1:
            raw = {next(iter(sg.nodes())): np.array([0.0, 0.0], dtype=float)}
        elif n == 2:
            nodes = list(sg.nodes())
            raw = {
                nodes[0]: np.array([-1.0, 0.0], dtype=float),
                nodes[1]: np.array([1.0, 0.0], dtype=float),
            }
        else:
            iters = 150 if n <= 120 else 90
            raw = nx.spring_layout(
                sg,
                seed=42,
                weight="weight",
                iterations=iters,
            )

        local = _normalize_positions({k: np.asarray(v, dtype=float) for k, v in raw.items()}, scale=step * 0.42)
        row = idx // cols
        col = idx % cols
        ox = col * (step + gap)
        oy = row * (step + gap)
        for node, (x, y) in local.items():
            packed[str(node)] = (x + ox, y + oy)
    return packed


def _prepare_network_payload(
    graph: nx.Graph,
    construct_meta: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    meta_by_id = {
        str(row["construct_id"]): row
        for _, row in construct_meta.iterrows()
    }
    layout_pos = _compute_layout_positions(graph)
    top_label_count = max(12, int(max(1, graph.number_of_nodes()) * 0.18))
    ranked_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    labeled_nodes = {str(node) for node, _ in ranked_nodes[:top_label_count]}

    nodes: list[dict[str, Any]] = []
    for node in graph.nodes():
        node_id = str(node)
        row = meta_by_id.get(node_id)
        neighbors = list(graph.neighbors(node))
        weights = [float(graph.edges[node, n].get("weight", 0.0)) for n in neighbors]
        degree = len(neighbors)
        avg_weight = float(np.mean(weights)) if weights else 0.0
        full_label = str(row["construct_label"]) if row is not None else node_id
        label = full_label if node_id in labeled_nodes else ""
        size = max(6, min(20, 7 + degree * 0.9))
        x, y = layout_pos.get(node_id, (0.0, 0.0))
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "full_label": full_label,
                "size": size,
                "degree": degree,
                "avg_weight": avg_weight,
                "framework_id": str(row["framework_id"]) if row is not None else "",
                "x": float(x),
                "y": float(y),
            }
        )

    edges: list[dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        su = str(u)
        sv = str(v)
        lu = str(meta_by_id.get(su, {}).get("construct_label", su))
        lv = str(meta_by_id.get(sv, {}).get("construct_label", sv))
        edges.append(
            {
                "id": f"{su}__{sv}",
                "source": su,
                "target": sv,
                "source_label": lu,
                "target_label": lv,
                "weight": float(data.get("weight", 0.0)),
            }
        )
    return nodes, edges


def _fallback_backbone_from_full(full_graph: nx.Graph) -> nx.Graph:
    """Fallback backbone when disparity filter removes everything.

    Keep a maximum spanning tree inside each connected component so the
    resulting graph stays sparse but still readable.
    """
    B = nx.Graph()
    B.add_nodes_from(full_graph.nodes(data=True))
    if full_graph.number_of_edges() == 0:
        return B

    for comp_nodes in nx.connected_components(full_graph):
        sub = full_graph.subgraph(comp_nodes).copy()
        if sub.number_of_edges() == 0:
            continue
        tree = nx.maximum_spanning_tree(sub, weight="weight")
        B.add_edges_from(tree.edges(data=True))
    return B


def _align_embeddings_to_meta(
    full_meta: pd.DataFrame,
    full_emb: np.ndarray,
    target_meta: pd.DataFrame,
    id_col: str,
) -> np.ndarray:
    if target_meta.empty:
        return np.zeros((0, full_emb.shape[1] if full_emb.ndim == 2 else 0), dtype=np.float32)
    idx_map = {str(v): i for i, v in enumerate(full_meta[id_col].astype(str).tolist())}
    idx = [idx_map[str(v)] for v in target_meta[id_col].astype(str).tolist() if str(v) in idx_map]
    if not idx:
        return np.zeros((0, full_emb.shape[1]), dtype=np.float32)
    return full_emb[np.asarray(idx, dtype=int)]


def _cluster_name_bundle(
    points_df: pd.DataFrame,
    labels: np.ndarray,
    embeddings: np.ndarray,
    use_llm: bool = True,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    arr = np.asarray(labels, dtype=int)
    unique_clusters = sorted({int(x) for x in arr if int(x) != -1})

    if use_llm and unique_clusters:
        interpreted = llm_interpret_cluster_labels(
            umap_df=points_df,
            labels=arr,
            embeddings=embeddings,
            model="claude-haiku-4-5-20251001",
            temperature=0.2,
            max_items_per_cluster=10,
        )
    else:
        interpreted = {
            int(cid): {"label": f"Cluster {int(cid)}", "rationale": ""}
            for cid in unique_clusters
        }

    name_map: dict[int, str] = {-1: "Noise"}
    seen_names: set[str] = {"Noise"}
    for cid in unique_clusters:
        label = str(interpreted.get(cid, {}).get("label") or f"Cluster {cid}").strip()
        clean = " ".join(label.split()) or f"Cluster {cid}"
        if clean in seen_names:
            clean = f"{clean} [C{cid}]"
        seen_names.add(clean)
        name_map[cid] = clean

    color_map: dict[str, str] = {"Noise": "#98A2B3"}
    for i, cid in enumerate(unique_clusters):
        display = name_map[cid]
        if i < len(_PALETTE):
            color_map[display] = _PALETTE[i]
        else:
            hue = float((i * 137.508) % 360.0)
            color_map[display] = f"hsl({hue:.1f}, 65%, 48%)"

    out = points_df.copy()
    out["hdbscan_label"] = arr.astype(int)
    out["cluster_display_name"] = out["hdbscan_label"].map(lambda x: name_map.get(int(x), f"Cluster {int(x)}"))
    out["cluster_label"] = out["hdbscan_label"].map(lambda x: interpreted.get(int(x), {}).get("label", f"Cluster {int(x)}"))
    out["cluster_rationale"] = out["hdbscan_label"].map(lambda x: interpreted.get(int(x), {}).get("rationale", ""))
    out["cluster_color"] = out["cluster_display_name"].map(lambda x: color_map.get(str(x), "#7F8C8D"))

    cluster_meta = {
        "cluster_display_by_id": {str(k): v for k, v in name_map.items()},
        "cluster_rationale_by_id": {str(k): interpreted.get(k, {}).get("rationale", "") for k in unique_clusters},
        "cluster_color_map": color_map,
    }
    return out, color_map, cluster_meta


def _auto_tune_hdbscan_params(
    embeddings: np.ndarray,
    target_min_clusters: int = 10,
    target_max_clusters: int = 15,
    min_cluster_size_grid: list[int] | None = None,
    min_samples_grid: list[int | None] | None = None,
) -> dict[str, Any]:
    """Pick HDBSCAN params with highest silhouette under target cluster-count range."""
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 10:
        return {
            "selected": {"min_cluster_size": 20, "min_samples": None},
            "best_result": {"n_clusters": 0, "silhouette": None},
            "search_count": 0,
        }

    mcs_grid = min_cluster_size_grid or [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40]
    ms_grid = min_samples_grid or [None, 1, 2, 3, 4, 5, 6, 8, 10]

    best_in_range: tuple[float, int, int, int | None] | None = None  # sil, n, mcs, ms
    best_any: tuple[float, int, int, int | None] | None = None
    n_eval = 0

    for mcs in mcs_grid:
        for ms in ms_grid:
            n_eval += 1
            res = run_hdbscan(X, min_cluster_size=int(mcs), min_samples=ms)
            n_clusters = int(res["n_clusters"])
            sil = float(res["silhouette"]) if res["silhouette"] is not None else -1.0
            row = (sil, n_clusters, int(mcs), ms)

            if best_any is None or row > best_any:
                best_any = row
            if target_min_clusters <= n_clusters <= target_max_clusters:
                if best_in_range is None or row > best_in_range:
                    best_in_range = row

    chosen = best_in_range if best_in_range is not None else best_any
    if chosen is None:
        return {
            "selected": {"min_cluster_size": 20, "min_samples": None},
            "best_result": {"n_clusters": 0, "silhouette": None},
            "search_count": n_eval,
        }

    sil, n_clusters, mcs, ms = chosen
    return {
        "selected": {"min_cluster_size": int(mcs), "min_samples": ms},
        "best_result": {
            "n_clusters": int(n_clusters),
            "silhouette": None if sil < 0 else float(sil),
        },
        "search_count": n_eval,
        "met_target_range": best_in_range is not None,
    }


def build_visualization_cache(
    output_dir: Path | None = None,
    min_weight: float = 0.25,
    alpha: float = 0.05,
    umap_n_neighbors: int = 20,
    umap_min_dist: float = 0.10,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int | None = None,
    auto_tune_hdbscan: bool = False,
    target_min_clusters: int = 10,
    target_max_clusters: int = 15,
    seed: int = 42,
    use_llm_labels: bool = True,
) -> dict[str, Any]:
    """Build cache artifacts for fast 04/06 rendering."""
    target = output_dir or VIS_CACHE_DIR
    target.mkdir(parents=True, exist_ok=True)
    file_paths = {
        "manifest": target / "manifest.json",
        "item_points": target / "item_points.csv",
        "construct_points": target / "construct_points.csv",
        "network_full_nodes": target / "network_full_nodes.json",
        "network_full_edges": target / "network_full_edges.json",
        "network_backbone_nodes": target / "network_backbone_nodes.json",
        "network_backbone_edges": target / "network_backbone_edges.json",
    }

    item_meta, item_emb = get_item_embedding_table()
    construct_meta, construct_emb = get_construct_embedding_table()
    if item_meta.empty or construct_meta.empty:
        raise RuntimeError("Cannot build visualization cache: missing item/construct embedding tables.")

    logger.info("Building visualization cache at %s", target)

    # Item-level UMAP + HDBSCAN + names
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*divide by zero encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*", category=RuntimeWarning)
        item_points = compute_umap(
            level="item",
            filters=None,
            n_neighbors=int(umap_n_neighbors),
            min_dist=float(umap_min_dist),
            seed=int(seed),
        )
    item_emb_aligned = _align_embeddings_to_meta(item_meta, item_emb, item_points, id_col="id")
    selected_mcs = int(hdbscan_min_cluster_size)
    selected_ms = hdbscan_min_samples
    tuning_meta: dict[str, Any] = {}
    if auto_tune_hdbscan:
        tuning_meta = _auto_tune_hdbscan_params(
            embeddings=item_emb_aligned,
            target_min_clusters=int(target_min_clusters),
            target_max_clusters=int(target_max_clusters),
        )
        sel = tuning_meta.get("selected", {})
        selected_mcs = int(sel.get("min_cluster_size", selected_mcs))
        selected_ms = sel.get("min_samples", selected_ms)
        logger.info(
            "Auto-tuned HDBSCAN: min_cluster_size=%s, min_samples=%s, target=%s-%s, best=%s",
            selected_mcs,
            selected_ms,
            target_min_clusters,
            target_max_clusters,
            tuning_meta.get("best_result"),
        )

    item_hdb = run_hdbscan(
        embeddings=item_emb_aligned,
        min_cluster_size=int(selected_mcs),
        min_samples=selected_ms,
    )
    item_labels = item_hdb["labels"]
    if len(item_labels) != len(item_points):
        item_labels = np.resize(item_labels, len(item_points))
    item_points, item_color_map, item_cluster_meta = _cluster_name_bundle(
        points_df=item_points,
        labels=item_labels,
        embeddings=item_emb_aligned,
        use_llm=use_llm_labels,
    )

    # Construct-level UMAP + HDBSCAN + names
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*divide by zero encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*", category=RuntimeWarning)
        construct_points = compute_umap(
            level="construct",
            filters=None,
            n_neighbors=int(umap_n_neighbors),
            min_dist=float(umap_min_dist),
            seed=int(seed),
        )
    construct_emb_aligned = _align_embeddings_to_meta(
        construct_meta,
        construct_emb,
        construct_points,
        id_col="construct_id",
    )
    construct_umap_coords = construct_points[["x", "y"]].values
    construct_hdb = run_hdbscan(
        embeddings=construct_umap_coords,
        min_cluster_size=max(2, int(selected_mcs)),
        min_samples=selected_ms,
    )
    construct_labels = construct_hdb["labels"]
    if len(construct_labels) != len(construct_points):
        construct_labels = np.resize(construct_labels, len(construct_points))
    construct_points, construct_color_map, construct_cluster_meta = _cluster_name_bundle(
        points_df=construct_points,
        labels=construct_labels,
        embeddings=construct_emb_aligned,
        use_llm=use_llm_labels,
    )

    # Networks
    n_construct = min(len(construct_meta), construct_emb.shape[0])
    construct_embeddings = {
        str(construct_meta.iloc[i]["construct_id"]): construct_emb[i]
        for i in range(n_construct)
    }
    graph_full = build_construct_network(
        construct_embeddings=construct_embeddings,
        mode="full",
        backbone_method="disparity",
        alpha=float(alpha),
        min_weight=float(min_weight),
    )
    graph_backbone = build_construct_network(
        construct_embeddings=construct_embeddings,
        mode="backbone",
        backbone_method="disparity",
        alpha=float(alpha),
        min_weight=float(min_weight),
    )
    if graph_backbone.number_of_edges() == 0 and graph_full.number_of_edges() > 0:
        logger.warning(
            "Backbone is empty with current params (alpha=%.4f, min_weight=%.4f); "
            "using max-spanning-forest fallback.",
            float(alpha),
            float(min_weight),
        )
        graph_backbone = _fallback_backbone_from_full(graph_full)

    full_nodes, full_edges = _prepare_network_payload(graph_full, construct_meta)
    backbone_nodes, backbone_edges = _prepare_network_payload(graph_backbone, construct_meta)

    # Persist
    item_points.to_csv(file_paths["item_points"], index=False, encoding="utf-8")
    construct_points.to_csv(file_paths["construct_points"], index=False, encoding="utf-8")
    file_paths["network_full_nodes"].write_text(json.dumps(full_nodes, ensure_ascii=False), encoding="utf-8")
    file_paths["network_full_edges"].write_text(json.dumps(full_edges, ensure_ascii=False), encoding="utf-8")
    file_paths["network_backbone_nodes"].write_text(json.dumps(backbone_nodes, ensure_ascii=False), encoding="utf-8")
    file_paths["network_backbone_edges"].write_text(json.dumps(backbone_edges, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(target),
        "params": {
            "min_weight": float(min_weight),
            "alpha": float(alpha),
            "umap_n_neighbors": int(umap_n_neighbors),
            "umap_min_dist": float(umap_min_dist),
            "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
            "hdbscan_min_samples": None if hdbscan_min_samples is None else int(hdbscan_min_samples),
            "hdbscan_selected_min_cluster_size": int(selected_mcs),
            "hdbscan_selected_min_samples": None if selected_ms is None else int(selected_ms),
            "auto_tune_hdbscan": bool(auto_tune_hdbscan),
            "target_min_clusters": int(target_min_clusters),
            "target_max_clusters": int(target_max_clusters),
            "seed": int(seed),
            "use_llm_labels": bool(use_llm_labels),
        },
        "hdbscan_tuning": tuning_meta,
        "stats": {
            "n_items": int(len(item_points)),
            "n_constructs": int(len(construct_points)),
            "network_full_edges": int(len(full_edges)),
            "network_backbone_edges": int(len(backbone_edges)),
            "item_hdbscan_clusters": int(item_hdb["n_clusters"]),
            "item_hdbscan_silhouette": item_hdb["silhouette"],
            "construct_hdbscan_clusters": int(construct_hdb["n_clusters"]),
            "construct_hdbscan_silhouette": construct_hdb["silhouette"],
        },
        "cluster_meta": {
            "item": {
                **item_cluster_meta,
                "cluster_color_map": item_color_map,
            },
            "construct": {
                **construct_cluster_meta,
                "cluster_color_map": construct_color_map,
            },
        },
    }
    file_paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    clear_visualization_cache_memory()
    logger.info("Visualization cache built successfully: %s", file_paths["manifest"])
    return manifest


def patch_llm_cluster_labels(
    cache_dir: Path | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_items_per_cluster: int = 10,
) -> dict[str, Any]:
    """Re-run LLM cluster labeling on existing cached UMAP/HDBSCAN results.

    Updates construct_points.csv and the manifest in-place without rebuilding
    the full UMAP/HDBSCAN/network artifacts.  Returns the updated manifest.
    """
    target = cache_dir or VIS_CACHE_DIR
    pts_path = target / "construct_points.csv"
    manifest_path = target / "manifest.json"

    if not pts_path.exists():
        raise FileNotFoundError(f"construct_points.csv not found at {pts_path}")

    pts = pd.read_csv(pts_path)
    pts["hdbscan_label"] = pd.to_numeric(pts["hdbscan_label"], errors="coerce").fillna(-1).astype(int)
    labels = pts["hdbscan_label"].values

    # Align full construct embeddings to pts row order
    construct_meta, construct_emb = get_construct_embedding_table()
    construct_emb_aligned = _align_embeddings_to_meta(
        construct_meta, construct_emb, pts, id_col="construct_id"
    )

    # Drop stale cluster columns so _cluster_name_bundle writes them fresh
    pts_base = pts.drop(
        columns=[c for c in ["cluster_display_name", "cluster_label", "cluster_rationale", "cluster_color", "hdbscan_label"] if c in pts.columns],
        errors="ignore",
    )

    updated_pts, color_map, cluster_meta = _cluster_name_bundle(
        points_df=pts_base,
        labels=labels,
        embeddings=construct_emb_aligned,
        use_llm=True,
    )
    updated_pts.to_csv(pts_path, index=False, encoding="utf-8")

    # Update manifest
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("params", {})["use_llm_labels"] = True
    manifest["cluster_meta"] = manifest.get("cluster_meta", {})
    manifest["cluster_meta"]["construct"] = {
        **cluster_meta,
        "cluster_color_map": color_map,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    clear_visualization_cache_memory()
    logger.info("LLM cluster labels patched into %s", pts_path)
    return manifest


def get_visualization_cache(cache_dir: Path | None = None) -> dict[str, Any] | None:
    """Load visualization cache bundle from disk with in-memory memoization."""
    global _cached_bundle, _cached_signature
    target = cache_dir or VIS_CACHE_DIR
    paths = [
        target / "manifest.json",
        target / "item_points.csv",
        target / "construct_points.csv",
        target / "network_full_nodes.json",
        target / "network_full_edges.json",
        target / "network_backbone_nodes.json",
        target / "network_backbone_edges.json",
    ]
    sig = _signature(paths)
    if sig is None:
        return None
    if _cached_bundle is not None and _cached_signature == sig:
        return _cached_bundle

    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    item_points = pd.read_csv(target / "item_points.csv")
    construct_points = pd.read_csv(target / "construct_points.csv")
    network_full_nodes = json.loads((target / "network_full_nodes.json").read_text(encoding="utf-8"))
    network_full_edges = json.loads((target / "network_full_edges.json").read_text(encoding="utf-8"))
    network_backbone_nodes = json.loads((target / "network_backbone_nodes.json").read_text(encoding="utf-8"))
    network_backbone_edges = json.loads((target / "network_backbone_edges.json").read_text(encoding="utf-8"))

    bundle = {
        "manifest": manifest,
        "item_points": item_points,
        "construct_points": construct_points,
        "network": {
            "full": {"nodes": network_full_nodes, "edges": network_full_edges},
            "backbone": {"nodes": network_backbone_nodes, "edges": network_backbone_edges},
        },
    }
    _cached_bundle = bundle
    _cached_signature = sig
    return bundle
