"""Offline pipeline to produce SSN graph visualization JSON (PRD Section 7).

Generates data/processed/ssn_graph_data.json from embeddings and DB hierarchy.
Run from project root: python -m ssn.services.graph_data_pipeline
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ssn.config import (
    HULL_PADDING_FRACTION,
    PRIMARY_FRAMEWORK_ID,
    PROCESSED_DIR,
    SSN_GRAPH_JSON_PATH,
    UMAP_CONSTRUCT_MIN_DIST,
    UMAP_CONSTRUCT_NEIGHBORS,
    UMAP_DOMAIN_MIN_DIST,
    UMAP_DOMAIN_NEIGHBORS,
    UMAP_ITEM_MIN_DIST,
    UMAP_ITEM_NEIGHBORS,
    UMAP_SEED,
)
from ssn.db.schema import (
    get_all_constructs,
    get_all_domains,
    get_all_frameworks,
    get_construct_to_primary_domain,
    get_construct_to_primary_domain_for_framework,
    get_constructs_by_domain,
    get_domains_by_framework,
    get_framework_id_for_construct,
    get_items_by_construct,
    leiden_get_communities_with_labels,
    leiden_get_construct_to_community,
    leiden_get_latest_run_id,
    leiden_insert_communities,
    leiden_insert_membership_bulk,
    leiden_insert_run,
)
from ssn.services.embedding_service import (
    get_construct_embeddings,
    get_domain_embeddings,
    load_item_embeddings,
)
from ssn.services.llm_interpret_graph import run_ssn_leiden_domain_interpretation
from ssn.services.similarity_service import (
    compute_pairwise_similarity,
    compute_similarity_distribution,
)
from ssn.services.network_service import build_construct_network, detect_communities

logger = logging.getLogger(__name__)

# Qualitative colors for frameworks (max ~12 distinguishable)
_FRAMEWORK_COLORS = [
    "#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12",
    "#1ABC9C", "#34495E", "#E91E63", "#00BCD4", "#8BC34A",
    "#FF5722", "#607D8B",
]


def _all_pairs_edges(
    sim_matrix: np.ndarray,
    ids: list[str],
) -> list[tuple[str, str, float]]:
    """All pairs (i,j) with i < j; weight = similarity. Returns list of (id1, id2, weight)."""
    n = sim_matrix.shape[0]
    result: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append((ids[i], ids[j], float(sim_matrix[i, j])))
    return result


def _percentile_rank(value: float, all_values: np.ndarray) -> float:
    """Percentile rank of value in all_values (0..1)."""
    if all_values.size == 0:
        return 0.0
    return float(np.mean(all_values <= value))


def _level_stats_from_sims(sims: np.ndarray) -> dict[str, Any]:
    """Per-level stats: mean, std, q25, q75, iqr_warning."""
    if sims.size == 0:
        return {"mean": 0.0, "std": 0.0, "q25": 0.0, "q75": 0.0, "iqr_warning": False}
    q25, q75 = float(np.percentile(sims, 25)), float(np.percentile(sims, 75))
    iqr = q75 - q25
    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "q25": q25,
        "q75": q75,
        "iqr_warning": iqr < 0.05,
    }


def _run_umap_on_similarity(
    sim_matrix: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP on a similarity matrix (converted to distance = 1 - sim). Fallback to PCA if umap not installed."""
    n = sim_matrix.shape[0]
    if n < 3:
        return np.random.RandomState(random_state).randn(n, 2).astype(np.float64)
    # Distance for UMAP: 1 - similarity (clamp similarity to [0,1])
    sim_clip = np.clip(sim_matrix, 0.0, 1.0)
    dist = 1.0 - sim_clip
    np.fill_diagonal(dist, 0.0)
    try:
        import umap
        n_neighbors = min(n_neighbors, n - 1)
        if n_neighbors < 2:
            n_neighbors = 2
        # For small N, spectral init triggers scipy eigsh "k >= N". Use random init
        # when n <= 50 to avoid that (UMAP spectral requests ~n_components+1 evs).
        use_random_init = n <= 50
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric="precomputed",
            init="random" if use_random_init else "spectral",
        )
        coords = reducer.fit_transform(dist)
        return coords.astype(np.float64)
    except ImportError:
        from ssn.services.decomposition_service import run_pca
        # Use similarity matrix as "embeddings" (rows); PCA on it gives 2D layout
        result = run_pca(sim_clip, n_components=2)
        return result["coords"].astype(np.float64)


def _map_local_to_circle(
    local_coords: np.ndarray,
    parent_x: float,
    parent_y: float,
    radius_scale: float,
) -> np.ndarray:
    """Map local 2D coords into a circle centered at (parent_x, parent_y). Radius scales with sqrt(n)."""
    if len(local_coords) == 0:
        return local_coords
    c = local_coords - local_coords.mean(axis=0)
    r = np.sqrt((c ** 2).sum(axis=1) + 1e-12)
    r_max = r.max()
    if r_max < 1e-9:
        r_max = 1.0
    scale = radius_scale / r_max
    out = c * scale
    out[:, 0] += parent_x
    out[:, 1] += parent_y
    return out


def _convex_hull_with_padding(points: np.ndarray, padding_fraction: float = 0.15) -> np.ndarray:
    """Convex hull of points, then expand by padding_fraction of hull diameter."""
    from scipy.spatial import ConvexHull
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        centroid = vertices.mean(axis=0)
        diam = np.sqrt(((vertices.max(axis=0) - vertices.min(axis=0)) ** 2).sum())
        if diam < 1e-9:
            diam = 1.0
        pad = diam * padding_fraction
        expanded = centroid + (vertices - centroid) * (1.0 + pad / (diam / 2 + 1e-9))
        return expanded
    except Exception:
        return points


LEIDEN_DERIVED_FRAMEWORK_ID = "leiden_derived"


def _reassign_domains_with_leiden(
    communities: dict[str, int],
    processed_dir: Path,
    top_k: int = 10,
) -> None:
    """Write Leiden CSVs, run LLM interpretation, persist to dedicated leiden_* tables (not frameworks/domains)."""
    import pandas as pd

    map_csv = processed_dir / "leiden_domain_construct_map.csv"
    labels_csv = processed_dir / "leiden_domain_llm_labels.csv"

    # Step 1: Write construct_id -> community_id CSV
    map_df = pd.DataFrame(
        [{"construct_id": cid, "community_id": comm_id} for cid, comm_id in communities.items()]
    )
    map_df.to_csv(map_csv, index=False, encoding="utf-8")
    logger.info("Wrote Leiden domain construct map to %s", map_csv)

    # Step 2 & 3: Build top items and run LLM interpretation (SSN-specific)
    labels_df = run_ssn_leiden_domain_interpretation(
        communities=communities,
        out_csv=str(labels_csv),
        top_k=top_k,
    )

    # Step 4: Persist to dedicated Leiden tables only
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = leiden_insert_run(created_at=created_at, note="pipeline reassign")
    label_by_cluster = labels_df.set_index("cluster")["label"].to_dict()
    rationale_by_cluster = labels_df.set_index("cluster")["rationale"].to_dict() if "rationale" in labels_df.columns else {}
    unique_comm_ids = sorted(labels_df["cluster"].unique(), key=lambda x: (x,))
    communities_rows = [
        (
            int(comm_id),
            (label_by_cluster.get(comm_id) or "").strip() or f"Community {comm_id}",
            rationale_by_cluster.get(comm_id),
        )
        for comm_id in unique_comm_ids
    ]
    leiden_insert_communities(run_id, communities_rows)
    leiden_insert_membership_bulk(run_id, communities)
    logger.info("Persisted Leiden run_id=%s (%s communities) to leiden_* tables", run_id, len(unique_comm_ids))


def _inject_leiden_into_hierarchy(
    domain_ids: list[str],
    domain_x: dict[str, float],
    domain_y: dict[str, float],
    domain_labels: dict[str, str],
    domain_to_fw: dict[str, str],
    domain_to_constructs: dict[str, list[str]],
    frameworks: list[dict],
    construct_x: dict[str, float],
    construct_y: dict[str, float],
) -> list[str]:
    """Load latest Leiden run from leiden_* tables and inject as synthetic domains/framework.
    Returns list of leiden domain_ids for member_domain_ids_by_fw.
    """
    run_id = leiden_get_latest_run_id()
    if run_id is None:
        return []
    c2c = leiden_get_construct_to_community(run_id)
    comms = leiden_get_communities_with_labels(run_id)
    if not c2c or not comms:
        return []
    # Invert: community_id -> [construct_id]
    comm_to_constructs: dict[int, list[str]] = {}
    for cid, comm_id in c2c.items():
        comm_to_constructs.setdefault(comm_id, []).append(cid)
    label_by_comm = {c["community_id"]: (c.get("label") or "").strip() or f"Community {c['community_id']}" for c in comms}
    leiden_domain_ids: list[str] = []
    for comm in comms:
        comm_id = comm["community_id"]
        domain_id = f"leiden_{comm_id}"
        cids = comm_to_constructs.get(comm_id, [])
        xs = [construct_x[c] for c in cids if c in construct_x]
        ys = [construct_y[c] for c in cids if c in construct_y]
        if not xs or not ys:
            cx, cy = 0.0, 0.0
        else:
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
        domain_ids.append(domain_id)
        domain_x[domain_id] = cx
        domain_y[domain_id] = cy
        domain_labels[domain_id] = label_by_comm.get(comm_id, domain_id)
        domain_to_fw[domain_id] = LEIDEN_DERIVED_FRAMEWORK_ID
        domain_to_constructs[domain_id] = cids
        leiden_domain_ids.append(domain_id)
    if not any(f.get("framework_id") == LEIDEN_DERIVED_FRAMEWORK_ID for f in frameworks):
        frameworks.append({
            "framework_id": LEIDEN_DERIVED_FRAMEWORK_ID,
            "name": "Leiden (data-driven)",
            "version": None,
            "source_url": None,
            "license": None,
            "citation": None,
        })
    return leiden_domain_ids


def run_leiden_domain_reassign(
    processed_dir: Path | None = None,
    top_k: int = 10,
) -> dict[str, int]:
    """Run only Leiden community detection and reassign domains (LLM naming + DB persist). No graph JSON."""
    processed_dir = processed_dir or PROCESSED_DIR
    construct_embs = get_construct_embeddings()
    G_construct = build_construct_network(construct_embs)
    communities = detect_communities(G_construct, method="leiden") if G_construct.number_of_nodes() > 0 else {}
    if not communities:
        logger.warning("No communities from Leiden; skipping reassign")
        return {}
    _reassign_domains_with_leiden(communities, processed_dir, top_k=top_k)
    return communities


def run_pipeline(
    output_path: Path | None = None,
    reassign_domains_with_leiden: bool = False,
) -> dict[str, Any]:
    """Run the full graph data pipeline and return the output dict (and write JSON if output_path set)."""
    output_path = output_path or SSN_GRAPH_JSON_PATH
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load embeddings and hierarchy ---
    item_embeddings_matrix, item_ids_ordered = load_item_embeddings()
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids_ordered)}
    construct_embs = get_construct_embeddings()
    domain_embs = get_domain_embeddings()

    frameworks = get_all_frameworks()
    domains = get_all_domains()
    constructs = get_all_constructs()
    if PRIMARY_FRAMEWORK_ID:
        domains = [d for d in domains if d["framework_id"] == PRIMARY_FRAMEWORK_ID]
        construct_to_domain = get_construct_to_primary_domain_for_framework(PRIMARY_FRAMEWORK_ID)
        if not domains:
            logger.warning("PRIMARY_FRAMEWORK_ID=%s has no domains; using all frameworks", PRIMARY_FRAMEWORK_ID)
            domains = get_all_domains()
            construct_to_domain = get_construct_to_primary_domain()
    else:
        construct_to_domain = get_construct_to_primary_domain()
    domain_to_fw: dict[str, str] = {d["domain_id"]: d["framework_id"] for d in domains}

    fw_ids = [f["framework_id"] for f in frameworks]
    fw_labels = {f["framework_id"]: f["name"] for f in frameworks}
    domain_labels = {d["domain_id"]: d["name"] for d in domains}
    construct_labels = {c["construct_id"]: c.get("name", c["construct_id"]) for c in constructs}
    construct_to_fw: dict[str, str] = {
        c["construct_id"]: (c.get("framework_id") or "") or (get_framework_id_for_construct(c["construct_id"]) or "")
        for c in constructs
    }

    # Domain order consistent with domain_embs (only real domains; may be empty)
    domain_ids_in_use = {d["domain_id"] for d in domains}
    domain_ids = [d for d in domain_embs.keys() if d in domain_ids_in_use]
    if not domain_ids:
        domain_ids = list(domain_embs.keys())
    if domain_ids:
        domain_matrix = np.array([domain_embs[d] for d in domain_ids], dtype=np.float64)
        domain_matrix = np.nan_to_num(domain_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(domain_matrix, axis=1, keepdims=True)
        domain_matrix = np.where(norms > 1e-12, domain_matrix / np.maximum(norms, 1e-12), 0.0)
    else:
        domain_matrix = np.zeros((0, 0), dtype=np.float64)

    construct_ids = [c for c in construct_embs.keys()]
    construct_matrix = np.array([construct_embs[c] for c in construct_ids], dtype=np.float64)
    construct_matrix = np.nan_to_num(construct_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    norms_c = np.linalg.norm(construct_matrix, axis=1, keepdims=True)
    construct_matrix = np.where(norms_c > 1e-12, construct_matrix / np.maximum(norms_c, 1e-12), 0.0)

    # --- 2. Similarity matrices and all-pairs edges (no threshold, no k-NN) ---
    # Domain-Domain: only when we have real domains
    if domain_ids:
        sim_domain = compute_pairwise_similarity(domain_matrix, domain_matrix, "cosine")
        sim_domain = np.nan_to_num(sim_domain, nan=0.0, posinf=1.0, neginf=0.0)
        sim_domain = np.clip(sim_domain, -1.0, 1.0)
        domain_edges = _all_pairs_edges(sim_domain, domain_ids)
        domain_sims_upper = sim_domain[np.triu_indices(len(domain_ids), k=1)]
        level_stats_domain = _level_stats_from_sims(domain_sims_upper)
    else:
        domain_edges = []
        level_stats_domain = _level_stats_from_sims(np.array([]))

    # Construct-Construct: TMFG-filtered edges (G_construct) for clearer visualization
    sim_construct = compute_pairwise_similarity(construct_matrix, construct_matrix, "cosine")
    sim_construct = np.nan_to_num(sim_construct, nan=0.0, posinf=1.0, neginf=0.0)
    sim_construct = np.clip(sim_construct, -1.0, 1.0)
    G_construct = build_construct_network(construct_embs)
    construct_edges = [
        (u, v, G_construct.edges[u, v]["weight"])
        for u, v in G_construct.edges()
    ]
    construct_sims_upper = sim_construct[np.triu_indices(len(construct_ids), k=1)]
    level_stats_construct = _level_stats_from_sims(construct_sims_upper)

    # --- 3. UMAP layouts ---
    # Step 1: Global Domain layout (only when we have real domains)
    domain_x = {}
    domain_y = {}
    if domain_ids:
        domain_coords = _run_umap_on_similarity(
            sim_domain.copy(),
            n_neighbors=UMAP_DOMAIN_NEIGHBORS,
            min_dist=UMAP_DOMAIN_MIN_DIST,
            random_state=UMAP_SEED,
        )
        domain_x = {domain_ids[i]: float(domain_coords[i, 0]) for i in range(len(domain_ids))}
        domain_y = {domain_ids[i]: float(domain_coords[i, 1]) for i in range(len(domain_ids))}

    # Step 2: Per-Domain Construct layout (local -> circle around domain) for constructs with domain
    construct_x: dict[str, float] = {}
    construct_y: dict[str, float] = {}
    domain_to_constructs: dict[str, list[str]] = {}
    for cid in construct_ids:
        did = construct_to_domain.get(cid)
        if did:
            domain_to_constructs.setdefault(did, []).append(cid)
    for did, cids in domain_to_constructs.items():
        if not cids or did not in domain_x:
            continue
        sub_matrix = np.array([construct_embs[c] for c in cids], dtype=np.float64)
        sim_sub = compute_pairwise_similarity(sub_matrix, sub_matrix, "cosine")
        local = _run_umap_on_similarity(
            sim_sub,
            n_neighbors=min(UMAP_CONSTRUCT_NEIGHBORS, len(cids) - 1 if len(cids) > 1 else 2),
            min_dist=UMAP_CONSTRUCT_MIN_DIST,
            random_state=UMAP_SEED,
        )
        radius = 1.0 * (len(cids) ** 0.5)
        mapped = _map_local_to_circle(
            local,
            domain_x[did],
            domain_y[did],
            radius,
        )
        for i, cid in enumerate(cids):
            construct_x[cid] = float(mapped[i, 0])
            construct_y[cid] = float(mapped[i, 1])

    # Step 2b: Global UMAP for constructs without domain (place offset from domain cluster)
    no_domain_cids = [c for c in construct_ids if c not in construct_x]
    if no_domain_cids:
        sub_matrix = np.array([construct_embs[c] for c in no_domain_cids], dtype=np.float64)
        sub_matrix = np.nan_to_num(sub_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(sub_matrix, axis=1, keepdims=True)
        sub_matrix = np.where(norms > 1e-12, sub_matrix / np.maximum(norms, 1e-12), 0.0)
        sim_sub = compute_pairwise_similarity(sub_matrix, sub_matrix, "cosine")
        local = _run_umap_on_similarity(
            sim_sub,
            n_neighbors=min(UMAP_CONSTRUCT_NEIGHBORS, len(no_domain_cids) - 1 if len(no_domain_cids) > 1 else 2),
            min_dist=UMAP_CONSTRUCT_MIN_DIST,
            random_state=UMAP_SEED,
        )
        # Center and scale; offset to the right of domain cluster to avoid overlap
        offset_x = (float(np.mean(list(domain_x.values()))) + 8.0) if domain_x else 0.0
        offset_y = float(np.mean(list(domain_y.values()))) if domain_y else 0.0
        c_centroid = local.mean(axis=0)
        r_max = np.sqrt(((local - c_centroid) ** 2).sum(axis=1) + 1e-12).max()
        if r_max < 1e-9:
            r_max = 1.0
        scale = 3.0 / r_max
        for i, cid in enumerate(no_domain_cids):
            construct_x[cid] = float((local[i, 0] - c_centroid[0]) * scale + offset_x)
            construct_y[cid] = float((local[i, 1] - c_centroid[1]) * scale + offset_y)

    # Inject latest Leiden run from dedicated tables as synthetic domains (centroids of member constructs)
    leiden_domain_ids = _inject_leiden_into_hierarchy(
        domain_ids, domain_x, domain_y, domain_labels, domain_to_fw, domain_to_constructs,
        frameworks, construct_x, construct_y,
    )

    # Step 3: Per-Construct Item layout (for Phase 2; we compute for JSON completeness)
    item_x: dict[str, float] = {}
    item_y: dict[str, float] = {}
    items_with_coords: list[dict[str, Any]] = []  # {item_id, construct_id, text} for nodes/search
    for cid in construct_ids:
        items = get_items_by_construct(cid)
        item_ids_c = [str(it["item_id"]) for it in items if str(it["item_id"]) in item_id_to_idx]
        if not item_ids_c:
            continue
        idx_c = [item_id_to_idx[iid] for iid in item_ids_c]
        sub_matrix = item_embeddings_matrix[idx_c]
        if len(item_ids_c) < 2:
            iid = item_ids_c[0]
            item_x[iid] = construct_x.get(cid, 0.0)
            item_y[iid] = construct_y.get(cid, 0.0)
            items_with_coords.append({
                "item_id": iid,
                "construct_id": cid,
                "text": next((it.get("text", "") for it in items if str(it["item_id"]) == iid), ""),
            })
            continue
        sim_item = compute_pairwise_similarity(sub_matrix, sub_matrix, "cosine")
        local = _run_umap_on_similarity(
            sim_item,
            n_neighbors=min(UMAP_ITEM_NEIGHBORS, len(item_ids_c) - 1),
            min_dist=UMAP_ITEM_MIN_DIST,
            random_state=UMAP_SEED,
        )
        cx, cy = construct_x.get(cid, 0.0), construct_y.get(cid, 0.0)
        radius = 0.8 * (len(item_ids_c) ** 0.5)
        mapped = _map_local_to_circle(local, cx, cy, radius)
        for i, iid in enumerate(item_ids_c):
            item_x[iid] = float(mapped[i, 0])
            item_y[iid] = float(mapped[i, 1])
        for i, iid in enumerate(item_ids_c):
            text = next((it.get("text", "") for it in items if str(it["item_id"]) == iid), "")
            items_with_coords.append({"item_id": iid, "construct_id": cid, "text": text})

    # --- 4. Framework convex hulls (from construct positions; domain membership for UI) ---
    hull_points_by_fw: dict[str, list[list[float]]] = {}
    member_domain_ids_by_fw: dict[str, list[str]] = {}
    for fw in frameworks:
        fid = fw["framework_id"]
        if fid == LEIDEN_DERIVED_FRAMEWORK_ID:
            dids = [d for d in leiden_domain_ids if d in domain_x]
        else:
            dids = [d["domain_id"] for d in get_domains_by_framework(fid) if d["domain_id"] in domain_x]
        member_domain_ids_by_fw[fid] = dids
        # Hull from all constructs in this framework (with and without domain)
        fw_constructs = [c for c in construct_ids if construct_to_fw.get(c) == fid and c in construct_x]
        if len(fw_constructs) < 3:
            continue
        pts = np.array([[construct_x[c], construct_y[c]] for c in fw_constructs])
        hull = _convex_hull_with_padding(pts, HULL_PADDING_FRACTION)
        hull_points_by_fw[fid] = [[float(hull[i, 0]), float(hull[i, 1])] for i in range(len(hull))]

    # --- 5. Visibility levels (ZMLT-inspired: 1-2 Domain, 3-4 Construct, 5-7 Item) ---
    # Use degree centrality on construct graph for within-level ordering (G_construct from step 2)
    if G_construct.number_of_nodes() > 0:
        import networkx as nx
        degree_c = nx.degree_centrality(G_construct)
        construct_degree = {str(n): degree_c.get(n, 0.0) for n in G_construct.nodes()}
    else:
        construct_degree = {c: 0.0 for c in construct_ids}

    domain_degrees = {}
    for did in domain_ids:
        cids = domain_to_constructs.get(did, [])
        domain_degrees[did] = sum(construct_degree.get(c, 0) for c in cids) / (len(cids) or 1)

    # Assign visibility_level: Domain 1-2, Construct 3-4, Item 5-7
    domain_sorted = sorted(domain_ids, key=lambda d: -domain_degrees.get(d, 0))
    n_d = len(domain_sorted)
    domain_vis: dict[str, int] = {}
    for i, d in enumerate(domain_sorted):
        domain_vis[d] = 1 if i < (n_d // 2) else 2

    construct_sorted = sorted(construct_ids, key=lambda c: -construct_degree.get(c, 0))
    n_c = len(construct_sorted)
    construct_vis: dict[str, int] = {}
    for i, c in enumerate(construct_sorted):
        if i < n_c // 2:
            construct_vis[c] = 3
        else:
            construct_vis[c] = 4

    item_vis: dict[str, int] = {}
    for cid in construct_ids:
        items = get_items_by_construct(cid)
        for it in items:
            iid = str(it["item_id"])
            item_vis[iid] = 5  # simplified: all items same level for Phase 1

    # --- 6. Community (Leiden) for theory/data grouping - single resolution for Phase 1 ---
    communities = detect_communities(G_construct, method="leiden") if G_construct.number_of_nodes() > 0 else {}
    if reassign_domains_with_leiden and communities:
        _reassign_domains_with_leiden(communities, PROCESSED_DIR, top_k=10)
    unique_comm = list(dict.fromkeys(communities.values()))
    one_res = {
        f"c_{cid}": {
            "label": f"Community {cid}",
            "color": _FRAMEWORK_COLORS[cid % len(_FRAMEWORK_COLORS)],
            "member_ids": [n for n, v in communities.items() if v == cid],
        }
        for cid in unique_comm
    }
    community_groups: dict[str, dict[str, Any]] = {
        "res_0.5": one_res,
        "res_1.0": one_res,
        "res_2.0": one_res,
    }
    node_community: dict[str, dict[str, str]] = {}
    for nid, cid in communities.items():
        node_community[nid] = {"res_0.5": f"c_{cid}", "res_1.0": f"c_{cid}", "res_2.0": f"c_{cid}"}

    # --- 7. Build nodes list ---
    nodes: list[dict[str, Any]] = []
    for d in domain_ids:
        fw_id = domain_to_fw.get(d, "")
        nodes.append({
            "id": d,
            "label": domain_labels.get(d, d),
            "level": "domain",
            "visibility_level": domain_vis.get(d, 1),
            "x": domain_x[d],
            "y": domain_y[d],
            "parent_id": fw_id,
            "framework_id": fw_id,
            "theory_group": fw_id,
            "community_group": node_community.get(d, {}),
            "n_children": len(domain_to_constructs.get(d, [])),
            "weight": float(domain_degrees.get(d, 0.5)),
        })
    for c in construct_ids:
        did = construct_to_domain.get(c, "")
        fw_id = construct_to_fw.get(c, "") or domain_to_fw.get(did, "")
        parent_id = did if did else fw_id
        n_items = len(get_items_by_construct(c))
        nodes.append({
            "id": c,
            "label": construct_labels.get(c, c),
            "level": "construct",
            "visibility_level": construct_vis.get(c, 3),
            "x": construct_x.get(c, 0.0),
            "y": construct_y.get(c, 0.0),
            "parent_id": parent_id,
            "framework_id": fw_id,
            "theory_group": did if did else fw_id,
            "community_group": node_community.get(c, {}),
            "n_children": n_items,
            "weight": float(construct_degree.get(c, 0.5)),
        })
    for it in items_with_coords:
        iid = it["item_id"]
        cid = it["construct_id"]
        text = it.get("text", "")
        did = construct_to_domain.get(cid, "")
        fw_id = construct_to_fw.get(cid, "") or domain_to_fw.get(did, "")
        nodes.append({
            "id": iid,
            "label": (text or iid)[:80] + ("..." if len(text) > 80 else ""),
            "level": "item",
            "visibility_level": item_vis.get(iid, 5),
            "x": item_x[iid],
            "y": item_y[iid],
            "parent_id": cid,
            "framework_id": fw_id,
            "theory_group": cid,
            "community_group": {},
            "n_children": 0,
            "weight": 0.2,
            "item_text": text,
            "instrument_id": "",
        })

    # --- 8. Build edges list ---
    edges: list[dict[str, Any]] = []
    # Domain-Domain meta-edges (aggregate construct-construct)
    for (a, b, w) in domain_edges:
        cids_a = set(domain_to_constructs.get(a, []))
        cids_b = set(domain_to_constructs.get(b, []))
        max_sim = 0.0
        max_pair = [a, b]
        n_underlying = 0
        total = 0.0
        for ca in cids_a:
            for cb in cids_b:
                if ca == cb:
                    continue
                ia, ib = construct_ids.index(ca), construct_ids.index(cb)
                s = sim_construct[ia, ib]
                n_underlying += 1
                total += s
                if s > max_sim:
                    max_sim = s
                    max_pair = [ca, cb]
        pct = _percentile_rank(max_sim, construct_sims_upper)
        edges.append({
            "source": a,
            "target": b,
            "weight": max_sim,
            "percentile": pct,
            "visibility_level": 1,
            "is_meta": True,
            "n_underlying": n_underlying,
            "max_pair": max_pair,
            "mean_weight": total / n_underlying if n_underlying else 0.0,
        })
    # Construct-Construct edges (actual); visibility_level = max of endpoints
    for (a, b, w) in construct_edges:
        pct = _percentile_rank(w, construct_sims_upper)
        vis = max(construct_vis.get(a, 3), construct_vis.get(b, 3))
        edges.append({
            "source": a,
            "target": b,
            "weight": w,
            "percentile": pct,
            "visibility_level": vis,
            "is_meta": False,
        })

    # --- 9. Level stats (item: from construct-level proxy for Phase 1) ---
    level_stats_item = _level_stats_from_sims(construct_sims_upper)  # placeholder
    level_stats = {
        "domain": level_stats_domain,
        "construct": level_stats_construct,
        "item": level_stats_item,
    }

    # --- 10. Search index ---
    search_index: list[dict[str, Any]] = []
    for d in domain_ids:
        fw_name = fw_labels.get(domain_to_fw.get(d, ""), "")
        search_index.append({
            "id": d,
            "label": domain_labels.get(d, d),
            "path": f"{fw_name} → {domain_labels.get(d, d)}",
            "level": "domain",
        })
    for c in construct_ids:
        did = construct_to_domain.get(c, "")
        fw_name = fw_labels.get(construct_to_fw.get(c, "") or domain_to_fw.get(did, ""), "")
        d_name = domain_labels.get(did, "")
        c_label = construct_labels.get(c, c)
        path = f"{fw_name} → {d_name} → {c_label}" if d_name else f"{fw_name} → {c_label}"
        search_index.append({
            "id": c,
            "label": c_label,
            "path": path,
            "level": "construct",
        })
    for it in items_with_coords:
        iid = it["item_id"]
        cid = it["construct_id"]
        text = (it.get("text", "") or "")[:40] + ("..." if len(it.get("text", "")) > 40 else "")
        did = construct_to_domain.get(cid, "")
        fw_name = fw_labels.get(construct_to_fw.get(cid, "") or domain_to_fw.get(did, ""), "")
        d_name = domain_labels.get(did, "")
        c_name = construct_labels.get(cid, cid)
        path = f"{fw_name} → {d_name} → {c_name} → {text}" if d_name else f"{fw_name} → {c_name} → {text}"
        search_index.append({
            "id": iid,
            "label": text,
            "path": path,
            "level": "item",
        })

    # --- 11. Frameworks list for JSON ---
    frameworks_out = []
    for i, fw in enumerate(frameworks):
        fid = fw["framework_id"]
        frameworks_out.append({
            "id": fid,
            "label": fw.get("name", fid),
            "color": _FRAMEWORK_COLORS[i % len(_FRAMEWORK_COLORS)],
            "hull_points": hull_points_by_fw.get(fid, []),
            "member_domain_ids": member_domain_ids_by_fw.get(fid, []),
        })

    out = {
        "metadata": {
            "version": "2.0",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": UMAP_SEED,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
        "frameworks": frameworks_out,
        "nodes": nodes,
        "edges": edges,
        "level_stats": level_stats,
        "community_groups": community_groups,
        "search_index": search_index,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote graph JSON to {output_path} ({len(nodes)} nodes, {len(edges)} edges)")
    return out


if __name__ == "__main__":
    import os as _os
    logging.basicConfig(level=logging.INFO)
    run_pipeline(reassign_domains_with_leiden=_os.environ.get("REASSIGN_DOMAINS_WITH_LEIDEN", "").strip() in ("1", "true", "yes"))
