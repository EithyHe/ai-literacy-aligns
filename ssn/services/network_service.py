"""Construct association network service for the Semantic Scale Network."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ssn.services.similarity_service import compute_pairwise_similarity

logger = logging.getLogger(__name__)

# Optional: Leiden via igraph + leidenalg
try:
    import igraph as ig
    import leidenalg as la

    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    ig = None  # type: ignore
    la = None  # type: ignore
    logger.debug("igraph/leidenalg not available; will use networkx Louvain fallback")

# Optional: TMFG for clearer network visualization (fewer, stronger edges)
try:
    from fast_tmfg import TMFG as _TMFG

    TMFG_AVAILABLE = True
except ImportError:
    TMFG_AVAILABLE = False
    _TMFG = None  # type: ignore
    logger.debug("fast-tmfg not available; network will be fully connected")


def _apply_tmfg(
    ids: list[str],
    matrix: np.ndarray,
    sim_matrix: np.ndarray,
) -> nx.Graph:
    """Build a Triangulated Maximally Filtered Graph from similarity matrix.

    TMFG keeps at most 3n-6 edges (planar triangulation), retaining the
    strongest similarities so the network visualization is clearer.
    """
    if not TMFG_AVAILABLE or _TMFG is None:
        return None
    n = len(ids)
    if n < 4:
        return None
    try:
        # TMFG expects non-negative weights; map similarity [-1,1] -> [0,1]
        weights = np.clip((sim_matrix + 1.0) / 2.0, 0.0, 1.0).astype(np.float64)
        # Covariance of constructs (rows = constructs); shape (n, n)
        cov = np.cov(matrix, rowvar=True)
        if cov.shape != (n, n):
            return None
        # fast_tmfg may call .to_numpy() on inputs; pass DataFrames to satisfy that
        weights_df = pd.DataFrame(weights)
        cov_df = pd.DataFrame(cov)
        model = _TMFG()
        _, _, adj = model.fit_transform(weights_df, "weighted_sparse_W_matrix", cov=cov_df)
        try:
            from scipy.sparse import issparse
            if issparse(adj):
                adj = adj.toarray()
        except ImportError:
            pass
        adj = np.asarray(adj)
        if adj.ndim != 2 or adj.shape[0] != n or adj.shape[1] != n:
            return None
        G = nx.Graph()
        G.add_nodes_from(ids)
        for i in range(n):
            for j in range(i + 1, n):
                w = adj[i, j]
                if w is not None and float(w) != 0:
                    # Restore original similarity as edge weight for display
                    G.add_edge(ids[i], ids[j], weight=float(sim_matrix[i, j]))
        if G.number_of_edges() == 0:
            return None
        logger.info(
            f"TMFG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
            f"(planar, clearer layout)"
        )
        return G
    except Exception as e:
        logger.warning(f"TMFG failed, using full graph: {e}")
        return None


def _build_threshold_network(
    ids: list[str],
    sim_matrix: np.ndarray,
    min_weight: float = 0.2,
) -> nx.Graph:
    """Build a thresholded construct graph from similarity matrix."""
    G = nx.Graph()
    G.add_nodes_from(ids)
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= min_weight:
                G.add_edge(ids[i], ids[j], weight=sim)
    return G


def _apply_disparity_backbone(G: nx.Graph, alpha: float = 0.05) -> nx.Graph:
    """Apply disparity filter backbone extraction on a weighted undirected graph.

    Keep edge (i, j) if alpha_ij < alpha OR alpha_ji < alpha.
    """
    B = nx.Graph()
    B.add_nodes_from(G.nodes(data=True))
    if G.number_of_edges() == 0:
        return B

    strengths: dict[str, float] = {}
    degrees: dict[str, int] = {}
    for node in G.nodes():
        weights = [float(data.get("weight", 0.0)) for _, _, data in G.edges(node, data=True)]
        strengths[node] = float(sum(weights))
        degrees[node] = int(G.degree(node))

    def edge_alpha(u: str, v: str, w: float) -> float:
        k = degrees.get(u, 0)
        s = strengths.get(u, 0.0)
        if k <= 1 or s <= 0:
            return 0.0
        p_ij = max(0.0, min(1.0, w / s))
        return float((1.0 - p_ij) ** (k - 1))

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 0.0))
        a_uv = edge_alpha(u, v, w)
        a_vu = edge_alpha(v, u, w)
        if a_uv < alpha or a_vu < alpha:
            B.add_edge(u, v, **data)
    return B


def build_construct_network(
    construct_embeddings: dict[str, np.ndarray],
    metric: str = "cosine",
    mode: str = "backbone",
    backbone_method: str = "disparity",
    alpha: float = 0.05,
    min_weight: float = 0.2,
    use_tmfg: bool | None = None,
) -> nx.Graph:
    """Build construct association network with full/backbone modes.

    Args:
        construct_embeddings: Dict mapping construct ID to embedding vector.
        metric: Similarity metric for pairwise comparison.
        mode: "full" or "backbone".
        backbone_method: "disparity" (default) or "tmfg".
        alpha: Disparity significance threshold.
        min_weight: Edge threshold for full graph construction.
        use_tmfg: Deprecated compatibility flag. If True and mode is not set to
            full explicitly, forces backbone_method="tmfg".

    Returns:
        Undirected networkx Graph with weighted construct edges.
    """
    if not construct_embeddings:
        logger.warning("Empty construct_embeddings; returning empty graph")
        return nx.Graph()

    mode_norm = (mode or "backbone").lower()
    if mode_norm not in {"full", "backbone"}:
        raise ValueError(f"mode must be 'full' or 'backbone', got {mode!r}")
    backbone_method_norm = (backbone_method or "disparity").lower()
    if backbone_method_norm not in {"disparity", "tmfg"}:
        raise ValueError(
            f"backbone_method must be 'disparity' or 'tmfg', got {backbone_method!r}"
        )
    if use_tmfg is True and mode_norm != "full":
        backbone_method_norm = "tmfg"

    ids = list(construct_embeddings.keys())
    matrix = np.array([construct_embeddings[i] for i in ids], dtype=np.float64)
    sim_matrix = compute_pairwise_similarity(matrix, matrix, metric)
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    full_graph = _build_threshold_network(ids, sim_matrix, min_weight=min_weight)
    if mode_norm == "full":
        logger.info(
            "Construct network(full): %s nodes, %s edges, min_weight=%.3f",
            full_graph.number_of_nodes(),
            full_graph.number_of_edges(),
            min_weight,
        )
        return full_graph

    if backbone_method_norm == "tmfg":
        tmfg_graph = _apply_tmfg(ids, matrix, sim_matrix)
        if tmfg_graph is not None:
            logger.info(
                "Construct network(backbone=tmfg): %s nodes, %s edges",
                tmfg_graph.number_of_nodes(),
                tmfg_graph.number_of_edges(),
            )
            return tmfg_graph
        logger.warning("TMFG unavailable/failed; falling back to disparity backbone")

    backbone = _apply_disparity_backbone(full_graph, alpha=alpha)
    logger.info(
        "Construct network(backbone=disparity): %s nodes, %s edges (full=%s edges)",
        backbone.number_of_nodes(),
        backbone.number_of_edges(),
        full_graph.number_of_edges(),
    )
    return backbone


def compute_network_metrics(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute topology metrics per node: degree centrality, betweenness,
    closeness, clustering coefficient.

    Args:
        G: Networkx graph (undirected).

    Returns:
        Dict mapping node_id -> {degree_centrality, betweenness, closeness,
        clustering_coefficient}. Values are 0 for disconnected graphs where
        metric is undefined.
    """
    if G.number_of_nodes() == 0:
        return {}

    result: dict[str, dict[str, float]] = {}
    weight_key = "weight" if nx.get_edge_attributes(G, "weight") else None

    try:
        degree_c = nx.degree_centrality(G)
    except Exception as e:
        logger.warning(f"degree_centrality failed: {e}")
        degree_c = {n: 0.0 for n in G.nodes()}

    try:
        betweenness = nx.betweenness_centrality(G, weight=weight_key)
    except Exception as e:
        logger.warning(f"betweenness_centrality failed: {e}")
        betweenness = {n: 0.0 for n in G.nodes()}

    try:
        # Use unweighted closeness: similarity weights would need 1/weight as distance
        closeness = nx.closeness_centrality(G)
    except Exception as e:
        logger.warning(f"closeness_centrality failed: {e}")
        closeness = {n: 0.0 for n in G.nodes()}

    try:
        clustering = nx.clustering(G, weight=weight_key)
    except Exception as e:
        logger.warning(f"clustering failed: {e}")
        clustering = {n: 0.0 for n in G.nodes()}

    for node in G.nodes():
        result[str(node)] = {
            "degree_centrality": float(degree_c.get(node, 0.0)),
            "betweenness": float(betweenness.get(node, 0.0)),
            "closeness": float(closeness.get(node, 0.0)),
            "clustering_coefficient": float(clustering.get(node, 0.0)),
        }

    return result


def _nx_to_igraph(G: nx.Graph) -> "ig.Graph":
    """Convert networkx Graph to igraph Graph (requires igraph)."""
    if not LEIDEN_AVAILABLE:
        raise ImportError("igraph and leidenalg required for Leiden community detection")

    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    weights = [
        G.edges[u, v].get("weight", 1.0) for u, v in G.edges()
    ]

    G_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)
    G_ig.vs["name"] = node_list
    G_ig.es["weight"] = weights
    return G_ig


def _igraph_partition_to_communities(
    G: nx.Graph, G_ig: "ig.Graph", partition: Any
) -> dict[str, int]:
    """Map igraph partition to {node_id: community_id} for original nx graph."""
    node_list = list(G.nodes())
    return {
        node_list[i]: int(partition.membership[i])
        for i in range(len(node_list))
    }


def detect_communities(
    G: nx.Graph,
    method: str = "leiden",
) -> dict[str, int]:
    """Detect communities using Leiden (default) or Louvain.

    Args:
        G: Networkx graph (undirected).
        method: "leiden" or "louvain".

    Returns:
        Dict mapping node_id -> community_id (0-based integer).
    """
    if G.number_of_nodes() == 0:
        return {}

    method = method.lower()
    if method not in ("leiden", "louvain"):
        raise ValueError(f"method must be 'leiden' or 'louvain', got {method!r}")

    if method == "leiden" and LEIDEN_AVAILABLE:
        try:
            G_ig = _nx_to_igraph(G)
            # ModularityVertexPartition requires non-negative weights. Similarity
            # (e.g. cosine) can be in [-1, 1]; map to [0, 1] for Leiden.
            if G.number_of_edges() > 0:
                raw = G_ig.es["weight"]
                weights = [max(0.0, min(1.0, (1.0 + float(w)) / 2.0)) for w in raw]
            else:
                weights = None
            partition = la.find_partition(
                G_ig,
                la.ModularityVertexPartition,
                weights=weights,
                seed=42,
            )
            return _igraph_partition_to_communities(G, G_ig, partition)
        except Exception as e:
            logger.warning(f"Leiden failed, falling back to Louvain: {e}")
            method = "louvain"

    # Louvain via networkx
    try:
        communities = nx.community.louvain_communities(
            G, weight="weight", seed=42
        )
    except Exception as e:
        logger.error(f"Louvain community detection failed: {e}")
        raise

    return {
        str(node): cid
        for cid, comm in enumerate(communities)
        for node in comm
    }


def get_bridge_nodes(
    G: nx.Graph,
    communities: dict[str, int],
) -> list[dict]:
    """Identify bridge nodes connecting different communities.

    A bridge node has neighbors in more than one community.

    Args:
        G: Networkx graph.
        communities: Dict mapping node_id -> community_id.

    Returns:
        List of dicts with keys: node_id, community_id, n_communities_adjacent,
        neighbor_communities. Sorted by n_communities_adjacent descending.
    """
    bridges: list[dict] = []
    for node in G.nodes():
        my_comm = communities.get(node, -1)
        neighbor_comms = {
            communities.get(neigh, -1)
            for neigh in G.neighbors(node)
        }
        neighbor_comms.discard(-1)
        n_adjacent = len(neighbor_comms)
        if n_adjacent > 1 or (n_adjacent == 1 and my_comm not in neighbor_comms):
            bridges.append({
                "node_id": str(node),
                "community_id": my_comm,
                "n_communities_adjacent": n_adjacent,
                "neighbor_communities": sorted(neighbor_comms),
            })

    bridges.sort(key=lambda x: x["n_communities_adjacent"], reverse=True)
    return bridges


def get_network_summary(G: nx.Graph) -> dict:
    """Return summary: n_nodes, n_edges, density, avg_clustering, modularity,
    connected_components.

    Args:
        G: Networkx graph.

    Returns:
        Dict with summary statistics.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    density = nx.density(G) if n_nodes > 0 else 0.0

    try:
        avg_clustering = nx.average_clustering(G, weight="weight")
    except Exception:
        avg_clustering = 0.0

    modularity = 0.0
    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
        modularity = nx.community.modularity(G, communities, weight="weight")
    except Exception as e:
        logger.debug(f"Modularity computation failed: {e}")

    n_components = nx.number_connected_components(G) if n_nodes > 0 else 0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(density),
        "avg_clustering": float(avg_clustering),
        "modularity": float(modularity),
        "connected_components": n_components,
    }


def get_ego_network(
    G: nx.Graph,
    node_id: str,
    radius: int = 2,
) -> nx.Graph:
    """Extract ego network (subgraph) around a specific node.

    Args:
        G: Full networkx graph.
        node_id: Center node.
        radius: Number of hops (1 = direct neighbors only).

    Returns:
        Subgraph induced by nodes within radius of node_id.
    """
    if node_id not in G:
        raise KeyError(f"Node {node_id!r} not in graph")

    nodes_in_ego = nx.ego_graph(G, node_id, radius=radius)
    return G.subgraph(nodes_in_ego).copy()


def _umap_layout_from_similarity(
    sim_matrix: np.ndarray,
    ids: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Run UMAP on similarity matrix (distance = 1 - sim). Same as main corpus pipeline. Returns id -> (x, y)."""
    n = sim_matrix.shape[0]
    if n < 2:
        rng = np.random.RandomState(random_state)
        return {ids[i]: (float(rng.randn()), float(rng.randn())) for i in range(n)}
    sim_clip = np.clip(sim_matrix, 0.0, 1.0)
    dist = 1.0 - sim_clip
    np.fill_diagonal(dist, 0.0)
    try:
        import umap
        n_neighbors = min(n_neighbors, n - 1)
        if n_neighbors < 2:
            n_neighbors = 2
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
        return {ids[i]: (float(coords[i, 0]), float(coords[i, 1])) for i in range(n)}
    except ImportError:
        from ssn.services.decomposition_service import run_pca
        result = run_pca(sim_clip, n_components=2)
        coords = result["coords"]
        return {ids[i]: (float(coords[i, 0]), float(coords[i, 1])) for i in range(n)}


def compute_umap_layout_for_constructs(
    construct_embeddings: dict[str, np.ndarray],
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute 2D UMAP layout from construct embeddings (cosine similarity). Matches main corpus pipeline layout."""
    if not construct_embeddings:
        return {}
    ids = list(construct_embeddings.keys())
    matrix = np.array([construct_embeddings[i] for i in ids], dtype=np.float64)
    sim_matrix = compute_pairwise_similarity(matrix, matrix, "cosine")
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
    return _umap_layout_from_similarity(
        sim_matrix, ids,
        n_neighbors=min(n_neighbors, len(ids) - 1) if len(ids) > 1 else 2,
        min_dist=min_dist,
        random_state=random_state,
    )


def network_to_plotly_data(
    G: nx.Graph,
    layout: str = "spring",
    communities: dict[str, int] | None = None,
    positions: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """Convert networkx graph to Plotly-compatible node/edge data for visualization.

    Args:
        G: Networkx graph.
        layout: "spring", "kamada_kawai", "circular", or "shell" (ignored if positions given).
        communities: Optional dict mapping node_id -> community_id for coloring.
        positions: Optional precomputed node positions (e.g. from graph_data or UMAP). When set, layout is ignored.

    Returns:
        Dict with keys: node_x, node_y, node_text, node_color, node_ids,
        edge_x, edge_y. Suitable for Plotly scatter (nodes) and scatter/line (edges).
    """
    if G.number_of_nodes() == 0:
        return {
            "node_x": [],
            "node_y": [],
            "node_text": [],
            "node_color": [],
            "node_ids": [],
            "edge_x": [],
            "edge_y": [],
            "edge_weights": [],
        }

    node_list = list(G.nodes())
    if positions is not None:
        # Use precomputed layout (e.g. from main corpus graph_data / UMAP)
        pos = {n: positions[n] for n in node_list if n in positions}
        missing = [n for n in node_list if n not in pos]
        if missing:
            weight_key = "weight" if nx.get_edge_attributes(G, "weight") else None
            try:
                fallback = nx.spring_layout(G.subgraph(missing), seed=42, weight=weight_key)
                for n in missing:
                    pos[n] = fallback.get(n, (0.0, 0.0))
            except Exception:
                for n in missing:
                    pos[n] = (0.0, 0.0)
    else:
        weight_key = "weight" if nx.get_edge_attributes(G, "weight") else None
        try:
            if layout == "spring":
                pos = nx.spring_layout(G, seed=42, weight=weight_key)
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G, weight=weight_key)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "shell":
                pos = nx.shell_layout(G)
            else:
                pos = nx.spring_layout(G, seed=42, weight=weight_key)
        except Exception as e:
            logger.warning(f"Layout {layout} failed, using spring: {e}")
            pos = nx.spring_layout(G, seed=42)

    node_x = [float(pos[n][0]) for n in node_list]
    node_y = [float(pos[n][1]) for n in node_list]
    node_text = [str(n) for n in node_list]
    node_color = (
        [communities.get(str(n), 0) for n in node_list]
        if communities is not None
        else [0] * len(node_list)
    )

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_weights: list[float] = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(float(data.get("weight", 1.0)))

    return {
        "node_x": node_x,
        "node_y": node_y,
        "node_text": node_text,
        "node_color": node_color,
        "node_ids": [str(n) for n in node_list],
        "edge_x": edge_x,
        "edge_y": edge_y,
        "edge_weights": edge_weights,
    }
