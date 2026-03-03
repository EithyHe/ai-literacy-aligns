"""Construct association network service for the Semantic Scale Network."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

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


def build_construct_network(
    construct_embeddings: dict[str, np.ndarray],
    metric: str = "cosine",
) -> nx.Graph:
    """Build a fully-connected construct association network with similarity as edge weight.

    Args:
        construct_embeddings: Dict mapping construct ID to embedding vector.
        metric: Similarity metric for pairwise comparison.

    Returns:
        Undirected networkx Graph with nodes = construct IDs and every pair
        connected by an edge whose weight is the pairwise similarity.
    """
    if not construct_embeddings:
        logger.warning("Empty construct_embeddings; returning empty graph")
        return nx.Graph()

    ids = list(construct_embeddings.keys())
    matrix = np.array([construct_embeddings[i] for i in ids], dtype=np.float64)
    sim_matrix = compute_pairwise_similarity(matrix, matrix, metric)

    G = nx.Graph()
    G.add_nodes_from(ids)

    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            G.add_edge(ids[i], ids[j], weight=sim)

    logger.info(
        f"Built construct network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


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
            partition = la.find_partition(
                G_ig,
                la.ModularityVertexPartition,
                weights=G_ig.es["weight"] if G.number_of_edges() > 0 else None,
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


def network_to_plotly_data(
    G: nx.Graph,
    layout: str = "spring",
    communities: dict[str, int] | None = None,
) -> dict:
    """Convert networkx graph to Plotly-compatible node/edge data for visualization.

    Args:
        G: Networkx graph.
        layout: "spring", "kamada_kawai", "circular", or "shell".
        communities: Optional dict mapping node_id -> community_id for coloring.

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

    # Layout
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

    node_list = list(G.nodes())
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
