"""Corpus Overview – Full corpus and framework-scoped Network and Density.

Presents Overview (and optional Framework) views for construct network and density.
Not tied to current Explorer search; use Explorer for result-set tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from ssn.services.embedding_service import get_construct_embeddings
from ssn.services.decomposition_service import run_decomposition
from ssn.services.density_service import (
    estimate_density,
    compute_redundancy_risk,
    intra_domain_density,
)
from ssn.services.network_service import (
    build_construct_network,
    compute_network_metrics,
    detect_communities,
    get_network_summary,
    network_to_plotly_data,
)
from ssn.components.network_viz import plot_network_from_plotly_data
from ssn.components.scatter_viz import plot_density_scatter
from ssn.db.schema import (
    get_all_constructs,
    get_all_domains,
    get_all_frameworks,
    get_constructs_by_domain,
    get_domains_by_framework,
    get_items_by_construct,
)


def _construct_id_to_name() -> dict[str, str]:
    return {c["construct_id"]: c.get("name", c["construct_id"]) for c in get_all_constructs()}


def _filtered_embeddings(framework_id: str | None, domain_id: str | None) -> dict:
    all_embs = get_construct_embeddings()
    if not all_embs:
        return {}
    if domain_id:
        valid = {c["construct_id"] for c in get_constructs_by_domain(domain_id) if c.get("construct_id")}
        return {k: v for k, v in all_embs.items() if k in valid}
    if framework_id:
        valid = set()
        for d in get_domains_by_framework(framework_id):
            valid.update(c["construct_id"] for c in get_constructs_by_domain(d["domain_id"]) if c.get("construct_id"))
        return {k: v for k, v in all_embs.items() if k in valid}
    return all_embs


def _show_network(construct_embs: dict, title: str) -> None:
    """Build, visualize, and summarize a network from a dict of embeddings."""
    G = build_construct_network(construct_embs)
    communities = detect_communities(G) if G.number_of_nodes() > 0 else {}

    id_to_name = _construct_id_to_name()
    G_renamed = G.copy()
    mapping = {n: id_to_name.get(n, n) for n in G.nodes()}
    import networkx as nx
    G_display = nx.relabel_nodes(G_renamed, mapping)
    communities_display = {id_to_name.get(k, k): v for k, v in communities.items()}

    plotly_data = network_to_plotly_data(G_display, layout="spring", communities=communities_display)

    st.plotly_chart(
        plot_network_from_plotly_data(plotly_data, title=title, height=600),
        use_container_width=True,
    )

    summary = get_network_summary(G)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", summary["n_nodes"])
    c2.metric("Edges", summary["n_edges"])
    c3.metric("Modularity", f"{summary['modularity']:.3f}")
    c4.metric("Communities", len(set(communities.values())) if communities else 0)

    with st.expander("Node metrics"):
        if G.number_of_nodes() > 0:
            metrics = compute_network_metrics(G)
            rows = [{"Construct": id_to_name.get(nid, nid), **m} for nid, m in metrics.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _show_density_overview(all_embs: dict) -> None:
    """Full corpus density map and redundancy risk table."""
    ids = list(all_embs.keys())
    matrix = np.array([all_embs[i] for i in ids], dtype=np.float64)

    try:
        result = run_decomposition(matrix, method="umap", n_components=2)
        coords = result["coords"]
    except Exception as e:
        st.error(f"Failed to compute coordinates: {e}")
        return

    id_to_name = _construct_id_to_name()
    labels = [id_to_name.get(i, i) for i in ids]

    try:
        density = estimate_density(coords, method="kde")
    except Exception as e:
        st.error(f"Density estimation failed: {e}")
        return

    st.subheader("Density map")
    st.plotly_chart(
        plot_density_scatter(coords, density, labels, title="Corpus Density Map", height=600),
        use_container_width=True,
    )

    # Community-based intra-group stats
    st.subheader("Community similarity")
    st.markdown("Pairwise cosine similarity within auto-detected communities.")

    G = build_construct_network(all_embs)
    communities = detect_communities(G) if G.number_of_nodes() > 0 else {}

    if communities:
        comm_groups: dict[str, list[str]] = {}
        for cid, comm_id in communities.items():
            comm_groups.setdefault(f"Community {comm_id}", []).append(cid)

        item_counts = {cid: len(get_items_by_construct(cid)) for cid in all_embs}

        try:
            intra_stats = intra_domain_density(all_embs, comm_groups, item_counts)
        except Exception as e:
            st.warning(f"Community analysis failed: {e}")
            intra_stats = {}

        if intra_stats:
            rows = [
                {
                    "Community": grp,
                    "Constructs": v["construct_count"],
                    "Mean Sim": v["mean_pairwise_cosine_sim"],
                    "Std Sim": v["std_pairwise_cosine_sim"],
                    "Min Sim": v["min_pairwise_cosine_sim"],
                    "Max Sim": v["max_pairwise_cosine_sim"],
                }
                for grp, v in intra_stats.items()
            ]
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.format(
                    {"Mean Sim": "{:.4f}", "Std Sim": "{:.4f}", "Min Sim": "{:.4f}", "Max Sim": "{:.4f}"},
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Redundancy Risk Index")
    try:
        risk = compute_redundancy_risk(density)
        risk_df = pd.DataFrame({"Construct": labels, "Redundancy Risk (0-1)": risk})
        st.dataframe(
            risk_df.sort_values("Redundancy Risk (0-1)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as e:
        st.warning(f"Could not compute redundancy risk: {e}")


def _tab_overview() -> None:
    st.markdown(
        "Full corpus network and density. "
        "Communities are discovered automatically via the Leiden algorithm."
    )
    st.caption("These views are independent of the Construct Explorer search.")

    try:
        all_embs = get_construct_embeddings()
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return

    if not all_embs:
        st.warning("No construct embeddings available.")
        return

    net_tab, dens_tab = st.tabs(["Network", "Density"])
    with net_tab:
        try:
            _show_network(all_embs, title="Full Corpus Network")
        except Exception as e:
            st.error(f"Network visualization failed: {e}")
    with dens_tab:
        try:
            _show_density_overview(all_embs)
        except Exception as e:
            st.error(f"Density visualization failed: {e}")


def _tab_framework() -> None:
    st.markdown("Explore network and density within a specific theoretical framework.")
    st.caption("Select a framework and optionally a domain to scope the views.")

    frameworks = get_all_frameworks()
    fw_options = [f["name"] for f in frameworks]
    fw_ids = [f["framework_id"] for f in frameworks]

    if not fw_options:
        st.warning("No frameworks in the corpus.")
        return

    fw_idx = st.selectbox("Framework", range(len(fw_options)), format_func=lambda i: fw_options[i], key="overview_fw_sel")
    framework_id = fw_ids[fw_idx]

    fw_domains = get_domains_by_framework(framework_id)
    domain_options = ["All"] + [d["name"] for d in fw_domains]
    domain_ids: list[str | None] = [None] + [d["domain_id"] for d in fw_domains]

    dom_idx = st.selectbox("Domain", range(len(domain_options)), format_func=lambda i: domain_options[i], key="overview_dom_sel")
    domain_id = domain_ids[dom_idx]

    try:
        embs = _filtered_embeddings(framework_id, domain_id)
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return

    if not embs:
        st.warning("No embeddings for this selection.")
        return

    net_tab, dens_tab = st.tabs(["Network", "Density"])
    with net_tab:
        try:
            _show_network(embs, title=f"Network: {fw_options[fw_idx]}")
        except Exception as e:
            st.error(f"Network visualization failed: {e}")
    with dens_tab:
        try:
            ids = list(embs.keys())
            matrix = np.array([embs[i] for i in ids], dtype=np.float64)
            try:
                result = run_decomposition(matrix, method="umap", n_components=2)
                coords = result["coords"]
            except Exception:
                result = run_decomposition(matrix, method="pca", n_components=2)
                coords = result["coords"]
            id_to_name = _construct_id_to_name()
            labels = [id_to_name.get(i, i) for i in ids]
            try:
                density = estimate_density(coords, method="kde")
            except Exception:
                density = np.ones(len(ids))
            st.subheader("Density map")
            st.plotly_chart(
                plot_density_scatter(coords, density, labels, title=f"Density: {fw_options[fw_idx]}", height=600),
                use_container_width=True,
            )
            try:
                risk = compute_redundancy_risk(density)
                risk_df = pd.DataFrame({"Construct": labels, "Redundancy Risk (0-1)": risk})
                st.dataframe(
                    risk_df.sort_values("Redundancy Risk (0-1)", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception:
                pass
        except Exception as e:
            st.error(f"Density visualization failed: {e}")


def render() -> None:
    st.title("Corpus Overview")
    st.markdown(
        "Full corpus and framework-scoped **network** and **density** views. "
        "Independent of the current Construct Explorer search."
    )

    tab_overview, tab_framework = st.tabs(["Full corpus", "By framework"])
    with tab_overview:
        _tab_overview()
    with tab_framework:
        _tab_framework()


render()
