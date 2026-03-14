"""Construct Explorer (PRD Stage 1, FR-4.1.1–4.1.6).

Exact-match only: search by construct or domain name. Shows matched name and definition,
then nearest constructs (or constructs in domain).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx

from ssn.services.embedding_service import get_construct_embeddings
from ssn.services.exploration_service import (
    hierarchical_lookup,
    nearest_constructs_for_id,
)
from ssn.services.decomposition_service import run_decomposition
from ssn.services.network_service import (
    build_construct_network,
    compute_network_metrics,
    detect_communities,
    get_network_summary,
    network_to_plotly_data,
)
from ssn.services.density_service import estimate_density, compute_redundancy_risk
from ssn.db.schema import (
    get_all_constructs,
    get_all_domains,
    get_all_frameworks,
    get_constructs_by_domain,
    search_by_name,
)
from ssn.components.scatter_viz import plot_density_scatter
from ssn.components.network_viz import plot_network_from_plotly_data


def _dummy_encode(_texts: list) -> np.ndarray:
    """Dummy encoder so hierarchical_lookup never runs semantic path; we only use exact match."""
    return np.zeros((len(_texts), 384), dtype=np.float32)


@st.cache_data(ttl=300)
def _get_construct_metadata() -> dict[str, dict]:
    """Build construct_id -> {name, framework, domain, item_count, description}. Uses constructs.framework_id; domain from domain_construct_map when present."""
    from ssn.db.schema import get_construct_to_primary_domain

    constructs = {c["construct_id"]: dict(c) for c in get_all_constructs()}
    frameworks = {f["framework_id"]: f["name"] for f in get_all_frameworks()}
    construct_to_domain = get_construct_to_primary_domain()
    domains_by_id = {d["domain_id"]: d.get("name", "") for d in get_all_domains()}
    meta: dict[str, dict] = {}
    for cid, c in constructs.items():
        fw_id = c.get("framework_id", "")
        domain_name = domains_by_id.get(construct_to_domain.get(cid, ""), "")
        meta[cid] = {
            "name": c.get("name", cid),
            "framework": frameworks.get(fw_id, ""),
            "domain": domain_name,
            "item_count": c.get("item_count", 0),
            "description": c.get("description") or "",
        }
    return meta


@st.cache_data(ttl=300)
def _get_umap_coords_and_ids(construct_ids: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Run UMAP on construct embeddings. Returns (coords, ids)."""
    if not construct_ids:
        return np.zeros((0, 2)), []
    construct_embeddings = get_construct_embeddings()
    emb_matrix = np.vstack([construct_embeddings[cid] for cid in construct_ids if cid in construct_embeddings])
    if emb_matrix.shape[0] == 0:
        return np.zeros((0, 2)), []
    result = run_decomposition(emb_matrix, method="umap", n_components=2)
    ids_used = [cid for cid in construct_ids if cid in construct_embeddings]
    return result["coords"], ids_used


def render() -> None:
    st.title("Construct Explorer")
    st.markdown(
        "Search by **construct name** (exact match). "
        "View the matched definition and related constructs. Searching by domains is not supported at this stage."
    )

    query_text = st.text_input(
        "Construct name",
        placeholder="E.g., Extraversion, Gregariousness, Imagination",
    )
    search_clicked = st.button("Search", type="primary")

    if not query_text or not query_text.strip():
        st.warning("Please enter a construct name.")
        return

    if not search_clicked:
        return

    with st.spinner("Searching database…"):
        try:
            construct_embeddings = get_construct_embeddings()
            construct_metadata = _get_construct_metadata()
        except FileNotFoundError as e:
            st.error(f"Embeddings not found: {e}. Run the embedding pipeline first.")
            return
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

        if not construct_embeddings:
            st.warning("No construct embeddings available. Import corpus data first.")
            return

        lookup = hierarchical_lookup(
            query_text.strip(),
            db_search_fn=lambda t, n: search_by_name(t, n, limit=20),
            encode_fn=_dummy_encode,
            construct_embeddings=construct_embeddings,
            construct_metadata=construct_metadata,
        )

    if lookup["match_type"] == "semantic":
        st.info(
            f"No exact match found for **{query_text.strip()}**. "
            "Please enter an existing construct name."
        )
        return

    matched = lookup.get("matched_entity") or {}
    related = lookup.get("related_constructs") or []

    # --- Matched entity: name and definition ---
    if lookup["match_type"] == "domain":
        st.subheader("Matched domain")
        st.markdown(f"**{matched.get('name', '')}**")
        if matched.get("description"):
            st.caption("Definition")
            st.markdown(matched.get("description", ""))
    else:
        st.subheader("Matched construct")
        st.markdown(f"**{matched.get('name', '')}**")
        if matched.get("description"):
            st.caption("Definition")
            st.markdown(matched.get("description", ""))
        # Enrich from metadata if description missing in matched_entity
        cid = matched.get("construct_id")
        if cid and not matched.get("description") and construct_metadata.get(cid, {}).get("description"):
            st.caption("Definition")
            st.markdown(construct_metadata[cid]["description"])

    # --- Nearest / related constructs ---
    if lookup["match_type"] == "construct" and related:
        cid = related[0].get("construct_id")
        neighbors = nearest_constructs_for_id(
            cid, construct_embeddings, construct_metadata, top_n=15
        )
        ids_for_umap = [cid] + [n["construct_id"] for n in neighbors]
        highlight_idx = 0
    else:
        neighbors = related
        ids_for_umap = [n.get("construct_id") for n in related if n.get("construct_id") and n.get("construct_id") in construct_embeddings]
        highlight_idx = None

    st.subheader("Nearest constructs" if lookup["match_type"] == "construct" else "Constructs in this domain")
    if not neighbors:
        st.info("No related constructs found.")
        return

    rows = []
    for n in neighbors:
        sim = n.get("similarity")
        rows.append({
            "Construct": n.get("name", n.get("construct_id", "")),
            "Similarity": f"{sim:.3f}" if sim is not None else "—",
            "Framework": n.get("framework", ""),
            "Domain": n.get("domain", ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # --- Tools: Network & Density on current result set ---
    if ids_for_umap:
        coords, ids_used = _get_umap_coords_and_ids(tuple(ids_for_umap))
        if len(coords) > 0:
            labels = [construct_metadata.get(cid, {}).get("name", cid) for cid in ids_used]
            sub_embs = {cid: construct_embeddings[cid] for cid in ids_used}
            name_map = {cid: construct_metadata.get(cid, {}).get("name", cid) for cid in ids_used}
            highlight_id = ids_used[highlight_idx] if highlight_idx is not None else None
            highlight_label = name_map.get(highlight_id, highlight_id) if highlight_id else None

            st.subheader("Tools")
            st.caption("Network and density views for the current search result set.")

            tab_net, tab_dens = st.tabs(["Network", "Density"])
            with tab_net:
                G = build_construct_network(sub_embs)
                communities = detect_communities(G) if G.number_of_nodes() > 0 else {}
                G_renamed = G.copy()
                mapping = {n: name_map.get(n, n) for n in G.nodes()}
                G_display = nx.relabel_nodes(G_renamed, mapping)
                communities_display = {name_map.get(k, k): v for k, v in communities.items()}
                plotly_data = network_to_plotly_data(G_display, layout="spring", communities=communities_display)
                st.plotly_chart(
                    plot_network_from_plotly_data(
                        plotly_data,
                        title="Construct network (current result set)",
                        height=600,
                        highlight_node_id=highlight_label,
                    ),
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
                        rows = [{"Construct": name_map.get(nid, nid), **m} for nid, m in metrics.items()]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with tab_dens:
                try:
                    density = estimate_density(coords, method="kde")
                except Exception:
                    density = np.ones(len(coords))
                risk = compute_redundancy_risk(density)
                if highlight_idx is not None:
                    user_risk = float(risk[highlight_idx])
                    st.metric("Redundancy risk (matched construct)", f"{user_risk:.0%}")
                st.plotly_chart(
                    plot_density_scatter(
                        coords,
                        density,
                        labels,
                        title="Density (current result set)",
                        height=550,
                    ),
                    use_container_width=True,
                )
                risk_df = pd.DataFrame({"Construct": labels, "Redundancy Risk (0-1)": risk})
                st.dataframe(
                    risk_df.sort_values("Redundancy Risk (0-1)", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )


render()
