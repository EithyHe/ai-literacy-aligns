"""Shared visualization workspace used by page 04 and page 06.

This page is intentionally render-only:
- expensive computations are done offline into visualization cache artifacts
- Streamlit runtime only loads cached data, filters, and renders
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ssn.components.cytoscape_viz import render_construct_network
from ssn.services.visualization_cache_service import (
    VIS_CACHE_DIR,
    build_visualization_cache,
    get_visualization_cache,
)
from ssn.services.visualization_service import build_cluster_diagnostics


_FALLBACK_CLUSTER_PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.D3
    + px.colors.qualitative.G10
    + px.colors.qualitative.Set3
)


def _get_query_param(name: str) -> str | None:
    val = st.query_params.get(name)
    if val is None:
        return None
    if isinstance(val, list):
        return str(val[0]) if val else None
    return str(val)


def _filter_df(df: pd.DataFrame, frameworks: list[str], instruments: list[str]) -> pd.DataFrame:
    out = df.copy()
    if frameworks and "framework_id" in out.columns:
        out = out[out["framework_id"].astype(str).isin({str(x) for x in frameworks})]
    if instruments and "instrument" in out.columns:
        out = out[out["instrument"].astype(str).isin({str(x) for x in instruments})]
    return out.reset_index(drop=True)


def _extract_selected_indices(selection_state: Any) -> list[int]:
    if selection_state is None:
        return []

    if isinstance(selection_state, dict):
        payload = selection_state
    else:
        payload = getattr(selection_state, "selection", None)
        if payload is None:
            return []

    if "selection" in payload and isinstance(payload["selection"], dict):
        payload = payload["selection"]

    points = payload.get("points", []) if isinstance(payload, dict) else []
    out: list[int] = []
    for p in points:
        if isinstance(p, dict) and isinstance(p.get("point_index"), int):
            out.append(int(p["point_index"]))
    return sorted(set(out))


def _cap_edges_payload(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    max_edges: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if max_edges <= 0 or len(edges) <= max_edges:
        return nodes, edges, False

    ranked = sorted(edges, key=lambda e: float(e.get("weight", 0.0)), reverse=True)
    kept = ranked[:max_edges]
    kept_nodes = {str(e.get("source")) for e in kept} | {str(e.get("target")) for e in kept}
    nodes_out = [n for n in nodes if str(n.get("id")) in kept_nodes]
    return nodes_out, kept, True


def _render_cache_incompatible_ui(missing_parts: list[str]) -> None:
    st.error("Cached visualization data is incompatible with cluster-labeled UMAP rendering.")
    st.warning("Please rebuild visualization cache before using this page.")
    st.caption("Missing cache fields:")
    for item in missing_parts:
        st.write(f"- {item}")
    st.code("python -m ssn.scripts.build_visualization_cache", language="bash")


def _validate_cluster_cache_contract(bundle: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    manifest = bundle.get("manifest", {})
    cluster_meta = manifest.get("cluster_meta", {})

    for level_key, table_key in (("item", "item_points"), ("construct", "construct_points")):
        points = bundle.get(table_key)
        cols = set(points.columns.tolist()) if isinstance(points, pd.DataFrame) else set()
        if "hdbscan_label" not in cols:
            missing.append(f"{table_key}.hdbscan_label")
        if "cluster_display_name" not in cols:
            missing.append(f"{table_key}.cluster_display_name")

        cmap = cluster_meta.get(level_key, {}).get("cluster_color_map")
        if not isinstance(cmap, dict) or len(cmap) == 0:
            missing.append(f"manifest.cluster_meta.{level_key}.cluster_color_map")

    return missing


def _pick_non_empty_mode(values: pd.Series) -> str:
    s = values.fillna("").astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return ""
    return str(s.mode().iloc[0])


def _prepare_cluster_umap_view(
    points_df: pd.DataFrame,
    manifest: dict[str, Any],
    level_key: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    out = points_df.copy()
    out["hdbscan_label"] = pd.to_numeric(out["hdbscan_label"], errors="coerce").fillna(-1).astype(int)

    display_col = out["cluster_display_name"] if "cluster_display_name" in out.columns else pd.Series("", index=out.index)
    label_col = out["cluster_label"] if "cluster_label" in out.columns else pd.Series("", index=out.index)

    level_meta = manifest.get("cluster_meta", {}).get(level_key, {})
    raw_id_to_display = level_meta.get("cluster_display_by_id", {})
    id_to_display_meta: dict[int, str] = {}
    for k, v in raw_id_to_display.items():
        try:
            id_to_display_meta[int(k)] = str(v).strip()
        except Exception:
            continue

    base_color_map = {
        str(k): str(v)
        for k, v in (level_meta.get("cluster_color_map", {}) or {}).items()
    }

    id_to_display: dict[int, str] = {}
    for cid, idx in out.groupby("hdbscan_label").groups.items():
        cid_int = int(cid)
        if cid_int == -1:
            id_to_display[cid_int] = "Noise"
            continue

        from_display = _pick_non_empty_mode(display_col.loc[idx])
        from_label = _pick_non_empty_mode(label_col.loc[idx])
        from_meta = str(id_to_display_meta.get(cid_int, "")).strip()
        final = from_display or from_label or from_meta or f"Cluster {cid_int}"
        id_to_display[cid_int] = final

    out["cluster_display_name"] = out["hdbscan_label"].map(
        lambda x: id_to_display.get(int(x), "Noise" if int(x) == -1 else f"Cluster {int(x)}")
    )

    color_map = dict(base_color_map)
    color_map["Noise"] = "#98A2B3"

    palette = _FALLBACK_CLUSTER_PALETTE
    for cid, display in sorted(id_to_display.items(), key=lambda p: p[0]):
        if display in color_map:
            continue
        meta_display = id_to_display_meta.get(int(cid), "")
        if meta_display and meta_display in base_color_map:
            color_map[display] = str(base_color_map[meta_display])
            continue
        if int(cid) == -1:
            color_map[display] = "#98A2B3"
            continue
        color_map[display] = palette[int(cid) % len(palette)]

    return out, color_map


def _render_selected_construct_sidebar(
    selected_construct_id: str | None,
    item_points: pd.DataFrame,
    construct_points: pd.DataFrame,
) -> None:
    with st.sidebar:
        st.subheader("Selected Construct")
        if not selected_construct_id:
            st.caption("Click a node in the network to inspect details.")
            return

        row = construct_points[construct_points["construct_id"].astype(str) == str(selected_construct_id)]
        if row.empty:
            st.warning("Selected construct is outside current filters.")
            return

        meta = row.iloc[0]
        st.markdown(f"**{meta.get('construct_label', selected_construct_id)}**")
        st.caption(f"Framework: {meta.get('framework_id', '') or 'N/A'}")

        items = item_points[item_points["construct_id"].astype(str) == str(selected_construct_id)].copy()
        if items.empty:
            st.caption("No items under this construct in current filters.")
            return

        st.metric("Items", int(len(items)))
        inst_counts = items["instrument"].fillna("").astype(str).value_counts().head(5)
        if not inst_counts.empty:
            st.caption("Top instruments")
            for inst, cnt in inst_counts.items():
                st.write(f"- {inst or 'Unknown'} ({int(cnt)})")

        st.caption("Items")
        for txt in items["text"].fillna("").astype(str).head(15).tolist():
            st.write(f"- {txt}")


def _render_network_section(
    cache_bundle: dict[str, Any],
    construct_points_filtered: pd.DataFrame,
    network_mode: str,
    max_render_edges: int,
    page_key: str,
) -> dict[str, float]:
    st.subheader("Construct Network")

    mode_key = "backbone" if network_mode == "Backbone" else "full"
    payload = cache_bundle["network"][mode_key]
    all_nodes: list[dict[str, Any]] = list(payload["nodes"])
    all_edges: list[dict[str, Any]] = list(payload["edges"])

    if construct_points_filtered.empty:
        st.info("No constructs available under current filters.")
        return {"n_nodes": 0.0, "n_edges": 0.0, "density": 0.0, "n_components": 0.0}

    keep_ids = set(construct_points_filtered["construct_id"].astype(str).tolist())
    nodes = [n for n in all_nodes if str(n.get("id")) in keep_ids]
    edges = [
        e
        for e in all_edges
        if str(e.get("source")) in keep_ids and str(e.get("target")) in keep_ids
    ]
    filtered_edge_count = len(edges)

    nodes, edges, clipped = _cap_edges_payload(nodes, edges, int(max_render_edges))
    if clipped:
        st.caption(
            f"Rendering top {int(max_render_edges)} strongest edges for responsiveness "
            f"(filtered network has {filtered_edge_count} edges)."
        )

    render_construct_network(nodes, edges, height=650, key=f"{page_key}_cy")

    g = nx.Graph()
    for n in nodes:
        g.add_node(str(n.get("id")))
    for e in edges:
        g.add_edge(str(e.get("source")), str(e.get("target")))

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    density = float(nx.density(g)) if n_nodes > 1 else 0.0
    n_components = int(nx.number_connected_components(g)) if n_nodes > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", n_nodes)
    c2.metric("Edges", n_edges)
    c3.metric("Density", f"{density:.3f}")
    c4.metric("Components", n_components)

    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "density": density,
        "n_components": float(n_components),
    }


def _build_umap_figure(
    points_df: pd.DataFrame,
    level: str,
    cluster_color_map: dict[str, str],
) -> go.Figure:
    data = points_df.copy()
    data["color_group"] = data["cluster_display_name"].astype(str)

    hover = ["cluster_display_name", "construct_label", "framework_id", "instrument"]
    if level == "Item" and "text" in data.columns:
        hover.append("text")

    fig = px.scatter(
        data,
        x="x",
        y="y",
        color="color_group",
        hover_data=hover,
        custom_data=["id", "construct_label", "instrument", "text", "cluster_display_name"],
        title=f"UMAP ({level})",
        color_discrete_map=cluster_color_map,
    )
    fig.update_traces(marker={"size": 8 if level == "Item" else 11, "opacity": 0.82})
    fig.update_layout(legend_title_text="Cluster", margin=dict(l=8, r=8, t=50, b=8))
    return fig


def _render_umap_section(
    points_df: pd.DataFrame,
    level: str,
    cluster_color_map: dict[str, str],
    page_key: str,
) -> None:
    st.subheader("UMAP Scatter")
    if points_df.empty:
        st.info("No UMAP points available under current filters.")
        return

    fig = _build_umap_figure(
        points_df=points_df,
        level=level,
        cluster_color_map=cluster_color_map,
    )
    selection_state = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{page_key}_umap",
        on_select="rerun",
        selection_mode=("box", "lasso"),
    )

    sel_indices = _extract_selected_indices(selection_state)
    if not sel_indices:
        return

    selected_rows = points_df.iloc[sel_indices].copy()
    st.markdown("**Selected Points**")
    show_cols = ["id", "cluster_display_name", "construct_label", "instrument", "text", "hdbscan_label"]
    show_cols = [c for c in show_cols if c in selected_rows.columns]
    st.dataframe(selected_rows[show_cols], use_container_width=True, hide_index=True)

    csv_buf = StringIO()
    selected_rows.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download selected points (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"{page_key}_umap_selected.csv",
        mime="text/csv",
        key=f"{page_key}_umap_selected_download",
    )


def _render_diagnostics_section(
    item_points_filtered: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    st.subheader("Clustering Diagnostics")
    if item_points_filtered.empty:
        st.info("No item-level data available for diagnostics.")
        return

    labels = item_points_filtered["hdbscan_label"].fillna(-1).astype(int).tolist()
    constructs = item_points_filtered["construct_label"].fillna("").astype(str).tolist()
    diag = build_cluster_diagnostics(construct_labels=constructs, cluster_labels=labels)
    links = diag["sankey_links"]
    if not links:
        st.info("Insufficient cluster structure to build Sankey.")
        return

    src_labels = sorted({x["source"] for x in links})
    tgt_labels = sorted({x["target"] for x in links})
    all_nodes = src_labels + tgt_labels
    node_index = {n: i for i, n in enumerate(all_nodes)}

    source_idx = [node_index[x["source"]] for x in links]
    target_idx = [node_index[x["target"]] for x in links]
    values = [int(x["value"]) for x in links]
    link_colors = [
        "rgba(192,57,43,0.62)" if (x["is_jingle"] or x["is_jangle"]) else "rgba(120,140,160,0.36)"
        for x in links
    ]

    node_colors = ["#2E86AB" for _ in src_labels] + ["#F39C12" for _ in tgt_labels]
    sankey = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=all_nodes, color=node_colors, pad=14, thickness=13),
                link=dict(source=source_idx, target=target_idx, value=values, color=link_colors),
            )
        ]
    )
    sankey.update_layout(
        title="Construct Label vs HDBSCAN Cluster",
        margin=dict(l=10, r=10, t=45, b=10),
        height=520,
    )
    st.plotly_chart(sankey, use_container_width=True)

    jingle = sorted(diag["jingle_flags"])
    jangle = sorted(diag["jangle_flags"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Jingle flags", len(jingle))
    c2.metric("Jangle flags", len(jangle))
    c3.metric("Global Silhouette", f"{manifest.get('stats', {}).get('item_hdbscan_silhouette', 0) or 0:.3f}")

    if jingle:
        st.warning("Jingle (same name, split semantics): " + ", ".join(jingle[:8]))
    if jangle:
        st.warning("Jangle (different names, merged semantics): " + ", ".join(jangle[:8]))

    st.markdown("**Construct-to-Cluster Summary**")
    st.dataframe(diag["construct_summary"], use_container_width=True, hide_index=True)

    st.markdown("**Cluster Purity Summary**")
    st.dataframe(diag["cluster_summary"], use_container_width=True, hide_index=True)


def _render_cache_bootstrap_ui() -> None:
    st.warning(
        "Visualization cache is missing. Build it once offline, then this page will be render-only and fast."
    )
    st.code("python -m ssn.scripts.build_visualization_cache", language="bash")

    with st.expander("Build cache in-app (slow once)", expanded=False):
        use_llm = st.checkbox("Use LLM cluster naming", value=True)
        if st.button("Build visualization cache now", type="primary"):
            with st.spinner("Building cache artifacts (this may take a while)..."):
                manifest = build_visualization_cache(use_llm_labels=bool(use_llm))
            st.success(
                "Cache built. "
                f"Items: {manifest.get('stats', {}).get('n_items', 0)}, "
                f"Constructs: {manifest.get('stats', {}).get('n_constructs', 0)}"
            )
            st.rerun()


def render_visualization_workspace(page_key: str, title: str, description: str) -> None:
    """Render the full 3-view visualization workspace from offline cache."""
    st.title(title)
    st.markdown(description)

    bundle = get_visualization_cache()
    if bundle is None:
        _render_cache_bootstrap_ui()
        return

    missing_contract_fields = _validate_cluster_cache_contract(bundle)
    if missing_contract_fields:
        _render_cache_incompatible_ui(missing_contract_fields)
        return

    manifest = bundle["manifest"]
    item_points = bundle["item_points"].copy()
    construct_points = bundle["construct_points"].copy()

    all_frameworks = sorted(x for x in item_points["framework_id"].fillna("").astype(str).unique().tolist() if x)
    all_instruments = sorted(x for x in item_points["instrument"].fillna("").astype(str).unique().tolist() if x)

    with st.sidebar:
        st.subheader("Visualization Controls")
        st.caption(f"Cache dir: `{VIS_CACHE_DIR}`")
        st.caption(f"Built at: {manifest.get('created_at_utc', 'N/A')}")

        active_view = st.radio(
            "Active view",
            ["Network", "UMAP", "Diagnostics", "All"],
            index=0,
            key=f"{page_key}_active_view",
            help="Render one view at a time for better responsiveness.",
        )

        selected_frameworks = st.multiselect(
            "Framework filter",
            options=all_frameworks,
            default=[],
            help="Empty means all frameworks.",
            key=f"{page_key}_framework_filter",
        )
        selected_instruments = st.multiselect(
            "Instrument filter",
            options=all_instruments,
            default=[],
            help="Empty means all instruments.",
            key=f"{page_key}_instrument_filter",
        )

        st.markdown("---")
        network_mode = st.radio(
            "Network mode",
            ["Backbone", "Full"],
            index=0,
            key=f"{page_key}_network_mode",
        )
        max_render_edges = st.slider(
            "Render edge cap",
            min_value=300,
            max_value=12000,
            value=2500,
            step=100,
            key=f"{page_key}_network_render_edge_cap",
            help="Only affects front-end rendering; strongest edges are kept first.",
        )

        st.markdown("---")
        umap_level = st.radio(
            "UMAP level",
            ["Item", "Construct"],
            index=0,
            key=f"{page_key}_umap_level",
        )

    item_filtered = _filter_df(item_points, selected_frameworks, selected_instruments)
    construct_filtered = _filter_df(construct_points, selected_frameworks, [])

    if selected_instruments:
        keep_constructs = set(item_filtered["construct_id"].astype(str).tolist())
        construct_filtered = construct_filtered[
            construct_filtered["construct_id"].astype(str).isin(keep_constructs)
        ].reset_index(drop=True)

    selected_id = _get_query_param("selected_construct_id")
    _render_selected_construct_sidebar(selected_id, item_filtered, construct_filtered)

    net_stats = {"n_nodes": 0.0, "n_edges": 0.0}
    if active_view in {"Network", "All"}:
        net_stats = _render_network_section(
            cache_bundle=bundle,
            construct_points_filtered=construct_filtered,
            network_mode=network_mode,
            max_render_edges=int(max_render_edges),
            page_key=page_key,
        )

    level_key = "item" if umap_level == "Item" else "construct"
    umap_points_raw = item_filtered if umap_level == "Item" else construct_filtered
    umap_points, cluster_color_map = _prepare_cluster_umap_view(
        points_df=umap_points_raw,
        manifest=manifest,
        level_key=level_key,
    )
    item_points_diag, _ = _prepare_cluster_umap_view(
        points_df=item_filtered,
        manifest=manifest,
        level_key="item",
    )

    if active_view in {"UMAP", "All"}:
        if active_view == "All":
            st.divider()
        _render_umap_section(
            points_df=umap_points,
            level=umap_level,
            cluster_color_map=cluster_color_map,
            page_key=page_key,
        )

    if active_view in {"Diagnostics", "All"}:
        if active_view == "All":
            st.divider()
        _render_diagnostics_section(item_points_filtered=item_points_diag, manifest=manifest)

    if active_view in {"Network", "All"} and net_stats["n_nodes"] == 0:
        st.info("Network view is empty for current filters.")
