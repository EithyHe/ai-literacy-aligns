"""Corpus Overview — three-tab analytical interface.

Tab 1: Semantic Map          — UMAP scatter, cluster colors, frequency sizing
Tab 2: Measurement Decomp    — Cytoscape.js co-measurement network, community-aware layout
Tab 3: Integrated View       — UMAP + cluster-pair edges, ego-network mode
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ssn.db.schema import get_all_constructs, get_all_frameworks
from ssn.services.visualization_cache_service import VIS_CACHE_DIR, get_visualization_cache, patch_llm_cluster_labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_key(s: str) -> str:
    """Convert an ID to a safe CSS class name."""
    return s.replace("-", "_").replace(" ", "_").replace(".", "_")


# ── Shared node tooltip definition (Tab 1 & Tab 2) ───────────────────────────
# Tab 1 uses Plotly hovertemplate; Tab 2 injects this JS function into Cytoscape.
# Fields: label / cluster_name / freq / all_scales  — keep both in sync.
_NODE_TOOLTIP_JS = """\
function buildNodeTooltipHtml(node) {
    var label   = node.data('label')       || '';
    var cluster = node.data('cluster_name')|| '';
    var freq    = node.data('freq')        || 0;
    var scales  = node.data('all_scales')  || '';
    var html = '<b>' + label + '</b>';
    html += '<br><span style="color:#666;font-size:11px">Cluster: ' + cluster + '</span>';
    html += '<br><span style="color:#666;font-size:11px">Frequency: ' + freq + ' scale(s)</span>';
    if (scales) {
        html += '<br><span style="color:#888;font-size:11px">Scales: ' + scales + '</span>';
    }
    return html;
}"""


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def _load_data() -> dict:
    cache_path = VIS_CACHE_DIR / "construct_points.csv"
    if not cache_path.exists():
        return {}

    pts = pd.read_csv(cache_path)
    pts["hdbscan_label"] = pd.to_numeric(pts["hdbscan_label"], errors="coerce").fillna(-1).astype(int)

    # Frameworks (= scales)
    fw_rows = get_all_frameworks()
    fw_df = pd.DataFrame(fw_rows) if fw_rows else pd.DataFrame(columns=["framework_id", "name"])
    fw_name: dict[str, str] = fw_df.set_index("framework_id")["name"].to_dict() if not fw_df.empty else {}

    # Constructs from DB
    c_rows = get_all_constructs()
    c_df = pd.DataFrame(c_rows) if c_rows else pd.DataFrame(
        columns=["construct_id", "name", "framework_id", "item_count"])

    # Frequency: how many distinct frameworks share a construct label
    if not c_df.empty:
        c_df["_key"] = c_df["name"].str.lower().str.strip()
        freq = c_df.groupby("_key")["framework_id"].nunique().rename("frequency")
        fw_name_tmp = fw_df.set_index("framework_id")["name"].to_dict() if not fw_df.empty else {}
        all_scales_map = (
            c_df.groupby("_key")["framework_id"]
            .apply(lambda ids: "; ".join(fw_name_tmp.get(fid, str(fid)) for fid in sorted(set(ids))))
            .to_dict()
        )
        pts["_key"] = pts["construct_label"].str.lower().str.strip()
        pts = pts.merge(freq, on="_key", how="left")
        pts["frequency"] = pts["frequency"].fillna(1).astype(int)
        pts["all_scales"] = pts["_key"].map(all_scales_map).fillna("")
        pts.drop(columns=["_key"], inplace=True)
    else:
        pts["frequency"] = 1
        pts["all_scales"] = ""

    max_freq = max(int(pts["frequency"].max()), 1)
    pts["freq_scaled"] = 5 + (pts["frequency"] / max_freq) * 13

    # Canonical cluster display name per row
    pts["cluster_name_final"] = pts.apply(
        lambda r: "Noise" if r["hdbscan_label"] == -1
        else (str(r["cluster_display_name"]).strip()
              if pd.notna(r.get("cluster_display_name")) and str(r.get("cluster_display_name", "")).strip()
              else f"Cluster {r['hdbscan_label']}"),
        axis=1,
    )

    # Cluster → color map (from cached colors)
    color_map: dict[str, str] = {"Noise": "#98A2B3"}
    for _, row in pts.iterrows():
        name = row["cluster_name_final"]
        if name not in color_map:
            color_map[name] = str(row.get("cluster_color", "#aaa") or "#aaa")

    # Co-measurement edges: all construct pairs sharing the same framework
    edges_list: list[dict] = []
    if not c_df.empty:
        for fw_id, grp in c_df.groupby("framework_id"):
            cids = sorted(grp["construct_id"].tolist())
            for a, b in combinations(cids, 2):
                edges_list.append({"construct_a": a, "construct_b": b, "scale_id": str(fw_id)})

    edges_df = pd.DataFrame(edges_list) if edges_list else pd.DataFrame(
        columns=["construct_a", "construct_b", "scale_id"])

    # Aggregate edges: one row per construct pair with shared_scale_count
    if not edges_df.empty:
        agg_edges = (
            edges_df.groupby(["construct_a", "construct_b"])
            .agg(shared_scale_count=("scale_id", "count"),
                 scale_ids=("scale_id", list))
            .reset_index()
        )
    else:
        agg_edges = pd.DataFrame(
            columns=["construct_a", "construct_b", "shared_scale_count", "scale_ids"])

    # Leiden community detection on label-level co-measurement graph.
    # Constructs are deduplicated by normalized label before graph construction,
    # so cross-scale co-measurement is captured (e.g. "Extraversion" in 16PF and
    # BFI becomes one node connected to everything it's co-measured with across
    # all scales). Community assignments are then mapped back to each construct_id
    # via its normalized label.
    pts["leiden_community"] = -1
    try:
        import igraph as ig
        import leidenalg

        c_df["_norm"] = c_df["name"].str.lower().str.strip()

        # Build label-level edge list: pairs of labels co-occurring in the same scale
        label_edges_list: list[dict] = []
        for fw_id, grp in c_df.groupby("framework_id"):
            labels = sorted(grp["_norm"].unique().tolist())
            for a, b in combinations(labels, 2):
                label_edges_list.append({"label_a": a, "label_b": b, "scale_id": str(fw_id)})

        if label_edges_list:
            label_edges_df = pd.DataFrame(label_edges_list)
            label_agg = (
                label_edges_df.groupby(["label_a", "label_b"])
                .agg(shared_scale_count=("scale_id", "count"))
                .reset_index()
            )

            all_labels = sorted(c_df["_norm"].unique().tolist())
            label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(all_labels)}

            ig_edges = []
            ig_weights = []
            for _, e in label_agg.iterrows():
                a_lbl, b_lbl = str(e["label_a"]), str(e["label_b"])
                if a_lbl in label_to_idx and b_lbl in label_to_idx:
                    ig_edges.append((label_to_idx[a_lbl], label_to_idx[b_lbl]))
                    ig_weights.append(int(e["shared_scale_count"]))

            g = ig.Graph(n=len(all_labels), edges=ig_edges, directed=False)
            if ig_weights:
                g.es["weight"] = ig_weights
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights="weight" if ig_weights else None,
                seed=42,
            )
            label_community_map: dict[str, int] = {
                all_labels[i]: partition.membership[i]
                for i in range(len(all_labels))
            }
            pts["_norm"] = pts["construct_label"].str.lower().str.strip()
            pts["leiden_community"] = (
                pts["_norm"].map(label_community_map).fillna(-1).astype(int)
            )
            pts.drop(columns=["_norm"], inplace=True)

        c_df.drop(columns=["_norm"], inplace=True)
    except ImportError:
        pass  # leidenalg not installed; leiden_community stays -1

    # Cluster-pair edge table for Tab 3
    if not agg_edges.empty:
        pts_cluster = pts.set_index("construct_id")["hdbscan_label"].to_dict()
        agg_c = agg_edges.copy()
        agg_c["src_cl"] = agg_c["construct_a"].astype(str).map(
            lambda c: pts_cluster.get(c, -1))
        agg_c["tgt_cl"] = agg_c["construct_b"].astype(str).map(
            lambda c: pts_cluster.get(c, -1))
        cross = agg_c[
            (agg_c["src_cl"] != agg_c["tgt_cl"]) &
            (agg_c["src_cl"] != -1) &
            (agg_c["tgt_cl"] != -1)
        ].copy()
        cross["cluster_a"] = cross[["src_cl", "tgt_cl"]].min(axis=1)
        cross["cluster_b"] = cross[["src_cl", "tgt_cl"]].max(axis=1)
        rows: list[dict] = []
        for (ca, cb), grp in cross.groupby(["cluster_a", "cluster_b"]):
            all_sids: set = set()
            for sids in grp["scale_ids"]:
                all_sids.update(sids)
            sc: dict[str, int] = {}
            for sids in grp["scale_ids"]:
                for sid in sids:
                    sc[sid] = sc.get(sid, 0) + 1
            dom = max(sc, key=sc.get) if sc else ""
            dom_cnt = sc.get(dom, 0)
            total = len(grp)
            rows.append({
                "cluster_a": int(ca),
                "cluster_b": int(cb),
                "total_construct_pairs": total,
                "n_scales": len(all_sids),
                "dominant_scale": dom,
                "dominant_scale_share": dom_cnt / total if total > 0 else 0.0,
            })
        cluster_edge_df = pd.DataFrame(rows)
    else:
        cluster_edge_df = pd.DataFrame(
            columns=["cluster_a", "cluster_b", "total_construct_pairs",
                     "n_scales", "dominant_scale", "dominant_scale_share"])

    # Attach UMAP coordinates to per-scale edges (for Tab 3 ego mode)
    coord_idx = pts.set_index("construct_id")[["x", "y", "hdbscan_label"]].to_dict("index")
    if not edges_df.empty:
        edges_df["source_x"] = edges_df["construct_a"].map(
            lambda c: coord_idx.get(c, {}).get("x", 0))
        edges_df["source_y"] = edges_df["construct_a"].map(
            lambda c: coord_idx.get(c, {}).get("y", 0))
        edges_df["target_x"] = edges_df["construct_b"].map(
            lambda c: coord_idx.get(c, {}).get("x", 0))
        edges_df["target_y"] = edges_df["construct_b"].map(
            lambda c: coord_idx.get(c, {}).get("y", 0))
        edges_df["source_cluster"] = edges_df["construct_a"].map(
            lambda c: coord_idx.get(c, {}).get("hdbscan_label", -1))
        edges_df["target_cluster"] = edges_df["construct_b"].map(
            lambda c: coord_idx.get(c, {}).get("hdbscan_label", -1))

    return {
        "pts": pts,
        "fw_df": fw_df,
        "fw_name": fw_name,
        "c_df": c_df,
        "edges_df": edges_df,
        "agg_edges": agg_edges,
        "cluster_edge_df": cluster_edge_df,
        "color_map": color_map,
    }


# ── Tab 1: Semantic Map ───────────────────────────────────────────────────────

def _render_tab1(pts: pd.DataFrame, color_map: dict[str, str], fw_df: pd.DataFrame) -> None:
    n_clusters = int(pts[pts["hdbscan_label"] != -1]["hdbscan_label"].nunique())
    c1, c2, c3 = st.columns(3)
    c1.metric("Constructs", len(pts))
    c2.metric("Semantic clusters", n_clusters)
    c3.metric("Scales", len(fw_df))

    col_a, col_b, col_c = st.columns([3, 2, 1])
    with col_a:
        non_noise = sorted(
            [n for n in pts["cluster_name_final"].unique() if n != "Noise"],
            key=lambda x: (x.startswith("Cluster"), x),
        )
        all_names = non_noise + (["Noise"] if "Noise" in pts["cluster_name_final"].values else [])
        selected_clusters = st.multiselect(
            "Clusters", all_names, default=all_names, key="tab1_clusters"
        )
    with col_b:
        color_by = st.radio("Color by", ["Cluster", "Frequency"], horizontal=True, key="tab1_color_by")
    with col_c:
        size_mult = st.slider("Size ×", 0.5, 3.0, 1.0, 0.5, key="tab1_size")

    df = pts[pts["cluster_name_final"].isin(selected_clusters)].copy() if selected_clusters else pts.copy()
    if df.empty:
        st.info("No constructs match the current filter.")
        return

    fig = go.Figure()
    if color_by == "Cluster":
        for cname in sorted(df["cluster_name_final"].unique(), key=lambda x: (x == "Noise", x)):
            sub = df[df["cluster_name_final"] == cname]
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="markers",
                name=cname,
                marker=dict(
                    size=sub["freq_scaled"] * size_mult,
                    color=color_map.get(cname, "#aaa"),
                    opacity=0.75,
                    line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Cluster: %{customdata[1]}<br>"
                    "Frequency: %{customdata[2]} scales<br>"
                    "Scales: %{customdata[3]}"
                    "<extra></extra>"
                ),
                customdata=sub[["construct_label", "cluster_name_final", "frequency", "all_scales"]].values,
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            name="Frequency",
            marker=dict(
                size=df["freq_scaled"] * size_mult,
                color=df["frequency"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Scales"),
                opacity=0.82,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Frequency: %{customdata[1]} scales<br>"
                "Cluster: %{customdata[2]}"
                "<extra></extra>"
            ),
            customdata=df[["construct_label", "frequency", "cluster_name_final"]].values,
        ))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.02, font=dict(size=11)),
        margin=dict(l=20, r=20, t=20, b=60),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True, key="tab1_chart")


# ── Tab 2: Measurement Decomposition ─────────────────────────────────────────

def _build_cytoscape_html(
    pts: pd.DataFrame,
    agg_edges: pd.DataFrame,
    fw_name: dict[str, str],
    edge_pct: int,
    height: int,
) -> str:
    if pts.empty:
        return "<p style='padding:16px;color:#666'>No data available.</p>"

    # Deduplicate nodes by normalized label: merge same-label constructs from
    # different scales into one node (canonical = first occurrence).
    max_freq_local = max(int(pts["frequency"].max()), 1)
    pts_sorted = pts.copy()
    pts_sorted["_norm"] = pts_sorted["construct_label"].str.lower().str.strip()

    # Leiden community → deterministic color assignment
    _LEIDEN_PALETTE = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        "#6b6ecf", "#b5cf6b", "#8ca252", "#bd9e39", "#e7ba52",
        "#843c39", "#ad494a", "#d6616b", "#7b4173", "#a55194",
    ]
    leiden_comm_ids = sorted(
        int(c) for c in pts_sorted["leiden_community"].dropna().astype(int).unique()
        if int(c) >= 0
    )
    leiden_color_map: dict[int, str] = {
        comm: _LEIDEN_PALETTE[i % len(_LEIDEN_PALETTE)]
        for i, comm in enumerate(leiden_comm_ids)
    }

    canonical_id: dict[str, str] = {}   # construct_id → canonical node id
    seen_labels: dict[str, dict] = {}   # norm_label → node data dict
    nodes_json = []

    for _, row in pts_sorted.iterrows():
        cid = str(row["construct_id"])
        norm = str(row["_norm"])
        freq = int(row.get("frequency", 1))
        item_count = int(row.get("item_count", 0))

        if norm not in seen_labels:
            size = round(5 + (freq / max_freq_local) * 10, 1)
            leiden_comm = int(row.get("leiden_community", -1))
            nd = {
                "id": cid,
                "label": str(row.get("construct_label", cid)),
                "freq": freq,
                "item_count": item_count,
                "leiden_community": leiden_comm,
                "leiden_color": leiden_color_map.get(leiden_comm, "#aaa"),
                "sc_color": str(row.get("cluster_color", "#aaa") or "#aaa"),
                "cluster_name": str(row.get("cluster_name_final", "Unknown")),
                "all_scales": str(row.get("all_scales", "") or ""),
                "size": size,
            }
            seen_labels[norm] = nd
            nodes_json.append({"data": nd, "classes": "construct-node"})
            canonical_id[cid] = cid
        else:
            # Map duplicate to existing canonical node; accumulate item_count
            canon_cid = seen_labels[norm]["id"]
            canonical_id[cid] = canon_cid
            seen_labels[norm]["item_count"] += item_count
            seen_labels[norm]["freq"] = max(seen_labels[norm]["freq"], freq)

    # Recompute sizes after accumulation
    max_freq_final = max((d["freq"] for d in seen_labels.values()), default=1)
    for nd in seen_labels.values():
        nd["size"] = round(5 + (nd["freq"] / max_freq_final) * 10, 1)

    # leiden_community lookup keyed by canonical id
    leiden_map: dict[str, int] = {nd["id"]: nd["leiden_community"] for nd in seen_labels.values()}
    canon_ids = set(canonical_id.values())

    # Filter edges by percentile, remap to canonical ids, tag community membership
    edges_json = []
    if not agg_edges.empty:
        visible_e = agg_edges[
            agg_edges["construct_a"].astype(str).map(
                lambda c: canonical_id.get(c, c)).isin(canon_ids) &
            agg_edges["construct_b"].astype(str).map(
                lambda c: canonical_id.get(c, c)).isin(canon_ids)
        ].copy()
        if not visible_e.empty:
            threshold = float(visible_e["shared_scale_count"].quantile(edge_pct / 100.0))
            visible_e = visible_e[visible_e["shared_scale_count"] >= threshold].copy()
        if not visible_e.empty:
            max_w = visible_e["shared_scale_count"].max()
            visible_e["weight_norm"] = (visible_e["shared_scale_count"] / max(max_w, 1)).round(3)

            seen: set[str] = set()
            for _, erow in visible_e.iterrows():
                src = canonical_id.get(str(erow["construct_a"]), str(erow["construct_a"]))
                tgt = canonical_id.get(str(erow["construct_b"]), str(erow["construct_b"]))
                if src == tgt:
                    continue  # skip self-loops from merging
                ek = f"{min(src, tgt)}__{max(src, tgt)}"
                if ek in seen:
                    continue
                seen.add(ek)
                src_comm = leiden_map.get(src, -1)
                tgt_comm = leiden_map.get(tgt, -1)
                same_comm = src_comm >= 0 and tgt_comm >= 0 and src_comm == tgt_comm
                comm_cls = "intra-community" if same_comm else "cross-community"
                comm_color = leiden_color_map.get(src_comm, "#aaa") if same_comm else "#999"
                wn = float(erow["weight_norm"])
                backbone_cls = "backbone" if wn >= 0.5 else "non-backbone"
                edges_json.append({"data": {
                    "id": ek,
                    "source": src,
                    "target": tgt,
                    "shared_scale_count": int(erow["shared_scale_count"]),
                    "weight_norm": wn,
                    "same_community": same_comm,
                    "comm_color": comm_color,
                }, "classes": f"co-measurement {comm_cls} {backbone_cls}"})

    elements_json = json.dumps(nodes_json + edges_json)

    tooltip_js = _NODE_TOOLTIP_JS

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent;}}
#cy-container{{position:relative;}}
#cy{{width:100%;height:{height}px;background:#f8f9fa;border-radius:8px;}}
#cy-tooltip{{
  display:none;position:absolute;background:rgba(255,255,255,0.97);
  border:1px solid #ddd;border-radius:6px;padding:8px 12px;
  font-size:12px;line-height:1.6;max-width:280px;
  box-shadow:0 2px 8px rgba(0,0,0,0.15);pointer-events:none;z-index:999;color:#333;
}}
#detail{{padding:12px 16px;min-height:56px;font-size:13px;line-height:1.5;
  background:#f0f2f6;border-radius:6px;margin-top:8px;color:#333;}}
.peer-tag{{display:inline-block;margin:2px 3px;padding:2px 8px;
  border-radius:3px;font-size:12px;color:#fff;font-weight:500;}}
</style></head>
<body>
<div id="cy-container">
<div id="cy"></div>
<div id="cy-tooltip"></div>
</div>
<div id="detail"><span style="color:#999">Click any construct node to see its co-measurement partners.
Mixed-color islands indicate constructs from different semantic clusters being habitually bundled together.</span></div>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/layout-base/layout-base.js"></script>
<script src="https://unpkg.com/cose-base/cose-base.js"></script>
<script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
<script>
(function(){{
var elements = {elements_json};

var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {{ selector: 'node.construct-node', style: {{
      'background-color': 'data(sc_color)',
      'background-opacity': 0.82,
      'width': 'data(size)', 'height': 'data(size)',
      'label': '',
      'border-width': 0.5, 'border-color': 'rgba(150,150,150,0.25)',
    }}}},
    {{ selector: 'node.community-label', style: {{
      'label': 'data(label)',
      'font-size': 10, 'color': 'rgba(80,80,80,0.28)',
      'background-opacity': 0, 'border-width': 0,
      'width': 1, 'height': 1, 'shape': 'rectangle',
      'text-valign': 'center', 'text-halign': 'center',
      'events': 'no', 'z-index': 0,
    }}}},
    {{ selector: 'node.dimmed', style: {{
      'background-opacity': 0.07, 'width': 4, 'height': 4,
      'label': '', 'border-width': 0,
    }}}},
    {{ selector: 'node.selected', style: {{
      'border-width': 2.5, 'z-index': 10,
    }}}},
    {{ selector: 'edge.non-backbone', style: {{ 'display': 'none' }} }},
    {{ selector: 'edge.cross-community.backbone', style: {{
      'line-color': '#b0b0b0',
      'opacity': 0.12,
      'width': 0.6,
      'curve-style': 'bezier',
      'z-index': 1,
    }}}},
    {{ selector: 'edge.intra-community.backbone', style: {{
      'line-color': 'data(comm_color)',
      'opacity': 'mapData(weight_norm, 0.5, 1, 0.28, 0.60)',
      'width': 'mapData(weight_norm, 0.5, 1, 0.7, 1.8)',
      'curve-style': 'bezier',
      'z-index': 2,
    }}}},
    {{ selector: 'edge.hidden', style: {{ 'display': 'none' }} }},
    {{ selector: 'edge.active-edge', style: {{
      'z-index': 5, 'curve-style': 'unbundled-bezier',
    }} }},
  ],
}});

function runLayout() {{
  var layout = cy.layout({{
    name: 'fcose',
    quality: 'proof',
    randomize: true,
    animate: false,
    idealEdgeLength: function(edge) {{
      return edge.data('same_community') ? 75 : 200;
    }},
    edgeElasticity: function(edge) {{
      return edge.data('same_community') ? 0.30 : 0.008;
    }},
    nodeRepulsion: 14000,
    gravityRange: 2.5,
    numIter: 2500,
    tile: true,
    tilingPaddingVertical: 30,
    tilingPaddingHorizontal: 30,
  }});
  layout.one('layoutstop', function() {{
    addCommunityLabels();
    cy.fit(cy.nodes(), 30);
  }});
  layout.run();
}}

function addCommunityLabels() {{
  var commData = {{}};
  cy.nodes('.construct-node').forEach(function(n) {{
    var c = n.data('leiden_community');
    if (c < 0) return;
    if (!commData[c]) commData[c] = {{x: 0, y: 0, count: 0}};
    commData[c].x += n.position('x');
    commData[c].y += n.position('y');
    commData[c].count++;
  }});
  Object.keys(commData).forEach(function(c) {{
    var d = commData[c];
    cy.add({{
      group: 'nodes',
      data: {{id: 'comm_lbl_' + c, label: 'Community ' + (parseInt(c) + 1), leiden_community: -999}},
      position: {{x: d.x / d.count, y: d.y / d.count}},
      classes: 'community-label',
    }});
  }});
}}

runLayout();

// ── Node hover tooltip (fields match Tab 1 Plotly hovertemplate) ───────────
{tooltip_js}

var ttip = document.getElementById('cy-tooltip');
cy.on('mouseover', 'node.construct-node', function(evt) {{
  var node = evt.target;
  var oe = evt.originalEvent;
  ttip.innerHTML = buildNodeTooltipHtml(node);
  ttip.style.display = 'block';
  ttip.style.left = (oe.offsetX + 14) + 'px';
  ttip.style.top  = (oe.offsetY + 14) + 'px';
}});
cy.on('mousemove', 'node.construct-node', function(evt) {{
  var oe = evt.originalEvent;
  ttip.style.left = (oe.offsetX + 14) + 'px';
  ttip.style.top  = (oe.offsetY + 14) + 'px';
}});
cy.on('mouseout', 'node.construct-node', function() {{
  ttip.style.display = 'none';
}});

cy.on('tap', 'node.construct-node', function(evt) {{
  var node = evt.target;
  var nodeComm = node.data('leiden_community');
  var scColor = node.data('sc_color');

  // Dim all construct nodes and hide all edges
  cy.nodes('.construct-node').addClass('dimmed').removeClass('selected');
  cy.edges().addClass('hidden').removeClass('active-edge').removeStyle('line-color opacity width');
  cy.nodes('.community-label').style({{'opacity': 0.05}});

  // Highlight selected node
  node.removeClass('dimmed').addClass('selected').style({{
    'border-color': scColor,
    'border-width': 2.5,
    'background-opacity': 0.75,
  }});

  var intraPartners = [];
  var crossPartners = [];

  var connEdges = cy.edges().filter(function(e) {{
    return e.hasClass('co-measurement') &&
      (e.source().id() === node.id() || e.target().id() === node.id());
  }});

  connEdges.forEach(function(e) {{
    var partner = e.source().id() === node.id() ? e.target() : e.source();
    if (!partner.hasClass('construct-node')) return;
    var partnerColor = partner.data('sc_color') || '#aaa';
    var sameComm = e.data('same_community');
    var wn = e.data('weight_norm') || 0;

    partner.removeClass('dimmed').style({{'background-opacity': 0.85}});
    e.removeClass('hidden').addClass('active-edge');

    if (sameComm) {{
      e.style({{
        'line-color': partnerColor,
        'opacity': 0.80,
        'width': 0.8 + wn * 0.28,
        'display': 'element',
      }});
      intraPartners.push(
        '<span class="peer-tag" style="background:' + partnerColor + '">' +
        partner.data('label') + '</span>'
      );
    }} else {{
      e.style({{
        'line-color': partnerColor,
        'opacity': 0.45,
        'width': 0.6,
        'display': 'element',
      }});
      crossPartners.push(
        '<span class="peer-tag" style="background:' + partnerColor + '">' +
        partner.data('label') + '</span>'
      );
    }}
  }});

  var commLabel = nodeComm >= 0 ? 'Community ' + (nodeComm + 1) : 'Unassigned';
  var html = '<b>' + node.data('label') + '</b>';
  html += ' <span style="color:#777;font-size:12px">— Semantic cluster: ' +
    node.data('cluster_name') + ' &nbsp;|&nbsp; ' + commLabel +
    ' &nbsp;|&nbsp; ' + node.data('freq') + ' scale(s)</span>';

  if (intraPartners.length > 0) {{
    html += '<br><span style="color:#555;font-size:12px">Same-community co-measurement (' +
      intraPartners.length + '):</span><br>' + intraPartners.join(' ');
  }}
  if (crossPartners.length > 0) {{
    html += '<br><span style="color:#999;font-size:12px">Cross-community co-measurement (' +
      crossPartners.length + '):</span><br>' + crossPartners.join(' ');
  }}
  if (intraPartners.length === 0 && crossPartners.length === 0) {{
    html += '<br><span style="color:#999;font-size:12px">No co-measurement partners in current view.</span>';
  }}
  document.getElementById('detail').innerHTML = html;
}});

cy.on('tap', function(evt) {{
  if (evt.target === cy) {{
    cy.nodes('.construct-node').removeClass('dimmed selected').removeStyle(
      'background-opacity border-color border-width');
    cy.nodes('.community-label').removeStyle('opacity');
    // restore backbone edges; keep non-backbone hidden
    cy.edges('.backbone').removeClass('hidden active-edge').removeStyle('line-color opacity width');
    cy.edges('.non-backbone').addClass('hidden').removeClass('active-edge').removeStyle('line-color opacity width');
    document.getElementById('detail').innerHTML =
      '<span style="color:#999">Click any construct node to see its co-measurement partners. ' +
      'Mixed-color islands indicate constructs from different semantic clusters being ' +
      'habitually bundled together.</span>';
  }}
}});
}})();
</script>
</body></html>"""


def _render_tab2(
    pts: pd.DataFrame,
    agg_edges: pd.DataFrame,
    fw_name: dict[str, str],
    fw_df: pd.DataFrame,
) -> None:
    has_leiden = "leiden_community" in pts.columns and pts["leiden_community"].gt(-1).any()

    col1, col2 = st.columns([2, 3])
    with col1:
        edge_pct = st.slider(
            "Edge strength percentile", 50, 95, 85, key="tab2_edge_pct",
            help="Only show edges above this percentile of co-measurement strength. Higher = fewer, stronger edges.",
        )
    with col2:
        n_edges = 0
        if not agg_edges.empty:
            threshold = float(agg_edges["shared_scale_count"].quantile(edge_pct / 100.0))
            n_edges = int((agg_edges["shared_scale_count"] >= threshold).sum())
        n_communities = (
            int(pts["leiden_community"].nunique() -
                (1 if (pts["leiden_community"] == -1).any() else 0))
            if has_leiden else 0
        )
        st.caption(
            f"{len(fw_df)} scales &nbsp;·&nbsp; {len(pts)} constructs &nbsp;·&nbsp; "
            f"{n_edges} edges shown &nbsp;·&nbsp; {n_communities} Leiden communities"
        )

    if not has_leiden:
        st.warning(
            "Leiden community detection requires `leidenalg` and `python-igraph`. "
            "Install them to enable community-aware layout: `pip install leidenalg python-igraph`"
        )

    html = _build_cytoscape_html(
        pts=pts, agg_edges=agg_edges, fw_name=fw_name,
        edge_pct=edge_pct, height=500,
    )
    components.html(html, height=620, scrolling=False)


def _geometric_median(coords: np.ndarray, eps: float = 1e-5, max_iter: int = 200) -> np.ndarray:
    """Weiszfeld algorithm — robust to outliers, result lies inside the point cloud."""
    median = coords.mean(axis=0)
    for _ in range(max_iter):
        dists = np.maximum(np.linalg.norm(coords - median, axis=1), eps)
        weights = 1.0 / dists
        new_median = (coords * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
        if np.linalg.norm(new_median - median) < eps:
            break
        median = new_median
    return median


def _build_sc_centroids(pts: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Returns (sc_centroids, cluster_name_map, cluster_sizes) for labeled clusters."""
    labeled = pts[pts["hdbscan_label"] != -1]
    sc_centroids = {
        int(label): {"x": float(pt[0]), "y": float(pt[1])}
        for label, grp in labeled.groupby("hdbscan_label")
        for pt in [_geometric_median(grp[["x", "y"]].values)]
    }
    cluster_name_map = (
        labeled.drop_duplicates("hdbscan_label")
        .set_index("hdbscan_label")["cluster_name_final"]
        .to_dict()
    )
    cluster_sizes = labeled.groupby("cluster_name_final").size().to_dict()
    return sc_centroids, cluster_name_map, cluster_sizes


# ── Tab 3: Integrated View ────────────────────────────────────────────────────

def _render_tab3(
    pts: pd.DataFrame,
    edges_df: pd.DataFrame,
    cluster_edge_df: pd.DataFrame,
    color_map: dict[str, str],
    fw_name: dict[str, str],
) -> None:
    if pts.empty:
        st.info("No data available.")
        return

    selected_id: str | None = st.session_state.get("tab3_selected", None)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        min_pairs = st.slider(
            "Min construct pairs per cluster edge", 1, 100, 10, key="tab3_min_pairs",
            help="Filter out weak cluster-to-cluster connections",
        )
        show_dominated = st.checkbox(
            "Show single-scale dominated edges", False, key="tab3_show_dominated",
            help="Show dashed edges (⚠️) where one scale contributes ≥50% of construct pairs",
        )
    with col2:
        cluster_focus = st.selectbox(
            "Focus cluster", ["All"] + sorted(
                [n for n in pts["cluster_name_final"].unique() if n != "Noise"],
                key=lambda x: (x.startswith("Cluster"), x),
            ),
            key="tab3_focus_cluster",
        )
    with col3:
        if selected_id is not None:
            st.markdown("")
            if st.button("← Back to overview", key="tab3_back"):
                st.session_state.tab3_selected = None
                st.rerun()

    if selected_id is None:
        # ── Mode A: Global overview (cluster-pair edges) ──────────────────────
        if cluster_focus != "All":
            pts = pts.copy()

        fig = _build_tab3_overview(
            pts, cluster_edge_df, color_map, fw_name,
            min_pairs, show_dominated, cluster_focus,
        )

        visible_ce = cluster_edge_df[cluster_edge_df["total_construct_pairs"] >= min_pairs] \
            if not cluster_edge_df.empty else cluster_edge_df
        if not show_dominated and not visible_ce.empty:
            visible_ce = visible_ce[visible_ce["dominant_scale_share"] < 0.5]
        strong = int((visible_ce["n_scales"] >= 8).sum()) if not visible_ce.empty else 0
        dominated_count = int((visible_ce["dominant_scale_share"] >= 0.5).sum()) if not visible_ce.empty else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Constructs", len(pts))
        m2.metric("Cluster edges shown", len(visible_ce))
        m3.metric("Strong consensus ✦", strong)
        m4.metric("Dominated ⚠️", dominated_count)
    else:
        # ── Mode C: Ego-network focus (cluster level) ─────────────────────────
        fig = _build_tab3_ego(
            selected_id, pts, cluster_edge_df, color_map, fw_name, min_pairs, show_dominated,
        )

        sel_pts = pts[pts["cluster_name_final"] == str(selected_id)]
        if not sel_pts.empty:
            n_constructs = len(sel_pts)
            sel_hlabel = int(sel_pts["hdbscan_label"].iloc[0])
            vis_ce = cluster_edge_df[
                cluster_edge_df["total_construct_pairs"] >= min_pairs
            ].copy() if not cluster_edge_df.empty else cluster_edge_df
            if not show_dominated and not vis_ce.empty:
                vis_ce = vis_ce[vis_ce["dominant_scale_share"] < 0.5]
            ego_edges = vis_ce[
                (vis_ce["cluster_a"] == sel_hlabel) | (vis_ce["cluster_b"] == sel_hlabel)
            ] if not vis_ce.empty else pd.DataFrame()
            total_pairs = int(ego_edges["total_construct_pairs"].sum()) if not ego_edges.empty else 0
            m1, m2, m3 = st.columns(3)
            m1.metric("Constructs in cluster", n_constructs)
            m2.metric("Connected clusters", len(ego_edges))
            m3.metric("Cross-cluster construct pairs", total_pairs)

    event = st.plotly_chart(
        fig, use_container_width=True,
        key="tab3_chart", on_select="rerun", selection_mode="points",
    )

    if event and hasattr(event, "selection") and event.selection:
        raw_pts = getattr(event.selection, "points", None)
        if raw_pts is None and isinstance(event.selection, dict):
            raw_pts = event.selection.get("points", [])
        if raw_pts:
            p0 = raw_pts[0]
            cd = p0.get("customdata") if isinstance(p0, dict) else None
            clicked_cid = str(cd[0]) if cd and cd[0] is not None else None
            if clicked_cid and not clicked_cid.startswith("__"):
                if clicked_cid == str(selected_id):
                    st.session_state.tab3_selected = None
                else:
                    st.session_state.tab3_selected = clicked_cid
                st.rerun()


def _build_tab3_overview(
    pts: pd.DataFrame,
    cluster_edge_df: pd.DataFrame,
    color_map: dict[str, str],
    fw_name: dict[str, str],
    min_pairs: int,
    show_dominated: bool,
    cluster_focus: str,
) -> go.Figure:
    fig = go.Figure()

    sc_centroids, cluster_name_map, cluster_sizes = _build_sc_centroids(pts)

    # ── Cluster-pair edges ────────────────────────────────────────────────────
    if not cluster_edge_df.empty:
        visible = cluster_edge_df[cluster_edge_df["total_construct_pairs"] >= min_pairs].copy()
        if not show_dominated:
            visible = visible[visible["dominant_scale_share"] < 0.5]

        if not visible.empty:
            max_pairs_val = visible["total_construct_pairs"].max()
            max_scales_val = visible["n_scales"].max()

            for _, e in visible.iterrows():
                ca, cb = int(e["cluster_a"]), int(e["cluster_b"])
                if ca not in sc_centroids or cb not in sc_centroids:
                    continue
                pa, pb = sc_centroids[ca], sc_centroids[cb]
                width = 0.5 + (e["total_construct_pairs"] / max(max_pairs_val, 1)) * 7
                alpha = 0.12 + (e["n_scales"] / max(max_scales_val, 1)) * 0.68
                dominated = float(e["dominant_scale_share"]) >= 0.5
                ca_name = cluster_name_map.get(ca, f"Cluster {ca}")
                cb_name = cluster_name_map.get(cb, f"Cluster {cb}")
                n_scales_val = int(e["n_scales"])
                if n_scales_val >= 8:
                    signal = "Strong consensus ✦"
                elif n_scales_val >= 4:
                    signal = "Moderate consensus"
                else:
                    signal = "Single-scale dominated ⚠️"
                dom_name = fw_name.get(str(e["dominant_scale"]), str(e["dominant_scale"]))
                fig.add_trace(go.Scatter(
                    x=[pa["x"], pb["x"], None],
                    y=[pa["y"], pb["y"], None],
                    mode="lines",
                    line=dict(
                        color=f"rgba(186,117,23,{alpha:.2f})",
                        width=width,
                        dash="dot" if dominated else "solid",
                    ),
                    hovertemplate=(
                        f"<b>{ca_name} ↔ {cb_name}</b><br>"
                        f"Co-measured construct pairs: {int(e['total_construct_pairs'])}<br>"
                        f"Source scales: {n_scales_val} → {signal}<br>"
                        f"Top contributing scale: {dom_name} ({float(e['dominant_scale_share']):.0%})"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ))

    # ── Cluster centroid nodes (one per cluster) ──────────────────────────────
    for label, centroid_pt in sc_centroids.items():
        cname = cluster_name_map.get(label, f"Cluster {label}")
        if cname == "Noise":
            continue
        color = color_map.get(cname, "#aaa")
        n_constructs = cluster_sizes.get(cname, 1)
        size = 12 + np.sqrt(n_constructs) * 2.5
        opacity = 0.90
        if cluster_focus != "All" and cname != cluster_focus:
            opacity = 0.20
            color = "rgba(150,150,150,0.4)"
        fig.add_trace(go.Scatter(
            x=[centroid_pt["x"]], y=[centroid_pt["y"]],
            mode="markers+text",
            name=cname,
            marker=dict(
                size=size, color=color, opacity=opacity,
                line=dict(width=1.5, color="rgba(255,255,255,0.7)"),
            ),
            text=[cname],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=(
                f"<b>{cname}</b><br>"
                f"Constructs: {n_constructs}<br>"
                "Click to explore connections"
                "<extra></extra>"
            ),
            customdata=[[cname, cname, n_constructs]],
        ))

    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.02, font=dict(size=10)),
        margin=dict(l=20, r=20, t=20, b=60),
        height=520,
    )
    return fig


def _build_tab3_ego(
    selected_cluster: str,
    pts: pd.DataFrame,
    cluster_edge_df: pd.DataFrame,
    color_map: dict[str, str],
    fw_name: dict[str, str],
    min_pairs: int,
    show_dominated: bool,
) -> go.Figure:
    """Cluster-level ego network: one centroid node per cluster, edges from cluster_edge_df."""
    fig = go.Figure()

    sc_centroids, cluster_name_map, cluster_sizes = _build_sc_centroids(pts)
    if not sc_centroids:
        return fig

    name_to_label = {v: k for k, v in cluster_name_map.items()}
    selected_label: int | None = name_to_label.get(selected_cluster)

    # Filter ego edges
    connected_labels: set[int] = set()
    if selected_label is not None and not cluster_edge_df.empty:
        visible = cluster_edge_df[cluster_edge_df["total_construct_pairs"] >= min_pairs].copy()
        if not show_dominated:
            visible = visible[visible["dominant_scale_share"] < 0.5]
        if not visible.empty:
            ego_edges = visible[
                (visible["cluster_a"] == selected_label) | (visible["cluster_b"] == selected_label)
            ]
            max_pairs_val = visible["total_construct_pairs"].max()
            max_scales_val = visible["n_scales"].max()
            for _, e in ego_edges.iterrows():
                ca, cb = int(e["cluster_a"]), int(e["cluster_b"])
                if ca not in sc_centroids or cb not in sc_centroids:
                    continue
                other = cb if ca == selected_label else ca
                connected_labels.add(other)
                pa, pb = sc_centroids[ca], sc_centroids[cb]
                width = 0.5 + (e["total_construct_pairs"] / max(max_pairs_val, 1)) * 7
                alpha = 0.35 + (e["n_scales"] / max(max_scales_val, 1)) * 0.55
                dominated = float(e["dominant_scale_share"]) >= 0.5
                ca_name = cluster_name_map.get(ca, f"Cluster {ca}")
                cb_name = cluster_name_map.get(cb, f"Cluster {cb}")
                n_scales_val = int(e["n_scales"])
                if n_scales_val >= 8:
                    signal = "Strong consensus ✦"
                elif n_scales_val >= 4:
                    signal = "Moderate consensus"
                else:
                    signal = "Single-scale dominated ⚠️"
                dom_name = fw_name.get(str(e["dominant_scale"]), str(e["dominant_scale"]))
                fig.add_trace(go.Scatter(
                    x=[pa["x"], pb["x"], None],
                    y=[pa["y"], pb["y"], None],
                    mode="lines",
                    line=dict(
                        color=f"rgba(186,117,23,{alpha:.2f})",
                        width=width,
                        dash="dot" if dominated else "solid",
                    ),
                    hovertemplate=(
                        f"<b>{ca_name} ↔ {cb_name}</b><br>"
                        f"Co-measured construct pairs: {int(e['total_construct_pairs'])}<br>"
                        f"Source scales: {n_scales_val} → {signal}<br>"
                        f"Top contributing scale: {dom_name} ({float(e['dominant_scale_share']):.0%})"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ))

    # Dimmed (unconnected, non-selected) cluster nodes
    for label, centroid_pt in sc_centroids.items():
        cname = cluster_name_map.get(label, f"Cluster {label}")
        if cname == "Noise" or label == selected_label or label in connected_labels:
            continue
        n_constructs = cluster_sizes.get(cname, 1)
        size = 8 + np.sqrt(n_constructs) * 1.5
        fig.add_trace(go.Scatter(
            x=[centroid_pt["x"]], y=[centroid_pt["y"]],
            mode="markers",
            marker=dict(size=size, color="rgba(150,150,150,0.12)"),
            hoverinfo="skip", showlegend=False,
            customdata=[[cname]],
        ))

    # Connected cluster nodes
    for label in connected_labels:
        if label not in sc_centroids:
            continue
        centroid_pt = sc_centroids[label]
        cname = cluster_name_map.get(label, f"Cluster {label}")
        color = color_map.get(cname, "#aaa")
        n_constructs = cluster_sizes.get(cname, 1)
        size = 12 + np.sqrt(n_constructs) * 2.0
        fig.add_trace(go.Scatter(
            x=[centroid_pt["x"]], y=[centroid_pt["y"]],
            mode="markers+text",
            name=cname,
            marker=dict(
                size=size, color=color, opacity=0.85,
                line=dict(width=1.5, color="rgba(255,255,255,0.6)"),
            ),
            text=[cname],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=f"<b>{cname}</b><br>Constructs: {n_constructs}<extra></extra>",
            customdata=[[cname]],
        ))

    # Selected cluster node
    if selected_label is not None and selected_label in sc_centroids:
        centroid_pt = sc_centroids[selected_label]
        color = color_map.get(selected_cluster, "#aaa")
        n_constructs = cluster_sizes.get(selected_cluster, 1)
        size = 16 + np.sqrt(n_constructs) * 2.5
        fig.add_trace(go.Scatter(
            x=[centroid_pt["x"]], y=[centroid_pt["y"]],
            mode="markers+text",
            name=selected_cluster,
            marker=dict(
                size=size, color=color, opacity=1.0,
                line=dict(width=3, color="#E24B4A"),
            ),
            text=[selected_cluster],
            textposition="top center",
            textfont=dict(size=11, color="#E24B4A"),
            hovertemplate=f"<b>{selected_cluster}</b><br>Constructs: {n_constructs}<extra></extra>",
            customdata=[[selected_cluster]],
        ))

    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.02),
        margin=dict(l=20, r=20, t=20, b=60),
        height=520,
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def render() -> None:
    st.title("Corpus Overview")

    if get_visualization_cache() is None:
        st.warning("Visualization cache is missing. Build it once offline:")
        st.code("python -m ssn.scripts.build_visualization_cache", language="bash")
        return

    # ── LLM cluster label patch ───────────────────────────────────────────────
    manifest = get_visualization_cache() or {}
    _params = manifest.get("manifest", {}).get("params", {}) if isinstance(manifest.get("manifest"), dict) else {}
    _has_llm = bool(_params.get("use_llm_labels", False))

    with st.expander("⚙️ Cluster labels", expanded=not _has_llm):
        if _has_llm:
            st.success("Cluster labels are LLM-interpreted.")
        else:
            st.warning("Cluster labels are generic (C0, C1 …). Run LLM interpretation to generate descriptive names.")
        if st.button("Run LLM cluster interpretation", type="primary" if not _has_llm else "secondary"):
            with st.spinner("Calling LLM to name clusters — this may take ~30 s …"):
                try:
                    patch_llm_cluster_labels()
                    st.cache_data.clear()
                    st.success("Done! Cluster labels updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"LLM labeling failed: {exc}")

    data = _load_data()
    if not data:
        st.error("Failed to load construct data. Check visualization cache path.")
        return

    pts = data["pts"]
    fw_df = data["fw_df"]
    fw_name = data["fw_name"]
    edges_df = data["edges_df"]
    color_map = data["color_map"]

    tab1, tab2, tab3 = st.tabs([
        "🗺️ Semantic Map",
        "🔬 Measurement Decomposition",
        "🔗 Integrated View",
    ])

    with tab1:
        _render_tab1(pts, color_map, fw_df)

    with tab2:
        _render_tab2(pts, data["agg_edges"], fw_name, fw_df)

    with tab3:
        _render_tab3(pts, edges_df, data["cluster_edge_df"], color_map, fw_name)


render()
