"""Plotly-based ZMLT multi-level view: Domain + expanded Constructs from pipeline JSON.

Uses pre-computed layout (nodes/edges/frameworks from ssn_graph_data.json).
Only Domain and Construct levels; no Item-level (Phase 1).
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# Framework colors (guide: Big Five, HEXACO, IPIP-NEO, Oregon Vocational, Clinical, Values; overflow grey)
_FRAMEWORK_PALETTE = [
    "#2563EB",  # blue
    "#D97706",  # amber/orange
    "#059669",  # green
    "#DC2626",  # red
    "#7C3AED",  # purple
    "#0891B2",  # cyan
    "#E91E63",  # pink
    "#34495E",  # dark grey-blue
    "#8BC34A",  # lime
    "#FF5722",  # deep orange
    "#607D8B",  # blue-grey
    "#9CA3AF",  # grey (overflow)
]


def _framework_color(framework_id: str) -> str:
    if not framework_id:
        return "#888888"
    idx = sum(ord(c) for c in framework_id) % len(_FRAMEWORK_PALETTE)
    return _FRAMEWORK_PALETTE[idx]


def build_zmlt_figure(
    graph_data: dict[str, Any],
    expanded_domains: set[str],
    min_edge_weight: float = 0.15,
    title: str = "Corpus Network",
    height: int = 600,
) -> go.Figure:
    """Build a Plotly figure for the ZMLT view: Domain nodes + Constructs of expanded domains.

    - Visible nodes: all level=domain + all level=construct whose parent_id in expanded_domains.
    - Visible edges: Domain–Domain (meta) + Construct–Construct where both endpoints visible,
      and weight >= min_edge_weight.
    - Framework hulls: layout.shapes polygons from frameworks[].hull_points.
    - Node size by n_children; color by framework_id.
    """
    nodes = graph_data.get("nodes") or []
    edges = graph_data.get("edges") or []
    frameworks = graph_data.get("frameworks") or []

    visible_ids = set()
    for n in nodes:
        if n.get("level") == "domain":
            visible_ids.add(n["id"])
        elif n.get("level") == "construct" and n.get("parent_id") in expanded_domains:
            visible_ids.add(n["id"])

    node_by_id = {n["id"]: n for n in nodes}
    # Build edge list for visible edges (weight >= threshold)
    edge_tuples: list[tuple[float, float, float, float, float, bool]] = []
    for e in edges:
        if e.get("weight", 0) < min_edge_weight:
            continue
        a, b = e.get("source"), e.get("target")
        if a not in visible_ids or b not in visible_ids:
            continue
        na, nb = node_by_id.get(a), node_by_id.get(b)
        if not na or not nb:
            continue
        edge_tuples.append(
            (na["x"], na["y"], nb["x"], nb["y"], e.get("weight", 0.5), e.get("is_meta", False))
        )

    # Layout shapes: framework hulls (polygon, light fill)
    shapes = []
    for fw in frameworks:
        hull = fw.get("hull_points") or []
        if len(hull) < 3:
            continue
        # Plotly polygon: closed path
        xs = [p[0] for p in hull] + [hull[0][0]]
        ys = [p[1] for p in hull] + [hull[0][1]]
        color = fw.get("color", "#cccccc")
        shapes.append(
            dict(
                type="path",
                path="M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in zip(xs, ys)) + " Z",
                line=dict(color=color, width=1, dash="dot"),
                fillcolor=color,
                opacity=0.08,
                layer="below",
            )
        )

    # Edge traces: one for meta (dashed), one for solid (or single trace with dash by is_meta)
    edge_x_meta: list[float | None] = []
    edge_y_meta: list[float | None] = []
    edge_x_solid: list[float | None] = []
    edge_y_solid: list[float | None] = []
    for x0, y0, x1, y1, w, is_meta in edge_tuples:
        width = 0.5 + 2.5 * w
        alpha = 0.2 + 0.5 * w
        if is_meta:
            edge_x_meta.extend([x0, x1, None])
            edge_y_meta.extend([y0, y1, None])
        else:
            edge_x_solid.extend([x0, x1, None])
            edge_y_solid.extend([y0, y1, None])

    fig = go.Figure()

    if edge_x_meta:
        fig.add_trace(
            go.Scatter(
                x=edge_x_meta,
                y=edge_y_meta,
                mode="lines",
                line=dict(width=1.5, color="rgba(100,100,100,0.5)", dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if edge_x_solid:
        fig.add_trace(
            go.Scatter(
                x=edge_x_solid,
                y=edge_y_solid,
                mode="lines",
                line=dict(width=1, color="rgba(120,120,120,0.4)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Visible nodes
    vis_nodes = [n for n in nodes if n["id"] in visible_ids]
    if not vis_nodes:
        fig.update_layout(title=title, height=height, shapes=shapes)
        return fig

    node_x = [n["x"] for n in vis_nodes]
    node_y = [n["y"] for n in vis_nodes]
    node_text = [n.get("label", n["id"]) for n in vis_nodes]
    node_ids = [n["id"] for n in vis_nodes]
    node_sizes = [12 + min((n.get("n_children") or 0) * 1.2, 18) for n in vis_nodes]
    node_colors = [_framework_color(n.get("framework_id") or "") for n in vis_nodes]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[t[:24] + ("…" if len(t) > 24 else "") for t in node_text],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color="white"),
            ),
            hovertext=[f"{n.get('label', n['id'])} ({n.get('level', '')})" for n in vis_nodes],
            hoverinfo="text",
            customdata=node_ids,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
        shapes=shapes,
    )
    return fig


def build_domains_only_figure(
    graph_data: dict[str, Any],
    min_edge_weight: float = 0.15,
    title: str = "Domain map",
    height: int = 600,
) -> go.Figure:
    """Build a Plotly figure with only Domain nodes and Domain–Domain edges.

    Visible nodes: all level=domain. Visible edges: both endpoints in that set
    (all Domain–Domain are meta). Framework hulls included. Node customdata is
    [domain_id] per point for selection handling.
    """
    nodes = graph_data.get("nodes") or []
    edges = graph_data.get("edges") or []
    frameworks = graph_data.get("frameworks") or []

    visible_ids = {n["id"] for n in nodes if n.get("level") == "domain"}
    vis_nodes = [n for n in nodes if n["id"] in visible_ids]
    node_by_id = {n["id"]: n for n in nodes}

    edge_tuples: list[tuple[float, float, float, float, float, bool]] = []
    for e in edges:
        if e.get("weight", 0) < min_edge_weight:
            continue
        a, b = e.get("source"), e.get("target")
        if a not in visible_ids or b not in visible_ids:
            continue
        na, nb = node_by_id.get(a), node_by_id.get(b)
        if not na or not nb:
            continue
        edge_tuples.append(
            (na["x"], na["y"], nb["x"], nb["y"], e.get("weight", 0.5), e.get("is_meta", False))
        )

    shapes = []
    for fw in frameworks:
        hull = fw.get("hull_points") or []
        if len(hull) < 3:
            continue
        xs = [p[0] for p in hull] + [hull[0][0]]
        ys = [p[1] for p in hull] + [hull[0][1]]
        color = fw.get("color", "#cccccc")
        shapes.append(
            dict(
                type="path",
                path="M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in zip(xs, ys)) + " Z",
                line=dict(color=color, width=1, dash="dot"),
                fillcolor=color,
                opacity=0.08,
                layer="below",
            )
        )

    edge_x_meta: list[float | None] = []
    edge_y_meta: list[float | None] = []
    edge_x_solid: list[float | None] = []
    edge_y_solid: list[float | None] = []
    for x0, y0, x1, y1, w, is_meta in edge_tuples:
        if is_meta:
            edge_x_meta.extend([x0, x1, None])
            edge_y_meta.extend([y0, y1, None])
        else:
            edge_x_solid.extend([x0, x1, None])
            edge_y_solid.extend([y0, y1, None])

    fig = go.Figure()
    if edge_x_meta:
        fig.add_trace(
            go.Scatter(
                x=edge_x_meta,
                y=edge_y_meta,
                mode="lines",
                line=dict(width=1.5, color="rgba(100,100,100,0.5)", dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if edge_x_solid:
        fig.add_trace(
            go.Scatter(
                x=edge_x_solid,
                y=edge_y_solid,
                mode="lines",
                line=dict(width=1, color="rgba(120,120,120,0.4)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Annotations: edge similarity labels at midpoints (domain map: meta edges only for clarity)
    edge_annotations = []
    for x0, y0, x1, y1, w, is_meta in edge_tuples:
        if is_meta:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            edge_annotations.append(
                dict(x=mx, y=my, text=f"{w:.2f}", showarrow=False, xref="x", yref="y", font=dict(size=8))
            )

    if not vis_nodes:
        fig.update_layout(title=title, height=height, shapes=shapes, annotations=edge_annotations)
        return fig

    node_x = [n["x"] for n in vis_nodes]
    node_y = [n["y"] for n in vis_nodes]
    node_text = [n.get("label", n["id"]) for n in vis_nodes]
    node_sizes = [12 + min((n.get("n_children") or 0) * 1.2, 18) for n in vis_nodes]
    node_colors = [_framework_color(n.get("framework_id") or "") for n in vis_nodes]
    customdata = [[n["id"]] for n in vis_nodes]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[t[:24] + ("…" if len(t) > 24 else "") for t in node_text],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color="white"),
            ),
            hovertext=[f"{n.get('label', n['id'])} (domain)" for n in vis_nodes],
            hoverinfo="text",
            customdata=customdata,
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
        shapes=shapes,
        annotations=edge_annotations,
    )
    return fig


def build_domain_constructs_figure(
    graph_data: dict[str, Any],
    focused_domain_id: str,
    min_edge_weight: float = 0.15,
    title: str = "Constructs",
    height: int = 600,
    cross_construct_links: list[tuple[str, str, float]] | None = None,
) -> go.Figure:
    """Build a Plotly figure with only one Domain's Constructs and their edges.

    Visible nodes: level=construct and parent_id=focused_domain_id. Visible edges:
    is_meta=False, both endpoints in that set, weight >= min_edge_weight. No hulls.
    If cross_construct_links is provided (source_cid, other_cid, sim), draw dashed
    lines to other-framework constructs and show those as smaller external nodes.
    """
    nodes = graph_data.get("nodes") or []
    edges = graph_data.get("edges") or []

    visible_ids = {n["id"] for n in nodes if n.get("level") == "construct" and n.get("parent_id") == focused_domain_id}
    vis_nodes = [n for n in nodes if n["id"] in visible_ids]
    node_by_id = {n["id"]: n for n in nodes}

    edge_tuples: list[tuple[float, float, float, float, float]] = []
    for e in edges:
        if e.get("is_meta", True):
            continue
        w = e.get("weight", 0)
        if w < min_edge_weight:
            continue
        a, b = e.get("source"), e.get("target")
        if a not in visible_ids or b not in visible_ids:
            continue
        na, nb = node_by_id.get(a), node_by_id.get(b)
        if not na or not nb:
            continue
        edge_tuples.append((na["x"], na["y"], nb["x"], nb["y"], w))

    fig = go.Figure()
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_annotations: list[dict] = []
    for x0, y0, x1, y1, w in edge_tuples:
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        edge_annotations.append(
            dict(x=mx, y=my, text=f"{w:.2f}", showarrow=False, xref="x", yref="y", font=dict(size=8))
        )
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="rgba(120,120,120,0.4)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if not vis_nodes:
        fig.update_layout(
            title=title,
            height=height,
            annotations=[dict(text="No constructs in this domain", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)] + edge_annotations,
            xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="rgba(248,248,248,0.5)",
        )
        return fig

    node_x = [n["x"] for n in vis_nodes]
    node_y = [n["y"] for n in vis_nodes]
    node_text = [n.get("label", n["id"]) for n in vis_nodes]
    node_sizes = [12 + min((n.get("n_children") or 0) * 1.2, 18) for n in vis_nodes]
    node_colors = [_framework_color(n.get("framework_id") or "") for n in vis_nodes]
    customdata = [[n["id"]] for n in vis_nodes]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[t[:24] + ("…" if len(t) > 24 else "") for t in node_text],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color="white"),
            ),
            hovertext=[f"{n.get('label', n['id'])} (construct)" for n in vis_nodes],
            hoverinfo="text",
            customdata=customdata,
            showlegend=False,
        )
    )

    if cross_construct_links:
        ext_ids = set()
        cross_edge_x: list[float | None] = []
        cross_edge_y: list[float | None] = []
        for cid_a, cid_b, _sim in cross_construct_links:
            na, nb = node_by_id.get(cid_a), node_by_id.get(cid_b)
            if na and nb:
                cross_edge_x.extend([na["x"], nb["x"], None])
                cross_edge_y.extend([na["y"], nb["y"], None])
                ext_ids.add(cid_b)
        if cross_edge_x:
            fig.add_trace(
                go.Scatter(
                    x=cross_edge_x,
                    y=cross_edge_y,
                    mode="lines",
                    line=dict(width=1, color="rgba(100,100,100,0.5)", dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        ext_nodes = [node_by_id[eid] for eid in ext_ids if eid in node_by_id]
        if ext_nodes:
            ext_x = [n["x"] for n in ext_nodes]
            ext_y = [n["y"] for n in ext_nodes]
            ext_text = [n.get("label", n["id"])[:20] + ("…" if len(n.get("label", n["id"])) > 20 else "") for n in ext_nodes]
            ext_colors = [_framework_color(n.get("framework_id") or "") for n in ext_nodes]
            fig.add_trace(
                go.Scatter(
                    x=ext_x,
                    y=ext_y,
                    mode="markers+text",
                    text=ext_text,
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(size=8, color=ext_colors, line=dict(width=1, color="white"), symbol="square"),
                    hovertext=[f"{n.get('label', n['id'])} (other framework)" for n in ext_nodes],
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
        annotations=edge_annotations,
    )
    return fig


def build_frameworks_figure(
    frameworks: list[dict[str, Any]],
    graph_data: dict[str, Any] | None,
    title: str = "Frameworks",
    height: int = 600,
) -> go.Figure:
    """Build a Plotly figure with one node per framework for By-framework Layer 1.

    Node positions: centroid of framework hull_points from graph_data if available;
    otherwise a simple grid. customdata is [framework_id] for selection.
    """
    fw_by_id = {f["framework_id"]: f for f in frameworks}
    fw_ids = [f["framework_id"] for f in frameworks]
    if not fw_ids:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig

    # Positions: from graph_data hull centroids or grid
    x_coords: list[float] = []
    y_coords: list[float] = []
    fw_list = [fw_by_id[fid] for fid in fw_ids]
    if graph_data:
        fw_meta = {fw["id"]: fw for fw in (graph_data.get("frameworks") or [])}
        for fid in fw_ids:
            fw = fw_meta.get(fid, {})
            hull = fw.get("hull_points") or []
            if len(hull) >= 3:
                cx = sum(p[0] for p in hull) / len(hull)
                cy = sum(p[1] for p in hull) / len(hull)
                x_coords.append(cx)
                y_coords.append(cy)
            else:
                x_coords.append(0.0)
                y_coords.append(0.0)
        if not x_coords or all(x == 0 and y == 0 for x, y in zip(x_coords, y_coords)):
            # Fallback grid if no valid hulls
            n = len(fw_ids)
            cols = 2 if n >= 2 else 1
            x_coords = [(i % cols) * 1.5 for i in range(n)]
            y_coords = [-(i // cols) * 1.5 for i in range(n)]
    else:
        n = len(fw_ids)
        cols = 2 if n >= 2 else 1
        x_coords = [(i % cols) * 1.5 for i in range(n)]
        y_coords = [-(i // cols) * 1.5 for i in range(n)]

    labels = [fw_by_id[fid].get("name", fid) for fid in fw_ids]
    customdata = [[fid] for fid in fw_ids]
    colors = [_framework_color(fid) for fid in fw_ids]
    sizes = [24 for _ in fw_ids]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            text=[t[:20] + ("…" if len(t) > 20 else "") for t in labels],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
            hovertext=[f"{l} (framework)" for l in labels],
            hoverinfo="text",
            customdata=customdata,
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig


def build_framework_domains_figure(
    graph_data: dict[str, Any],
    framework_id: str,
    min_edge_weight: float = 0.15,
    title: str = "Domains",
    height: int = 600,
) -> go.Figure:
    """Build a Plotly figure with domain nodes for one framework (By-framework Layer 2 when >1 domain).

    Visible nodes: level=domain and framework_id=framework_id. Edges between these domains.
    customdata is [domain_id] for selection.
    """
    nodes = graph_data.get("nodes") or []
    edges = graph_data.get("edges") or []
    vis_nodes = [
        n for n in nodes
        if n.get("level") == "domain" and n.get("framework_id") == framework_id
    ]
    visible_ids = {n["id"] for n in vis_nodes}
    node_by_id = {n["id"]: n for n in nodes}

    edge_tuples: list[tuple[float, float, float, float, float, bool]] = []
    for e in edges:
        if e.get("weight", 0) < min_edge_weight:
            continue
        a, b = e.get("source"), e.get("target")
        if a not in visible_ids or b not in visible_ids:
            continue
        na, nb = node_by_id.get(a), node_by_id.get(b)
        if not na or not nb:
            continue
        edge_tuples.append(
            (na["x"], na["y"], nb["x"], nb["y"], e.get("weight", 0.5), e.get("is_meta", False))
        )

    fig = go.Figure()
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for x0, y0, x1, y1, _w, _m in edge_tuples:
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        all_x.extend([x0, x1])
        all_y.extend([y0, y1])
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=1, color="rgba(120,120,120,0.4)"),
                hoverinfo="skip", showlegend=False,
            )
        )

    if not vis_nodes:
        fig.update_layout(
            title=title, height=height,
            xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="rgba(248,248,248,0.5)",
        )
        return fig

    node_x = [n["x"] for n in vis_nodes]
    node_y = [n["y"] for n in vis_nodes]
    all_x.extend(node_x)
    all_y.extend(node_y)
    pad = 0.15
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = max(x_max - x_min, 0.5)
    y_span = max(y_max - y_min, 0.5)
    x_range = [x_min - pad * x_span, x_max + pad * x_span]
    y_range = [y_min - pad * y_span, y_max + pad * y_span]

    node_text = [n.get("label", n["id"]) for n in vis_nodes]
    node_sizes = [12 + min((n.get("n_children") or 0) * 1.2, 18) for n in vis_nodes]
    node_colors = [_framework_color(n.get("framework_id") or "") for n in vis_nodes]
    customdata = [[n["id"]] for n in vis_nodes]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[t[:24] + ("…" if len(t) > 24 else "") for t in node_text],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="white")),
            hovertext=[f"{n.get('label', n['id'])} (domain)" for n in vis_nodes],
            hoverinfo="text",
            customdata=customdata,
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(
            showgrid=True, zeroline=False, showticklabels=False,
            range=x_range,
        ),
        yaxis=dict(
            showgrid=True, zeroline=False, showticklabels=False,
            range=y_range,
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig


def _hull_2d(x: list[float], y: list[float]) -> list[tuple[float, float]]:
    """Convex hull vertices for 2D points. Returns [] if fewer than 3 points or scipy unavailable."""
    if len(x) < 3:
        return []
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
        pts = np.column_stack([x, y])
        hull = ConvexHull(pts)
        return [(float(pts[i, 0]), float(pts[i, 1])) for i in hull.vertices]
    except Exception:
        return []


def _expand_hull(
    hull: list[tuple[float, float]],
    centroid: tuple[float, float],
    padding: float = 0.35,
) -> list[tuple[float, float]]:
    """Expand hull vertices outward from centroid by (1 + padding). Guide: ~35% breathing space."""
    if not hull:
        return []
    cx, cy = centroid
    return [
        (cx + (hx - cx) * (1 + padding), cy + (hy - cy) * (1 + padding))
        for hx, hy in hull
    ]


def build_onestream_hypergraph_figure(
    frameworks: list[dict[str, Any]],
    fw_to_domains: dict[str, list[str]],
    domain_id_to_label: dict[str, str],
    graph_data: dict[str, Any] | None,
    title: str = "Frameworks & Domains",
    height: int = 600,
) -> go.Figure:
    """One-graph hypergraph: all domains as nodes, each framework as a hyperedge (hull).

    Nodes = domains; each framework draws a hull around its domains. Framework centroids
    are drawn as points for click (drill to domains). Domain points are clickable (drill to constructs).
    Layout: from graph_data domain positions if available, else HyperNetX force-directed.
    """
    fw_by_id = {f["framework_id"]: f for f in frameworks}
    all_domain_ids = set()
    for dom_ids in fw_to_domains.values():
        all_domain_ids.update(dom_ids)
    if not all_domain_ids:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig

    nodes_data = (graph_data or {}).get("nodes") or []
    domain_nodes = {n["id"]: n for n in nodes_data if n.get("level") == "domain" and n["id"] in all_domain_ids}
    if len(domain_nodes) >= len(all_domain_ids):
        node_pos = {did: (domain_nodes[did]["x"], domain_nodes[did]["y"]) for did in all_domain_ids if did in domain_nodes}
    else:
        node_pos = {}
    if len(node_pos) < len(all_domain_ids):
        try:
            import hypernetx as hnx
            from hypernetx.drawing import layout as hnx_layout
            H = hnx.Hypergraph(fw_to_domains)
            pos = hnx_layout.layout(H, method="force_directed", seed=42)
            if isinstance(pos, dict):
                for nid in all_domain_ids:
                    p = pos.get(nid, (0.0, 0.0))
                    if hasattr(p, "__getitem__") and len(p) >= 2:
                        node_pos[nid] = (float(p[0]), float(p[1]))
                    else:
                        node_pos[nid] = (0.0, 0.0)
            if len(node_pos) < len(all_domain_ids):
                for nid in all_domain_ids:
                    if nid not in node_pos:
                        node_pos[nid] = (0.0, 0.0)
        except Exception:
            pass
        if len(node_pos) < len(all_domain_ids):
            import math
            n = len(all_domain_ids)
            cols = max(1, int(math.sqrt(n)))
            for i, did in enumerate(sorted(all_domain_ids)):
                node_pos[did] = ((i % cols) * 1.2, -(i // cols) * 1.2)

    hulls: list[tuple[list[float], list[float], str]] = []
    fw_centroids: list[tuple[float, float, str, str]] = []
    for fw in frameworks:
        fid = fw["framework_id"]
        name = fw.get("name", fid)
        member_ids = fw_to_domains.get(fid, [])
        xs = [node_pos[d][0] for d in member_ids if d in node_pos]
        ys = [node_pos[d][1] for d in member_ids if d in node_pos]
        if not xs:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        fw_centroids.append((cx, cy, fid, name))
        hull_pts = _hull_2d(xs, ys)
        if len(hull_pts) >= 3:
            hx = [hull_pts[i][0] for i in range(len(hull_pts))] + [hull_pts[0][0]]
            hy = [hull_pts[i][1] for i in range(len(hull_pts))] + [hull_pts[0][1]]
            hulls.append((hx, hy, fid))

    fig = go.Figure()
    for hx, hy, fid in hulls:
        color = _framework_color(fid)
        fig.add_trace(
            go.Scatter(
                x=hx, y=hy, mode="lines", fill="toself",
                line=dict(color=color, width=1.5, dash="dot"),
                fillcolor=color, opacity=0.12,
                hoverinfo="skip", showlegend=False,
            )
        )
    dom_ids_plot = [did for did in sorted(all_domain_ids) if did in node_pos]
    dom_x = [node_pos[did][0] for did in dom_ids_plot]
    dom_y = [node_pos[did][1] for did in dom_ids_plot]
    dom_text = [domain_id_to_label.get(did, did)[:16] + ("…" if len(domain_id_to_label.get(did, did)) > 16 else "") for did in dom_ids_plot]
    dom_colors = [_framework_color(next((fid for fid, dlist in fw_to_domains.items() if did in dlist), "")) for did in dom_ids_plot]
    dom_customdata = [[did, next((fid for fid, dlist in fw_to_domains.items() if did in dlist), ""), "domain"] for did in dom_ids_plot]
    dom_hover = [domain_id_to_label.get(did, did) + " (domain)" for did in dom_ids_plot]
    if dom_x:
        fig.add_trace(
            go.Scatter(
                x=dom_x, y=dom_y, mode="markers+text",
                text=dom_text, textposition="top center", textfont=dict(size=8),
                marker=dict(size=12, color=dom_colors, line=dict(width=1, color="white")),
                hovertext=dom_hover, hoverinfo="text", customdata=dom_customdata, showlegend=False,
            )
        )
    fw_x = [c[0] for c in fw_centroids]
    fw_y = [c[1] for c in fw_centroids]
    fw_names = [c[3] for c in fw_centroids]
    fw_ids = [c[2] for c in fw_centroids]
    fw_customdata = [[fid, "framework"] for fid in fw_ids]
    fw_colors = [_framework_color(fid) for fid in fw_ids]
    fig.add_trace(
        go.Scatter(
            x=fw_x, y=fw_y, mode="markers+text",
            text=[t[:18] + ("…" if len(t) > 18 else "") for t in fw_names],
            textposition="bottom center", textfont=dict(size=10),
            marker=dict(size=22, color=fw_colors, line=dict(width=2, color="white"), symbol="diamond"),
            hovertext=[n + " (framework)" for n in fw_names],
            hoverinfo="text", customdata=fw_customdata, showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig


def build_expand_in_place_figure(
    frameworks: list[dict[str, Any]],
    fw_to_domains: dict[str, list[str]],
    domain_id_to_label: dict[str, str],
    graph_data: dict[str, Any] | None,
    expanded_fw_ids: set[str],
    cross_links: list[tuple[str, str, float]],
    title: str = "Frameworks (click to expand)",
    height: int = 600,
) -> go.Figure:
    """Expand-in-place: collapsed frameworks as diamonds; expanded show hull + domains.

    cross_links: [(domain_id_a, domain_id_b, similarity), ...]. Only edges where at
    least one endpoint is in an expanded framework's domain set are drawn.
    """
    fw_by_id = {f["framework_id"]: f for f in frameworks}
    all_domain_ids = set()
    for dom_ids in fw_to_domains.values():
        all_domain_ids.update(dom_ids)

    node_pos: dict[str, tuple[float, float]] = {}
    if all_domain_ids:
        nodes_data = (graph_data or {}).get("nodes") or []
        domain_nodes = {
            n["id"]: n for n in nodes_data
            if n.get("level") == "domain" and n["id"] in all_domain_ids
        }
        if len(domain_nodes) >= len(all_domain_ids):
            node_pos = {
                did: (domain_nodes[did]["x"], domain_nodes[did]["y"])
                for did in all_domain_ids if did in domain_nodes
            }
        if len(node_pos) < len(all_domain_ids):
            try:
                import hypernetx as hnx
                from hypernetx.drawing import layout as hnx_layout
                H = hnx.Hypergraph(fw_to_domains)
                pos = hnx_layout.layout(H, method="force_directed", seed=42)
                if isinstance(pos, dict):
                    for nid in all_domain_ids:
                        p = pos.get(nid, (0.0, 0.0))
                        if hasattr(p, "__getitem__") and len(p) >= 2:
                            node_pos[nid] = (float(p[0]), float(p[1]))
                        else:
                            node_pos[nid] = (0.0, 0.0)
                for nid in all_domain_ids:
                    if nid not in node_pos:
                        node_pos[nid] = (0.0, 0.0)
            except Exception:
                pass
            if len(node_pos) < len(all_domain_ids):
                import math
                n = len(all_domain_ids)
                cols = max(1, int(math.sqrt(n)))
                for i, did in enumerate(sorted(all_domain_ids)):
                    if did not in node_pos:
                        node_pos[did] = ((i % cols) * 1.2, -(i // cols) * 1.2)

    # Construct positions by framework (from graph_data) for no-domain frameworks
    construct_pos_by_fw: dict[str, list[tuple[float, float]]] = {}
    if graph_data:
        for n in (graph_data.get("nodes") or []):
            if n.get("level") != "construct":
                continue
            fid = n.get("framework_id")
            if not fid:
                continue
            if "x" in n and "y" in n:
                construct_pos_by_fw.setdefault(fid, []).append((float(n["x"]), float(n["y"])))

    # Compute (cx, cy) for every framework; use domain centroid, else construct centroid, else grid
    fw_pos: dict[str, tuple[float, float]] = {}
    no_domain_fws: list[tuple[str, str, str]] = []
    for fw in frameworks:
        fid = fw["framework_id"]
        name = fw.get("name", fid)
        member_ids = fw_to_domains.get(fid, [])
        xs = [node_pos[d][0] for d in member_ids if d in node_pos]
        ys = [node_pos[d][1] for d in member_ids if d in node_pos]
        if xs:
            fw_pos[fid] = (sum(xs) / len(xs), sum(ys) / len(ys))
        else:
            no_domain_fws.append((fid, name, _framework_color(fid)))

    if no_domain_fws:
        import math
        for fid, _name, _color in no_domain_fws:
            pts = construct_pos_by_fw.get(fid) or []
            if pts:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                fw_pos[fid] = (cx, cy)
            else:
                fw_pos[fid] = None  # mark for grid fallback
        # Grid fallback for those still without position
        need_grid = [(fid, _n, _c) for (fid, _n, _c) in no_domain_fws if fw_pos.get(fid) is None]
        if need_grid:
            valid = [p for p in fw_pos.values() if p is not None]
            if valid:
                all_x = [p[0] for p in valid]
                all_y = [p[1] for p in valid]
                max_x, min_y = max(all_x), min(all_y)
                x_start = max_x + 1.5
            else:
                x_start, min_y = 0.0, 0.0
            ncols = max(1, min(8, int(math.sqrt(len(need_grid))) + 1))
            for i, (fid, _name, _color) in enumerate(need_grid):
                fw_pos[fid] = (x_start + (i % ncols) * 1.5, min_y - (i // ncols) * 1.2)
        # Ensure no framework is left with None (safety: assign (0,0) if any missed)
        for fid in list(fw_pos):
            if fw_pos[fid] is None:
                fw_pos[fid] = (0.0, 0.0)

    expanded_domain_ids = set()
    for fid in expanded_fw_ids:
        expanded_domain_ids.update(fw_to_domains.get(fid, []))

    fig = go.Figure()

    # Cross-bridge lines: only where at least one endpoint is in an expanded framework
    if cross_links and expanded_domain_ids and node_pos:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for did_a, did_b, sim in cross_links:
            if did_a not in expanded_domain_ids and did_b not in expanded_domain_ids:
                continue
            if did_a not in node_pos or did_b not in node_pos:
                continue
            x0, y0 = node_pos[did_a]
            x1, y1 = node_pos[did_b]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        if edge_x:
            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=1.5, color="rgba(100,100,100,0.4)", dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # When any framework is expanded, dim collapsed ones (smaller, lower opacity)
    has_expanded = bool(expanded_fw_ids)

    # Draw every framework: expanded (hull + domains + toggle) or collapsed diamond
    for fw in frameworks:
        fid = fw["framework_id"]
        name = fw.get("name", fid)
        member_ids = fw_to_domains.get(fid, [])
        cx, cy = fw_pos[fid]
        color = _framework_color(fid)

        if fid in expanded_fw_ids and member_ids:
            xs = [node_pos[d][0] for d in member_ids if d in node_pos]
            ys = [node_pos[d][1] for d in member_ids if d in node_pos]
            hull_pts = _hull_2d(xs, ys)
            if len(hull_pts) >= 3:
                hx = [hull_pts[i][0] for i in range(len(hull_pts))] + [hull_pts[0][0]]
                hy = [hull_pts[i][1] for i in range(len(hull_pts))] + [hull_pts[0][1]]
                fig.add_trace(
                    go.Scatter(
                        x=hx, y=hy, mode="lines", fill="toself",
                        line=dict(color=color, width=1.5, dash="dot"),
                        fillcolor=color, opacity=0.12,
                        hoverinfo="skip", showlegend=False,
                    )
                )
            dom_x = [node_pos[d][0] for d in member_ids if d in node_pos]
            dom_y = [node_pos[d][1] for d in member_ids if d in node_pos]
            dom_text = [
                domain_id_to_label.get(d, d)[:16] + ("…" if len(domain_id_to_label.get(d, d)) > 16 else "")
                for d in member_ids if d in node_pos
            ]
            dom_customdata = [[d, fid, "domain"] for d in member_ids if d in node_pos]
            dom_hover = [domain_id_to_label.get(d, d) + " (domain)" for d in member_ids if d in node_pos]
            if dom_x:
                fig.add_trace(
                    go.Scatter(
                        x=dom_x, y=dom_y, mode="markers+text",
                        text=dom_text, textposition="top center", textfont=dict(size=8),
                        marker=dict(size=12, color=color, line=dict(width=1, color="white")),
                        hovertext=dom_hover, hoverinfo="text", customdata=dom_customdata,
                        showlegend=False,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[cx], y=[cy], mode="markers+text",
                    text=[name[:18] + ("…" if len(name) > 18 else "")],
                    textposition="bottom center", textfont=dict(size=9),
                    marker=dict(size=16, color=color, line=dict(width=1, color="white"), symbol="diamond"),
                    hovertext=[name + " (click to collapse)"],
                    hoverinfo="text",
                    customdata=[[fid, "framework_toggle"]],
                    showlegend=False,
                )
            )
        elif fid in expanded_fw_ids and not member_ids and graph_data:
            # No domains: expand to show constructs
            nodes_data = graph_data.get("nodes") or []
            construct_nodes = [
                n for n in nodes_data
                if n.get("level") == "construct" and n.get("framework_id") == fid
                and "x" in n and "y" in n
            ]
            if construct_nodes:
                c_xs = [float(n["x"]) for n in construct_nodes]
                c_ys = [float(n["y"]) for n in construct_nodes]
                hull_pts = _hull_2d(c_xs, c_ys)
                if len(hull_pts) >= 3:
                    hx = [hull_pts[i][0] for i in range(len(hull_pts))] + [hull_pts[0][0]]
                    hy = [hull_pts[i][1] for i in range(len(hull_pts))] + [hull_pts[0][1]]
                    fig.add_trace(
                        go.Scatter(
                            x=hx, y=hy, mode="lines", fill="toself",
                            line=dict(color=color, width=1.5, dash="dot"),
                            fillcolor=color, opacity=0.12,
                            hoverinfo="skip", showlegend=False,
                        )
                    )
                con_x = [float(n["x"]) for n in construct_nodes]
                con_y = [float(n["y"]) for n in construct_nodes]
                con_text = [n.get("label", n["id"])[:16] + ("…" if len(n.get("label", n["id"])) > 16 else "") for n in construct_nodes]
                con_hover = [n.get("label", n["id"]) + " (construct)" for n in construct_nodes]
                fig.add_trace(
                    go.Scatter(
                        x=con_x, y=con_y, mode="markers+text",
                        text=con_text, textposition="top center", textfont=dict(size=8),
                        marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                        hovertext=con_hover, hoverinfo="text",
                        showlegend=False,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[cx], y=[cy], mode="markers+text",
                    text=[name[:18] + ("…" if len(name) > 18 else "")],
                    textposition="bottom center", textfont=dict(size=9),
                    marker=dict(size=16, color=color, line=dict(width=1, color="white"), symbol="diamond"),
                    hovertext=[name + " (click to collapse)"],
                    hoverinfo="text",
                    customdata=[[fid, "framework_toggle"]],
                    showlegend=False,
                )
            )
        else:
            # Dim collapsed frameworks when any framework is expanded
            if has_expanded:
                size = 14
                text_size = 8
                marker_opacity = 0.28
            else:
                size = 22
                text_size = 10
                marker_opacity = 1.0
            fig.add_trace(
                go.Scatter(
                    x=[cx], y=[cy], mode="markers+text",
                    text=[name[:18] + ("…" if len(name) > 18 else "")],
                    textposition="bottom center", textfont=dict(size=text_size),
                    marker=dict(
                        size=size,
                        color=color,
                        line=dict(width=2, color="white"),
                        symbol="diamond",
                        opacity=marker_opacity,
                    ),
                    hovertext=[name + " (framework, click to expand)"],
                    hoverinfo="text",
                    customdata=[[fid, "framework"]],
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig


def get_search_options(graph_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return search_index filtered to domain and construct only (for selectbox)."""
    idx = graph_data.get("search_index") or []
    return [e for e in idx if e.get("level") in ("domain", "construct")]


def build_all_frameworks_umap_figure(
    graph_data: dict[str, Any],
    frameworks: list[dict[str, Any]],
    active_framework_id: str | None = None,
    show_labels: bool = False,
    title: str = "Frameworks in UMAP semantic space",
    height: int = 600,
    show_legend: bool = True,
) -> go.Figure:
    """Two-layer UMAP view: default = framework-level only (diamond + hull); expand = construct-level (points + edges).

    Default view: only framework centroid (diamond marker), framework name, and convex hull (fill + outline).
    No construct-level points or labels.

    When active_framework_id is set (click framework in sidebar): that framework keeps diamond and shows
    all constructs as circles with names; construct-construct edges drawn with line width/opacity by weight.
    Other frameworks stay as diamond + hull with reduced opacity.
    """
    nodes = graph_data.get("nodes") or []
    edges = graph_data.get("edges") or []
    fw_by_id = {f["framework_id"]: f for f in frameworks}

    # Group construct nodes by framework_id; node_by_id for edge endpoints
    constructs_by_fw: dict[str, list[dict[str, Any]]] = {}
    node_by_id: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if n.get("level") != "construct" or "x" not in n or "y" not in n:
            continue
        fid = n.get("framework_id")
        if not fid or fid not in fw_by_id:
            continue
        constructs_by_fw.setdefault(fid, []).append(n)
        node_by_id[n["id"]] = n

    # Construct-construct edges (is_meta False) for drawing in expanded view
    construct_edges = [
        (e["source"], e["target"], float(e.get("weight", 0.5)))
        for e in edges
        if e.get("is_meta") is False and e.get("source") and e.get("target")
    ]

    fig = go.Figure()
    has_active = active_framework_id is not None

    for fw in frameworks:
        fid = fw["framework_id"]
        name = fw.get("name", fid)
        construct_nodes = constructs_by_fw.get(fid, [])
        if not construct_nodes:
            continue

        color = _framework_color(fid)
        is_active = fid == active_framework_id
        xs = [float(n["x"]) for n in construct_nodes]
        ys = [float(n["y"]) for n in construct_nodes]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        hull_pts = _hull_2d(xs, ys)
        if len(hull_pts) >= 3:
            hull_pts = _expand_hull(hull_pts, (cx, cy), 0.35)

        # Opacity: unselected frameworks dimmed when one is selected
        if has_active and not is_active:
            fill_opacity = 0.02
            outline_opacity = 0.12
            hull_dash = "dot"
            hull_width = 1.5
            label_opacity = 0.2
        else:
            fill_opacity = 0.07
            outline_opacity = 0.7
            hull_dash = "solid" if is_active else "dot"
            hull_width = 2.5 if is_active else 1.5
            label_opacity = 1.0

        # Hull fill + outline (all frameworks)
        if len(hull_pts) >= 3:
            hx = [hull_pts[i][0] for i in range(len(hull_pts))] + [hull_pts[0][0]]
            hy = [hull_pts[i][1] for i in range(len(hull_pts))] + [hull_pts[0][1]]
            fig.add_trace(
                go.Scatter(
                    x=hx,
                    y=hy,
                    mode="lines",
                    fill="toself",
                    line=dict(width=0),
                    fillcolor=color,
                    opacity=fill_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=hx,
                    y=hy,
                    mode="lines",
                    line=dict(color=color, width=hull_width, dash=hull_dash),
                    fill="none",
                    opacity=outline_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Framework-level: diamond marker at centroid (always)
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=14,
                    color=color,
                    line=dict(width=0.8, color="white"),
                    opacity=1.0 if (is_active or not has_active) else label_opacity,
                ),
                hovertext=name,
                hoverinfo="text",
                showlegend=show_legend,
                legendgroup=fid,
                name=name,
            )
        )

        # Framework name annotation beside centroid
        fig.add_annotation(
            x=cx,
            y=cy,
            text=name,
            showarrow=False,
            xref="x",
            yref="y",
            xshift=18,
            font=dict(size=13, color=color, family="sans-serif"),
            opacity=label_opacity,
        )

        # Default view: no construct points. Expanded view: only active framework shows constructs + edges.
        if is_active:
            # Edges within this framework (weight -> line width and opacity)
            cids = {n["id"] for n in construct_nodes}
            for a, b, w in construct_edges:
                if a not in cids or b not in cids:
                    continue
                na, nb = node_by_id.get(a), node_by_id.get(b)
                if not na or not nb:
                    continue
                # Map weight to width ~0.5–2.5 and opacity ~0.2–0.7
                width = 0.5 + 2.0 * max(0.0, min(1.0, w))
                opacity = 0.25 + 0.45 * max(0.0, min(1.0, w))
                fig.add_trace(
                    go.Scatter(
                        x=[na["x"], nb["x"], None],
                        y=[na["y"], nb["y"], None],
                        mode="lines",
                        line=dict(color=color, width=width),
                        opacity=opacity,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            # Construct points (circles) + labels for active framework
            c_x = [float(n["x"]) for n in construct_nodes]
            c_y = [float(n["y"]) for n in construct_nodes]
            labels = [n.get("label", n["id"]) for n in construct_nodes]
            hover_text = [f"{l} — {name}" for l in labels]
            show_text = True  # 展开层始终显示 construct 名称；侧边栏「Show construct labels」可关闭
            text_vals = [t[:18] + ("…" if len(t) > 18 else "") for t in labels]
            fig.add_trace(
                go.Scatter(
                    x=c_x,
                    y=c_y,
                    mode="markers+text" if show_text else "markers",
                    text=text_vals,
                    textposition="top center",
                    textfont=dict(size=10, color="#444"),
                    marker=dict(
                        size=7,
                        color=color,
                        line=dict(width=0.8, color="white"),
                        opacity=0.9,
                    ),
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title=dict(text="Framework"),
            font=dict(size=11),
        ) if show_legend else {},
        xaxis=dict(
            showgrid=True,
            gridcolor="#F0F0F0",
            zeroline=False,
            showticklabels=False,
            title="UMAP Dimension 1",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F0F0F0",
            zeroline=False,
            showticklabels=False,
            title="UMAP Dimension 2",
        ),
        margin=dict(l=20, r=20 if not show_legend else 140, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig


def build_all_frameworks_umap_figure_for_client(
    graph_data: dict[str, Any],
    frameworks: list[dict[str, Any]],
    title: str = "Frameworks in UMAP semantic space",
    height: int = 600,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build full figure for client-side restyle (no rerun). First level shows only hulls and
    diamonds; no edge traces to avoid black-line clutter. Construct trace is present but
    visible=False. Returns (figure_dict, trace_map) for use in HTML/JS component.
    """
    nodes = graph_data.get("nodes") or []
    fw_by_id = {f["framework_id"]: f for f in frameworks}

    constructs_by_fw: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        if n.get("level") != "construct" or "x" not in n or "y" not in n:
            continue
        fid = n.get("framework_id")
        if not fid or fid not in fw_by_id:
            continue
        constructs_by_fw.setdefault(fid, []).append(n)

    fig = go.Figure()
    trace_map: list[dict[str, Any]] = []
    ann_idx = 0

    for fw in frameworks:
        fid = fw["framework_id"]
        name = fw.get("name", fid)
        construct_nodes = constructs_by_fw.get(fid, [])
        if not construct_nodes:
            continue

        color = _framework_color(fid)
        xs = [float(n["x"]) for n in construct_nodes]
        ys = [float(n["y"]) for n in construct_nodes]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        hull_pts = _hull_2d(xs, ys)
        if len(hull_pts) >= 3:
            hull_pts = _expand_hull(hull_pts, (cx, cy), 0.35)

        indices: dict[str, Any] = {
            "hullFill": None,
            "hullOutline": None,
            "hullClick": None,  # 透明 overlay，用于接收凸包区域内点击
            "diamond": None,
            "construct": None,
            "edges": [],
        }

        if len(hull_pts) >= 3:
            hx = [hull_pts[i][0] for i in range(len(hull_pts))] + [hull_pts[0][0]]
            hy = [hull_pts[i][1] for i in range(len(hull_pts))] + [hull_pts[0][1]]
            fig.add_trace(
                go.Scatter(
                    x=hx, y=hy, mode="lines", fill="toself",
                    line=dict(width=0), fillcolor=color, opacity=0.07,
                    hoverinfo="skip", showlegend=False,
                )
            )
            indices["hullFill"] = len(fig.data) - 1
            fig.add_trace(
                go.Scatter(
                    x=hx, y=hy, mode="lines", fill="none",
                    line=dict(color=color, width=1.5, dash="dot"), opacity=0.7,
                    hoverinfo="skip", showlegend=False,
                )
            )
            indices["hullOutline"] = len(fig.data) - 1
            # 透明填充 overlay：与凸包同形，仅用于扩大可点击区域（点击凸包内部即可触发）
            fig.add_trace(
                go.Scatter(
                    x=hx, y=hy, mode="lines", fill="toself",
                    line=dict(width=0), fillcolor="rgba(0,0,0,0)",
                    hovertext=name, hoverinfo="text", showlegend=False,
                )
            )
            indices["hullClick"] = len(fig.data) - 1

        fig.add_trace(
            go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(symbol="diamond", size=14, color=color, line=dict(width=0.8, color="white"), opacity=1.0),
                hovertext=name, hoverinfo="text", showlegend=False,
            )
        )
        indices["diamond"] = len(fig.data) - 1

        c_x = [float(n["x"]) for n in construct_nodes]
        c_y = [float(n["y"]) for n in construct_nodes]
        labels = [n.get("label", n["id"]) for n in construct_nodes]
        hover_text = [f"{l} — {name}" for l in labels]
        text_vals = [t[:18] + ("…" if len(t) > 18 else "") for t in labels]
        fig.add_trace(
            go.Scatter(
                x=c_x, y=c_y, mode="markers+text",
                text=text_vals, textposition="top center", textfont=dict(size=10, color="#444"),
                marker=dict(size=7, color=color, line=dict(width=0.8, color="white"), opacity=0.9),
                hovertext=hover_text, hoverinfo="text", showlegend=False,
                visible=False,
            )
        )
        indices["construct"] = len(fig.data) - 1

        fig.add_annotation(
            x=cx, y=cy, text=name, showarrow=False, xref="x", yref="y", xshift=18,
            font=dict(size=13, color=color, family="sans-serif"), opacity=1.0,
        )
        indices["annotationIndex"] = ann_idx
        ann_idx += 1

        trace_map.append({
            "id": fid,
            "name": name,
            "color": color,
            "count": len(construct_nodes),
            "traceIndices": indices,
        })

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False, showticklabels=False, title="UMAP Dimension 1"),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False, showticklabels=False, title="UMAP Dimension 2"),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig.to_dict(), trace_map
