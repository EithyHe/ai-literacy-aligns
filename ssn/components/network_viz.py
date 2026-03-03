"""Interactive network graph visualization components for the Semantic Scale Network."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

# Qualitative color palette for community coloring (Plotly Set2-like)
_QUALITATIVE_PALETTE = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]


def _get_group_colors(groups: list[Any]) -> list[str]:
    """Map group values to qualitative colors."""
    unique = list(dict.fromkeys(groups))
    color_map = {g: _QUALITATIVE_PALETTE[i % len(_QUALITATIVE_PALETTE)] for i, g in enumerate(unique)}
    return [color_map[g] for g in groups]


def plot_network(
    nodes: list[dict],
    edges: list[dict],
    title: str = "",
    height: int = 600,
    color_by: str | None = None,
) -> go.Figure:
    """Plot interactive network graph.

    Args:
        nodes: List of dicts with keys: id, x, y, label; optional: group, size.
        edges: List of dicts with keys: source_x, source_y, target_x, target_y; optional: weight.
        title: Plot title.
        height: Figure height in pixels.
        color_by: If set, color nodes by their 'group' field; otherwise single color.

    Returns:
        Plotly Figure ready for display.
    """
    if not nodes:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig

    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_text = [n.get("label", str(n.get("id", ""))) for n in nodes]
    node_sizes = [n.get("size", 12) for n in nodes]

    # Edges as line segments (source -> target -> None for disjoint segments)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for e in edges:
        edge_x.extend([e["source_x"], e["target_x"], None])
        edge_y.extend([e["source_y"], e["target_y"], None])

    fig = go.Figure()

    # Edge trace (lines)
    if edge_x and edge_y:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="rgba(150,150,150,0.5)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Node trace
    if color_by and all("group" in n for n in nodes):
        groups = [n["group"] for n in nodes]
        colors = _get_group_colors(groups)
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[t[:20] + ("…" if len(t) > 20 else "") for t in node_text],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=node_sizes,
                    color=colors,
                    line=dict(width=1, color="white"),
                ),
                hovertext=node_text,
                hoverinfo="text",
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[t[:20] + ("…" if len(t) > 20 else "") for t in node_text],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=node_sizes,
                    color="#4a90d9",
                    line=dict(width=1, color="white"),
                ),
                hovertext=node_text,
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
    )
    return fig


def plot_network_from_plotly_data(
    plotly_data: dict,
    title: str = "",
    height: int = 600,
    highlight_node_id: str | None = None,
) -> go.Figure:
    """Plot network from the dict returned by network_service.network_to_plotly_data().

    plotly_data has keys: node_x, node_y, node_text, node_color, node_ids,
    edge_x, edge_y, edge_weights.

    Args:
        plotly_data: Dict from network_to_plotly_data().
        title: Plot title.
        height: Figure height in pixels.
        highlight_node_id: If set, this node is drawn larger with a red star marker.

    Returns:
        Plotly Figure ready for display.
    """
    node_x = plotly_data.get("node_x", [])
    node_y = plotly_data.get("node_y", [])
    node_text = plotly_data.get("node_text", [])
    node_color = plotly_data.get("node_color", [])
    node_ids = plotly_data.get("node_ids", [])
    edge_x = plotly_data.get("edge_x", [])
    edge_y = plotly_data.get("edge_y", [])
    edge_weights = plotly_data.get("edge_weights", [])

    if not node_x and not node_y:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig

    fig = go.Figure()

    # Edges — width and opacity proportional to similarity weight
    if edge_x and edge_y and edge_weights:
        for idx, w in enumerate(edge_weights):
            i0 = idx * 3
            lw = 0.5 + 2.5 * w
            alpha = 0.15 + 0.65 * w
            fig.add_trace(
                go.Scatter(
                    x=[edge_x[i0], edge_x[i0 + 1], None],
                    y=[edge_y[i0], edge_y[i0 + 1], None],
                    mode="lines",
                    line=dict(width=lw, color=f"rgba(100,100,100,{alpha:.2f})"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    elif edge_x and edge_y:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="rgba(150,150,150,0.5)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Separate highlighted node from regular nodes
    highlight_idx = None
    if highlight_node_id and highlight_node_id in node_ids:
        highlight_idx = node_ids.index(highlight_node_id)

    # Map community ids to colors
    unique_ids = list(dict.fromkeys(node_color))
    color_map = {
        cid: _QUALITATIVE_PALETTE[i % len(_QUALITATIVE_PALETTE)]
        for i, cid in enumerate(unique_ids)
    }
    colors = [color_map[cid] for cid in node_color]

    # Regular nodes (exclude highlighted)
    reg_x, reg_y, reg_text, reg_colors = [], [], [], []
    for i in range(len(node_x)):
        if i == highlight_idx:
            continue
        reg_x.append(node_x[i])
        reg_y.append(node_y[i])
        reg_text.append(node_text[i])
        reg_colors.append(colors[i])

    if reg_x:
        fig.add_trace(
            go.Scatter(
                x=reg_x,
                y=reg_y,
                mode="markers+text",
                text=[(t[:20] + "…") if len(t) > 20 else t for t in reg_text],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=12,
                    color=reg_colors,
                    line=dict(width=1, color="white"),
                ),
                hovertext=reg_text,
                hoverinfo="text",
                showlegend=False,
            )
        )

    # Highlighted node (user construct)
    if highlight_idx is not None:
        fig.add_trace(
            go.Scatter(
                x=[node_x[highlight_idx]],
                y=[node_y[highlight_idx]],
                mode="markers+text",
                text=[node_text[highlight_idx]],
                textposition="top center",
                textfont=dict(size=11, color="#d62728"),
                marker=dict(
                    size=20,
                    color="#d62728",
                    symbol="star",
                    line=dict(width=2, color="white"),
                ),
                hovertext=[node_text[highlight_idx]],
                hoverinfo="text",
                name="Your construct",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=highlight_idx is not None,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig
