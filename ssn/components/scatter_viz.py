"""Scatter plot visualization components for the Semantic Scale Network."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Qualitative palette for group coloring (Plotly Set2)
_SET2 = px.colors.qualitative.Set2


def _get_set2_colors(groups: list[Any]) -> list[str]:
    """Map group values to Set2 qualitative colors."""
    unique = list(dict.fromkeys(groups))
    color_map = {g: _SET2[i % len(_SET2)] for i, g in enumerate(unique)}
    return [color_map[g] for g in groups]


def plot_embedding_scatter(
    coords: np.ndarray,
    labels: list[str],
    groups: list[str] | None = None,
    highlight_indices: list[int] | None = None,
    highlight_label: str = "Query",
    title: str = "",
    height: int = 600,
) -> go.Figure:
    """Plot 2D scatter of embeddings with optional group coloring and highlighted points.

    Args:
        coords: (n, 2) array of 2D coordinates.
        labels: Hover labels for each point.
        groups: Categorical group for coloring (e.g. domain names).
        highlight_indices: Indices of special points to highlight (e.g. user's query).
        highlight_label: Legend label for highlighted points.
        title: Plot title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure ready for display.
    """
    n = len(coords)
    labels = labels[:n] if len(labels) >= n else labels + [""] * (n - len(labels))
    highlight_indices = set(highlight_indices or [])

    # Split into regular and highlighted
    reg_mask = [i not in highlight_indices for i in range(n)]
    hi_mask = [i in highlight_indices for i in range(n)]

    fig = go.Figure()

    if groups and len(groups) >= n:
        # Color by group for regular points
        reg_groups = [groups[i] for i in range(n) if reg_mask[i]]
        reg_colors = _get_set2_colors(reg_groups)
        reg_x = coords[reg_mask, 0].tolist()
        reg_y = coords[reg_mask, 1].tolist()
        reg_labels = [labels[i] for i in range(n) if reg_mask[i]]
        fig.add_trace(
            go.Scatter(
                x=reg_x,
                y=reg_y,
                mode="markers",
                marker=dict(size=8, color=reg_colors, opacity=0.8, line=dict(width=0.5, color="white")),
                text=reg_labels,
                hoverinfo="text",
                name="Corpus",
            )
        )
    else:
        reg_x = coords[reg_mask, 0].tolist()
        reg_y = coords[reg_mask, 1].tolist()
        reg_labels = [labels[i] for i in range(n) if reg_mask[i]]
        fig.add_trace(
            go.Scatter(
                x=reg_x,
                y=reg_y,
                mode="markers",
                marker=dict(size=8, color="#4a90d9", opacity=0.8, line=dict(width=0.5, color="white")),
                text=reg_labels,
                hoverinfo="text",
                name="Corpus",
            )
        )

    if any(hi_mask):
        hi_x = coords[hi_mask, 0].tolist()
        hi_y = coords[hi_mask, 1].tolist()
        hi_labels = [labels[i] for i in range(n) if hi_mask[i]]
        fig.add_trace(
            go.Scatter(
                x=hi_x,
                y=hi_y,
                mode="markers",
                marker=dict(size=14, color="#e74c3c", symbol="star", line=dict(width=2, color="white")),
                text=hi_labels,
                hoverinfo="text",
                name=highlight_label,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(title="Dimension 1", showgrid=True),
        yaxis=dict(title="Dimension 2", showgrid=True),
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_density_scatter(
    coords: np.ndarray,
    density: np.ndarray,
    labels: list[str],
    title: str = "",
    height: int = 600,
) -> go.Figure:
    """Plot scatter with density as color intensity. Uses continuous colorscale.

    Args:
        coords: (n, 2) array of 2D coordinates.
        density: (n,) array of density values per point.
        labels: Hover labels for each point.
        title: Plot title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure ready for display.
    """
    n = len(coords)
    labels = labels[:n] if len(labels) >= n else labels + [""] * (n - len(labels))
    density = np.asarray(density).ravel()[:n]
    if len(density) < n:
        density = np.pad(density, (0, n - len(density)), constant_values=0)

    fig = go.Figure(
        data=go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=10,
                color=density,
                colorscale="Viridis",
                colorbar=dict(title="Density"),
                line=dict(width=0.5, color="white"),
            ),
            text=labels,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(title="Dimension 1", showgrid=True),
        yaxis=dict(title="Dimension 2", showgrid=True),
        margin=dict(l=50, r=80, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig


def plot_scree(
    explained_variance: np.ndarray,
    title: str = "Scree Plot",
    height: int = 400,
) -> go.Figure:
    """Plot scree plot (bar + cumulative line) for PCA variance explained.

    Uses dual y-axis: left for individual variance, right for cumulative.

    Args:
        explained_variance: 1D array of variance explained per component.
        title: Plot title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure ready for display.
    """
    n = len(explained_variance)
    cumsum = np.cumsum(explained_variance)
    x_labels = [f"PC{i+1}" for i in range(n)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=explained_variance,
            name="Individual",
            marker_color="#4a90d9",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=cumsum,
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=8),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(title="Component"),
        yaxis=dict(
            title="Variance Explained",
            side="left",
            range=[0, max(explained_variance) * 1.1] if n else [0, 1],
            tickformat=".2%",
        ),
        yaxis2=dict(
            title="Cumulative",
            side="right",
            overlaying="y",
            range=[0, min(1.05, max(cumsum) * 1.1)] if n else [0, 1],
            tickformat=".0%",
        ),
        margin=dict(l=60, r=60, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_item_clusters(
    coords: np.ndarray,
    item_texts: list[str],
    cluster_labels: list[int],
    title: str = "",
    height: int = 600,
) -> go.Figure:
    """Plot item clustering results: colored by cluster, text on hover, boundary indicators.

    Args:
        coords: (n, 2) array of 2D coordinates.
        item_texts: Hover text for each item.
        cluster_labels: Cluster assignment per item (0-based integers).
        title: Plot title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure ready for display.
    """
    n = len(coords)
    item_texts = item_texts[:n] if len(item_texts) >= n else item_texts + [""] * (n - len(item_texts))
    cluster_labels = list(cluster_labels)[:n]
    if len(cluster_labels) < n:
        cluster_labels = cluster_labels + [0] * (n - len(cluster_labels))

    # Map cluster ids to Set2 colors
    unique_clusters = sorted(set(cluster_labels))
    color_map = {c: _SET2[i % len(_SET2)] for i, c in enumerate(unique_clusters)}
    colors = [color_map[c] for c in cluster_labels]

    fig = go.Figure(
        data=go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=1, color="white"),
            ),
            text=[f"Cluster {c}: {t}" for c, t in zip(cluster_labels, item_texts)],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(title="Dimension 1", showgrid=True),
        yaxis=dict(title="Dimension 2", showgrid=True),
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,248,248,0.5)",
    )
    return fig
