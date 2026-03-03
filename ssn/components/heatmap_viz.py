"""Heatmap visualization components for the Semantic Scale Network."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_similarity_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    height: int = 600,
    colorscale: str = "RdBu_r",
) -> go.Figure:
    """Plot a similarity heatmap with hover showing labels and values.

    Args:
        matrix: 2D array of similarity values.
        row_labels: Labels for rows.
        col_labels: Labels for columns.
        title: Plot title.
        height: Figure height in pixels.
        colorscale: Plotly colorscale name (default: RdBu_r for red-blue reversed).

    Returns:
        Plotly Figure ready for display.
    """
    n_rows, n_cols = matrix.shape
    row_labels = row_labels[:n_rows] if len(row_labels) >= n_rows else row_labels + [""] * (n_rows - len(row_labels))
    col_labels = col_labels[:n_cols] if len(col_labels) >= n_cols else col_labels + [""] * (n_cols - len(col_labels))

    hover_text = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            val = float(matrix[i, j])
            row.append(f"Row: {row_labels[i]}<br>Col: {col_labels[j]}<br>Similarity: {val:.3f}")
        hover_text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=col_labels,
            y=row_labels,
            colorscale=colorscale,
            text=hover_text,
            hoverinfo="text",
            texttemplate="",
            colorbar=dict(title="Similarity"),
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=120, r=40, t=50, b=120),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def plot_cross_similarity_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    height: int = 500,
) -> go.Figure:
    """Plot item-level cross-similarity heatmap between two scales.

    Args:
        matrix: 2D array of cross-similarity values (scale A items x scale B items).
        row_labels: Item labels for rows (scale A).
        col_labels: Item labels for columns (scale B).
        title: Plot title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure ready for display.
    """
    return plot_similarity_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title=title,
        height=height,
        colorscale="RdBu_r",
    )
