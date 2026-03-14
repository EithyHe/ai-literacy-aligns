"""Sigma.js v3 + graphology network visualization component for Streamlit.

Phase 1: Domain/Construct semantic zoom, framework hulls, search.
Requires frontend build: cd frontend && npm install && npm run build.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_RELEASE = (Path(__file__).resolve().parent / "frontend" / "build").exists()
if _RELEASE:
    _path = Path(__file__).resolve().parent / "frontend" / "build"
else:
    _path = Path(__file__).resolve().parent / "frontend"

_component = components.declare_component(
    "sigma_network",
    path=str(_path),
)


def sigma_network(
    graph_data: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Render the SSN multi-level graph with Sigma.js.

    Args:
        graph_data: Full graph JSON (nodes, edges, frameworks, search_index, etc.).
        config: Optional config (minEdgeWeight, etc.).
        key: Streamlit key for the component instance.

    Returns:
        Event payload from the frontend (e.g. {"type": "click", "nodeId": "..."})
        or None if no event or graph_data is None.
    """
    if graph_data is None:
        return None
    payload = _component(
        graph_data=graph_data,
        config=config or {},
        key=key,
        default=None,
    )
    return payload
