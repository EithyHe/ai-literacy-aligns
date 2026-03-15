"""Internal Dimensions - unified visualization workspace."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ssn.components.visualization_workspace import render_visualization_workspace


def render() -> None:
    render_visualization_workspace(
        page_key="internal_dimensions",
        title="Internal Dimensions",
        description=(
            "Unified visualization workspace for **Construct Network**, **UMAP + HDBSCAN**, "
            "and **cluster diagnostics**."
        ),
    )


render()
