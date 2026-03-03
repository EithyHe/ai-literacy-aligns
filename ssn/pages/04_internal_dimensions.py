"""Internal Dimensions (PRD Stage 4, FR-4.3.1–4.3.7).

Explore internal dimension structure of user's scale items.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np

from ssn.services.embedding_service import encode_texts
from ssn.services.decomposition_service import run_decomposition
from ssn.components.scatter_viz import plot_item_clusters, plot_scree


def _parse_items(text: str) -> list[str]:
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


def _cluster_items(
    coords: np.ndarray,
    method: str,
    n_clusters: int | None = None,
) -> np.ndarray:
    """Apply clustering to 2D coords. Returns cluster labels (0-based)."""
    if method == "kmeans":
        from sklearn.cluster import KMeans
        n = coords.shape[0]
        k = n_clusters or max(2, min(5, n // 2))
        k = min(k, n)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return km.fit_predict(coords)
    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            st.warning("hdbscan not installed. Using K-Means. pip install hdbscan")
            return _cluster_items(coords, "kmeans", n_clusters)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        labels = clusterer.fit_predict(coords)
        return labels
    return np.zeros(coords.shape[0], dtype=int)


def render() -> None:
    st.title("Internal Dimensions")
    st.markdown(
        "Explore the internal structure of your scale items via dimensionality "
        "reduction and clustering."
    )

    # Reuse from session state if available
    default_items = "\n".join(st.session_state.get("user_item_texts", []))
    items_text = st.text_area(
        "Scale items",
        value=default_items,
        height=200,
        placeholder="Enter one item per line, or use items from Similarity Detection.",
    )
    items = _parse_items(items_text)

    with st.sidebar:
        st.subheader("Dimensionality reduction")
        method = st.selectbox(
            "Method",
            ["PCA", "t-SNE", "UMAP"],
            index=0,
        )
        st.subheader("Clustering")
        cluster_method = st.selectbox(
            "Clustering method",
            ["K-Means", "HDBSCAN"],
            index=0,
        )
        n_clusters_auto = st.checkbox("Auto number of clusters", value=True)
        n_clusters = None
        if not n_clusters_auto:
            n_clusters = st.number_input(
                "Number of clusters",
                min_value=2,
                max_value=20,
                value=3,
            )

    analyze_clicked = st.button("Analyze", type="primary")

    if not items:
        st.warning("Please enter at least one item.")
        return

    if len(items) < 3:
        st.warning("Enter at least 3 items for dimensionality reduction.")

    if not analyze_clicked:
        return

    with st.spinner("Encoding and reducing dimensions…"):
        try:
            embeddings = encode_texts(items, use_openai=False)
        except Exception as e:
            st.error(f"Encoding failed: {e}")
            return

        method_lower = method.lower().replace("-", "_")
        try:
            result = run_decomposition(
                embeddings,
                method=method_lower,
                n_components=2,
            )
        except ImportError as e:
            st.error(f"Method requires extra package: {e}")
            return
        except Exception as e:
            st.error(f"Decomposition failed: {e}")
            return

        coords = result["coords"]
        explained_var = result.get("explained_variance")

    if explained_var is not None and len(explained_var) > 0:
        st.subheader("Scree plot (PCA variance)")
        fig_scree = plot_scree(
            np.array(explained_var),
            title="Variance explained by components",
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    st.subheader("2D item scatter (colored by cluster)")
    cluster_method_lower = cluster_method.lower().replace("-", "")
    k = int(n_clusters) if n_clusters is not None else None
    if not n_clusters_auto and n_clusters is not None:
        k = n_clusters
    labels = _cluster_items(coords, cluster_method_lower, k)
    fig = plot_item_clusters(
        coords,
        items,
        labels.tolist(),
        title=f"{method} + {cluster_method}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster summary")
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)
        unique_labels.append(-1)
    rows = []
    for lab in unique_labels:
        count = int(np.sum(labels == lab))
        cluster_items = [items[i] for i in range(len(items)) if labels[i] == lab]
        core = cluster_items[0] if cluster_items else ""
        rows.append({
            "Cluster": lab if lab >= 0 else "Noise",
            "Count": count,
            "Core/Edge": core[:50] + ("…" if len(core) > 50 else ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Cluster semantic summary")
    for lab in unique_labels:
        cluster_items = [items[i] for i in range(len(items)) if labels[i] == lab]
        lab_name = f"Cluster {lab}" if lab >= 0 else "Noise"
        with st.expander(f"{lab_name} ({len(cluster_items)} items)"):
            for it in cluster_items:
                st.write(f"- {it}")


render()
