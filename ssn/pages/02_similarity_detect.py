"""Similarity Detection (PRD Stage 2, FR-1.1.1–1.2.5).

User inputs scale items (text area or CSV). System computes scale embedding,
Top-N similar constructs, and item-level cross-similarity heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from ssn.services.embedding_service import (
    encode_texts,
    encode_scale,
    get_construct_embeddings,
    load_item_embeddings,
)
from ssn.services.similarity_service import (
    find_scale_neighbors,
    compute_cross_similarity_matrix,
    find_top_n_neighbors,
)
from ssn.db.schema import get_all_constructs, get_items_by_construct, get_all_items
from ssn.components.heatmap_viz import plot_similarity_heatmap, plot_cross_similarity_heatmap


def _parse_items_from_text(text: str) -> list[str]:
    """Parse items from text (one per line)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines


def _parse_items_from_csv(uploaded_file) -> list[str]:
    """Parse items from CSV. Tries 'text', 'item', 'item_text' columns."""
    df = pd.read_csv(uploaded_file)
    for col in ("text", "item", "item_text", "Text", "Item"):
        if col in df.columns:
            return [str(x).strip() for x in df[col].dropna() if str(x).strip()]
    if len(df.columns) >= 1:
        return [str(x).strip() for x in df.iloc[:, 0].dropna() if str(x).strip()]
    return []


@st.cache_data(ttl=300)
def _get_construct_metadata() -> dict[str, dict]:
    """Build construct_id -> {name, framework} from constructs.framework_id."""
    constructs = {c["construct_id"]: dict(c) for c in get_all_constructs()}
    from ssn.db.schema import get_all_frameworks
    frameworks = {f["framework_id"]: f["name"] for f in get_all_frameworks()}
    meta: dict[str, dict] = {}
    for cid, c in constructs.items():
        meta[cid] = {
            "name": c.get("name", cid),
            "framework": frameworks.get(c.get("framework_id", ""), ""),
        }
    return meta


def render() -> None:
    st.title("Similarity Detection")
    st.markdown(
        "Input your scale items to find the most similar existing constructs/scales "
        "and view item-level cross-similarity."
    )

    # Reuse from session state if available
    default_items = "\n".join(st.session_state.get("user_item_texts", []))
    if not default_items and "analysis_results" in st.session_state:
        res = st.session_state.get("analysis_results", {})
        if res.get("similarity_detect", {}).get("items"):
            default_items = "\n".join(res["similarity_detect"]["items"])

    input_mode = st.radio(
        "Input mode",
        ["Text area (one item per line)", "CSV upload"],
        horizontal=True,
    )

    items: list[str] = []
    if input_mode == "Text area (one item per line)":
        items_text = st.text_area(
            "Scale items",
            value=default_items,
            height=200,
            placeholder="Enter one item per line, e.g.:\nI am the life of the party.\nI don't talk a lot.\n...",
        )
        items = _parse_items_from_text(items_text)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            items = _parse_items_from_csv(uploaded)
        else:
            st.info("Upload a CSV with a 'text', 'item', or 'item_text' column.")

    st.subheader("Settings")
    metric = st.selectbox(
        "Similarity metric",
        ["cosine", "euclidean", "manhattan", "dot"],
        index=0,
    )
    top_n = st.slider("Top-N neighbors", min_value=1, max_value=15, value=5)

    analyze_clicked = st.button("Analyze", type="primary")

    if not items:
        st.warning("Please enter at least one item (text area or CSV).")
        return

    if len(items) < 2:
        st.warning("Enter at least 2 items for meaningful scale-level analysis.")

    if not analyze_clicked:
        return

    with st.spinner("Encoding scale and finding neighbors…"):
        try:
            scale_emb = encode_scale(items, method="mean")
            construct_embeddings = get_construct_embeddings()
            construct_metadata = _get_construct_metadata()
        except FileNotFoundError as e:
            st.error(f"Embeddings not found: {e}. Run the embedding pipeline first.")
            return
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

        if not construct_embeddings:
            st.warning("No construct embeddings available.")
            return

        neighbors = find_scale_neighbors(
            scale_emb,
            construct_embeddings,
            top_n=top_n,
            metric=metric,
        )

    st.subheader("Top-N neighbor scales")
    if not neighbors:
        st.info("No similar constructs found.")
    else:
        max_items_per_construct = 5

        def _top_items_text(cid: str) -> str:
            items_list = get_items_by_construct(cid)
            texts = [i.get("text", "") or "" for i in items_list[:max_items_per_construct]]
            truncated = [t[:70] + "…" if len(t) > 70 else t for t in texts]
            return " | ".join(truncated) if truncated else "—"

        rows = []
        for n in neighbors:
            cid = n.get("id", "")
            meta = construct_metadata.get(cid, {})
            rows.append({
                "Construct": meta.get("name", cid),
                "Similarity": f"{n.get('similarity', 0):.3f}",
                "Framework": meta.get("framework", ""),
                "Top-5 items": _top_items_text(cid),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Cross-similarity heatmap")
    if neighbors and items:
        try:
            embeddings, item_ids = load_item_embeddings()
            item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
            all_items_db = get_all_items()
            item_id_to_text = {i.get("item_id"): i.get("text", "") for i in all_items_db if i.get("item_id")}

            # User items: encode and create embeddings dict
            user_embs = encode_texts(items, use_openai=False)
            user_embeddings = {f"user_{i}": user_embs[i] for i in range(len(items))}

            # Top neighbor: pick first neighbor's items
            top_cid = neighbors[0]["id"]
            neighbor_items = get_items_by_construct(top_cid)
            corpus_item_ids = [i["item_id"] for i in neighbor_items if i["item_id"] in item_id_to_idx]
            if not corpus_item_ids:
                st.info("Top neighbor has no items with embeddings. Heatmap skipped.")
            else:
                corpus_embeddings = {
                    iid: embeddings[item_id_to_idx[iid]]
                    for iid in corpus_item_ids
                }
                user_item_dicts = [{"id": f"user_{i}"} for i in range(len(items))]
                corpus_item_dicts = [{"id": iid} for iid in corpus_item_ids]
                combined_embeddings = {**user_embeddings, **corpus_embeddings}

                matrix, row_lbls, col_lbls = compute_cross_similarity_matrix(
                    user_item_dicts,
                    corpus_item_dicts,
                    combined_embeddings,
                    metric=metric,
                )
                def _truncate(t: str, max_len: int = 50) -> str:
                    return (t[:max_len] + "…") if len(t) > max_len else t

                row_display = []
                for lbl in row_lbls:
                    if lbl.startswith("user_") and "_" in lbl:
                        try:
                            i = int(lbl.split("_")[1])
                            row_display.append(_truncate(items[i]))
                        except (ValueError, IndexError):
                            row_display.append(lbl)
                    else:
                        row_display.append(lbl)
                col_display = [
                    (item_id_to_text.get(cid, cid))[:50] + ("…" if len(item_id_to_text.get(cid, cid) or "") > 50 else "")
                    for cid in col_lbls
                ]
                fig = plot_cross_similarity_heatmap(
                    matrix,
                    row_display,
                    col_display,
                    title=f"Your items vs {construct_metadata.get(top_cid, {}).get('name', top_cid)}",
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap failed: {e}")

    # Store in session state for other pages
    st.session_state["user_item_texts"] = items
    st.session_state["user_embeddings"] = encode_texts(items, use_openai=False) if items else None
    st.session_state.setdefault("analysis_results", {})["similarity_detect"] = {
        "items": items,
        "neighbors": neighbors,
        "top_n": top_n,
        "metric": metric,
    }
    st.success("Results stored in session. Use Item Diagnosis or Internal Dimensions next.")


render()
