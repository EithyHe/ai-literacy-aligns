"""Item Diagnosis (PRD Stage 3, FR-4.2.1–4.2.5).

Per-item redundancy diagnosis with modification suggestions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np

from ssn.services.embedding_service import encode_texts, load_item_embeddings
from ssn.services.diagnosis_service import (
    diagnose_scale,
    generate_modification_suggestion,
    find_redundant_pairs,
)
from ssn.db.schema import get_all_items
from ssn.components.heatmap_viz import plot_cross_similarity_heatmap


def _parse_items(text: str) -> list[str]:
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


def render() -> None:
    st.title("Item Diagnosis")
    st.markdown(
        "Diagnose per-item redundancy against the corpus. Get risk levels, "
        "top similar items, and modification suggestions."
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
        st.subheader("Thresholds")
        high_thresh = st.slider(
            "High risk threshold (similarity ≥)",
            min_value=0.70,
            max_value=0.98,
            value=0.85,
            step=0.01,
        )
        medium_thresh = st.slider(
            "Medium risk threshold (similarity ≥)",
            min_value=0.50,
            max_value=0.90,
            value=0.65,
            step=0.01,
        )
        if medium_thresh >= high_thresh:
            st.warning("Medium threshold should be lower than high.")

    diagnose_clicked = st.button("Diagnose", type="primary")

    if not items:
        st.warning("Please enter at least one item.")
        return

    if not diagnose_clicked:
        return

    with st.spinner("Encoding items and diagnosing…"):
        try:
            item_embeddings = encode_texts(items, use_openai=False)
            embeddings, item_ids = load_item_embeddings()
            all_items_db = get_all_items()
            corpus_ids = [i["item_id"] for i in all_items_db if i["item_id"] in item_ids]
            item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
            corpus_indices = [item_id_to_idx[iid] for iid in corpus_ids]
            corpus_embeddings = embeddings[corpus_indices]
            corpus_texts = [
                next((x["text"] for x in all_items_db if x.get("item_id") == iid), "")
                for iid in corpus_ids
            ]
        except FileNotFoundError as e:
            st.error(f"Embeddings not found: {e}. Run the embedding pipeline first.")
            return
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

        thresholds = {"high": high_thresh, "medium": medium_thresh}
        result = diagnose_scale(
            item_embeddings,
            items,
            corpus_embeddings,
            corpus_ids,
            corpus_texts,
            top_k=3,
            thresholds=thresholds,
        )

    summary = result["summary"]
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total items", summary["total"])
    c2.metric("High risk", summary["high_risk"], delta=None)
    c3.metric("Medium risk", summary["medium_risk"], delta=None)
    c4.metric("Low risk", summary["low_risk"], delta=None)

    st.subheader("Per-item diagnosis")
    risk_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    for diag in result["items"]:
        risk = diag["risk_level"]
        badge = risk_colors.get(risk, "⚪")
        with st.expander(f"{badge} {risk.upper()}: {diag['item_text'][:60]}{'…' if len(diag['item_text']) > 60 else ''}"):
            st.write("**Item:**", diag["item_text"])
            st.write("**Max similarity:**", f"{diag['max_similarity']:.3f}")
            st.write("**Top similar corpus items:**")
            for m in diag.get("top_matches", []):
                st.write(f"- {m.get('text', m.get('id', ''))} (sim: {m.get('similarity', 0):.3f})")
            suggestion = generate_modification_suggestion(
                diag["item_text"],
                diag.get("top_matches", []),
            )
            st.write("**Modification suggestion:**")
            st.info(suggestion)

    st.subheader("Redundant pairs within scale")
    pairs = find_redundant_pairs(
        item_embeddings,
        items,
        threshold=high_thresh,
    )
    if not pairs:
        st.info("No highly redundant pairs found within your scale.")
    else:
        for p in pairs:
            st.write(f"- **{p['item_a'][:50]}…** ↔ **{p['item_b'][:50]}…** (sim: {p['similarity']:.3f})")

    # Store for Report Export page
    st.session_state.setdefault("analysis_results", {})["item_diagnosis"] = result


render()
