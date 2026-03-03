"""Home page – overview dashboard and corpus statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ssn.db.schema import get_corpus_stats, get_all_frameworks


def render() -> None:
    st.title("Semantic Scale Network")
    st.markdown(
        "An **LLM-based** upgrade of the Semantic Scale Network for detecting "
        "semantic overlap between psychological scales, exploring construct spaces, "
        "and supporting scale development."
    )

    try:
        stats = get_corpus_stats()
    except Exception as e:
        st.error(f"Could not load corpus statistics: {e}")
        stats = {"frameworks": 0, "domains": 0, "constructs": 0, "items": 0, "instruments": 0}

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Frameworks", stats["frameworks"])
    col2.metric("Domains", stats["domains"])
    col3.metric("Constructs", stats["constructs"])
    col4.metric("Items", stats["items"])
    col5.metric("Instruments", stats["instruments"])

    st.divider()

    st.subheader("Scale Development Workflow")
    st.markdown("""
    | Stage | Description | Page |
    |-------|------------|------|
    | **1. Construct Explorer** | Describe your construct in natural language, explore the semantic landscape | Construct Explorer |
    | **2. Similarity Detection** | Input your scale items, find most similar existing scales | Similarity Detection |
    | **3. Item Diagnosis** | Get per-item redundancy diagnosis and modification suggestions | Item Diagnosis |
    | **4. Internal Dimensions** | Discover sub-dimensions within your scale items | Internal Dimensions |
    | **5. Report Export** | Generate a standardized report for your paper | Report Export |
    """)

    st.divider()

    st.subheader("Analysis Tools")
    st.markdown("""
    | Tool | Description |
    |------|------------|
    | **Construct Network** | Visualize construct association networks with topology metrics |
    | **Density Analysis** | Spatial density analysis of the construct space |
    | **Corpus Management** | Browse the corpus, view hierarchy, cross-framework domain mapping, submit new scales |
    """)

    st.divider()

    with st.expander("Corpus Frameworks", expanded=False):
        try:
            frameworks = get_all_frameworks()
        except Exception:
            frameworks = []
        for fw in frameworks:
            st.markdown(f"- **{fw['name']}** ({fw['framework_id']})")


render()
