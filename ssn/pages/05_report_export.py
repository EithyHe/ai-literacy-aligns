"""Report Export (PRD Stage 5, FR-4.4.1-4.4.5) – Generate standardized semantic analysis report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from ssn.db.schema import get_corpus_stats
from ssn.services.report_service import (
    generate_full_report,
    export_to_word,
    export_to_pdf,
)


def _collect_session_summary() -> dict:
    """Collect analysis results from session state."""
    return {
        "embedding_model": st.session_state.get("embedding_model", "MPNet (dwulff/mpnet-personality)"),
        "corpus_size": st.session_state.get("corpus_size"),
        "similarity_method": st.session_state.get("similarity_method", "cosine"),
        "neighbor_comparison_table": st.session_state.get("neighbor_comparison_table"),
        "heatmap_image": st.session_state.get("heatmap_image"),
        "per_item_diagnosis": st.session_state.get("per_item_diagnosis"),
        "construct_positioning_map": st.session_state.get("construct_positioning_map"),
        "density_analysis_results": st.session_state.get("density_analysis_results"),
        "internal_dimension_exploration": st.session_state.get("internal_dimension_exploration"),
    }


def _neighbors_from_session() -> list[dict] | None:
    """Build neighbors list for report from analysis_results.similarity_detect."""
    ar = st.session_state.get("analysis_results", {})
    sim = ar.get("similarity_detect", {})
    neighbors = sim.get("neighbors")
    if not neighbors:
        return None
    from ssn.db.schema import get_all_constructs, get_all_frameworks, get_constructs_by_domain, get_all_domains
    constructs = {c["construct_id"]: c for c in get_all_constructs()}
    frameworks = {f["framework_id"]: f["name"] for f in get_all_frameworks()}
    construct_to_fw = {}
    for d in get_all_domains():
        for c in get_constructs_by_domain(d["domain_id"]):
            if c.get("construct_id"):
                construct_to_fw[c["construct_id"]] = frameworks.get(d.get("framework_id", ""), "")
    result = []
    for n in neighbors:
        cid = n.get("id", "")
        result.append({
            "name": constructs.get(cid, {}).get("name", cid),
            "framework": construct_to_fw.get(cid, ""),
            "similarity": n.get("similarity", 0),
        })
    return result


def _diagnosis_from_session() -> dict | None:
    """Get diagnosis from analysis_results.item_diagnosis."""
    return st.session_state.get("analysis_results", {}).get("item_diagnosis")


def _build_markdown_report(data: dict) -> str:
    """Build markdown report from session + report_service."""
    stats = get_corpus_stats()
    neighbors = _neighbors_from_session()
    diagnosis = _diagnosis_from_session()
    return generate_full_report(
        corpus_stats=stats,
        neighbors=neighbors or [],
        diagnosis=diagnosis,
        density_results=data.get("density_analysis_results"),
        dimension_clusters=data.get("internal_dimension_exploration"),
        embedding_model="dwulff/mpnet-personality",
        similarity_metric=data.get("similarity_method", "cosine"),
    )


def render() -> None:
    st.title("Report Export")
    st.markdown("Generate a standardized semantic analysis report from your session results.")

    data = _collect_session_summary()

    # 1. Summary of analysis results
    st.subheader("Session Summary")
    with st.expander("View collected analysis results", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Embedding Model", data.get("embedding_model", "N/A")[:40] + ("…" if len(str(data.get("embedding_model", ""))) > 40 else ""))
            st.metric("Similarity Method", data.get("similarity_method", "cosine"))
        with col2:
            stats = get_corpus_stats()
            st.metric("Corpus Items", stats.get("items", "N/A"))
            st.metric("Corpus Constructs", stats.get("constructs", "N/A"))

        has_data = any([
            _neighbors_from_session() is not None,
            data.get("heatmap_image") is not None,
            _diagnosis_from_session() is not None,
            data.get("construct_positioning_map") is not None,
            data.get("density_analysis_results") is not None,
            data.get("internal_dimension_exploration") is not None,
        ])
        if not has_data:
            st.info("Run analyses on previous pages (Similarity Detection, Item Diagnosis, etc.) to populate the report.")

    # 2. Generate report
    if st.button("Generate Report", type="primary"):
        try:
            report_md = _build_markdown_report(data)
            st.session_state["generated_report"] = report_md
            st.success("Report generated successfully.")
        except Exception as e:
            st.error(f"Failed to generate report: {e}")

    generated = st.session_state.get("generated_report")
    if not generated:
        st.subheader("Report Preview")
        st.caption("Generate a report to see the preview and download options.")
        return

    # 3. Download: Markdown, Word, PDF
    st.subheader("Download")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="Download Markdown (.md)",
            data=generated,
            file_name="semantic_analysis_report.md",
            mime="text/markdown",
        )
    with c2:
        try:
            word_bytes = export_to_word(generated)
            st.download_button(
                label="Download Word (.docx)",
                data=word_bytes,
                file_name="semantic_analysis_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_word",
            )
        except ImportError:
            st.caption("Word: install python-docx")
    with c3:
        try:
            pdf_bytes = export_to_pdf(generated)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="semantic_analysis_report.pdf",
                mime="application/pdf",
                key="dl_pdf",
            )
        except ImportError:
            st.caption("PDF: install weasyprint")

    # 4. Preview
    st.subheader("Report Preview")
    st.markdown("---")
    st.markdown(generated)


render()
