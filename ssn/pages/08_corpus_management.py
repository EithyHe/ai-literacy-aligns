"""Corpus Management (PRD FR-1.4.1-1.4.5) – Browse, search, and manage the corpus."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from ssn.db.schema import (
    get_corpus_stats,
    get_all_frameworks,
    get_domains_by_framework,
    get_constructs_by_domain,
    get_items_by_construct,
    get_hierarchy_tree,
    search_by_name,
    get_cross_framework_mappings,
    get_all_domains,
)


@st.cache_data
def _get_items_per_framework() -> list[dict]:
    """Items per framework for bar chart."""
    frameworks = get_all_frameworks()
    result = []
    for fw in frameworks:
        domains = get_domains_by_framework(fw["framework_id"])
        total = 0
        for d in domains:
            constructs = get_constructs_by_domain(d["domain_id"])
            for c in constructs:
                items = get_items_by_construct(c["construct_id"])
                total += len(items)
        result.append({"Framework": fw["name"], "Items": total})
    return result


@st.cache_data(ttl=60)
def _get_cross_framework_mappings_cached() -> list[dict]:
    """Cached cross-framework mappings for the Sankey viz."""
    return get_cross_framework_mappings()


@st.cache_data
def _get_items_per_domain() -> list[dict]:
    """Items per domain for distribution."""
    from ssn.db.schema import get_all_domains

    domains = get_all_domains()
    result = []
    for d in domains:
        constructs = get_constructs_by_domain(d["domain_id"])
        total = sum(len(get_items_by_construct(c["construct_id"])) for c in constructs)
        result.append({"Domain": d["name"], "Items": total})
    return result


def render() -> None:
    st.title("Corpus Management")
    st.markdown("Browse, search, and manage the Semantic Scale Network corpus.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Browse Hierarchy", "Search", "Statistics", "Submit New Scale", "Cross-Framework Mapping"
    ])

    with tab1:
        st.subheader("Browse Hierarchy")
        frameworks = get_all_frameworks()
        if not frameworks:
            st.info("No frameworks in the corpus.")
        else:
            fw_options = [f["name"] for f in frameworks]
            fw_ids = [f["framework_id"] for f in frameworks]
            fw_idx = st.selectbox("Framework", range(len(fw_options)), format_func=lambda i: fw_options[i])
            framework_id = fw_ids[fw_idx] if fw_idx is not None else None

            if framework_id:
                tree = get_hierarchy_tree(framework_id)
                # Build domain -> constructs mapping
                domain_constructs: dict[str, list[dict]] = {}
                for row in tree:
                    d_id = row.get("domain_id")
                    d_name = row.get("domain_name", d_id or "Unknown")
                    c_id = row.get("construct_id")
                    c_name = row.get("construct_name", c_id or "Unknown")
                    icount = row.get("item_count", 0)
                    if not d_id or not c_id:
                        continue
                    key = (d_id, d_name)
                    if key not in domain_constructs:
                        domain_constructs[key] = []
                    domain_constructs[key].append({"construct_id": c_id, "name": c_name, "item_count": icount})

                for (d_id, d_name), constructs in domain_constructs.items():
                    with st.expander(f"**{d_name}** (Domain)", expanded=False):
                        for c in constructs:
                            with st.expander(f"{c['name']} ({c['item_count']} items)", expanded=False):
                                items = get_items_by_construct(c["construct_id"])
                                for it in items[:15]:
                                    text = it.get("text", it.get("item_id", ""))
                                    st.caption(f"• {text[:80]}{'...' if len(text) > 80 else ''}")
                                if len(items) > 15:
                                    st.caption(f"... and {len(items) - 15} more items")

    with tab2:
        st.subheader("Search")
        query = st.text_input("Search", placeholder="Enter search term...")
        scope_options = ["Frameworks", "Domains", "Constructs"]
        scope_map = {"Frameworks": "frameworks", "Domains": "domains", "Constructs": "constructs"}
        scope = st.radio("Search scope", scope_options, horizontal=True)
        table = scope_map[scope]

        if query.strip():
            try:
                results = search_by_name(table, query.strip(), limit=50)
            except Exception as e:
                st.error(f"Search failed: {e}")
                results = []

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No results found.")
        else:
            st.caption("Enter a search term to find frameworks, domains, or constructs.")

    with tab3:
        st.subheader("Statistics")
        stats = get_corpus_stats()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Frameworks", stats["frameworks"])
        col2.metric("Domains", stats["domains"])
        col3.metric("Constructs", stats["constructs"])
        col4.metric("Items", stats["items"])
        col5.metric("Instruments", stats["instruments"])

        items_per_fw = _get_items_per_framework()
        if items_per_fw:
            df_fw = pd.DataFrame(items_per_fw)
            fig = px.bar(df_fw, x="Framework", y="Items", title="Items per Framework")
            st.plotly_chart(fig, use_container_width=True)

        items_per_dom = _get_items_per_domain()
        if items_per_dom:
            df_dom = pd.DataFrame(items_per_dom)
            fig2 = px.bar(df_dom, x="Domain", y="Items", title="Items per Domain")
            st.plotly_chart(fig2, use_container_width=True)

        st.caption("Similarity distribution: run similarity analyses on other pages to view.")

    with tab4:
        st.subheader("Submit New Scale")
        st.info("Submissions will be reviewed and added to the corpus.")

        with st.form("submit_scale_form"):
            scale_name = st.text_input("Scale name", placeholder="e.g. My New Scale")
            construct_name = st.text_input("Construct name", placeholder="e.g. Openness to Experience")
            framework = st.text_input("Framework", placeholder="e.g. Big Five")
            items_text = st.text_area("Items (one per line)", placeholder="Item 1\nItem 2\nItem 3")
            submitted = st.form_submit_button("Submit")

            if submitted:
                st.success("Thank you for your submission! It will be reviewed by the team.")

    with tab5:
        st.subheader("Cross-Framework Domain Mapping")
        st.markdown(
            "Domain alignments across frameworks (e.g. IPIP-NEO Extraversion ≈ HEXACO eXtraversion). "
            "Use this to compare constructs across instruments (PRD FR-DS-3)."
        )
        try:
            mappings = _get_cross_framework_mappings_cached()
        except Exception as e:
            st.error(f"Failed to load cross-framework mappings: {e}")
            mappings = []

        if mappings:
            # Build domain_id -> display name for labels
            all_domains = get_all_domains()
            domain_names = {d["domain_id"]: d["name"] for d in all_domains}
            frameworks = {f["framework_id"]: f["name"] for f in get_all_frameworks()}

            # Sankey: left = Framework A (name + domain), right = Framework B (name + domain)
            node_labels: list[str] = []
            node_to_idx: dict[str, int] = {}
            links_src: list[int] = []
            links_tgt: list[int] = []
            links_val: list[int] = []

            for m in mappings:
                da = m.get("domain_id_a", "")
                db = m.get("domain_id_b", "")
                fa = m.get("framework_id_a", "")
                fb = m.get("framework_id_b", "")
                theme = m.get("theme", "")
                fw_a_name = frameworks.get(fa, fa)
                fw_b_name = frameworks.get(fb, fb)
                d_a_name = domain_names.get(da, da)
                d_b_name = domain_names.get(db, db)
                left_label = f"{fw_a_name}: {d_a_name}"
                right_label = f"{fw_b_name}: {d_b_name}"
                if left_label not in node_to_idx:
                    node_to_idx[left_label] = len(node_labels)
                    node_labels.append(left_label)
                if right_label not in node_to_idx:
                    node_to_idx[right_label] = len(node_labels)
                    node_labels.append(right_label)
                links_src.append(node_to_idx[left_label])
                links_tgt.append(node_to_idx[right_label])
                links_val.append(1)

            if node_labels and links_src:
                import plotly.graph_objects as go
                fig = go.Figure(
                    data=[
                        go.Sankey(
                            node=dict(label=node_labels),
                            link=dict(
                                source=links_src,
                                target=links_tgt,
                                value=links_val,
                            ),
                        )
                    ]
                )
                fig.update_layout(
                    title="Cross-framework domain alignments",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(pd.DataFrame(mappings), use_container_width=True, hide_index=True)
        else:
            st.info(
                "No cross-framework mappings loaded. Run the IPIP import to seed NEO–HEXACO and NEO–BFAS alignments, "
                "or add rows to the cross_framework_map table for your analyses."
            )


render()
