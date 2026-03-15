"""Semantic Scale Network – Streamlit Application Entry Point.

Run with:  streamlit run ssn/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Use a broadly available Numba threading layer by default.
# Users can still override with NUMBA_THREADING_LAYER in their shell.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import streamlit as st

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ssn.config import STREAMLIT_TITLE


def _init_session_state() -> None:
    """Initialise shared session-state keys used across pages."""
    defaults = {
        "db_ready": False,
        "user_items": [],
        "user_item_texts": [],
        "user_embeddings": None,
        "exploration_history": [],
        "analysis_results": {},
        "current_stage": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _ensure_db() -> None:
    """Initialise the database and import IPIP data if not done yet."""
    if st.session_state.get("db_ready"):
        return
    try:
        from ssn.db.schema import get_item_count, init_db
        init_db()
        if get_item_count() == 0:
            with st.spinner("Importing IPIP corpus (first run only)…"):
                from ssn.db.import_ipip import import_ipip
                import_ipip()
        st.session_state["db_ready"] = True
    except FileNotFoundError as e:
        st.error(
            f"Corpus file not found: {e}. "
            "Ensure `data/interim/items_master_ipip.csv` exists, or run the data pipeline first."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialise the database: {e}")
        st.stop()


def _ensure_required_dependencies() -> None:
    """Fail fast for hard dependencies required by visualization pages."""
    try:
        import hdbscan  # noqa: F401
    except ImportError:
        st.error(
            "Missing dependency `hdbscan`. "
            "Please run: `pip install -r requirements.txt -r ssn/requirements.txt`"
        )
        st.stop()


# ── Page configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title=STREAMLIT_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_session_state()
_ensure_db()
_ensure_required_dependencies()

# ── Sidebar navigation ─────────────────────────────────────────────────────

PAGES = {
    "Home": "pages/00_home.py",
    "1 - Construct Explorer": "pages/01_construct_explorer.py",
    "2 - Similarity Detection": "pages/02_similarity_detect.py",
    "3 - Item Diagnosis": "pages/03_item_diagnosis.py",
    "4 - Internal Dimensions": "pages/04_internal_dimensions.py",
    "5 - Report Export": "pages/05_report_export.py",
    "Corpus Overview": "pages/06_corpus_overview.py",
    "Corpus Management": "pages/08_corpus_management.py",
}

page = st.navigation([
    st.Page("pages/00_home.py", title="Home", icon="🏠"),
    st.Page("pages/01_construct_explorer.py", title="Construct Explorer", icon="🔍"),
    st.Page("pages/06_corpus_overview.py", title="Corpus Overview", icon="📋"),
    st.Page("pages/02_similarity_detect.py", title="Similarity Detection", icon="📊"),
    st.Page("pages/03_item_diagnosis.py", title="Item Diagnosis", icon="🩺"),
    st.Page("pages/04_internal_dimensions.py", title="Internal Dimensions", icon="📐"),
    st.Page("pages/05_report_export.py", title="Report Export", icon="📄"),
    st.Page("pages/08_corpus_management.py", title="Corpus Management", icon="📚"),
])
page.run()
