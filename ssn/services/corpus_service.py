"""Corpus management service (PRD FR-1.4.1–1.4.5).

Provides corpus statistics, hierarchy browsing, search, and submission handling.
Wraps schema and adds convenience methods for the Corpus Management page.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_corpus_stats() -> dict[str, int]:
    """Return summary statistics: frameworks, domains, constructs, items, instruments."""
    from ssn.db.schema import get_corpus_stats as _get
    return _get()


def get_hierarchy_tree(framework_id: str | None = None) -> list[dict]:
    """Return the full hierarchy as a flat list of rows (framework, domain, construct, item_count)."""
    from ssn.db.schema import get_hierarchy_tree as _get
    return _get(framework_id)


def search_corpus(table: str, name_query: str, limit: int = 50) -> list[dict]:
    """Search frameworks, domains, or constructs by name (case-insensitive LIKE)."""
    from ssn.db.schema import search_by_name
    return search_by_name(table, name_query, limit=limit)


def get_frameworks() -> list[dict]:
    """Return all frameworks."""
    from ssn.db.schema import get_all_frameworks
    return get_all_frameworks()


def get_domains(framework_id: str | None = None) -> list[dict]:
    """Return domains, optionally filtered by framework."""
    from ssn.db.schema import get_all_domains, get_domains_by_framework
    if framework_id:
        return get_domains_by_framework(framework_id)
    return get_all_domains()


def get_constructs(domain_id: str | None = None, framework_id: str | None = None) -> list[dict]:
    """Return constructs, optionally filtered by domain or framework."""
    from ssn.db.schema import get_all_constructs, get_constructs_by_domain, get_constructs_by_framework_id
    if domain_id:
        return get_constructs_by_domain(domain_id)
    if framework_id:
        return get_constructs_by_framework_id(framework_id)
    return get_all_constructs()


def get_items(construct_id: str | None = None) -> list[dict]:
    """Return items, optionally filtered by construct."""
    from ssn.db.schema import get_all_items, get_items_by_construct
    if construct_id:
        return get_items_by_construct(construct_id)
    return get_all_items()


def submit_new_scale(
    scale_name: str,
    construct_name: str,
    framework: str,
    items_text: str,
) -> dict[str, Any]:
    """Record a new scale submission for review (stub: no persistence beyond session).

    PRD FR-1.4.5: submissions will be reviewed. This returns a confirmation payload.
    """
    items = [ln.strip() for ln in items_text.strip().splitlines() if ln.strip()]
    return {
        "scale_name": scale_name,
        "construct_name": construct_name,
        "framework": framework,
        "item_count": len(items),
        "status": "submitted_for_review",
        "message": "Thank you for your submission. It will be reviewed by the team.",
    }
