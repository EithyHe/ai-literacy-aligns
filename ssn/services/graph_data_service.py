"""Load and cache SSN graph visualization JSON for the frontend (PRD Section 7)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ssn.config import SSN_GRAPH_JSON_PATH

logger = logging.getLogger(__name__)

_cached_graph_data: dict[str, Any] | None = None
_cached_path: Path | None = None


def get_graph_data(path: Path | None = None) -> dict[str, Any] | None:
    """Load the graph JSON from disk. Returns None if file does not exist or is invalid.

    Results are cached in memory; pass path to force a specific file or bypass cache.
    """
    global _cached_graph_data, _cached_path
    target = path or SSN_GRAPH_JSON_PATH
    if path is None and _cached_graph_data is not None and _cached_path == target:
        return _cached_graph_data
    if not target.exists():
        logger.debug(f"Graph JSON not found: {target}")
        return None
    try:
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        if path is None:
            _cached_graph_data = data
            _cached_path = target
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load graph JSON from {target}: {e}")
        return None


def clear_graph_cache() -> None:
    """Clear the in-memory cache (e.g. after pipeline regenerates the file)."""
    global _cached_graph_data, _cached_path
    _cached_graph_data = None
    _cached_path = None
