"""SQLite schema and data-access helpers for the unified hierarchy model.

Tables follow PRD Section 2.2.3:
  Framework -> Domain -> Construct -> Item
  with N:M Domain-Construct mapping to support PID-5 cross-loadings.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ssn.config import DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS frameworks (
    framework_id   TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    version        TEXT,
    source_url     TEXT,
    license        TEXT,
    citation       TEXT
);

CREATE TABLE IF NOT EXISTS domains (
    domain_id      TEXT PRIMARY KEY,
    framework_id   TEXT NOT NULL REFERENCES frameworks(framework_id),
    name           TEXT NOT NULL,
    original_label TEXT,          -- e.g. "Domain", "Virtue"
    description    TEXT
);

CREATE TABLE IF NOT EXISTS constructs (
    construct_id   TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    original_label TEXT,          -- e.g. "Facet", "Character Strength"
    description    TEXT,
    scale_name     TEXT,
    item_count     INTEGER,
    scale_source_url TEXT
);

CREATE TABLE IF NOT EXISTS items (
    item_id        TEXT PRIMARY KEY,
    construct_id   TEXT REFERENCES constructs(construct_id),
    text           TEXT NOT NULL,
    text_norm      TEXT,
    direction      TEXT DEFAULT '+',  -- '+' or '-'
    alpha          REAL,
    instrument     TEXT,
    framework_id   TEXT REFERENCES frameworks(framework_id)
);

CREATE TABLE IF NOT EXISTS domain_construct_map (
    domain_id      TEXT NOT NULL REFERENCES domains(domain_id),
    construct_id   TEXT NOT NULL REFERENCES constructs(construct_id),
    loading_type   TEXT DEFAULT 'primary',  -- 'primary' / 'secondary'
    loading_value  REAL,
    PRIMARY KEY (domain_id, construct_id)
);

CREATE TABLE IF NOT EXISTS cross_framework_map (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    theme          TEXT NOT NULL,   -- e.g. "Extraversion"
    framework_id_a TEXT NOT NULL REFERENCES frameworks(framework_id),
    domain_id_a    TEXT NOT NULL REFERENCES domains(domain_id),
    framework_id_b TEXT NOT NULL REFERENCES frameworks(framework_id),
    domain_id_b    TEXT NOT NULL REFERENCES domains(domain_id),
    mapping_type   TEXT NOT NULL    -- 'equivalent' / 'approximate' / 'inverse' / 'partial'
);

CREATE INDEX IF NOT EXISTS idx_items_construct ON items(construct_id);
CREATE INDEX IF NOT EXISTS idx_items_framework ON items(framework_id);
CREATE INDEX IF NOT EXISTS idx_items_instrument ON items(instrument);
CREATE INDEX IF NOT EXISTS idx_dcmap_domain ON domain_construct_map(domain_id);
CREATE INDEX IF NOT EXISTS idx_dcmap_construct ON domain_construct_map(construct_id);
CREATE INDEX IF NOT EXISTS idx_domains_framework ON domains(framework_id);
"""


def _db_path() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_connection():
    """Yield a SQLite connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(str(_db_path()))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)


def reset_db() -> None:
    """Drop and recreate all tables (destructive)."""
    db = _db_path()
    if db.exists():
        db.unlink()
    init_db()


# ---------------------------------------------------------------------------
# Generic CRUD helpers
# ---------------------------------------------------------------------------

def upsert_framework(fw: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO frameworks "
            "(framework_id, name, version, source_url, license, citation) "
            "VALUES (:framework_id, :name, :version, :source_url, :license, :citation)",
            fw,
        )


def upsert_domain(d: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO domains "
            "(domain_id, framework_id, name, original_label, description) "
            "VALUES (:domain_id, :framework_id, :name, :original_label, :description)",
            d,
        )


def upsert_construct(c: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO constructs "
            "(construct_id, name, original_label, description, scale_name, item_count, scale_source_url) "
            "VALUES (:construct_id, :name, :original_label, :description, "
            ":scale_name, :item_count, :scale_source_url)",
            c,
        )


def upsert_item(item: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO items "
            "(item_id, construct_id, text, text_norm, direction, alpha, instrument, framework_id) "
            "VALUES (:item_id, :construct_id, :text, :text_norm, :direction, :alpha, "
            ":instrument, :framework_id)",
            item,
        )


def bulk_upsert_items(items: list[dict[str, Any]]) -> None:
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO items "
            "(item_id, construct_id, text, text_norm, direction, alpha, instrument, framework_id) "
            "VALUES (:item_id, :construct_id, :text, :text_norm, :direction, :alpha, "
            ":instrument, :framework_id)",
            items,
        )


def link_domain_construct(domain_id: str, construct_id: str,
                          loading_type: str = "primary",
                          loading_value: float | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO domain_construct_map "
            "(domain_id, construct_id, loading_type, loading_value) "
            "VALUES (?, ?, ?, ?)",
            (domain_id, construct_id, loading_type, loading_value),
        )


def insert_cross_framework_mapping(
    theme: str,
    framework_id_a: str,
    domain_id_a: str,
    framework_id_b: str,
    domain_id_b: str,
    mapping_type: str = "equivalent",
) -> None:
    """Insert a cross-framework domain alignment (PRD 2.3.2)."""
    allowed = ("equivalent", "approximate", "inverse", "partial")
    if mapping_type not in allowed:
        raise ValueError(f"mapping_type must be one of {allowed}")
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO cross_framework_map "
            "(theme, framework_id_a, domain_id_a, framework_id_b, domain_id_b, mapping_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (theme, framework_id_a, domain_id_a, framework_id_b, domain_id_b, mapping_type),
        )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_all_frameworks() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM frameworks").fetchall()
        return [dict(r) for r in rows]


def get_domains_by_framework(framework_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM domains WHERE framework_id = ?", (framework_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_constructs_by_domain(domain_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT c.* FROM constructs c "
            "JOIN domain_construct_map dcm ON c.construct_id = dcm.construct_id "
            "WHERE dcm.domain_id = ?",
            (domain_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_items_by_construct(construct_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM items WHERE construct_id = ?", (construct_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_items() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM items").fetchall()
        return [dict(r) for r in rows]


def get_all_constructs() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM constructs").fetchall()
        return [dict(r) for r in rows]


def get_all_domains() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM domains").fetchall()
        return [dict(r) for r in rows]


def search_by_name(table: str, name: str, limit: int = 20) -> list[dict]:
    """Case-insensitive LIKE search on the name column."""
    allowed = {"frameworks", "domains", "constructs"}
    if table not in allowed:
        raise ValueError(f"table must be one of {allowed}")
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE name LIKE ? LIMIT ?",
            (f"%{name}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_item_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]


def get_construct_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM constructs").fetchone()[0]


def get_framework_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM frameworks").fetchone()[0]


def get_corpus_stats() -> dict:
    """Return summary statistics about the corpus."""
    with get_connection() as conn:
        return {
            "frameworks": conn.execute("SELECT COUNT(*) FROM frameworks").fetchone()[0],
            "domains": conn.execute("SELECT COUNT(*) FROM domains").fetchone()[0],
            "constructs": conn.execute("SELECT COUNT(*) FROM constructs").fetchone()[0],
            "items": conn.execute("SELECT COUNT(*) FROM items").fetchone()[0],
            "instruments": conn.execute(
                "SELECT COUNT(DISTINCT instrument) FROM items"
            ).fetchone()[0],
        }


def get_cross_framework_mappings() -> list[dict]:
    """Return all cross-framework domain alignments (PRD 2.3.2)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, theme, framework_id_a, domain_id_a, framework_id_b, domain_id_b, mapping_type "
            "FROM cross_framework_map ORDER BY theme, framework_id_a"
        ).fetchall()
        return [dict(r) for r in rows]


def get_hierarchy_tree(framework_id: str | None = None) -> list[dict]:
    """Return the full hierarchy as a nested structure for display."""
    with get_connection() as conn:
        fw_clause = "WHERE f.framework_id = ?" if framework_id else ""
        params = (framework_id,) if framework_id else ()
        rows = conn.execute(
            f"""
            SELECT f.framework_id, f.name AS framework_name,
                   d.domain_id, d.name AS domain_name,
                   c.construct_id, c.name AS construct_name,
                   COUNT(i.item_id) AS item_count
            FROM frameworks f
            LEFT JOIN domains d ON d.framework_id = f.framework_id
            LEFT JOIN domain_construct_map dcm ON dcm.domain_id = d.domain_id
            LEFT JOIN constructs c ON c.construct_id = dcm.construct_id
            LEFT JOIN items i ON i.construct_id = c.construct_id
                AND i.framework_id = f.framework_id
            {fw_clause}
            GROUP BY f.framework_id, d.domain_id, c.construct_id
            ORDER BY f.name, d.name, c.name
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]
