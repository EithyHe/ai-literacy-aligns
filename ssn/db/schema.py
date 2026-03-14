"""SQLite schema and data-access helpers for the unified hierarchy model.

Primary path: Framework -> Construct -> Item (constructs.framework_id NOT NULL).
Optional: Domain is only for frameworks with theoretical domains (e.g. IPIP-NEO, HEXACO, VIA, BFAS);
  domain_construct_map links constructs to domains where defined. No synthetic General/Other domains.
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
    construct_id     TEXT PRIMARY KEY,
    framework_id     TEXT NOT NULL REFERENCES frameworks(framework_id),
    name             TEXT NOT NULL,
    original_label   TEXT,          -- e.g. "Facet", "Character Strength"
    description      TEXT,
    scale_name       TEXT,
    item_count       INTEGER,
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

-- Leiden (data-driven) clustering: separate from theory hierarchy to avoid polluting frameworks/domains.
-- One run = one partition of constructs into communities; labels come from LLM interpretation.
CREATE TABLE IF NOT EXISTS leiden_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,     -- ISO 8601
    resolution   REAL,              -- Leiden resolution param (if used)
    random_seed  INTEGER,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS leiden_communities (
    run_id       INTEGER NOT NULL REFERENCES leiden_runs(run_id) ON DELETE CASCADE,
    community_id INTEGER NOT NULL,  -- 0, 1, 2, ... from algorithm
    label        TEXT,              -- LLM-generated name
    rationale    TEXT,
    PRIMARY KEY (run_id, community_id)
);

CREATE TABLE IF NOT EXISTS leiden_construct_membership (
    run_id       INTEGER NOT NULL REFERENCES leiden_runs(run_id) ON DELETE CASCADE,
    construct_id TEXT NOT NULL REFERENCES constructs(construct_id),
    community_id INTEGER NOT NULL,
    PRIMARY KEY (run_id, construct_id),
    FOREIGN KEY (run_id, community_id) REFERENCES leiden_communities(run_id, community_id)
);

CREATE INDEX IF NOT EXISTS idx_items_construct ON items(construct_id);
CREATE INDEX IF NOT EXISTS idx_items_framework ON items(framework_id);
CREATE INDEX IF NOT EXISTS idx_items_instrument ON items(instrument);
CREATE INDEX IF NOT EXISTS idx_dcmap_domain ON domain_construct_map(domain_id);
CREATE INDEX IF NOT EXISTS idx_dcmap_construct ON domain_construct_map(construct_id);
CREATE INDEX IF NOT EXISTS idx_domains_framework ON domains(framework_id);
CREATE INDEX IF NOT EXISTS idx_constructs_framework ON constructs(framework_id);
CREATE INDEX IF NOT EXISTS idx_leiden_communities_run ON leiden_communities(run_id);
CREATE INDEX IF NOT EXISTS idx_leiden_membership_run ON leiden_construct_membership(run_id);
CREATE INDEX IF NOT EXISTS idx_leiden_membership_construct ON leiden_construct_membership(construct_id);
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


def _migrate_leiden_tables(conn: sqlite3.Connection) -> None:
    """Create Leiden tables if missing (for DBs created before Leiden tables were in SCHEMA_SQL)."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='leiden_runs'"
    )
    if cur.fetchone() is not None:
        return
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS leiden_runs (
            run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT NOT NULL,
            resolution   REAL,
            random_seed  INTEGER,
            note         TEXT
        );
        CREATE TABLE IF NOT EXISTS leiden_communities (
            run_id       INTEGER NOT NULL REFERENCES leiden_runs(run_id) ON DELETE CASCADE,
            community_id INTEGER NOT NULL,
            label        TEXT,
            rationale    TEXT,
            PRIMARY KEY (run_id, community_id)
        );
        CREATE TABLE IF NOT EXISTS leiden_construct_membership (
            run_id       INTEGER NOT NULL REFERENCES leiden_runs(run_id) ON DELETE CASCADE,
            construct_id TEXT NOT NULL REFERENCES constructs(construct_id),
            community_id INTEGER NOT NULL,
            PRIMARY KEY (run_id, construct_id)
        );
        CREATE INDEX IF NOT EXISTS idx_leiden_communities_run ON leiden_communities(run_id);
        CREATE INDEX IF NOT EXISTS idx_leiden_membership_run ON leiden_construct_membership(run_id);
        CREATE INDEX IF NOT EXISTS idx_leiden_membership_construct ON leiden_construct_membership(construct_id);
    """)


def init_db() -> None:
    """Create all tables if they don't exist; run migrations for existing DBs."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        _migrate_leiden_tables(conn)


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
            "(construct_id, name, original_label, description, scale_name, item_count, scale_source_url, framework_id) "
            "VALUES (:construct_id, :name, :original_label, :description, "
            ":scale_name, :item_count, :scale_source_url, :framework_id)",
            {
                **c,
                "framework_id": c.get("framework_id"),
            },
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
# Leiden (data-driven clustering) – dedicated tables, not mixed with frameworks/domains
# ---------------------------------------------------------------------------

def leiden_insert_run(
    created_at: str,
    resolution: float | None = None,
    random_seed: int | None = None,
    note: str | None = None,
) -> int:
    """Insert a new Leiden run; returns run_id."""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO leiden_runs (created_at, resolution, random_seed, note) VALUES (?, ?, ?, ?)",
            (created_at, resolution, random_seed, note),
        )
        return cur.lastrowid


def leiden_insert_communities(
    run_id: int,
    communities: list[tuple[int, str | None, str | None]],
) -> None:
    """Insert (community_id, label, rationale) for a run. Replaces any existing for this run."""
    with get_connection() as conn:
        conn.execute("DELETE FROM leiden_communities WHERE run_id = ?", (run_id,))
        for community_id, label, rationale in communities:
            conn.execute(
                "INSERT INTO leiden_communities (run_id, community_id, label, rationale) VALUES (?, ?, ?, ?)",
                (run_id, community_id, label or None, rationale or None),
            )


def leiden_insert_membership(
    run_id: int,
    construct_id: str,
    community_id: int,
) -> None:
    """Assign one construct to a community in a run."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO leiden_construct_membership (run_id, construct_id, community_id) VALUES (?, ?, ?)",
            (run_id, construct_id, community_id),
        )


def leiden_insert_membership_bulk(run_id: int, construct_to_community: dict[str, int]) -> None:
    """Assign all construct_id -> community_id for a run. Replaces any existing membership for this run."""
    with get_connection() as conn:
        conn.execute("DELETE FROM leiden_construct_membership WHERE run_id = ?", (run_id,))
        conn.executemany(
            "INSERT INTO leiden_construct_membership (run_id, construct_id, community_id) VALUES (?, ?, ?)",
            [(run_id, cid, comm_id) for cid, comm_id in construct_to_community.items()],
        )


def leiden_get_latest_run_id() -> int | None:
    """Return the most recent run_id, or None if no runs exist."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT run_id FROM leiden_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None


def leiden_get_run(run_id: int) -> dict | None:
    """Return a single run row or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM leiden_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


def leiden_get_construct_to_community(run_id: int) -> dict[str, int]:
    """Return construct_id -> community_id for the given run."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT construct_id, community_id FROM leiden_construct_membership WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {r["construct_id"]: r["community_id"] for r in rows}


def leiden_get_communities_with_labels(run_id: int) -> list[dict]:
    """Return list of {community_id, label, rationale} for the run, ordered by community_id."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT community_id, label, rationale FROM leiden_communities WHERE run_id = ? ORDER BY community_id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def leiden_delete_run(run_id: int) -> None:
    """Delete a run and its communities/membership (CASCADE)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM leiden_runs WHERE run_id = ?", (run_id,))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_all_frameworks() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM frameworks").fetchall()
        return [dict(r) for r in rows]


# Domain names to exclude from reads (case-sensitive: "General", "Other")
_EXCLUDED_DOMAIN_NAMES = ("General", "Other")


def get_domains_by_framework(framework_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM domains WHERE framework_id = ?", (framework_id,)
        ).fetchall()
        result = [dict(r) for r in rows]
    return [d for d in result if d.get("name") not in _EXCLUDED_DOMAIN_NAMES]


def get_constructs_by_domain(domain_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT c.* FROM constructs c "
            "JOIN domain_construct_map dcm ON c.construct_id = dcm.construct_id "
            "WHERE dcm.domain_id = ?",
            (domain_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_construct_to_primary_domain() -> dict[str, str]:
    """Return mapping construct_id -> domain_id (primary domain, or first if no primary)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT construct_id, domain_id, loading_type FROM domain_construct_map "
            "ORDER BY CASE WHEN loading_type = 'primary' THEN 0 ELSE 1 END, domain_id"
        ).fetchall()
    result: dict[str, str] = {}
    for r in rows:
        cid = r["construct_id"]
        if cid not in result:
            result[cid] = r["domain_id"]
    return result


def get_construct_to_primary_domain_for_framework(framework_id: str) -> dict[str, str]:
    """Return construct_id -> domain_id for domains in the given framework only (primary loading)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT dcm.construct_id, dcm.domain_id, dcm.loading_type "
            "FROM domain_construct_map dcm "
            "JOIN domains d ON d.domain_id = dcm.domain_id AND d.framework_id = ? "
            "ORDER BY CASE WHEN dcm.loading_type = 'primary' THEN 0 ELSE 1 END, dcm.domain_id",
            (framework_id,),
        ).fetchall()
    result: dict[str, str] = {}
    for r in rows:
        cid = r["construct_id"]
        if cid not in result:
            result[cid] = r["domain_id"]
    return result


def delete_domains_by_framework(framework_id: str) -> None:
    """Remove all domains for a framework and their domain_construct_map entries (for full replace)."""
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM domain_construct_map WHERE domain_id IN "
            "(SELECT domain_id FROM domains WHERE framework_id = ?)",
            (framework_id,),
        )
        conn.execute("DELETE FROM domains WHERE framework_id = ?", (framework_id,))


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


def get_constructs_by_framework(framework_id: str) -> list[dict]:
    """Return constructs that belong to this framework (direct framework_id)."""
    return get_constructs_by_framework_id(framework_id)


def get_constructs_by_framework_id(framework_id: str) -> list[dict]:
    """Return constructs for the given framework (direct query on constructs.framework_id)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM constructs WHERE framework_id = ?", (framework_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_framework_id_for_construct(construct_id: str) -> str | None:
    """Return framework_id for a construct (direct column; domain fallback for legacy DBs)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT framework_id FROM constructs WHERE construct_id = ?",
            (construct_id,),
        ).fetchone()
        if row and row[0]:
            return row[0]
        row = conn.execute(
            "SELECT d.framework_id FROM domain_construct_map dcm "
            "JOIN domains d ON d.domain_id = dcm.domain_id "
            "WHERE dcm.construct_id = ? "
            "ORDER BY CASE WHEN dcm.loading_type = 'primary' THEN 0 ELSE 1 END LIMIT 1",
            (construct_id,),
        ).fetchone()
        return row[0] if row else None


def get_all_domains() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM domains").fetchall()
        result = [dict(r) for r in rows]
    return [d for d in result if d.get("name") not in _EXCLUDED_DOMAIN_NAMES]


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
    """Return summary statistics about the corpus. Domains count excludes synthetic General/Other."""
    with get_connection() as conn:
        domain_count = conn.execute(
            "SELECT COUNT(*) FROM domains WHERE name NOT IN (?, ?)",
            _EXCLUDED_DOMAIN_NAMES,
        ).fetchone()[0]
        return {
            "frameworks": conn.execute("SELECT COUNT(*) FROM frameworks").fetchone()[0],
            "domains": domain_count,
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
    """Return the full hierarchy: framework -> (optional domain) -> construct -> item_count."""
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
            JOIN constructs c ON c.framework_id = f.framework_id
            LEFT JOIN domain_construct_map dcm ON dcm.construct_id = c.construct_id
            LEFT JOIN domains d ON d.domain_id = dcm.domain_id
            LEFT JOIN items i ON i.construct_id = c.construct_id
                AND i.framework_id = f.framework_id
            {fw_clause}
            GROUP BY f.framework_id, d.domain_id, c.construct_id
            ORDER BY f.name, d.name, c.name
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]
