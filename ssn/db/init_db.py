"""Database initialization script for the Semantic Scale Network.

Run from project root:
  python -m ssn.db.init_db
  python -m ssn.db.init_db --import-ipip   # also import IPIP corpus
  python -m ssn.db.init_db --reset          # drop and recreate, then import
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ssn.db.schema import init_db, reset_db, get_item_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize SSN SQLite database")
    parser.add_argument(
        "--import-ipip",
        action="store_true",
        help="Import IPIP corpus after creating schema",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables (destructive). Implies --import-ipip if used alone.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_db()
        print("Database reset (tables dropped and recreated).")
        args.import_ipip = True
    else:
        init_db()
        print("Database initialized (tables created if not exist).")

    if args.import_ipip:
        from ssn.db.import_ipip import import_ipip
        try:
            stats = import_ipip()
            print(f"IPIP import complete: {stats}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Import failed: {e}", file=sys.stderr)
            return 1
    else:
        n = get_item_count()
        print(f"Current item count: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
