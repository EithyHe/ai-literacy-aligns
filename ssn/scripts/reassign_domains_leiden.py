"""Standalone script to reassign domains via Leiden and LLM naming (no full graph pipeline).

Run from project root:
  python -m ssn.scripts.reassign_domains_leiden

Requires: construct embeddings (from DB + item embeddings), OPENAI_API_KEY for LLM labels.
Writes: data/processed/leiden_domain_construct_map.csv, leiden_domain_llm_labels.csv.
Persists: framework leiden_derived, domains leiden_0, leiden_1, ..., domain_construct_map.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ssn.config import PROCESSED_DIR
from ssn.services.graph_data_pipeline import run_leiden_domain_reassign

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reassign domains with Leiden and LLM naming")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for CSV artifacts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Representative items per community for LLM prompt",
    )
    args = parser.parse_args()
    communities = run_leiden_domain_reassign(processed_dir=args.processed_dir, top_k=args.top_k)
    logger.info("Done. %s communities reassigned.", len(communities))


if __name__ == "__main__":
    main()
