"""CLI: build offline visualization cache for pages 04 and 06.

Run from project root:
  python -m ssn.scripts.build_visualization_cache
"""

from __future__ import annotations

import argparse
import json
import warnings

from ssn.services.visualization_cache_service import VIS_CACHE_DIR, build_visualization_cache

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*divide by zero encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden to 1 by setting random_state.*", category=UserWarning)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline visualization cache for SSN 04/06 pages.")
    parser.add_argument("--min-weight", type=float, default=0.25, help="Construct network min similarity edge threshold.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Disparity backbone alpha.")
    parser.add_argument("--umap-neighbors", type=int, default=20, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.10, help="UMAP min_dist.")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=20, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--hdbscan-min-samples", type=int, default=0, help="HDBSCAN min_samples (0 means auto).")
    parser.add_argument("--auto-tune-hdbscan", action="store_true", help="Auto-search HDBSCAN params.")
    parser.add_argument("--target-min-clusters", type=int, default=10, help="Target minimum cluster count for auto-tuning.")
    parser.add_argument("--target-max-clusters", type=int, default=15, help="Target maximum cluster count for auto-tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--skip-llm-labels",
        action="store_true",
        help="Skip LLM naming and keep default cluster labels (faster, no API calls).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    min_samples = None if int(args.hdbscan_min_samples) == 0 else int(args.hdbscan_min_samples)
    manifest = build_visualization_cache(
        output_dir=VIS_CACHE_DIR,
        min_weight=float(args.min_weight),
        alpha=float(args.alpha),
        umap_n_neighbors=int(args.umap_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        hdbscan_min_cluster_size=int(args.hdbscan_min_cluster_size),
        hdbscan_min_samples=min_samples,
        auto_tune_hdbscan=bool(args.auto_tune_hdbscan),
        target_min_clusters=int(args.target_min_clusters),
        target_max_clusters=int(args.target_max_clusters),
        seed=int(args.seed),
        use_llm_labels=not bool(args.skip_llm_labels),
    )
    print(json.dumps({"cache_dir": str(VIS_CACHE_DIR), "stats": manifest.get("stats", {})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
