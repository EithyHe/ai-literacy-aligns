from src.preprocessing import run_normalize, run_qc, run_add_item_ids
from src.embeddings import run_embed
from src.contrastive import run_contrastive
from src.sanity_rev import run_sanity_rev
from src.cluster import run_cluster
from src.construct_clustering import run_construct_clustering
from src.umap_plot import run_umap
from src.anchors import run_anchors
from src.cluster_select import run_k_selection, run_cluster_apply
from src.pca_selection import run_pca_selection
from src.pca_analysis import run_pca_analysis
from src.llm_interpret import run_llm_interpretation
from src.umap_from_scores import run_umap_from_scores
from src.neighbors_pca import run_sim_from_scores
from src.hypergraph import run_hypergraph

import os
import pandas as pd
def main():
    for d in ["data/docs", "data/processed", "data/interim"]:
        os.makedirs(d, exist_ok=True)

    items_file = "data/interim/items_master.csv"
    items_csv = "data/interim/items_master.csv"
    emb_file = "data/processed/emb_base.npy"
    ids_file = "data/processed/item_ids.txt"
    ids_csv = "data/processed/item_ids.csv"

    nn_file_pca   = "data/processed/neighbors_pca.parquet"
    sparse_file_pca = "data/processed/S_pca_topk20.npz"

    pca_metrics_csv = "data/processed/pca_metrics.csv"
    pca_scree_png = "data/docs/pca_scree.png"
    pca_scores_npy = "data/processed/pca_scores.npy"
    pca_scores_csv = "data/processed/pca_scores.csv"
    pca_loadings   = "data/processed/pca_loadings.csv"
    pca_top_items  = "data/processed/pca_top_items.csv"
    interpret_csv  = "data/processed/pca_component_labels.csv"

    umap_pca_png   = "data/docs/umap_pca.png"
    umap_emb_png   = "data/docs/umap.png"
    sanity_report  = "data/docs/revkey_sanity.md"

    labels_file = "data/processed/labels.parquet"
    labels_csv  = "data/processed/labels.csv"

    anchors_file = "data/processed/anchors.npy"

    cluster_metrics_csv = "data/processed/cluster_metrics.csv"
    cluster_elbow_png = "data/docs/cluster_elbow.png"
    cluster_labels_csv = "data/processed/cluster_labels.csv"

    print("== Step 1 | Preprocess ==")
    run_add_item_ids(items_file, items_csv, ids_file, ids_csv)
    run_normalize(items_file, items_file)
    run_qc(items_file, items_file)
    print("✓ Preprocess done")

    print("== Step 2 | Embeddings ==")
    run_embed(items_file, "text_norm", emb_file, ids_file)
    print("✓ Embeddings done:", emb_file)

    out_emb = "data/processed/emb_contrastive.npy"
    out_model = "data/models/proj_head.pt"
    run_contrastive(
        emb_file=emb_file,
        ids_file=ids_file,
        items_file=items_file,
        out_emb=out_emb,
        out_model=out_model
    )

    print("== Step 3 | Reverse-key sanity ==")
    run_sanity_rev(ids_file, emb_file, items_file, sanity_report)
    print("✓ Sanity report:", sanity_report)

    print("== Step 4 | PCA Selection ==")
    pca_selection_result = run_pca_selection(
        emb_file=out_emb,
        out_metrics_csv=pca_metrics_csv,
        out_scree_png=pca_scree_png,
        max_components=min(50, len(pd.read_csv(ids_csv)) - 1),
        variance_threshold=0.95,
        min_components=2,
        random_state=42
    )
    best_n_components = pca_selection_result["best_n_components"]
    print(f"✓ Best PCA components: {best_n_components}")
    print(f"✓ PCA Selection metrics: {pca_metrics_csv}")
    print(f"✓ PCA Selection scree: {pca_scree_png}")

    print("== Step 5 | PCA → LLM → UMAP ==")
    run_pca_analysis(
        emb_file=out_emb,
        ids_file=ids_file,
        items_file=items_file,
        pca_scores_npy=pca_scores_npy,
        pca_scores_csv=pca_scores_csv,
        pca_loadings_csv=pca_loadings,
        pca_top_items_csv=pca_top_items,
        n_components=best_n_components,  # 使用选择的主成分数
        top_k=min(25, best_n_components)  # 相应调整top_k
    )
    print("✓ PCA:", pca_scores_csv)

    run_llm_interpretation(
        pca_top_items_csv=pca_top_items,
        items_file=items_file,
        out_csv=interpret_csv,
        model="gpt-4o-mini",
        temperature=0.2
    )
    print("✓ LLM labels:", interpret_csv)

    run_umap_from_scores(
        pca_scores_npy=pca_scores_npy,
        labels_csv=interpret_csv,
        out_png=umap_pca_png,
        n_neighbors=30,
        min_dist=0.1
    )
    print("✓ UMAP (PCA):", umap_pca_png)

    print("== Step 6 | kNN on PCA scores ==")
    run_sim_from_scores(
        pca_scores_npy=pca_scores_npy,
        k=20,
        out_sparse_npz=sparse_file_pca,
        out_neighbors_parquet=nn_file_pca,
        ids_file=ids_file,
        metric="cosine"
    )
    print("✓ PCA-kNN:", nn_file_pca, "|", sparse_file_pca)

    print("== Step 7 | Automatic K Selection ==")
    try:
        # 自动选择最佳k
        k_result = run_k_selection(
            pca_scores_csv=pca_scores_csv,
            out_metrics_csv=cluster_metrics_csv,
            out_elbow_png=cluster_elbow_png,
            k_min=2,
            k_max=min(10, len(pd.read_csv(pca_scores_csv)) // 2),  # 动态设置k_max
            random_state=42
        )
        best_k = k_result["best_k"]
        print(f"✓ Best k selected: {best_k}")
        print(f"✓ K selection metrics: {cluster_metrics_csv}")
        print(f"✓ Elbow plot: {cluster_elbow_png}")
        
        # 使用最佳k进行聚类
        run_cluster_apply(
            pca_scores_csv=pca_scores_csv,
            out_labels_csv=cluster_labels_csv,
            k=best_k,
            random_state=42
        )
        print(f"✓ Cluster labels (k={best_k}): {cluster_labels_csv}")
        
    except Exception as e:
        print(f"! K selection failed, using default k=3: {e}")
        # 回退到默认聚类
        try:
            run_cluster(emb_file, 3, labels_file)
            print("✓ Cluster labels (fallback):", labels_file)
        except Exception as e2:
            print(f"! Clustering failed: {e2}")

    print("== Step 8 | UMAP on embeddings (legacy) ==")
    try:
        # 尝试使用新的聚类标签
        if os.path.exists(cluster_labels_csv):
            run_umap(emb_file, cluster_labels_csv, umap_emb_png)
        else:
            run_umap(emb_file, labels_file, umap_emb_png)
    except Exception:
        run_umap(emb_file, None, umap_emb_png)
    print("✓ UMAP (embeddings):", umap_emb_png)

    print("== Step 9 | Optional: anchors ==")
    try:
        run_anchors(items_file, ids_file, emb_file, None, anchors_file)
        print("✓ Anchors:", anchors_file)
    except Exception as e:
        print("! Anchors optional, skipped:", e)

    print("\n=== Artifacts ===")
    print(f"- Embeddings: {emb_file}")
    print(f"- PCA scores: {pca_scores_csv}")
    print(f"- PCA labels: {interpret_csv}")
    print(f"- PCA-UMAP:   {umap_pca_png}")
    print(f"- kNN (PCA):  {nn_file_pca} | {sparse_file_pca}")
    print(f"- Sanity:     {sanity_report}")
    print(f"- UMAP (emb): {umap_emb_png}")
    print(f"- Cluster metrics: {cluster_metrics_csv}")
    print(f"- Cluster elbow: {cluster_elbow_png}")
    print(f"- Cluster labels: {cluster_labels_csv}")
    try:
        hg = run_hypergraph(
            items_file=items_file,
            cluster_labels_csv=cluster_labels_csv,
            out_nodes_csv="data/processed/hyper_nodes.csv",
            out_edges_csv="data/processed/hyper_edges.csv",
            construct_col="construct_reported",
        )
        print(f"- Hypergraph nodes: {hg['nodes_csv']}")
        print(f"- Hypergraph edges: {hg['edges_csv']}")
    except Exception as e:
        print(f"! Hypergraph build skipped: {e}")

if __name__ == "__main__":
    main()
