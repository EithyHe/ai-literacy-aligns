import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def _safe_read_ids(ids_file: str) -> list:
    with open(ids_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def run_pca_analysis(
    emb_file: str,
    ids_file: str,
    items_file: str = "data/interim/items_master.parquet",
    pca_scores_npy: str = "data/processed/pca_scores.npy",
    pca_scores_csv: str = "data/processed/pca_scores.csv",
    pca_loadings_csv: str = "data/processed/pca_loadings.csv",
    pca_top_items_csv: str = "data/processed/pca_top_items.csv",
    n_components: int = 5,
    top_k: int = 5
):
    X = np.load(emb_file)
    ids = _safe_read_ids(ids_file)
    assert len(ids) == X.shape[0], "ids_file and emb_file row counts differ"

    n_components = min(n_components, X.shape[1])
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    scores = pca.fit_transform(X)
    comps  = pca.components_.T
    expvar = pca.explained_variance_ratio_

    np.save(pca_scores_npy, scores)
    df_scores = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    df_scores.insert(0, "item_id", ids)
    df_scores.to_csv(pca_scores_csv, index=False, encoding="utf-8")

    df_load = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    df_load.to_csv(pca_loadings_csv, index=False, encoding="utf-8")

    try:
        items = pd.read_parquet(items_file)
    except Exception:
        try:
            items = pd.read_csv(items_file)
        except Exception:
            items = pd.DataFrame({"item_id": ids, "text": ""})
    candidate_text_cols = ["text_norm","text","item_text","stem"]
    text_col = next((c for c in candidate_text_cols if c in items.columns), None)
    if text_col is None:
        items[text_col := "text"] = ""
    key_cols = ["item_id", text_col]
    if "construct" in items.columns:
        key_cols.append("construct")
    view = items[key_cols].copy()

    tops = []
    for j in range(scores.shape[1]):
        comp_name = f"PC{j+1}"
        order = np.argsort(-np.abs(scores[:, j]))
        pick = order[:top_k]
        df_top = pd.DataFrame({
            "component": comp_name,
            "rank": np.arange(1, len(pick)+1),
            "item_id": [ids[i] for i in pick],
            "score": scores[pick, j]
        })
        tops.append(df_top)
    df_top_all = pd.concat(tops, ignore_index=True)
    df_top_all = df_top_all.merge(view, on="item_id", how="left")
    df_top_all["explained_variance_ratio"] = df_top_all["component"].map(
        {f"PC{i+1}": float(expvar[i]) for i in range(len(expvar))}
    )
    df_top_all.to_csv(pca_top_items_csv, index=False, encoding="utf-8")
