import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import load_npz

def run_cluster(emb_file: str, n_clusters: int, out_labels_path: str):
    try:
        A = load_npz("data/processed/S_topk20.npz")
    except Exception:
        A = None
    X = np.load(emb_file)
    # For scikit-learn >=1.4, use affinity parameter name change guard
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average", connectivity=A)
    labels = model.fit_predict(X)
    pd.DataFrame({"label": labels}).to_parquet(out_labels_path, index=False)
