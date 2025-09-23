import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import NearestNeighbors

def run_sim_from_scores(
    pca_scores_npy: str,
    k: int,
    out_sparse_npz: str,
    out_neighbors_parquet: str,
    ids_file: str = None,
    metric: str = "cosine"
):
    X = np.load(pca_scores_npy)  # [N, d]
    n = X.shape[0]
    if k >= n:
        k = max(1, n - 1)

    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nbrs.fit(X)
    dists, inds = nbrs.kneighbors(X, return_distance=True)  # includes self at column 0

    # Strip self-neighbor (distance=0)
    dists = dists[:, 1:]
    inds  = inds[:, 1:]

    # Build CSR
    rows = np.repeat(np.arange(n), k)
    cols = inds.reshape(-1)
    data = dists.reshape(-1)

    if metric == "cosine":
        sim = 1.0 - data  # cosine distance -> similarity
    else:
        sim = 1.0 / (1e-9 + data)

    # Row-normalize similarities
    sim_mat = sim.reshape(n, k)
    row_sums = sim_mat.sum(axis=1, keepdims=True) + 1e-9
    sim_norm = (sim_mat / row_sums).reshape(-1)

    S = csr_matrix((sim_norm, (rows, cols)), shape=(n, n))
    save_npz(out_sparse_npz, S)

    # Build neighbors table
    if ids_file is not None:
        with open(ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        assert len(ids) == n, "ids_file length does not match PCA scores"
        src = np.repeat(ids, k)
        dst = [ids[i] for i in cols]
    else:
        src = rows
        dst = cols

    df = pd.DataFrame({
        "src": src,
        "dst": dst,
        "dist": data,
        "sim": sim_norm
    })
    df.to_parquet(out_neighbors_parquet, index=False)
    return out_neighbors_parquet
